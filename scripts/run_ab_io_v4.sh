#!/bin/bash -eu
# A/B of Quoridor I/O v4 (docs/QuoridorIOv4.md) on the home GPU. Fixed data, one starting checkpoint, equal samples:
#   A  control   : v4 net, wall-effect inputs zeroed, draws mapped back to win 0.5 / loss 0.5   (= the v3 net)
#   B  wall effects on, draw target as A
#   C  draw target on (noResult), wall-effect inputs zeroed
#   D  B + C (optional)
# Start: run4's checkpoint (~/q1_run/run4/train/run4/checkpoint.ckpt), migrated to v4 (bias -8 for the noResult logits).
# Data: one fresh shuffle of run4's window (the loop's settings, all ~1.9M rows kept), converted to v4.
# Each arm trains SAMPLES_PER_ARM (default 2M) samples at the loop's settings (Aurora, lr x2, batch 256), as two train.py
# instances of 10 epochs x 100k samples (about 40 min each at ~430 samples/s on the RTX 3070), so no process runs for
# more than an hour; the second part resumes from the first part's checkpoint.
#
# Usage: scripts/run_ab_io_v4.sh [STEP ...]     (default: prep A B C eval)
#   prep            shuffle, convert, migrate (a few minutes, CPU)
#   A | B | C | D   train that arm (both parts, ~80 min) and export its SWA net to ab_v4/models
#   A1 A2 ...       one part of an arm (~40 min); X_A ... exports an arm after its two parts
#   eval            held-out losses (S = the start net, A, B, C, D) + blind-spot suite
#   ladder          kq_ladder with the standard openings (v256), arms + the start net
# Run from anywhere; needs the venv's python and a CUDA katago (KATAGO_BIN, default cpp/build-cuda/katago).
export PATH=/home/kenny/ml_venv/bin:$PATH
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=$REPO/python
KATAGO_BIN=${KATAGO_BIN:-$REPO/cpp/build-cuda/katago}
R4=${R4:-$HOME/q1_run/run4}
A=${AB:-$HOME/q1_run/ab_v4}
MODELKIND=tf3_b5c256_quoridor_v4
STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(prep A B C eval)
mkdir -p "$A"/{logs,models,start}
log() { echo "$(date '+%F %T') $*" | tee -a "$A"/log.txt; }

prep() {
  [ -s "$A"/v4/v3/train.json ] && { echo "prep exists"; return; }
  cd "$PY"
  # the loop's shuffle settings (continue_loop.sh / synchronous_loop.sh), run on run4's own selfplay path so that the
  # md5 validation split is the loop's (1% of the files), with every window row kept
  C="-expand-window-per-row 0.3 -taper-window-exponent 0.65 -taper-window-scale 500000 -min-rows 563962 -approx-rows-per-out-file 70000 -num-processes 12 -keep-target-rows 2000000"
  mkdir -p "$A"/shuffleddata/v3 "$A"/scratch/train "$A"/scratch/val
  python shuffle.py "$R4"/selfplay/ $C -out-dir "$A"/shuffleddata/v3/val -out-tmp-dir "$A"/scratch/val \
    -only-include-md5-path-prop-lbound 0.99 -only-include-md5-path-prop-ubound 1.00 > "$A"/logs/shuffle_val.txt 2>&1
  python shuffle.py "$R4"/selfplay/ $C -out-dir "$A"/shuffleddata/v3/train -out-tmp-dir "$A"/scratch/train \
    -only-include-md5-path-prop-lbound 0.00 -only-include-md5-path-prop-ubound 0.99 > "$A"/logs/shuffle_train.txt 2>&1
  python quoridor_convert_tdata_v3_to_v4.py -katago "$KATAGO_BIN" -num-processes 12 -out-dir "$A"/v4 "$A"/shuffleddata/v3
  # the fixed unseen sets of ~/q1_run/analysis/eval_sets.py, converted
  python - <<PYEOF
import json, os, shutil
d = json.load(open(os.path.expanduser('~/q1_run/analysis/fixed_val_files.json')))
for k, v in d.items():
    os.makedirs('$A/fixed3/' + k, exist_ok=True)
    for f in v: shutil.copy2(f, '$A/fixed3/' + k + '/' + os.path.basename(f))
PYEOF
  python quoridor_convert_tdata_v3_to_v4.py -katago "$KATAGO_BIN" -out-dir "$A"/fixed4 "$A"/fixed3/old "$A"/fixed3/new "$A"/fixed3/late
  cp "$R4"/train/run4/checkpoint.ckpt "$A"/start/ckpt_v3.ckpt
  python quoridor_upgrade_v3_to_v4.py -checkpoint "$A"/start/ckpt_v3.ckpt -output "$A"/start/ckpt_v4.ckpt
  log "prep done"
}

arm_flags() {  # $1 = arm
  case $1 in
    A) echo "-zero-wall-effect -quoridor-draw-target half" ;;
    B) echo "-quoridor-draw-target half" ;;
    C) echo "-zero-wall-effect -quoridor-draw-target noresult" ;;
    D) echo "-quoridor-draw-target noresult" ;;
  esac
}

part() {  # $1 = arm, $2 = part (1|2)
  local arm=$1 N=ab$1 B=$A/$1
  if [ ! -e "$B"/train/$N/checkpoint.ckpt ]; then
    mkdir -p "$B"/train/$N "$B"/logs
    cp "$A"/start/ckpt_v4.ckpt "$B"/train/$N/checkpoint.ckpt
    mkdir -p "$B"/shuffleddata
    cp -al "$A"/v4/v3 "$B"/shuffleddata/v4
  fi
  log "$arm part $2 start"
  cd "$PY"
  ./selfplay/train.sh "$B" $N $MODELKIND 256 main -samples-per-epoch 100000 \
    -swa-period-samples 80000 -quit-if-no-data -export-only-at-end -max-epochs-this-instance ${EPOCHS:-10} -max-val-samples 10000 \
    -use-aurora -wd-floor-frac 0.5 -lr-scale 2 -no-lr-warmup $(arm_flags $arm) > "$B"/logs/train_part$2.txt 2>&1
  log "$arm part $2 done"
}

arm() { part $1 1; part $1 2; export_arm $1; }  # both parts and the export

export_arm() {
  local B=$A/$1
  cd "$PY"
  ./selfplay/export_model_for_selfplay.sh abv4-$1 "$B" 0 >> "$B"/logs/export.txt 2>&1
  for m in "$B"/models/*; do ln -sfn "$m" "$A"/models/ab$1-$(basename "$m"); done
  log "$1 exported: $(ls "$B"/models)"
}

evals() {
  cd "$PY"
  local out=$A/eval_losses.jsonl; : > $out
  for arm in S A B C D; do
    if [ $arm = S ]; then ck=$A/start/ckpt_v4.ckpt; else ck=$A/$arm/train/ab$arm/checkpoint.ckpt; fi
    [ -e "$ck" ] || continue
    z="-zero-wall-effect"; [ $arm = B ] || [ $arm = D ] && z=""
    python quoridor_eval_sets_v4.py "$ck" 100000 $z -swa val="$A"/v4/v3/val old="$A"/fixed4/old new="$A"/fixed4/new late="$A"/fixed4/late \
      | python -c "import sys,json
for l in sys.stdin:
    d=json.loads(l); d['arm']='$arm'; print(json.dumps(d))" >> $out
  done
  python - <<PYEOF
import json
rows=[json.loads(l) for l in open('$out')]
print("%-4s %-5s %7s %8s %7s %8s %8s %7s" % ("arm","set","rows","KL","pacc1","vhalf","vloss","lead"))
for r in rows: print("%-4s %-5s %7d %8.4f %7.4f %8.4f %8.4f %7.4f" % (r['arm'],r['name'],r['rows'],r['kl'],r['pacc1'],r['vhalfloss'],r['vloss'],r['leadloss']))
PYEOF
}

blindspot() {  # the newest labelled suite, v256 / v1000, gtp params, one net at a time
  cd "$PY"
  local S=$HOME/arena/blindspot2/job_eval/out/suite_v2.jsonl O=$A/blindspot; mkdir -p $O
  local nets="start:$R4/models/run4-s28602368-d9798994"
  for arm in A B C D; do m=$(ls -dt "$A"/$arm/models/* 2>/dev/null | head -1); [ -n "$m" ] && nets="$nets $arm:$m"; done  # the newest export of each arm
  for nm in $nets; do
    python -m quoridor_arena.blindspot eval --suite "$S" --model "${nm#*:}/model.bin.gz" --net "${nm%%:*}" \
      --config "$REPO"/cpp/configs/gtp_quoridor.cfg --katago "$KATAGO_BIN" --visits 256,1000 --params gtp \
      --engines 4 --out-dir "$O" --log-dir "$O"/logs
  done
  python -m quoridor_arena.blindspot report "$O"/eval_*.json > "$O"/report.md
  cat "$O"/report.md
}

ladder() {
  python3 - <<PYEOF
import json, glob, os
c = json.load(open('$HOME/arena/kq_std2/config.json'))
A = '$A'
ser = [{"name": "start", "prefix": "s", "models": ["$R4/models/run4-s28602368-d9798994"]}]
for arm in "ABCD":
    ms = sorted(glob.glob(f'{A}/{arm}/models/*'), key=os.path.getmtime)[-1:]  # the newest export
    if ms: ser.append({"name": arm, "prefix": arm.lower(), "models": ms})
c['series'] = ser
c['reference'] = "s-s28602368-v256"
c['openings'] = {"file": os.path.expanduser("~/arena/openings/std_run4_s187-s204_p2468.json")}  # as in kq_std2
c.pop('compare_ladder', None)
c['links'] = {"generation_distances": [], "cross": False}
c['katago'] = '$KATAGO_BIN'
c['extra_pairs'] = [{"a": f"start:latest@256", "b": f"{s['name']}:latest@256", "games": 230} for s in ser[1:]] + \
    [{"a": f"{x['name']}:latest@256", "b": f"{y['name']}:latest@256", "games": 230} for i, x in enumerate(ser[1:]) for y in ser[2+i:]]
json.dump(c, open(A + '/ladder_config.json', 'w'), indent=1)
PYEOF
  cd "$PY"
  python -m quoridor_arena.kq_ladder run --config "$A"/ladder_config.json --out "$A"/ladder > "$A"/ladder.log 2>&1
  log "ladder done"
}

for s in "${STEPS[@]}"; do
  case $s in
    prep) prep ;; A|B|C|D) arm $s ;; X_A|X_B|X_C|X_D) export_arm ${s:2:1} ;; A1|B1|C1|D1) part ${s:0:1} 1 ;; A2|B2|C2|D2|A3|B3|C3|D3) part ${s:0:1} ${s:1:1} ;;
    losses) evals ;; blindspot) blindspot ;; eval) evals; blindspot ;; ladder) ladder ;; *) echo "unknown step $s"; exit 1 ;;
  esac
done
