#!/bin/bash
# Home side of the remote self-play workers (README.md): every few minutes, pulls the workers' finished games and
# training rows from the workstation share into the run's selfplay dir (where shuffle picks them up), and publishes
# the run's newest net to the share (workers load it by themselves).
#   python/colab/share_sync.sh BASEDIR                 (runs until killed; log in $BASEDIR/share_sync.log)
#   SHARE=ws4:tmp2/kq_remote/share INTERVAL=120 ...  (defaults)
# One-off: publish the binary bundle and the selfplay config with publish_share.sh.
# $BASEDIR/remote_rows.txt keeps a running count of the training rows pulled; the loop's self-play counts its increase
# during self-play towards the cycle's rows (selfplay -extra-rows-file, synchronous_loop.sh ROWS_PER_CYCLE). Rows that
# arrive while the loop shuffles, trains or is stopped are found on disk by rows_since_shuffle.py at the next cycle.
export PATH=/home/kenny/ml_venv/bin:$PATH   # count_rows needs numpy
BASEDIR="$(realpath "$1")"
SHARE="${SHARE:-ws4:tmp2/kq_remote/share}"
INTERVAL="${INTERVAL:-120}"
HOSTPART="${SHARE%%:*}"; PATHPART="${SHARE#*:}"
exec 9> "$BASEDIR/share_sync.lock"
flock -n 9 || { echo "a share_sync.sh for $BASEDIR is already running"; exit 1; }
LOG="$BASEDIR/share_sync.log"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
COUNT="$BASEDIR/remote_rows.txt"
[[ -f "$COUNT" ]] || echo 0 > "$COUNT"
# rows in the .npz files listed on stdin (paths relative to $BASEDIR/selfplay), read from the array headers only
count_rows() {
  python3 -c '
import sys, zipfile, numpy as np
n = 0
for f in sys.stdin.read().split():
    if not f.endswith(".npz"): continue
    try:
        with zipfile.ZipFile(sys.argv[1] + "/" + f) as z, z.open([m for m in z.namelist() if m.startswith("binaryInputNCHWPacked")][0]) as a:
            v = np.lib.format.read_magic(a)
            h = np.lib.format.read_array_header_1_0(a) if v == (1, 0) else np.lib.format.read_array_header_2_0(a)
            n += h[0][0]
    except Exception as e:
        print("cannot count", f, e, file=sys.stderr)
print(n)' "$BASEDIR/selfplay"
}
STAGE="$BASEDIR/share_models_stage"
published=""
while true; do
  # 1. workers' data: incoming/<worker>/<model>/{tdata,sgfs}/... -> selfplay/<model>/{tdata,sgfs}/...
  for w in $(ssh -o ConnectTimeout=20 "$HOSTPART" "ls $PATHPART/incoming 2>/dev/null"); do
    files=$(rsync -a --timeout=120 --remove-source-files --out-format='%n' "$SHARE/incoming/$w/" "$BASEDIR/selfplay/" 2>>"$LOG" | grep -v '/$')
    if [[ -n "$files" ]]; then
      rows=$(count_rows <<< "$files" 2>>"$LOG")
      total=$(( $(cat "$COUNT") + ${rows:-0} ))
      echo "$total" > "$COUNT.tmp" && mv "$COUNT.tmp" "$COUNT"
      log "pulled $(wc -l <<< "$files") files, $rows rows from $w (total $total)"
    fi
  done
  # rsync --remove-source-files leaves the directories: drop the empty ones untouched for 10 minutes (a worker's
  # rsync recreates what it needs; the age keeps clear of one that is just being created)
  ssh -o ConnectTimeout=20 "$HOSTPART" "find $PATHPART/incoming -mindepth 2 -type d -empty -mmin +10 -delete" 2>>"$LOG"
  # 2. the newest net (the newest dir in models/), when it changed
  newest=$(ls -t "$BASEDIR/models" 2>/dev/null | head -1)
  if [[ -n "$newest" && "$newest" != "$published" && -f "$BASEDIR/models/$newest/model.bin.gz" ]]; then
    rm -rf "$STAGE" && mkdir -p "$STAGE/$newest" && cp "$BASEDIR/models/$newest/model.bin.gz" "$STAGE/$newest/"
    if rsync -a --timeout=120 --delete "$STAGE/" "$SHARE/models/" 2>>"$LOG"; then
      published="$newest"; log "published $newest"
    fi
  fi
  sleep "$INTERVAL"
done
