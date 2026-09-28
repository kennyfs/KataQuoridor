#!/usr/bin/env bash
# CUDA backend acceptance check for KataQuoridor (docs/KataQuoridor_Review_and_Roadmap.md §5 Phase 1 step 5).
#
#   1. Builds katago with -DUSE_BACKEND=CUDA, RelWithDebInfo, into cpp/build-cuda (incremental).
#   2. Runs python/tests/test_nn_parity.py against that binary:
#        - FP32, NCHW   (cudaUseFP16=false, cudaUseNHWC=false), tolerance 1e-4
#        - FP32, NHWC   (cudaUseFP16=false, cudaUseNHWC=true),  tolerance 1e-4
#        - FP16         (cudaUseFP16=true),                      tolerance 2e-2
#      Each covers a conv and a transformer net at symmetries 0 and 1.
#   3. Plays 60 plies of GTP genmove with numSearchThreads=4 (restarting the game if it ends early),
#      using the conv net exported by the parity test.
#   4. Prints a PASS/FAIL summary; exit status is 0 only if everything passed.
#
# Usage (from anywhere):  scripts/cuda_parity.sh
# Env: SKIP_BUILD=1 to reuse an existing cpp/build-cuda/katago, KATAGO_BIN to use another binary,
#      JOBS=N for make parallelism, CUDA_PARITY_OUT=dir to keep all intermediate files and logs.

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO/cpp/build-cuda"
OUT="${CUDA_PARITY_OUT:-$(mktemp -d -t kq-cuda-parity-XXXXXX)}"
mkdir -p "$OUT"
JOBS="${JOBS:-$(nproc)}"

declare -a NAMES RESULTS
record() { NAMES+=("$1"); RESULTS+=("$2"); echo "==> $1: $2"; }

# 1. Build
if [ -z "${KATAGO_BIN:-}" ] && [ -z "${SKIP_BUILD:-}" ]; then
  echo "==> Building CUDA RelWithDebInfo in $BUILD_DIR (log: $OUT/build.log)"
  mkdir -p "$BUILD_DIR"
  if (cd "$BUILD_DIR" && cmake .. -DUSE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNO_GIT_REVISION=1 \
        && make -j"$JOBS") >"$OUT/build.log" 2>&1; then
    record "build (CUDA, RelWithDebInfo)" PASS
  else
    tail -n 30 "$OUT/build.log"
    record "build (CUDA, RelWithDebInfo)" FAIL
  fi
fi
KATAGO_BIN="${KATAGO_BIN:-$BUILD_DIR/katago}"
export KATAGO_BIN

if [ ! -x "$KATAGO_BIN" ]; then
  record "katago binary at $KATAGO_BIN" FAIL
else
  # 2. NN parity, FP32 (NCHW and NHWC) and FP16
  run_parity() {  # label fp16 nhwc tol
    local label="$1" cfg="$OUT/parity_$1.cfg" dir="$OUT/parity_$1"
    printf 'cudaUseFP16 = %s\ncudaUseNHWC = %s\ncudaDeviceToUse = 0\n' "$2" "$3" > "$cfg"
    echo "==> NN parity $label (tolerance $4; log: $OUT/parity_$label.log)"
    if (cd "$REPO/python" && NN_PARITY_CONFIG="$cfg" NN_PARITY_TOL="$4" NN_PARITY_OUTDIR="$dir" \
          python -m pytest -q -rA tests/test_nn_parity.py) >"$OUT/parity_$label.log" 2>&1; then
      record "NN parity $label (tol $4)" PASS
    else
      grep -E "max_|rows_|FAILED|Error|error" "$OUT/parity_$label.log" | tail -n 30
      record "NN parity $label (tol $4)" FAIL
    fi
    grep -E "max_policy_diff|max_value_diff|max_misc_reldiff" "$OUT/parity_$label.log" | sort | tail -n 3 | sed 's/^/     worst /'
  }
  run_parity fp32_nchw false false 1e-4
  run_parity fp32_nhwc false true 1e-4
  run_parity fp16 true auto 2e-2

  # 3. 60 plies of genmove with 4 search threads
  MODEL="$OUT/parity_fp32_nchw/b2c64_quoridor/model.bin.gz"
  echo "==> genmove: 60 plies, numSearchThreads=4 (log: $OUT/genmove.log)"
  if [ ! -f "$MODEL" ]; then
    record "genmove 60 plies, 4 threads" "FAIL (no exported model at $MODEL)"
  elif python3 - "$KATAGO_BIN" "$MODEL" "$REPO/cpp/configs/gtp_example.cfg" >"$OUT/genmove.log" 2>&1 <<'EOF'
import subprocess, sys
katago, model, cfg = sys.argv[1:4]
overrides = ("numSearchThreads=4,maxVisits=200,rules=quoridor,logToStderr=false,logAllGTPCommunication=false,"
             "ponderingEnabled=false,cudaUseFP16=false,cudaDeviceToUse=0")
p = subprocess.Popen([katago, "gtp", "-model", model, "-config", cfg, "-override-config", overrides],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def gtp(cmd):
    p.stdin.write(cmd + "\n"); p.stdin.flush()
    lines = []
    while True:
        line = p.stdout.readline()
        if line == "":
            raise SystemExit(f"katago exited during '{cmd}' (code {p.wait()})")
        if line.strip() == "" and lines:
            break
        if line.strip():
            lines.append(line.strip())
    resp = " ".join(lines)
    if not resp.startswith("="):
        raise SystemExit(f"GTP error on '{cmd}': {resp}")
    return resp[1:].strip()
gtp("clear_board")
plies, games, color = 0, 1, "b"
while plies < 60:
    move = gtp(f"genmove {color}")
    plies += 1
    print(f"ply {plies} game {games} {color} {move}")
    if move.lower() == "resign" or gtp("winner") not in ("", "none"):
        print(f"game {games} ended after ply {plies}")
        gtp("clear_board"); games += 1; color = "b"
    else:
        color = "w" if color == "b" else "b"
gtp("quit")
p.wait()
print(f"OK: {plies} plies over {games} game(s)")
EOF
  then
    record "genmove 60 plies, 4 threads" "PASS ($(tail -n 1 "$OUT/genmove.log"))"
  else
    tail -n 15 "$OUT/genmove.log"
    record "genmove 60 plies, 4 threads" FAIL
  fi
fi

# 4. Summary
echo
echo "================ KataQuoridor CUDA parity summary ================"
fail=0
for i in "${!NAMES[@]}"; do
  printf '  %-40s %s\n' "${NAMES[$i]}" "${RESULTS[$i]}"
  case "${RESULTS[$i]}" in PASS*) ;; *) fail=1 ;; esac
done
echo "  logs and intermediate files: $OUT"
if [ "$fail" = 0 ]; then echo "OVERALL: PASS"; else echo "OVERALL: FAIL"; fi
exit "$fail"
