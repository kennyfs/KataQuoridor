#!/usr/bin/env bash
# Quoridor I/O v2 time-bonus (lambda = timeBonusPerPly) experiment, docs/QuoridorIOv2.md section 5.
#
# Trains one small v2 net per lambda from scratch, round-robin: every run does one synchronous_loop.sh cycle
# (gatekeeper, self-play, shuffle, train, export) in turn, so all runs have made the same progress whenever you stop.
# Stops after HOURS of wall time (a cycle in progress is finished first). Rerunning resumes all runs.
#
#   run0: lambda 0      run1: lambda 0.05      run2: lambda 0.15
#
# Usage (from anywhere):  bash scripts/lambda_experiment.sh
# Env:
#   BASE=~/q1_run          where run0..run2 live
#   HOURS=7                wall-time budget
#   MODELKIND=b2c64_quoridor_v2
#   LAMBDAS="0 0.05 0.15"
#   KATAGO_BIN=<repo>/cpp/katago   must be built from this repo's HEAD (checked below; CUDA Release recommended)
#   NUM_GAMES_PER_CYCLE=2000, MAX_TRAIN_SAMPLES_PER_CYCLE=250000, ...  (passed through to synchronous_loop.sh)
#
# Afterwards, compare per run: <run>/selfplay/stdout.txt "Quoridor stats" lines (draw rate, avg plies, Black win
# rate by komi), gatekeepersgf/stdout.txt, train/<run>/metrics_train.json; then an arena ladder of the final nets
# (not run here).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE:-$HOME/q1_run}"
HOURS="${HOURS:-7}"
MODELKIND="${MODELKIND:-b2c64_quoridor_v2}"
LAMBDAS=(${LAMBDAS:-0 0.05 0.15})
export KATAGO_BIN="${KATAGO_BIN:-$REPO/cpp/katago}"

# Smaller cycles than the default loop, so a few hours give each run ~10+ generations.
export NUM_GAMES_PER_CYCLE="${NUM_GAMES_PER_CYCLE:-2000}"
export MAX_TRAIN_SAMPLES_PER_CYCLE="${MAX_TRAIN_SAMPLES_PER_CYCLE:-250000}"
export SHUFFLE_MINROWS="${SHUFFLE_MINROWS:-50000}"
export SHUFFLE_KEEPROWS="${SHUFFLE_KEEPROWS:-300000}"
export TAPER_WINDOW_SCALE="${TAPER_WINDOW_SCALE:-250000}"
export NUM_TRAIN_SAMPLES_PER_EPOCH="${NUM_TRAIN_SAMPLES_PER_EPOCH:-50000}"
export NUM_TRAIN_SAMPLES_PER_SWA="${NUM_TRAIN_SAMPLES_PER_SWA:-40000}"
export MAX_CYCLES=1

# The binary must support I/O v2 (built from a commit containing 0c702aa3).
[[ -x "$KATAGO_BIN" ]] || { echo "No katago at $KATAGO_BIN (build it, or set KATAGO_BIN)"; exit 1; }
BIN_SHA="$("$KATAGO_BIN" version | sed -n 's/.*git \([0-9a-f]\{8\}\).*/\1/p' | head -1)"
if [[ -z "$BIN_SHA" ]] || ! git -C "$REPO" merge-base --is-ancestor 0c702aa3 "$BIN_SHA" 2>/dev/null; then
  echo "$KATAGO_BIN is not built from a commit with Quoridor I/O v2 (version: $("$KATAGO_BIN" version | head -1))"
  exit 1
fi
if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]; then
  echo "Warning: $REPO has uncommitted changes; they are archived in each run's scripts/dated/*/diff.txt"
fi

# Per-run configs: the v2 configs with this run's lambda (self-play and gatekeeper must agree).
mkdir -p "$BASE"
for i in "${!LAMBDAS[@]}"; do
  RUN="$BASE/run$i"
  mkdir -p "$RUN/configs"
  for c in selfplay gatekeeper; do
    sed -E "s/^timeBonusPerPly *=.*/timeBonusPerPly = ${LAMBDAS[$i]}  # lambda experiment run$i/" \
      "$REPO/cpp/configs/training/${c}_quoridor_v2.cfg" > "$RUN/configs/$c.cfg"
    grep -q "^timeBonusPerPly = ${LAMBDAS[$i]} " "$RUN/configs/$c.cfg" || { echo "failed to set lambda in $RUN/configs/$c.cfg"; exit 1; }
  done
  echo "run$i: lambda ${LAMBDAS[$i]}" > "$RUN/LAMBDA.txt"
done

END=$(( $(date +%s) + $(awk "BEGIN{print int($HOURS*3600)}") ))
ROUND=0
cd "$REPO/python/selfplay"
while [[ $(date +%s) -lt $END ]]; do
  ROUND=$((ROUND + 1))
  for i in "${!LAMBDAS[@]}"; do
    [[ $(date +%s) -lt $END ]] || break
    RUN="$BASE/run$i"
    echo "=== $(date '+%F %T') round $ROUND: run$i (lambda ${LAMBDAS[$i]}) ===" | tee -a "$BASE/experiment.log"
    SELFPLAY_CONFIG="$RUN/configs/selfplay.cfg" GATING_CONFIG="$RUN/configs/gatekeeper.cfg" \
      bash ./synchronous_loop.sh "q1-run$i" "$RUN" "run$i" "$MODELKIND" 1 >> "$RUN/loop.log" 2>&1 \
      || { echo "run$i cycle failed, see $RUN/loop.log" | tee -a "$BASE/experiment.log"; exit 1; }
    grep "Quoridor stats" "$RUN/selfplay/stdout.txt" 2>/dev/null | tail -1 | tee -a "$BASE/experiment.log" || true
  done
done
echo "=== $(date '+%F %T') done after $ROUND rounds ===" | tee -a "$BASE/experiment.log"
