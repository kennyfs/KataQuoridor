#!/bin/bash -eu
# Q4 run1 settings on top of q4_formal_run.sh. Every value can still be overridden from the environment.
#   python/selfplay/q4_run1.sh [BASEDIR]     (default BASEDIR: /home/kenny/q4/run1)
BASEDIR="${1:-/home/kenny/q4/run1}"
HERE="$(dirname "$(realpath "$0")")"
export KATAGO_BIN="${KATAGO_BIN:-$(realpath "$HERE/../../cpp/katago")}"   # the binary built in cpp/ (cmake . && make)

export MAX_CYCLES="${MAX_CYCLES:-1}"
export START_AT="${START_AT:-shuffle}"
export SHUFFLE_MINROWS="${SHUFFLE_MINROWS:-400000}"      # the data on disk has 432579 rows
export NUM_GAMES_PER_CYCLE="${NUM_GAMES_PER_CYCLE:-100000}"
export SELFPLAY_EXTRA_ARGS="${SELFPLAY_EXTRA_ARGS--max-rows-total 150000 -override-config maxVisits=600,cheapSearchVisits=100}"
export MAX_TRAIN_PER_DATA="${MAX_TRAIN_PER_DATA:-6}"
export MAX_TRAIN_SAMPLES_PER_CYCLE="${MAX_TRAIN_SAMPLES_PER_CYCLE:-1500000}"
LR_SCHEDULE="${LR_SCHEDULE:-(0,0.1),(125K,0.143),(250K,0.2),(375K,0.286),(500K,0.4),(625K,0.667),(750K,1.0),(875K,1.43),(1M,2.0)}"
export TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS--use-aurora -wd-floor-frac 0.5 -no-lr-warmup -lr-schedule $LR_SCHEDULE -max-val-samples 10000 -initial-train-bucket-level 600000}"

mkdir -p "$BASEDIR"
"$HERE"/q4_formal_run.sh "$BASEDIR" q4f q4f 2>&1 | tee -a "$BASEDIR"/whole.log
