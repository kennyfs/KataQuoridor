#!/bin/bash -eu
set -o pipefail
{

# The Q4 formal run (Plan §15, docs/q4/rounds/R8.md): q4_synchronous_loop.sh with the settings fixed in round 8, plus
# an evaluation every EVAL_EVERY exports (q4_formal_eval.sh) and a per-cycle summary, both run after each export.
#
#   python/selfplay/q4_formal_run.sh BASEDIR [NAMEPREFIX] [TRAININGNAME]
#
# Every knob below can be overridden from the environment (the smoke test uses a b1c32 net and tiny cycles).

if [[ $# -lt 1 ]]
then
    echo "Usage: $0 BASEDIR [NAMEPREFIX] [TRAININGNAME]"
    exit 0
fi
BASEDIR="$(realpath -m "$1")"
NAMEPREFIX="${2:-q4f}"
TRAININGNAME="${3:-q4f}"

export VENV_BIN="${VENV_BIN:-/home/kenny/ml_venv/bin}"
export PATH="$VENV_BIN:$PATH"
GITROOTDIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

MODELKIND="${MODELKIND:-tf2_b4c192_q4-meta}"
export KATAGO_BIN="${KATAGO_BIN:-$GITROOTDIR/cpp/build-cuda/katago}"

# Self-play: q4_selfplay.cfg as committed (numGameThreads / NN batch settings from the R8 benchmark). A cycle's self-play
# takes about 30 minutes at the R8 throughput.
export NUM_GAMES_PER_CYCLE="${NUM_GAMES_PER_CYCLE:-2500}"
export SELFPLAY_EXTRA_ARGS="${SELFPLAY_EXTRA_ARGS:-}"

# Training (Plan §15): Aurora from scratch with KataGo's warmup, LR scale x8 -> x4 -> x2 by training samples
# (train.py keeps global_step_samples in the checkpoint, so the schedule continues across cycles), batch 256,
# MAX_TRAIN_PER_DATA 6, validation on 5% of the files with at most 10k validation samples per epoch.
export BATCHSIZE="${BATCHSIZE:-256}"
export MAX_TRAIN_PER_DATA="${MAX_TRAIN_PER_DATA:-6}"
export VALIDATE="${VALIDATE:-1}"
export VALIDATION_PROP="${VALIDATION_PROP:-0.05}"
LR_SCHEDULE="${LR_SCHEDULE:-(0,8.0),(4M,4.0),(12M,2.0)}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-10000}"
export TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:--use-aurora -wd-floor-frac 0.5 -lr-schedule $LR_SCHEDULE -max-val-samples $MAX_VAL_SAMPLES}"
export USEGATING=0

# Evaluation every EVAL_EVERY exports (q4_formal_eval.sh): EVAL_OPENINGS openings per table at EVAL_VISITS visits.
export EVAL_EVERY="${EVAL_EVERY:-10}"
export EVAL_VISITS="${EVAL_VISITS:-100}"
export EVAL_OPENINGS="${EVAL_OPENINGS:-13}"
export EVAL_OVERRIDES="${EVAL_OVERRIDES:-numGameThreads=32}"   # -override-config of the evaluation q4match
export Q4_FORMAL_BASEDIR="$BASEDIR"
export POST_EXPORT_HOOK="$GITROOTDIR/python/selfplay/q4_formal_eval.sh"

mkdir -p "$BASEDIR"
{
    echo "$(date '+%F %T') q4_formal_run.sh $*"
    echo "MODELKIND=$MODELKIND KATAGO_BIN=$KATAGO_BIN NUM_GAMES_PER_CYCLE=$NUM_GAMES_PER_CYCLE"
    echo "TRAIN_EXTRA_ARGS=$TRAIN_EXTRA_ARGS BATCHSIZE=$BATCHSIZE MAX_TRAIN_PER_DATA=$MAX_TRAIN_PER_DATA VALIDATION_PROP=$VALIDATION_PROP"
    echo "EVAL_EVERY=$EVAL_EVERY EVAL_VISITS=$EVAL_VISITS EVAL_OPENINGS=$EVAL_OPENINGS SELFPLAY_EXTRA_ARGS=$SELFPLAY_EXTRA_ARGS EVAL_OVERRIDES=$EVAL_OVERRIDES"
} | tee -a "$BASEDIR"/formal_run.log

cd "$GITROOTDIR"/python/selfplay
exec ./q4_synchronous_loop.sh "$NAMEPREFIX" "$BASEDIR" "$TRAININGNAME" "$MODELKIND"
}
