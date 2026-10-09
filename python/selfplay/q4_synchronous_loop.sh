#!/bin/bash -eu
set -o pipefail
{

# Quoridor Four-at-a-Table (Q4) version of synchronous_loop.sh: self-play (katago q4selfplay) -> shuffle -> train ->
# export, in a loop, with no gatekeeper unless USEGATING=1 (docs/q4/rounds/R5.md, R6.md). The steps, the environment overrides and the dated
# archive are those of synchronous_loop.sh; the differences are:
# - katago q4selfplay with cpp/configs/q4/training/q4_selfplay.cfg;
# - train.py gets -pos-len 11 (train.sh passes -pos-len 9 first; argparse keeps the last value);
# - no gatekeeper by default (exports go straight to models/); USEGATING=1 exports to modelstobetested/ and runs
#   `katago q4gatekeeper` (cpp/configs/q4/match/q4_gatekeeper.cfg, edit gateBenchStrong there) before each self-play;
# - if models/ is empty, a randomly initialized net of MODELKIND is exported there first (q4selfplay has no "random"
#   net), named NAMEPREFIX-init;
# - PATH starts with the venv's bin (VENV_BIN, default /home/kenny/ml_venv/bin), so every python step finds numpy/torch.

if [[ $# -lt 4 ]]
then
    echo "Usage: $0 NAMEPREFIX BASEDIR TRAININGNAME MODELKIND"
    echo "NAMEPREFIX string prefix for this training run, shown when katago loads the model."
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TRAININGNAME name to prefix models with, specific to this training daemon"
    echo "MODELKIND what size model to train: tf2_b4c192_q4 (the intended net) or b2c64_q4 (smoke tests)"
    exit 0
fi
NAMEPREFIX="$1"
shift
BASEDIRRAW="$1"
shift
TRAININGNAME="$1"
shift
MODELKIND="$1"
shift

VENV_BIN="${VENV_BIN:-/home/kenny/ml_venv/bin}"
export PATH="$VENV_BIN:$PATH"

BASEDIR="$(realpath "$BASEDIRRAW")"
GITROOTDIR="$(git rev-parse --show-toplevel)"
LOGSDIR="$BASEDIR"/logs
SCRATCHDIR="$BASEDIR"/shufflescratch

mkdir -p "$BASEDIR"
mkdir -p "$LOGSDIR"
mkdir -p "$SCRATCHDIR"
mkdir -p "$BASEDIR"/selfplay
mkdir -p "$BASEDIR"/models
mkdir -p "$BASEDIR"/modelstobetested "$BASEDIR"/rejectedmodels "$BASEDIR"/gatekeeper

# Same knobs as synchronous_loop.sh (each can be overridden from the environment).
NUM_GAMES_PER_CYCLE="${NUM_GAMES_PER_CYCLE:-10000}"
NUM_THREADS_FOR_SHUFFLING="${NUM_THREADS_FOR_SHUFFLING:-12}"
NUM_TRAIN_SAMPLES_PER_EPOCH="${NUM_TRAIN_SAMPLES_PER_EPOCH:-100000}"
MAX_TRAIN_PER_DATA="${MAX_TRAIN_PER_DATA:-8}"
NUM_TRAIN_SAMPLES_PER_SWA="${NUM_TRAIN_SAMPLES_PER_SWA:-80000}"
BATCHSIZE="${BATCHSIZE:-256}"
SHUFFLE_MINROWS="${SHUFFLE_MINROWS:-100000}"
MAX_TRAIN_SAMPLES_PER_CYCLE="${MAX_TRAIN_SAMPLES_PER_CYCLE:-1000000}"
TAPER_WINDOW_SCALE="${TAPER_WINDOW_SCALE:-500000}"
SHUFFLE_KEEPROWS="${SHUFFLE_KEEPROWS:-1200000}"
EXPAND_WINDOW_PER_ROW="${EXPAND_WINDOW_PER_ROW:-0.6}"
MAX_CYCLES="${MAX_CYCLES:-0}" # Stop after this many cycles (0 = run forever).
VALIDATE="${VALIDATE:-1}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
SELFPLAY_EXTRA_ARGS="${SELFPLAY_EXTRA_ARGS:-}" # e.g. -override-config "numGameThreads=64"
USEGATING="${USEGATING:-0}" # 1 = gatekeeper (off by default, Plan §9)
GATEKEEPER_TIMEOUT="${GATEKEEPER_TIMEOUT:-3600}" # Kill a gatekeeper that runs longer than this many seconds; 0 disables.
GATEKEEPER_EXTRA_ARGS="${GATEKEEPER_EXTRA_ARGS:-}" # e.g. -override-config "gateBenchStrong=search:/path/model.bin.gz@100"
START_AT="${START_AT:-selfplay}" # First cycle only: gatekeeper (with USEGATING=1), selfplay, shuffle, train or export.

SELFPLAY_CONFIG="${SELFPLAY_CONFIG:-$GITROOTDIR/cpp/configs/q4/training/q4_selfplay.cfg}"
GATING_CONFIG="${GATING_CONFIG:-$GITROOTDIR/cpp/configs/q4/match/q4_gatekeeper.cfg}"
KATAGO_BIN="${KATAGO_BIN:-$GITROOTDIR/cpp/katago}"

DATE_FOR_FILENAME=$(date "+%Y%m%d-%H%M%S")
DATED_ARCHIVE="$BASEDIR"/scripts/dated/"$DATE_FOR_FILENAME"
mkdir -p "$DATED_ARCHIVE"/bin
cp "$GITROOTDIR"/python/*.py "$GITROOTDIR"/python/selfplay/*.py "$GITROOTDIR"/python/selfplay/*.sh "$DATED_ARCHIVE"
cp -r "$GITROOTDIR"/python/katago "$DATED_ARCHIVE"
cp -r "$GITROOTDIR"/python/muon "$DATED_ARCHIVE"
cp -r "$GITROOTDIR"/python/q4 "$DATED_ARCHIVE"
cp "$KATAGO_BIN" "$DATED_ARCHIVE"/bin/katago
cp "$SELFPLAY_CONFIG" "$DATED_ARCHIVE"/selfplay.cfg
mkdir -p "$DATED_ARCHIVE"/match
cp "$(dirname "$GATING_CONFIG")"/*.cfg "$DATED_ARCHIVE"/match/ # the gatekeeper config @includes q4_match_common.cfg
git show --no-patch --no-color > "$DATED_ARCHIVE"/version.txt
git diff --no-color > "$DATED_ARCHIVE"/diff.txt
git diff --staged --no-color > "$DATED_ARCHIVE"/diffstaged.txt

cd "$DATED_ARCHIVE"

if [[ -z "$(ls -A "$BASEDIR"/models)" ]]; then
    echo "No model in $BASEDIR/models: exporting a randomly initialized $MODELKIND as $NAMEPREFIX-init"
    python -c "import sys; sys.path.insert(0, '.'); from q4.make_random_model import export_random_model; \
print(export_random_model('$MODELKIND', '$BASEDIR/models_init', '$NAMEPREFIX-init', seed=0))"
    mv "$BASEDIR"/models_init/"$NAMEPREFIX-init" "$BASEDIR"/models/
fi

set -x
CYCLE=0
while [[ "$MAX_CYCLES" -le 0 || "$CYCLE" -lt "$MAX_CYCLES" ]]
do
    CYCLE=$((CYCLE + 1))
    SKIP=""
    if [[ "$CYCLE" == 1 ]]; then
        case "$START_AT" in
            gatekeeper) ;; selfplay) SKIP="gatekeeper" ;; shuffle) SKIP="gatekeeper selfplay" ;;
            train) SKIP="gatekeeper selfplay shuffle" ;; export) SKIP="gatekeeper selfplay shuffle train" ;;
            *) echo "unknown START_AT=$START_AT"; exit 1 ;;
        esac
    fi
    skip() { [[ " $SKIP " == *" $1 "* ]]; }

    if [[ "$USEGATING" == "1" ]] && ! skip gatekeeper; then
    echo "Gatekeeper"
    time timeout -k 60 "$GATEKEEPER_TIMEOUT" ./bin/katago q4gatekeeper -rejected-models-dir "$BASEDIR"/rejectedmodels -accepted-models-dir "$BASEDIR"/models/ -output-dir "$BASEDIR"/gatekeeper/ -test-models-dir "$BASEDIR"/modelstobetested/ -config "$DATED_ARCHIVE"/match/"$(basename "$GATING_CONFIG")" -quit-if-no-nets-to-test $GATEKEEPER_EXTRA_ARGS | tee -a "$BASEDIR"/gatekeeper/stdout.txt \
        || echo "$(date '+%F %T') gatekeeper failed or timed out (status $?), continuing with the current model" | tee -a "$BASEDIR"/gatekeeper/stdout.txt
    fi

    if ! skip selfplay; then
    echo "Selfplay"
    time ./bin/katago q4selfplay -max-games-total "$NUM_GAMES_PER_CYCLE" -output-dir "$BASEDIR"/selfplay -models-dir "$BASEDIR"/models -config "$DATED_ARCHIVE"/selfplay.cfg $SELFPLAY_EXTRA_ARGS | tee -a "$BASEDIR"/selfplay/stdout.txt
    fi

    if ! skip shuffle; then
    echo "Shuffle"
    (
        if [[ "$VALIDATE" == "0" ]]; then export SKIP_VALIDATE=1; else unset SKIP_VALIDATE; fi
        time ./shuffle.sh "$BASEDIR" "$SCRATCHDIR" "$NUM_THREADS_FOR_SHUFFLING" -include-meta -min-rows "$SHUFFLE_MINROWS" -keep-target-rows "$SHUFFLE_KEEPROWS" -taper-window-scale "$TAPER_WINDOW_SCALE" -expand-window-per-row "$EXPAND_WINDOW_PER_ROW" | tee -a "$BASEDIR"/logs/outshuffle.txt
    )
    fi

    if ! skip train; then
    echo "Train"
    time ./train.sh "$BASEDIR" "$TRAININGNAME" "$MODELKIND" "$BATCHSIZE" main -pos-len 11 -samples-per-epoch "$NUM_TRAIN_SAMPLES_PER_EPOCH" -swa-period-samples "$NUM_TRAIN_SAMPLES_PER_SWA" -quit-if-no-data -stop-when-train-bucket-limited -export-only-at-end -max-train-bucket-per-new-data "$MAX_TRAIN_PER_DATA" -max-train-bucket-size "$MAX_TRAIN_SAMPLES_PER_CYCLE" $TRAIN_EXTRA_ARGS
    fi

    echo "Export"
    (
        time ./export_model_for_selfplay.sh "$NAMEPREFIX" "$BASEDIR" "$USEGATING" | tee -a "$BASEDIR"/logs/outexport.txt
    )

done

exit 0
}
