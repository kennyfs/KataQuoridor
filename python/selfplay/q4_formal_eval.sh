#!/bin/bash -eu
set -o pipefail

# Post-export hook of the Q4 formal run (q4_formal_run.sh, Plan §15), called as: q4_formal_eval.sh CYCLE
# - every cycle: one line per cycle in $BASEDIR/summary.tsv (python q4/formal_summary.py: losses, self-play learner
#   win rates by opponent kind, draw / cutoff rates, rows);
# - every EVAL_EVERY exports: q4match of the newest export against the previous evaluated export (the random init net
#   the first time) and the bots (cpp/configs/q4/match/formal_eval.cfg), appended to $BASEDIR/eval/ladder.jsonl, and
#   python q4/rating.py on the whole file written to $BASEDIR/eval/ratings.md.

CYCLE="$1"
BASEDIR="$Q4_FORMAL_BASEDIR"
GITROOTDIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PYDIR="$GITROOTDIR/python"
EVALDIR="$BASEDIR"/eval
mkdir -p "$EVALDIR"

python "$PYDIR"/q4/formal_summary.py "$BASEDIR" "$CYCLE" >> "$BASEDIR"/summary.tsv

# Exports in models/ in mtime order (oldest first); the first one is the random init net.
mapfile -t MODELS < <(python "$PYDIR"/selfplay/list_by_mtime.py "$BASEDIR"/models | xargs -n1 basename)
NUM_EXPORTS=$(( ${#MODELS[@]} - 1 ))
if (( NUM_EXPORTS <= 0 || NUM_EXPORTS % EVAL_EVERY != 0 )); then
    exit 0
fi
NEW="${MODELS[-1]}"
PREV="${MODELS[0]}"
if [[ -f "$EVALDIR"/last_evaluated.txt ]]; then
    PREV="$(cat "$EVALDIR"/last_evaluated.txt)"
fi
if [[ "$PREV" == "$NEW" ]]; then
    exit 0
fi

echo "$(date '+%F %T') cycle $CYCLE: evaluating $NEW (export $NUM_EXPORTS) against $PREV and the bots"
# The match config of this evaluation: formal_eval.cfg (bots, tables) with the two nets filled in.
cp "$GITROOTDIR"/cpp/configs/q4/match/q4_match_common.cfg "$EVALDIR"/q4_match_common.cfg
CFG="$EVALDIR"/eval_"$NUM_EXPORTS".cfg
sed -e "s|NEWNET|$NEW|g" -e "s|PREVNET|$PREV|g" -e "s|MODELSDIR|$BASEDIR/models|g" -e "s|EVALVISITS|$EVAL_VISITS|g" \
    -e "s|EVALOPENINGS2|$((EVAL_OPENINGS * 2))|g" -e "s|EVALOPENINGS|$EVAL_OPENINGS|g" \
    "$GITROOTDIR"/cpp/configs/q4/match/formal_eval.cfg > "$CFG"
OUT="$EVALDIR"/eval_"$NUM_EXPORTS"_"$NEW".jsonl
"$KATAGO_BIN" q4match -config "$CFG" -output "$OUT" -summary "$EVALDIR"/eval_"$NUM_EXPORTS".summary.json \
    -override-config "seed=$NUM_EXPORTS,$EVAL_OVERRIDES" > "$EVALDIR"/eval_"$NUM_EXPORTS".log 2>&1
cat "$OUT" >> "$EVALDIR"/ladder.jsonl
echo "$NEW" > "$EVALDIR"/last_evaluated.txt
{
    echo "# Q4 formal run ratings ($(date '+%F %T'), after export $NUM_EXPORTS = $NEW)"
    echo
    echo '```'
    python "$PYDIR"/q4/rating.py "$EVALDIR"/ladder.jsonl --anchor random --bootstrap 200
    echo '```'
    echo
    echo "## Match report of export $NUM_EXPORTS (bench (c) = table newVsGrudge)"
    echo
    echo '```'
    python "$PYDIR"/q4/match_report.py "$OUT"
    echo '```'
} > "$EVALDIR"/ratings.md.tmp
mv "$EVALDIR"/ratings.md.tmp "$EVALDIR"/ratings.md
echo "$(date '+%F %T') wrote $EVALDIR/ratings.md"
