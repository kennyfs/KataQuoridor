#!/bin/bash -eu
# Evaluates every net in models/ on the suite (suite.jsonl) at v256 and v1000, GTP and self-play params
# (docs/BlindSpots.md). Output: out/eval_*.json (+ .jsonl per position) and out/report.md.
for NET in $(ls models); do
  for P in ${PARAMS:-gtp selfplay-nonoise}; do
    python3 -m quoridor_arena.blindspot eval --suite suite.jsonl --model "models/$NET/model.bin.gz" --net "$NET" \
      --config gtp_quoridor.cfg --katago "$KATAGO" --visits "${VISITS:-256,1000}" --params "$P" \
      --engines "${ENGINES:-4}" --out-dir out --log-dir out/logs
  done
done
python3 -m quoridor_arena.blindspot report out/eval_*.json > out/report.md
cat out/report.md
