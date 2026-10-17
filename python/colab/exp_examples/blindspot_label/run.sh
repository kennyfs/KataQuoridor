#!/bin/bash -eu
# Deep-search labels for the blind-spot suite (docs/BlindSpots.md). Inputs: positions.jsonl, models/<NET>/model.bin.gz
# (one net: the labeller), quoridor_arena/, gtp_quoridor.cfg. Output: out/suite.jsonl (resumes if run again).
NET="${NET:-$(ls models | head -1)}"
python3 -m quoridor_arena.blindspot label --positions positions.jsonl --model "models/$NET/model.bin.gz" --net "$NET" \
  --config gtp_quoridor.cfg --katago "$KATAGO" --out out/suite.jsonl --visits "${VISITS:-50000}" \
  --engines "${ENGINES:-2}" --threads "${THREADS:-16}" --cache-bits 22 --log-dir out/logs
