#!/bin/bash -eu
# Mines hint poses with the net in models/ from the self-play SGFs in selfplay/*/ (docs/BlindSpots.md). hints_in/ holds
# the hint files of the earlier runs (they are re-checked and expire); the new file and the kept ones are in out/hints/.
NET="${NET:-$(ls models | head -1)}"
mkdir -p out/hints
cp hints_in/*.hintposes.txt out/hints/ 2>/dev/null || true
python3 -m quoridor_arena.hintminer mine --dirs selfplay/* --model "models/$NET/model.bin.gz" --net "$NET" \
  --config gtp_quoridor.cfg --katago "$KATAGO" --out-dir out/hints --fresh-visits 1000 --deep-visits "${DEEP:-10000}" \
  --n-opening "${N_OPENING:-250}" --n-swing "${N_SWING:-150}" --engines "${ENGINES:-2}" --threads "${THREADS:-16}" \
  --cache-bits 22 --log-dir out/logs
