#!/bin/bash -eu
# Stages an experiment job directory for the blind-spot tools (README.md, "Experiment jobs"):
#   stage_exp.sh JOBDIR TEMPLATE [NET_DIR|FILE]...
# TEMPLATE is a directory of exp_examples/ (blindspot_label, blindspot_eval, hint_mine). JOBDIR gets its run.sh, the
# quoridor_arena package, gtp_quoridor.cfg, and the inputs: a directory with model.bin.gz goes to models/<name>/,
# a *.jsonl to the top, a directory of hint files to hints_in/, a directory of self-play SGFs (name/sgfs/*.sgfs) to
# selfplay/<name>/sgfs/. Then: rsync -a JOBDIR/ ws4:tmp2/kq_remote/share/exp/NAME/ and colab_job.sh NAME L4.
HERE="$(dirname "$(realpath "$0")")"
REPO="$HERE/../.."
JOB="$1"; T="$HERE/exp_examples/$2"; shift 2
mkdir -p "$JOB/out"
cp "$T/run.sh" "$JOB/run.sh"
rsync -a --exclude tests --exclude __pycache__ "$REPO/python/quoridor_arena" "$JOB/"
cp "$REPO/cpp/configs/gtp_quoridor.cfg" "$JOB/"
for x in "$@"; do
  x="$(realpath "$x")"
  if [[ -f "$x" ]]; then cp "$x" "$JOB/"
  elif [[ -f "$x/model.bin.gz" ]]; then mkdir -p "$JOB/models/$(basename "$x")"; cp "$x/model.bin.gz" "$JOB/models/$(basename "$x")/"
  elif [[ -d "$x/sgfs" ]]; then mkdir -p "$JOB/selfplay/$(basename "$x")"; rsync -a "$x/sgfs" "$JOB/selfplay/$(basename "$x")/"
  else mkdir -p "$JOB/hints_in"; cp "$x"/*.hintposes.txt "$JOB/hints_in/" 2>/dev/null || true
  fi
done
du -sh "$JOB"; ls "$JOB"
