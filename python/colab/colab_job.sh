#!/bin/bash
# Runs one experiment job on a Colab GPU VM and releases it (README.md, "Experiment jobs"). On the workstation:
#   colab_job.sh NAME GPU        e.g.  colab_job.sh vtd_ladder L4
# The job is the share's exp/NAME/ (run.sh and its inputs; run.sh writes into out/, $KATAGO is the binary); results
# come back to exp/NAME/out/ every 2 minutes. Logs: $LOGROOT/job_NAME/. Stop early: touch $LOGROOT/job_NAME/STOP.
# Another Colab account: HOME=<dir> colab_job.sh ...
S="$1"; G="$2"
HERE="$(dirname "$(realpath "$0")")"
REALHOME="$(getent passwd "$USER" | cut -d: -f6)"
KEY="${KEY:-$REALHOME/.ssh/kq_worker}"
LOGROOT="${LOGROOT:-$REALHOME/tmp2/kq_remote/colab_logs}"
L="$LOGROOT/job_$S"; mkdir -p "$L"; rm -f "$L/STOP"
export PATH="$REALHOME/.local/bin:$PATH"
log() { echo "$(date '+%F %T') $*" | tee -a "$L/launch.log"; }
until timeout 900 colab new -s "$S" --gpu "$G" >> "$L/launch.log" 2>&1; do
  log "no $G VM; retrying in 10 minutes"; timeout 120 colab stop -s "$S" >/dev/null 2>&1
  for i in $(seq 20); do [[ -f "$L/STOP" ]] && { rm -f "$L/STOP"; log "stopped before getting a VM"; exit 0; }; sleep 30; done
done
log "VM up"
timeout 300 colab upload -s "$S" "$KEY" /content/kq_worker_key >> "$L/launch.log" 2>&1
timeout 300 colab upload -s "$S" "$HERE/job_runner.py" /content/job_runner.py >> "$L/launch.log" 2>&1
echo "import subprocess; subprocess.Popen('python3 /content/job_runner.py $S >> /content/runner.log 2>&1', shell=True, start_new_session=True); print('started')" \
  | timeout 300 colab exec -s "$S" >> "$L/launch.log" 2>&1
fails=0
while true; do
  for i in $(seq 10); do [[ -f "$L/STOP" ]] && break; sleep 30; done
  out=$(echo "import subprocess; print(open('/content/runner.log').read()[-600:]); print('RUNNER', 'ALIVE' if subprocess.run(['pgrep', '-f', 'job_runner.py'], capture_output=True).returncode == 0 else 'DEAD')" \
    | timeout 180 colab exec -s "$S" 2>&1)
  if ! grep -q "^RUNNER" <<< "$out"; then
    fails=$((fails + 1)); log "poll failed ($fails)"; (( fails >= 3 )) && { log "VM gone"; break; }; continue
  fi
  fails=0
  echo "$(date '+%F %T') $(grep -v '^\s*$' <<< "$out" | tail -2 | tr '\n' ' ')" >> "$L/poll.log"
  grep -q "RUNNER DEAD" <<< "$out" && { log "job ended"; break; }
  if [[ -f "$L/STOP" ]]; then
    log "stopping"; echo "import subprocess; subprocess.run(['pkill', '-TERM', '-f', 'run.sh'])" | timeout 120 colab exec -s "$S" >/dev/null 2>&1
    sleep 60
  fi
done
timeout 300 colab download -s "$S" /content/runner.log "$L/runner.log" >/dev/null 2>&1
timeout 300 colab download -s "$S" /content/job/out/run.log "$L/run.log" >/dev/null 2>&1
timeout 120 colab stop -s "$S" >> "$L/launch.log" 2>&1
rm -f "$L/STOP"
log "VM released"
