#!/bin/bash
# Keeps one Colab GPU VM running kq_worker.py (README.md). Run it on the workstation (stable network), one per VM:
#   colab_launch.sh NAME GPU [HOURS]        e.g.  colab_launch.sh l4a L4 20
# NAME: the Colab session name and the worker's incoming/ dir; GPU: T4 / L4 / A100 / G4 / H100; HOURS: stop after
# this many hours in total (0 = until stopped). Each VM: upload the worker and the share key, start the worker
# detached, then every 5 minutes a short exec that prints the end of its log (this also keeps the kernel active). A VM
# that is gone (Colab's time limit, a preemption) or a worker that died is replaced, after saving its log.
# Stop: touch $LOGROOT/worker_NAME/STOP (the worker stops, sends its last files, the VM is released; the launcher
# removes STOP when it ends, and at its start) or kill this.
# Another Colab account: HOME=<dir with that account's colab login> colab_launch.sh ...   (colab keeps its login and
# sessions under ~/.config/colab-cli)
S="$1"; G="$2"; HOURS="${3:-0}"
HERE="$(dirname "$(realpath "$0")")"
REALHOME="$(getent passwd "$USER" | cut -d: -f6)"   # HOME may point at another Colab account's login
KEY="${KEY:-$REALHOME/.ssh/kq_worker}"
LOGROOT="${LOGROOT:-$REALHOME/tmp2/kq_remote/colab_logs}"
L="$LOGROOT/worker_$S"; mkdir -p "$L"
rm -f "$L/STOP"   # a STOP left from an earlier run with this name
export PATH="$REALHOME/.local/bin:$PATH"
log() { echo "$(date '+%F %T') $*" | tee -a "$L/launch.log"; }
T0=$(date +%s)
over() { [[ -f "$L/STOP" ]] || { [[ "$HOURS" != 0 ]] && (( $(date +%s) - T0 > ${HOURS%.*} * 3600 )); }; }
vm=0
while ! over; do
  vm=$((vm + 1))
  log "VM $vm: new $G session $S"
  if ! timeout 900 colab new -s "$S" --gpu "$G" >> "$L/launch.log" 2>&1; then
    log "no $G VM; retrying in 10 minutes"; timeout 120 colab stop -s "$S" >/dev/null 2>&1; sleep 600; continue
  fi
  timeout 300 colab upload -s "$S" "$KEY" /content/kq_worker_key >> "$L/launch.log" 2>&1
  timeout 300 colab upload -s "$S" "$HERE/kq_worker.py" /content/kq_worker.py >> "$L/launch.log" 2>&1
  echo "import subprocess, os; e = dict(os.environ, KQ_WORKER='$S'); subprocess.Popen('python3 /content/kq_worker.py >> /content/worker.log 2>&1', shell=True, env=e, start_new_session=True); print('started')" \
    | timeout 300 colab exec -s "$S" >> "$L/launch.log" 2>&1
  fails=0
  while true; do
    for i in $(seq 10); do over && break; sleep 30; done   # a poll every 5 minutes, STOP noticed within 30 s
    out=$(echo "import subprocess; print(open('/content/worker.log').read()[-1500:]); print('WORKER', 'ALIVE' if subprocess.run(['pgrep', '-f', 'kq_worker.py'], capture_output=True).returncode == 0 else 'DEAD')" \
      | timeout 180 colab exec -s "$S" 2>&1)
    if ! grep -q "^WORKER" <<< "$out"; then
      fails=$((fails + 1)); log "poll failed ($fails): $(tail -c 300 <<< "$out" | tr '\n' ' ')"
      (( fails >= 3 )) && { log "VM $vm gone"; break; }
      continue
    fi
    fails=0
    echo "$(date '+%F %T') $(grep -v '^\s*$' <<< "$out" | tail -2 | tr '\n' ' ')" >> "$L/poll.log"
    grep -q "WORKER DEAD" <<< "$out" && { log "worker ended"; break; }
    if over; then
      log "stopping the worker"
      echo "import subprocess; subprocess.run(['pkill', '-TERM', '-f', 'kq_worker.py'])" | timeout 120 colab exec -s "$S" >/dev/null 2>&1
      for i in $(seq 12); do   # up to 6 minutes for the worker to send its last files
        sleep 30
        echo "import subprocess; print('WORKER', 'ALIVE' if subprocess.run(['pgrep', '-f', 'kq_worker.py'], capture_output=True).returncode == 0 else 'DEAD')" \
          | timeout 120 colab exec -s "$S" 2>&1 | grep -q "WORKER DEAD" && break
      done
      break
    fi
  done
  timeout 300 colab download -s "$S" /content/worker.log "$L/worker_vm$vm.log" >/dev/null 2>&1
  timeout 300 colab download -s "$S" /content/kq/selfplay.log "$L/selfplay_vm$vm.log" >/dev/null 2>&1
  timeout 120 colab stop -s "$S" >> "$L/launch.log" 2>&1
  log "VM $vm released"
  over || sleep 60
done
rm -f "$L/STOP"
log "done"
