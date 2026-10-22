#!/bin/bash
# Home side of the remote self-play workers (README.md): every few minutes, publishes the run's newest net to the
# workstation share (workers load it by themselves) and pulls the workers' finished games and training rows into the
# run's selfplay dir (where shuffle picks them up).
#   python/colab/share_sync.sh BASEDIR                 (runs until killed; log in $BASEDIR/share_sync.log)
#   SHARE=user@host: INTERVAL=120 ...  (defaults: ws4, 120 s)
# One-off: publish the binary bundle and the selfplay config with publish_share.sh.
# $BASEDIR/remote_rows.txt keeps a running count of the training rows pulled; the loop's self-play counts its increase
# during self-play towards the cycle's rows (selfplay -extra-rows-file, synchronous_loop.sh ROWS_PER_CYCLE). Rows that
# arrive while the loop shuffles, trains or is stopped are found on disk by rows_since_shuffle.py at the next cycle.
# Ported from KataCDC's cdc_share_sync.sh (2026-10-23): restricted key, net first and published atomically, rows of a
# worker that ran on a random net rejected, back-off after failed logins, no stdin reads.
export PATH=/home/kenny/ml_venv/bin:$PATH   # count_rows needs numpy
BASEDIR="$(realpath "$1")"
SHARE="${SHARE:-r15922041@ws4.csie.ntu.edu.tw:}"   # the key is rooted at the share dir; SHARE ends with : or /
INTERVAL="${INTERVAL:-120}"
exec 9> "$BASEDIR/share_sync.lock"
flock -n 9 || { echo "a share_sync.sh for $BASEDIR is already running"; exit 1; }
# Logs in with ~/.ssh/kq_sync: a copy of the workers' restricted key (ws4 authorized_keys: rrsync rooted at
# tmp2/kq_remote/share, no passphrase). The personal key id_ed25519 has a passphrase and works only through the
# ssh-agent; a process started without the agent fell back to password prompts, and repeated failures got the IP banned
# (KataCDC, 2026-10-20). BatchMode: a failed key login fails at once, never asks for a password. The restricted key can
# only rsync: the empty incoming/ dirs are removed on ws4 by colab_launch.sh.
SSH_KEY="${SSH_KEY:-$HOME/.ssh/kq_sync}"
export RSYNC_RSH="ssh -i $SSH_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o PreferredAuthentications=publickey -o ConnectTimeout=20 -o ServerAliveInterval=30"
LOG="$BASEDIR/share_sync.log"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
COUNT="$BASEDIR/remote_rows.txt"
[[ -f "$COUNT" ]] || echo 0 > "$COUNT"
# rows in the .npz files listed on stdin (paths relative to $BASEDIR/selfplay), read from the array headers only
count_rows() {
  python3 -c '
import sys, zipfile, numpy as np
n = 0
for f in sys.stdin.read().split():
    if not f.endswith(".npz"): continue
    try:
        with zipfile.ZipFile(sys.argv[1] + "/" + f) as z, z.open([m for m in z.namelist() if m.startswith("binaryInputNCHWPacked")][0]) as a:
            v = np.lib.format.read_magic(a)
            h = np.lib.format.read_array_header_1_0(a) if v == (1, 0) else np.lib.format.read_array_header_2_0(a)
            n += h[0][0]
    except Exception as e:
        print("cannot count", f, e, file=sys.stderr)
print(n)' "$BASEDIR/selfplay"
}
STAGE="$BASEDIR/share_models_stage"
published=""; fails=0
while true; do
  # 1. the newest net (the newest dir in models/), when it changed
  newest=$(ls -t "$BASEDIR/models" 2>/dev/null | head -1)
  if [[ -n "$newest" && "$newest" != "$published" && -f "$BASEDIR/models/$newest/model.bin.gz" ]]; then
    rm -rf "$STAGE" && mkdir -p "$STAGE/$newest" && cp -p "$BASEDIR/models/$newest/model.bin.gz" "$STAGE/$newest/"
    # --delay-updates --delete-after: the new net is moved in place when complete and the old one removed after, so
    # a worker starting meanwhile never finds models/ without a net (its selfplay would start on a random net)
    if rsync -a --timeout=120 --delay-updates --delete-after "$STAGE/" "${SHARE}models/" 2>>"$LOG" < /dev/null; then
      published="$newest"; log "published $newest"
    fi
  fi
  # 2. workers' data: incoming/<worker>/<model>/{tdata,sgfs}/... -> selfplay/<model>/{tdata,sgfs}/...
  # ssh -n / rsync < /dev/null: never read stdin (started with nohup ... & from a terminal, a read stops the
  # process with SIGTTIN and the loop hangs)
  # when the login fails, skip the round and wait longer each time (up to 30 min) instead of retrying quickly
  if ! workers=$(set -o pipefail; rsync --list-only --timeout=60 "${SHARE}incoming/" 2>>"$LOG" < /dev/null | awk '/^d/ && $NF != "." {print $NF}'); then
    fails=$((fails + 1)); wait=$(( INTERVAL * 2 ** (fails < 4 ? fails : 4) )); (( wait > 1800 )) && wait=1800
    log "listing $SHARE failed ($fails in a row), next try in ${wait}s"; sleep "$wait"; continue
  fi
  fails=0
  for w in $workers; do
    # games of a worker whose selfplay started without a net (incoming/<worker>/random/) never reach selfplay/: they
    # go to rejected/<worker>/ (2026-10-22: a worker caught the share's models/ mid-update and played on a random net)
    rej=$(rsync -az --skip-compress=npz/gz/zip --timeout=120 --remove-source-files --exclude='.*' --out-format='%n' "${SHARE}incoming/$w/random/" "$BASEDIR/rejected/$w/" 2>/dev/null < /dev/null | grep -v '/$')
    [[ -n "$rej" ]] && log "rejected $(wc -l <<< "$rej") files of a random net from $w (rejected/$w)"
    files=$(rsync -az --skip-compress=npz/gz/zip --timeout=120 --remove-source-files --exclude='.*' --exclude='/random/' --out-format='%n' "${SHARE}incoming/$w/" "$BASEDIR/selfplay/" 2>>"$LOG" < /dev/null | grep -v '/$')
    if [[ -n "$files" ]]; then
      rows=$(count_rows <<< "$files" 2>>"$LOG")
      total=$(( $(cat "$COUNT") + ${rows:-0} ))
      echo "$total" > "$COUNT.tmp" && mv "$COUNT.tmp" "$COUNT"
      log "pulled $(wc -l <<< "$files") files, $rows rows from $w (total $total)"
    fi
  done
  sleep "$INTERVAL"
done
