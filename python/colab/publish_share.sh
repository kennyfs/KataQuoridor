#!/bin/bash -eu
# Publishes what the remote workers need besides the net to the workstation share (README.md):
#   publish_share.sh [BUNDLE_TAR] [SELFPLAY_CFG]
# BUNDLE_TAR: the build_portable.sh bundle (default: already on ws4 at tmp2/kq_remote/build/katago-kq-colab.tar.gz,
# copied there on the workstation itself); SELFPLAY_CFG: default cpp/configs/training/selfplay_quoridor_v2.cfg.
HERE="$(dirname "$(realpath "$0")")"
SHARE="${SHARE:-ws4:tmp2/kq_remote/share}"
HOSTPART="${SHARE%%:*}"; PATHPART="${SHARE#*:}"
CFG="${2:-$HERE/../../cpp/configs/training/selfplay_quoridor_v2.cfg}"
ssh "$HOSTPART" "mkdir -p $PATHPART/bin $PATHPART/models $PATHPART/incoming"
if [[ -n "${1:-}" ]]; then
  rsync -a "$1" "$SHARE/bin/katago-kq-colab.tar.gz"
else
  ssh "$HOSTPART" "cp ~/tmp2/kq_remote/build/katago-kq-colab.tar.gz $PATHPART/bin/"
fi
rsync -a "$CFG" "$SHARE/selfplay_quoridor_v2.cfg"
ssh "$HOSTPART" "ls -la $PATHPART $PATHPART/bin"
