# Remote self-play workers (Colab)

Extra self-play for the KataQuoridor loop from Colab GPU VMs, with the workstation (ws4) as a relay. Ported from
KataCDC (`TCG/final_old/katago/python/colab`), with its own key, share and logs, so the two projects never see each
other's files. The loop at home keeps running as before (gatekeeper, self-play, shuffle, train, export); the workers
only add games and training rows.

```text
 home (loop, share_sync.sh)               ws4: ~/tmp2/kq_remote/share            Colab VM (kq_worker.py)
 newest net in models/  ── publish ──▶   models/<net>/model.bin.gz   ── pull ──▶  models/ (selfplay loads it)
 selfplay/<net>/{tdata,sgfs} ◀── pull ── incoming/<worker>/<net>/...  ◀── push ── out/<net>/{tdata,sgfs}
                                          bin/katago-kq-colab.tar.gz, selfplay_quoridor_v2.cfg (publish_share.sh)
 ws4: colab_launch.sh keeps a VM running the worker (new VM when one ends), polls it, keeps its logs
```

- **Worker** (`kq_worker.py`, on the VM): fetches the binary, the selfplay config and the newest net from the share,
  runs `katago selfplay` with the GPU's thread / batch settings (`TUNING`), and every 2 minutes moves the files
  untouched for 3 minutes (finished: npz and sgfs files are never written again) to `incoming/<worker>/`, and pulls
  `models/` (selfplay loads a newer net by itself). On SIGTERM it stops selfplay and sends all its files at once.
- **Launcher** (`colab_launch.sh`, on ws4): `colab new`, uploads the worker and the share key, starts the worker
  detached, then a short `colab exec` every 5 minutes (prints the end of the worker's log into `poll.log`, and keeps
  the kernel active). A VM that is gone or a worker that died is replaced; logs go to
  `~/tmp2/kq_remote/colab_logs/worker_<name>/`.
- **Home** (`share_sync.sh BASEDIR`): every 2 minutes, pulls `incoming/*` into `$BASEDIR/selfplay/` (rsync writes each
  file under a temporary name and renames it, so shuffle never sees half a file) and publishes the newest net in
  `$BASEDIR/models/` to the share. It keeps a running count of the rows pulled in `$BASEDIR/remote_rows.txt`.
- **Share key**: `~/.ssh/kq_worker` on ws4, in `authorized_keys` as
  `command="/usr/bin/rrsync -no-lock /tmp2/r15922041/kq_remote/share",restrict`: rsync inside the share only, no
  shell (`-no-lock`: rrsync otherwise lets only one connection in at a time, and workers syncing together fail).
  KataCDC's key (`~/.ssh/cdc_worker`) and share (`~/tmp2/cdc_remote`) are separate.

## Rows per cycle with remote rows

`synchronous_loop.sh` with `ROWS_PER_CYCLE=N` trains every cycle on about N new rows, wherever they come from:

- before self-play, `rows_since_shuffle.py` counts the self-play rows on disk that the last shuffle did not see (the
  last shuffle's total is `range[1]` of its `train.json`, the d of the model names). That includes remote rows pulled
  while the loop shuffled, trained or exported, rows from a cycle that a stopped loop never trained on, and remote
  rows that arrived while the loop was stopped;
- self-play then runs until its own rows plus the remote rows pulled meanwhile (`-extra-rows-file remote_rows.txt`)
  make up the rest, or is skipped when there are N new rows already;
- train.py's bucket gives `MAX_TRAIN_PER_DATA` samples per new row whatever the cycle's size (up to
  `MAX_TRAIN_SAMPLES_PER_CYCLE`, default 1M, i.e. about 166K new rows at 6 per row).

With enough remote workers the local self-play gets short or is skipped, and a cycle is about shuffle + train.

## Running

Once (and after changing the C++ code or the selfplay config):

```bash
# sources on ws4: src/cpp came from a clone of github.com/kennyfs/KataQuoridor (the home uplink is slow); refresh it
# with the tracked files only, so rsync sends just the changes (seconds), at home:
git ls-files cpp | sed 's#^cpp/##' > /tmp/cppfiles.txt && rsync -a --files-from=/tmp/cppfiles.txt cpp/ ws4:tmp2/kq_remote/src/cpp/
rsync -a python/colab/build_portable.sh ws4:tmp2/kq_remote/
rsync -a python/colab/{colab_launch.sh,colab_job.sh,job_runner.py,kq_worker.py} ws4:tmp2/kq_remote/launch/
# on ws4 (rootless docker, PATH=~/bin:$PATH): build the binary for Colab (CUDA 13.0, Colab's cuBLAS / cuDNN / NVRTC
# are used); ws4 is shared: at most 32 make jobs (JOBS, default 32); ~2 minutes
cd ~/tmp2/kq_remote && ./build_portable.sh src/cpp build
# at home: binary + selfplay config to the share
python/colab/publish_share.sh
```

Each training session:

```bash
# at home, next to the loop
nohup python/colab/share_sync.sh /home/kenny/q1_run/run4 > /dev/null 2>&1 &
# on ws4, one per VM (NAME is also the worker's incoming dir), e.g. an L4 for 20 hours
cd ~/tmp2/kq_remote/launch && setsid -f ./colab_launch.sh l4a L4 20 > /dev/null 2>&1 < /dev/null
# stop a worker: noticed within 30 s, it sends its last files, the VM is released (~1-2 minutes)
touch ~/tmp2/kq_remote/colab_logs/worker_l4a/STOP
```

Other Colab accounts: log in with that account under another HOME and run the launcher there, e.g.
`HOME=~/colab_acct2 ./colab_launch.sh t4b T4` (colab keeps its login under `$HOME/.config/colab-cli`).

Speed test on a VM without uploading anything: `KQ_BENCH_ROWS=50000` in the worker's environment.

## Experiment jobs

One-off GPU jobs (ladders, evaluations) on a Colab VM, results kept on ws4:

```bash
# put the job in the share: exp/NAME/run.sh plus its inputs (run.sh runs in the job dir, writes into out/,
# $KATAGO is the binary); e.g. a ladder: models/<net>/model.bin.gz, a config with relative model paths
rsync -a NAME ws4:tmp2/kq_remote/share/exp/
# on ws4: a VM, the job, its results to exp/NAME/out/ every 2 minutes, the VM released when run.sh ends
cd ~/tmp2/kq_remote/launch && setsid -f ./colab_job.sh NAME L4 > /dev/null 2>&1 < /dev/null
```

`job_runner.py` runs on the VM (fetches the binary bundle and exp/NAME with the share key, syncs out/ back, writes
out/DONE at the end). Logs: `~/tmp2/kq_remote/colab_logs/job_NAME/`.

## Notes (from KataCDC)

- `colab exec` has a client timeout: long jobs run detached on the VM and are polled with short execs.
- In a remote `subprocess.run(..., shell=True)`, `[k]q_worker.py` is a shell glob; pgrep / pkill are called without a
  shell.
- Costly runs: logs to files, copied back before the VM is released (the launcher does it).
