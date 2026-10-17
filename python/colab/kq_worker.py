#!/usr/bin/env python3
"""KataQuoridor self-play worker for a Colab GPU VM (or any Linux GPU machine).

Data flow (see README.md next to this file): the trainer at home publishes the newest net to a share on the
workstation; workers rsync the share's binary, config and net, run `katago selfplay` (which loads a newer net by
itself whenever one appears in its models dir), and every few minutes move their finished games and training rows
to the share's incoming/<worker>/. The home side pulls those into the run's selfplay dir.

The SSH key gives access to the share only (rrsync in the workstation's authorized_keys).

Usage on Colab (CLI): colab upload KEY /content/kq_worker_key; colab exec -f kq_worker.py   (settings below or as
environment variables: KQ_WORKER, KQ_MINUTES, KQ_BENCH_ROWS, KQ_THREADS, KQ_BATCH)
In a notebook: paste this file into a cell, set the settings, run.
"""
import os, re, shutil, signal, subprocess, sys, threading, time

# ---- settings -------------------------------------------------------------------------------------------------
HOST = os.environ.get("KQ_HOST", "r15922041@ws4.csie.ntu.edu.tw")
# the share's path on HOST: "" with the restricted worker key (rrsync roots it at the share), else e.g. "tmp2/kq_remote/share/"
ROOT = os.environ.get("KQ_ROOT", "")
KEY = os.environ.get("KQ_KEY", "/content/kq_worker_key")
WORKER = os.environ.get("KQ_WORKER", "")          # name of this worker's incoming dir; default: colab-<gpu>-<time>
MINUTES = float(os.environ.get("KQ_MINUTES", "0"))  # stop after this many minutes (0 = until stopped)
BENCH_ROWS = int(os.environ.get("KQ_BENCH_ROWS", "0"))  # > 0: speed test only, this many rows, nothing uploaded
THREADS = int(os.environ.get("KQ_THREADS", "0"))   # numGameThreads (0 = by GPU, see TUNING)
BATCH = int(os.environ.get("KQ_BATCH", "0"))       # nnMaxBatchSize (0 = by GPU)
SYNC_SECONDS = 120
WORK = "/content/kq" if os.path.isdir("/content") else os.path.expanduser("~/kq_worker")
# GPU name substring -> (numGameThreads, nnMaxBatchSize); the RTX 3070 at home runs 256 / 256 (2026-10-18, 12 CPUs:
# 512 game threads was 18% slower than 256). KataCDC's L4 (12 CPUs, a smaller board) was faster at 512; L4 starts at
# 256 here until measured (KQ_BENCH_ROWS, or KQ_THREADS / KQ_BATCH for a worker).
TUNING = [("T4", 256, 256), ("L4", 256, 256), ("A100", 512, 512), ("H100", 768, 768), ("RTX PRO 6000", 768, 768),
          ("", 256, 256)]

SSH = f"ssh -i {KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20"


def sh(cmd, check=True, **kw):
    return subprocess.run(cmd, shell=True, check=check, text=True, capture_output=True, **kw)


def rsync(args, tries=5):
    for i in range(tries):
        r = sh(f"rsync -a --timeout=120 -e '{SSH}' {args}", check=False)
        if r.returncode == 0:
            return True
        print(f"rsync failed ({r.returncode}): {r.stderr.strip()[-300:]}", flush=True)
        time.sleep(10 * (i + 1))
    return False


def gpu_name():
    r = sh("nvidia-smi --query-gpu=name --format=csv,noheader", check=False)
    return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""


def setup():
    os.makedirs(WORK, exist_ok=True)
    os.chmod(KEY, 0o600)
    if shutil.which("rsync") is None:
        sh("apt-get install -y -qq rsync > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync)")
    # binary bundle (only when it changed), config, newest net
    os.chdir(WORK)
    if not rsync(f"{HOST}:{ROOT}bin/katago-kq-colab.tar.gz {WORK}/"):
        sys.exit("cannot fetch the binary from the share")
    stamp = os.path.getmtime("katago-kq-colab.tar.gz")
    if not os.path.exists("bin/katago") or os.path.getmtime("bin/katago") < stamp:
        shutil.rmtree("bin", ignore_errors=True)
        os.makedirs("bin")
        sh("tar -C bin -xzf katago-kq-colab.tar.gz && touch bin/katago")
    rsync(f"{HOST}:{ROOT}selfplay_quoridor_v2.cfg {WORK}/")
    os.makedirs("models", exist_ok=True)
    rsync(f"--delete {HOST}:{ROOT}models/ {WORK}/models/")
    # hint poses (hintPosesDir in the config points at the home run; publish_share.sh copies them to the share)
    os.makedirs("hintposes", exist_ok=True)
    rsync(f"--delete {HOST}:{ROOT}hintposes/ {WORK}/hintposes/", tries=2)
    if not os.listdir("models"):
        sys.exit("no net in the share's models/")


def push_finished(out_dir, dest, all_files=False):
    """Moves files untouched for 3 minutes (finished: selfplay never writes to them again), or every file once selfplay
    has ended, to the share."""
    lst = os.path.join(WORK, "finished.txt")
    age = "" if all_files else "-mmin +3"
    r = sh(f"cd {out_dir} && find . -type f {age} \\( -name '*.npz' -o -name '*.sgfs' \\) -printf '%P\\n'", check=False)
    files = [f for f in r.stdout.split() if f]
    if not files:
        return 0
    with open(lst, "w") as f:
        f.write("\n".join(files) + "\n")
    ok = rsync(f"--files-from={lst} --remove-source-files {out_dir}/ {HOST}:{ROOT}incoming/{dest}/")
    return len(files) if ok else 0


def main():
    # SIGTERM (colab_launch.sh stopping the worker) ends like Ctrl-C: selfplay stops, the last files are sent
    signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    gpu = gpu_name()
    if not gpu:
        sys.exit("no NVIDIA GPU")
    th, bt = next((t, b) for k, t, b in TUNING if k in gpu)
    th, bt = THREADS or th, BATCH or bt
    worker = WORKER or "colab-" + re.sub(r"[^A-Za-z0-9]+", "", gpu.split()[-1]) + time.strftime("-%m%d%H%M")
    print(f"GPU {gpu} · {os.cpu_count()} CPUs · numGameThreads {th} · nnMaxBatchSize {bt} · worker {worker}", flush=True)
    setup()
    out = os.path.join(WORK, "out")
    os.makedirs(out, exist_ok=True)
    over = f"numGameThreads={th},nnMaxBatchSize={bt},numNNServerThreadsPerModel=2"
    if re.search(r"^\s*hintPosesDir\s*=", open(os.path.join(WORK, "selfplay_quoridor_v2.cfg")).read(), re.M):
        over += f",hintPosesDir={os.path.join(WORK, 'hintposes')}"
    cmd = [os.path.join(WORK, "bin", "run_katago.sh"), "selfplay", "-models-dir", os.path.join(WORK, "models"),
           "-output-dir", out, "-config", os.path.join(WORK, "selfplay_quoridor_v2.cfg"), "-override-config", over,
           "-max-games-total", "100000000"]
    if BENCH_ROWS:
        cmd += ["-max-rows-total", str(BENCH_ROWS)]
    log = open(os.path.join(WORK, "selfplay.log"), "a")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=WORK)

    stop = threading.Event()

    def sync_loop():
        moved = 0
        while not stop.wait(SYNC_SECONDS):
            if BENCH_ROWS:
                continue
            moved += push_finished(out, worker)
            rsync(f"--delete {HOST}:{ROOT}models/ {WORK}/models/", tries=2)   # selfplay picks a newer net up itself
            print(f"{time.strftime('%H:%M')} {(time.time() - t0) / 60:.0f} min · {moved} files sent", flush=True)
    th_sync = threading.Thread(target=sync_loop, daemon=True)
    th_sync.start()
    try:
        while proc.poll() is None:
            time.sleep(10)
            if MINUTES and time.time() - t0 > MINUTES * 60:
                proc.terminate()   # unfinished games are lost; finished files stay and are sent below
                proc.wait(timeout=120)
    except KeyboardInterrupt:
        print("stopping", flush=True)
        proc.terminate()
        proc.wait(timeout=120)
    stop.set()
    if not BENCH_ROWS:
        # selfplay has ended: every file is finished
        print(f"final sync: {push_finished(out, worker, all_files=True)} files", flush=True)
    log.close()
    # speed summary from selfplay's final stats
    text = open(os.path.join(WORK, "selfplay.log")).read()
    rows = re.findall(r"Final NN rows: (\d+)", text)
    secs = re.findall(r"Total selfplay runtime \(seconds\): ([\d.]+)", text)
    if rows and secs:
        print(f"{gpu}: {int(rows[-1]) / float(secs[-1]):.0f} NN evals/s over {float(secs[-1]):.0f} s "
              f"(numGameThreads {th}, nnMaxBatchSize {bt})", flush=True)
    else:
        print("selfplay ended; last log lines:\n" + "\n".join(text.splitlines()[-15:]), flush=True)


if __name__ == "__main__":
    main()
