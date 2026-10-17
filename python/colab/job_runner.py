#!/usr/bin/env python3
"""Runs an experiment job on a Colab VM (README.md, "Experiment jobs"): fetches the KataQuoridor binary bundle and the
share's exp/<NAME>/ (run.sh plus its inputs) into /content/job, runs `bash run.sh` there (env KATAGO = the binary),
and copies /content/job/out/ back to exp/<NAME>/out/ every 2 minutes and at the end (then out/DONE).
Usage: job_runner.py NAME   (the share key at /content/kq_worker_key, as for kq_worker.py)"""
import os, subprocess, sys, threading, time, shutil
NAME = sys.argv[1]
HOST = os.environ.get("KQ_HOST", "r15922041@ws4.csie.ntu.edu.tw")
KEY = os.environ.get("KQ_KEY", "/content/kq_worker_key")
JOB = "/content/job"
SSH = f"ssh -i {KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20"

def rsync(args, tries=5):
    for i in range(tries):
        r = subprocess.run(f"rsync -a --timeout=120 -e '{SSH}' {args}", shell=True, capture_output=True, text=True)
        if r.returncode == 0: return True
        print(f"rsync failed ({r.returncode}): {r.stderr.strip()[-300:]}", flush=True); time.sleep(10 * (i + 1))
    return False

os.chmod(KEY, 0o600)
if shutil.which("rsync") is None:
    subprocess.run("apt-get install -y -qq rsync > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync)", shell=True)
os.makedirs(f"{JOB}/out", exist_ok=True)
if not rsync(f"{HOST}:bin/katago-kq-colab.tar.gz {JOB}/") or not rsync(f"--exclude out {HOST}:exp/{NAME}/ {JOB}/"):
    sys.exit("cannot fetch the binary or the job")
os.makedirs(f"{JOB}/bin", exist_ok=True)
subprocess.run(f"tar -C {JOB}/bin -xzf {JOB}/katago-kq-colab.tar.gz", shell=True, check=True)
stop = threading.Event()
def sync():
    while not stop.wait(120):
        rsync(f"{JOB}/out/ {HOST}:exp/{NAME}/out/", tries=2)
threading.Thread(target=sync, daemon=True).start()
print(f"{time.strftime('%F %T')} running run.sh", flush=True)
env = dict(os.environ, KATAGO=f"{JOB}/bin/run_katago.sh")
code = subprocess.run("bash run.sh > out/run.log 2>&1", shell=True, cwd=JOB, env=env).returncode
stop.set()
open(f"{JOB}/out/DONE", "w").write(f"exit {code} {time.strftime('%F %T')}\n")
print(f"{time.strftime('%F %T')} run.sh exited {code}; final sync {rsync(f'{JOB}/out/ {HOST}:exp/{NAME}/out/')}", flush=True)
