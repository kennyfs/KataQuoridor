"""train.py -stop-at-samples N: stops once the run's total trained samples reach N, in the middle of an epoch if need be, with
validation, a checkpoint and a forced export; a resumed instance continues from there and one at or past N does nothing
(docs/QuoridorIOv4.md). Runs train.py on the CPU on a few writer rows (about a minute)."""
import glob
import os
import subprocess
import sys
import tempfile

import pytest
import torch

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PYTHON_DIR)


def _katago_bin():
    candidates = [os.environ["KATAGO_BIN"]] if os.environ.get("KATAGO_BIN") else (
        glob.glob(os.path.join(REPO_ROOT, "build", "katago")) + glob.glob(os.path.join(REPO_ROOT, "cpp", "build*", "katago")))
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        pytest.skip("KataGo binary not found (set KATAGO_BIN or build under cpp/build*/).")
    return max(candidates, key=os.path.getmtime)


def _run(cmd, **kw):
    res = subprocess.run(cmd, capture_output=True, text=True, **kw)
    assert res.returncode == 0, f"{cmd} failed:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}"
    return res


def test_stop_at_samples():
    with tempfile.TemporaryDirectory() as d:
        raw, data, scratch = (os.path.join(d, x) for x in ("raw", "shuffleddata/cur", "scratch"))
        os.makedirs(raw)
        os.makedirs(data)
        os.makedirs(scratch)
        _run([_katago_bin(), "writesampletrainquoridor", raw, "4", "200"])
        for sub in ("train", "val"):
            _run([sys.executable, os.path.join(PYTHON_DIR, "shuffle.py"), raw, "-min-rows", "1", "-keep-target-rows", "all",
                  "-num-processes", "1", "-approx-rows-per-out-file", "200", "-out-dir", os.path.join(data, sub),
                  "-out-tmp-dir", os.path.join(scratch, sub)])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")  # CPU

        def train(stop):
            args = [sys.executable, os.path.join(PYTHON_DIR, "train.py"), "-traindir", os.path.join(d, "train"),
                    "-latestdatadir", os.path.join(d, "shuffleddata"), "-exportdir", os.path.join(d, "export"), "-exportprefix", "t",
                    "-pos-len", "9", "-batch-size", "50", "-model-kind", "b2c64_quoridor_v4", "-samples-per-epoch", "400",
                    "-quit-if-no-data", "-max-val-samples", "100", "-swa-period-samples", "100", "-lr-scale", "1"]
            if stop is not None:
                args += ["-stop-at-samples", stop]
            _run(args, env=env, cwd=PYTHON_DIR)
            ckpt = torch.load(os.path.join(d, "train", "checkpoint.ckpt"), map_location="cpu", weights_only=False)
            return ckpt["train_state"]["global_step_samples"], sorted(os.listdir(os.path.join(d, "export")))

        n, exports = train("300")  # mid-epoch (an epoch is >= 400 samples)
        assert n == 300 and any(e.startswith("t-s300-") for e in exports), (n, exports)
        n, exports = train("450")  # resumes at 300
        assert n == 450 and any(e.startswith("t-s450-") for e in exports), (n, exports)
        n2, exports2 = train("450")  # at N already: nothing to do
        assert n2 == 450 and exports2 == exports
