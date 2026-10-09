"""Helpers shared by the Q4 tests: binaries, positions from `q4tool dumpinputs`, random exported models, evalnn."""

import json
import os
import subprocess
import sys

import numpy as np
import torch

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)

from q4.make_random_model import export_model, make_random_model  # noqa: E402,F401  (re-exported for the tests)
from q4.reference import Pos, eliminate, play, str_to_action  # noqa: E402


def find_katago(kind):
    """The katago binary of a backend: 'eigen' (build-eigen, else build-eigen-release) or 'cuda' (build-cuda).
    $Q4_EIGEN_BIN / $Q4_CUDA_BIN override. Returns None if there is none."""
    env = os.environ.get("Q4_EIGEN_BIN" if kind == "eigen" else "Q4_CUDA_BIN")
    if env:
        return env
    names = ["build-eigen", "build-eigen-release"] if kind == "eigen" else ["build-cuda"]
    for name in names:
        path = os.path.join(REPO_DIR, "cpp", name, "katago")
        if os.path.isfile(path):
            return path
    return None


def run(cmd, **kwargs):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        raise AssertionError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}\n{tail}")
    return proc


def pos_of_dump(data):
    """The python reference position of a `q4tool dumpinputs` line (rules + events)."""
    rules = data.get("rules", {})
    pos = Pos(
        max_plies=rules.get("maxPlies", 400),
        repetition_draw_count=rules.get("repetitionDrawCount", 0),
        initial_walls=rules.get("initialWalls", [7, 7, 7, 7]),
    )
    for ev in data.get("events", []):
        if "elim" in ev:
            pos = eliminate(pos, ev["elim"])
        else:
            pos = play(pos, str_to_action(ev["action"]))
    return pos


def dump_positions(katago, n, seed):
    """n positions from random games (random walks, pawn shuffles for repetitions, positions near maxPlies, walls,
    eliminated seats): a list of (json line dict, Pos)."""
    out = run([katago, "q4tool", "dumpinputs", "-random", "-n", str(n), "-seed", str(seed)]).stdout
    result = []
    for line in out.splitlines():
        if line.strip():
            data = json.loads(line)
            result.append((data, pos_of_dump(data)))
    return result


def run_evalnn(katago, model_path, queries, config_lines=None, env=None):
    """Runs `q4tool evalnn` on query dicts ({"events": [...], "rules": {...}, "sym": s}); returns the result dicts."""
    cmd = [katago, "q4tool", "evalnn", "-model", model_path, "-json"]
    config_path = None
    if config_lines:
        config_path = model_path + ".evalnn.cfg"
        with open(config_path, "w") as f:
            f.write("\n".join(config_lines) + "\n")
        cmd += ["-config", config_path]
    text = "\n".join(json.dumps(q) for q in queries) + "\n"
    proc = subprocess.run(cmd, input=text, capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    if proc.returncode != 0:
        raise AssertionError(f"evalnn failed ({proc.returncode}): {proc.stderr[-2000:]}")
    results = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    assert len(results) == len(queries), f"{len(results)} results for {len(queries)} queries: {proc.stderr[-1000:]}"
    return results


def style_metadata_of_dump(data):
    """The metadata row [192] of a Q4 I/O v2 net for the position of a `q4tool dumpinputs` line: the style features
    (python/q4/style.py, computed from the events) of the seat to move at slots 0..75, zeros elsewhere."""
    from q4.style import record_features
    events = [{"elim": ev["elim"] + 1} if "elim" in ev else {"a": ev["action"]} for ev in data.get("events", [])]
    rules = data.get("rules", {})
    persp, feats = record_features({"rules": rules, "events": events})
    meta = np.zeros(192, dtype=np.float32)
    meta[:76] = feats[-1]
    return meta


def torch_raw_outputs(model, spatial, glob_, meta=None):
    """The model's raw outputs for numpy inputs [n,27,11,11], [n,28] (and [n,192] metadata for a Q4 I/O v2 net):
    policy [n,6,11,11] (channel = variant*3 + plane), value logits [n,5], misc [n,6], trajectory logits [n,11,11], as
    float64 arrays."""
    with torch.no_grad():
        out = model(torch.from_numpy(np.ascontiguousarray(spatial, dtype=np.float32)),
                    torch.from_numpy(np.ascontiguousarray(glob_, dtype=np.float32)),
                    None if meta is None else torch.from_numpy(np.ascontiguousarray(meta, dtype=np.float32)))
    policy, value, misc, traj = out[0][0], out[0][1], out[0][2], out[0][3]
    return (policy.reshape(policy.shape[0], 6, 11, 11).double().numpy(), value.double().numpy(),
            misc.double().numpy(), traj[:, 0].double().numpy())
