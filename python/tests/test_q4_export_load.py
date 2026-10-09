"""T16: export a random Q4 model, load it with the C++ engine (Eigen and CUDA), and the error cases:
a Q4 net on the Duel path, a Duel or Go net on the Q4 path, wrong channel counts and unsupported versions."""

import gzip
import json
import os
import subprocess
import sys

import pytest

from q4_testutil import REPO_DIR, find_katago, run
from q4.make_random_model import export_random_model

PYTHON_DIR = os.path.join(REPO_DIR, "python")
GO_MODEL = os.path.join(REPO_DIR, "cpp", "tests", "models", "g170-b6c96-s175395328-d26788732.bin.gz")


@pytest.fixture(scope="module")
def q4_model(tmp_path_factory):
    return export_random_model("b1c32_q4", str(tmp_path_factory.mktemp("q4model")), "q4test", seed=123)


@pytest.fixture(scope="module")
def duel_model(tmp_path_factory):
    outdir = str(tmp_path_factory.mktemp("duelmodel"))
    run([sys.executable, os.path.join(PYTHON_DIR, "export_model_pytorch.py"),
         "-export-random-initialized-model", "b2c64_quoridor_v3", "-export-dir", outdir, "-model-name", "duel",
         "-filename-prefix", "model"])
    return os.path.join(outdir, "model.bin.gz")


@pytest.fixture(scope="module")
def eigen():
    path = find_katago("eigen")
    if path is None:
        pytest.skip("no Eigen katago build")
    return path


def evalnn(katago, model):
    return subprocess.run([katago, "q4tool", "evalnn", "-model", model, "-json"], input="{}\n",
                          capture_output=True, text=True)


def test_exported_header_is_q4(q4_model):
    with gzip.open(q4_model, "rb") as f:
        head = f.read(400).split(b"\n")
    # model name, version 17, spatial and global input channels, ... option D (101) a few lines later
    assert head[1] == b"17"
    assert head[2] == b"27" and head[3] == b"28"
    assert b"101" in head[4:16]


@pytest.mark.parametrize("kind", ["eigen", "cuda"])
def test_export_and_load_t16(q4_model, kind):
    katago = find_katago(kind)
    if katago is None:
        pytest.skip(f"no {kind} katago build")
    res = evalnn(katago, q4_model)
    assert res.returncode == 0, res.stderr
    out = json.loads([l for l in res.stdout.splitlines() if l.startswith("{")][0])
    assert len(out["raw"]["policy"]) == 726 and len(out["raw"]["value"]) == 5
    assert len(out["raw"]["misc"]) == 6 and len(out["raw"]["trajectory"]) == 121
    assert abs(sum(out["decoded"]["valueAbs"]) - 1.0) < 1e-5
    # the start position has 4 pawn moves... and 200 walls: 3 pawn moves of seat 1 (f1: e1, g1, f2) + 200 walls
    assert len(out["decoded"]["legalActions"]) == 203


def test_q4_net_on_duel_path_is_refused(eigen, q4_model, tmp_path):
    res = subprocess.run([eigen, "evalnnparity", "-model", q4_model, "-n", "2", "-output", str(tmp_path / "o.npz")],
                         capture_output=True, text=True)
    text = res.stdout + res.stderr
    assert res.returncode != 0
    assert "Q4 (four-player) network" in text and "Duel path" in text, text[-1500:]


def test_duel_net_on_q4_path_is_refused(eigen, duel_model):
    res = evalnn(eigen, duel_model)
    text = res.stdout + res.stderr
    assert res.returncode != 0
    assert "not a Q4" in text and "option D 3" in text and ">= 100" in text, text[-1500:]


def test_go_net_on_q4_path_is_refused(eigen):
    if not os.path.isfile(GO_MODEL):
        pytest.skip("no Go test model")
    res = evalnn(eigen, GO_MODEL)
    text = res.stdout + res.stderr
    assert res.returncode != 0
    assert "unsupported Quoridor I/O version 0" in text, text[-1500:]


def patched_model(src, dst, old, new):
    """Copy of an exported model with the first occurrence of the header bytes `old` replaced by `new`."""
    with gzip.open(src, "rb") as f:
        data = f.read()
    assert old in data[:2000]
    with gzip.open(dst, "wb") as f:
        f.write(data.replace(old, new, 1))
    return dst


def test_wrong_channel_counts_are_refused(eigen, q4_model, duel_model, tmp_path):
    # 26 instead of 27 spatial input channels in a Q4 model; the header lines are "17\n27\n28\n"
    bad = patched_model(q4_model, str(tmp_path / "spatial26.bin.gz"), b"17\n27\n28\n", b"17\n26\n28\n")
    text = (lambda r: r.stdout + r.stderr)(evalnn(eigen, bad))
    assert "Q4 I/O version 101 expects 27 spatial and 28 global input channels, model has 26 and 28" in text, text[-1500:]
    # a Duel (I/O v3, 21 spatial and 19 global channels) net claiming to be Q4
    with gzip.open(duel_model, "rb") as f:
        head = f.read(2000)
    assert b"\n21\n19\n" in head
    lines = head.split(b"\n")
    option_d = [i for i, l in enumerate(lines[:30]) if l == b"3"]
    assert option_d, "option D = 3 not found in the header"
    with gzip.open(duel_model, "rb") as f:
        data = f.read()
    pos = 0
    for i in option_d[-1:]:
        pos = len(b"\n".join(lines[:i])) + 1
    patched = data[:pos] + b"101" + data[pos + 1:]
    bad_path = str(tmp_path / "duel_as_q4.bin.gz")
    with gzip.open(bad_path, "wb") as f:
        f.write(patched)
    text = (lambda r: r.stdout + r.stderr)(evalnn(eigen, bad_path))
    assert "Q4 I/O version 101 expects 27 spatial and 28 global input channels, model has 21 and 19" in text, text[-1500:]


def patch_option_d(q4_model, path, new_value):
    with gzip.open(q4_model, "rb") as f:
        data = f.read()
    lines = data[:2000].split(b"\n")
    idx = [i for i, l in enumerate(lines[:30]) if l == b"101"]
    assert len(idx) == 1
    pos = len(b"\n".join(lines[:idx[0]])) + 1
    with gzip.open(path, "wb") as f:
        f.write(data[:pos] + new_value + data[pos + 3:])
    return path


def test_unsupported_q4_version_is_refused(eigen, q4_model, tmp_path):
    res = evalnn(eigen, patch_option_d(q4_model, str(tmp_path / "v103.bin.gz"), b"103"))
    assert res.returncode != 0 and "Q4 model option D (quoridorIOVersion) unsupported" in res.stdout + res.stderr


def test_q4_v2_without_a_metadata_encoder_is_refused(eigen, q4_model, tmp_path):
    """I/O v2 = v1 + the metadata encoder: a v1 net relabeled 102 has none (Round 7)."""
    res = evalnn(eigen, patch_option_d(q4_model, str(tmp_path / "v102.bin.gz"), b"102"))
    assert res.returncode != 0 and "Q4 I/O version 102 needs a metadata encoder but the model has no one" in res.stdout + res.stderr


def test_truncated_model_is_refused(eigen, q4_model, tmp_path):
    with gzip.open(q4_model, "rb") as f:
        data = f.read()
    path = str(tmp_path / "truncated.bin.gz")
    with gzip.open(path, "wb") as f:
        f.write(data[: len(data) // 2])
    assert evalnn(eigen, path).returncode != 0
