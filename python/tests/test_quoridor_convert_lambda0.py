"""Tests for quoridor_convert_tdata_lambda0.py (lambda > 0 training rows -> lambda = 0)."""
import glob
import os
import shutil
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import quoridor_convert_tdata_lambda0 as conv  # noqa: E402

LAMBDA = 0.05
RUN3_SELFPLAY = os.path.expanduser("~/q1_run/run3/selfplay")


def synthetic_arrays(rng, n=64):
    """Rows like TrainingWriteBuffers::addRow writes them: decisive main rows, draws, side rows."""
    g = np.zeros((n, 80), dtype=np.float32)
    g[:, 63] = 3.0
    kind = rng.integers(0, 3, size=n)  # 0 decisive main, 1 draw main, 2 side
    kind[:3] = [0, 1, 2]
    for i in range(n):
        if kind[i] == 2:
            g[i, 3] = rng.normal() * 10  # the side search's u
            continue
        g[i, 27] = 1.0
        if kind[i] == 0:
            s = (rng.integers(0, 8) + 0.5) * rng.choice([-1, 1])
            t_end = rng.integers(20, 300)
            g[i, 21] = s
            g[i, 29] = 1.0
            g[i, 20] = s + np.sign(s) * LAMBDA * (300 - t_end)
            g[i, 3] = g[i, 20]
    g[:, 15] = rng.normal(size=n) * 10
    g[:, 19] = rng.normal(size=n) * 10
    return {
        "binaryInputNCHWPacked": rng.integers(0, 256, size=(n, 21, 11), dtype=np.uint8),
        "globalInputNC": rng.normal(size=(n, 19)).astype(np.float32),
        "policyTargetsNCMove": rng.integers(0, 100, size=(n, 2, 18 * 81), dtype=np.int16),
        "globalTargetsNC": g,
    }, kind


def write(path, arrs, compressed):
    (np.savez_compressed if compressed else np.savez)(path, **arrs)


def load(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


@pytest.mark.parametrize("compressed", [False, True])
def test_synthetic_round_trip(tmp_path, compressed):
    rng = np.random.default_rng(1)
    arrs, kind = synthetic_arrays(rng)
    path = str(tmp_path / "a.npz")
    write(path, arrs, compressed)

    assert conv.main(["-dry-run", str(tmp_path)]) == 0
    assert np.array_equal(load(path)["globalTargetsNC"], arrs["globalTargetsNC"])

    assert conv.main([str(tmp_path)]) == 0
    out = load(path)
    assert sorted(out) == sorted(arrs)
    for k in arrs:
        if k != "globalTargetsNC":
            assert np.array_equal(out[k], arrs[k]), k
    g0, g1 = arrs["globalTargetsNC"], out["globalTargetsNC"]
    dec, draw, side = kind == 0, kind == 1, kind == 2
    assert np.array_equal(g1[dec, 20], g0[dec, 21])
    assert np.all(g1[draw, 20] == 0) and np.all(g1[side, 20] == 0)
    assert np.array_equal(g1[dec | draw, 3], g1[dec | draw, 20])
    assert np.array_equal(g1[side, 3], g0[side, 3])
    assert np.all(g1[:, 70] == 1.0)
    changed = np.zeros(80, dtype=bool)
    changed[[3, 20, 70]] = True
    assert np.array_equal(g1[:, ~changed], g0[:, ~changed])
    with zipfile.ZipFile(path) as z:
        assert any(i.compress_type != zipfile.ZIP_STORED for i in z.infolist()) == compressed

    # Idempotent: a second run skips the file.
    status, _, _ = conv.classify_and_convert(out, LAMBDA, 300)
    assert status == "converted"
    assert conv.main([str(tmp_path)]) == 0
    assert np.array_equal(load(path)["globalTargetsNC"], g1)


def test_native_lambda0_file_is_skipped():
    arrs, kind = synthetic_arrays(np.random.default_rng(2))
    g = arrs["globalTargetsNC"]
    g[:, 20] = np.where(g[:, 29] > 0, g[:, 21], 0)
    g[g[:, 27] > 0, 3] = g[g[:, 27] > 0, 20]
    status, stats, _ = conv.classify_and_convert(arrs, LAMBDA, 300)
    assert status == "lambda0" and stats["decisive"] == int((kind == 0).sum())


@pytest.mark.parametrize("breakage", ["v2_inputs", "targets79", "wrong_lambda", "lead_estimate", "half_marked", "c3"])
def test_refuses_unknown_formats(tmp_path, breakage):
    arrs, kind = synthetic_arrays(np.random.default_rng(3))
    g = arrs["globalTargetsNC"]
    if breakage == "v2_inputs":
        arrs["binaryInputNCHWPacked"] = arrs["binaryInputNCHWPacked"][:, :19]
    elif breakage == "targets79":
        arrs["globalTargetsNC"] = g[:, :79]
    elif breakage == "wrong_lambda":
        dec = kind == 0
        g[dec, 20] = g[dec, 21] + np.sign(g[dec, 21]) * 0.03 * 101  # a 0.03 multiple, not 0.05
    elif breakage == "lead_estimate":
        g[kind == 2, 29] = 1.0
    elif breakage == "half_marked":
        g[:5, 70] = 1.0
    elif breakage == "c3":
        g[kind == 1, 3] = 0.25
    path = str(tmp_path / "b.npz")
    write(path, arrs, False)
    before = load(path)
    assert conv.main([str(tmp_path)]) == 1
    after = load(path)
    for k in before:
        assert np.array_equal(before[k], after[k])


def real_files():
    return sorted(glob.glob(os.path.join(RUN3_SELFPLAY, "run3-*", "tdata", "*.npz")))[:3]


@pytest.mark.skipif(not real_files(), reason="no run3 self-play data on this machine")
def test_real_file_only_intended_columns_change(tmp_path):
    for i, src in enumerate(real_files()):
        dst = str(tmp_path / f"r{i}.npz")
        shutil.copyfile(src, dst)
        before = load(dst)
        if np.all(before["globalTargetsNC"][:, 70] == 1.0):
            pytest.skip("run3 data already converted")
        assert conv.main([str(tmp_path)]) == 0
        after = load(dst)
        assert sorted(after) == sorted(before)
        for k in before:
            if k != "globalTargetsNC":
                assert np.array_equal(before[k], after[k]), k
        g0, g1 = before["globalTargetsNC"], after["globalTargetsNC"]
        diff_cols = set(np.nonzero(np.any(g0 != g1, axis=0))[0].tolist())
        assert diff_cols <= {3, 20, 70} and {20, 70} <= diff_cols
        dec = g0[:, 29] > 0
        main = g0[:, 27] > 0
        assert np.array_equal(g1[dec, 20], g0[dec, 21])
        assert np.array_equal(g1[~dec, 20], g0[~dec, 20])
        assert np.array_equal(g1[main, 3], g1[main, 20]) and np.array_equal(g1[~main, 3], g0[~main, 3])
        rows_changed = np.any(g0 != g1, axis=1)
        assert rows_changed.sum() == len(g0)  # C70 on every row
        assert os.path.getsize(dst) < 1.2 * os.path.getsize(src)
