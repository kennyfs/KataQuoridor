"""Quoridor I/O v4 (docs/QuoridorIOv4.md): the data converter (identical to what the C++ writer produces, on writer rows and
on real self-play rows), the wall-effect planes under the mirror symmetry, the v3 -> v4 checkpoint migration (same outputs on
v3-equivalent inputs, optimizer state loads), and the shuffler carrying wallEffectNCHW."""
import copy
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch
from torch.optim.swa_utils import AveragedModel

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)

from katago.train import modelconfigs  # noqa: E402
from katago.train.model_pytorch import Model  # noqa: E402
from katago.train.data_processing_pytorch import (  # noqa: E402
    apply_symmetry_quoridor,
    decode_binary_input,
    zero_quoridor_wall_effect_inputs,
)
import quoridor_convert_tdata_v3_to_v4 as convert  # noqa: E402
import quoridor_upgrade_v3_to_v4 as upgrade  # noqa: E402
import shuffle  # noqa: E402


def _katago_bin():
    candidates = [os.environ["KATAGO_BIN"]] if os.environ.get("KATAGO_BIN") else (
        glob.glob(os.path.join(REPO_ROOT, "build", "katago")) + glob.glob(os.path.join(REPO_ROOT, "cpp", "build*", "katago")))
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        pytest.skip("KataGo binary not found (set KATAGO_BIN or build under cpp/build*/).")
    return max(candidates, key=os.path.getmtime)


def _run(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"{cmd} failed:\n{res.stdout}\n{res.stderr}"


def _load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def dumped_rows():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rows.npz")
        _run([_katago_bin(), "dumpnninputs", "-n", "400", "-seed", "v4tests", "-output", path])
        yield _load(path)


# --- an independent brute force of the wall effects, from the stored canonical planes only -------------------------------

def _bfs_dist(open_edges, sources_row, pawn):
    """Shortest distance from pawn (c, r) to row sources_row with open_edges[(c, r, dc, dr)] -> bool, -1 if unreachable."""
    from collections import deque
    seen = {pawn: 0}
    q = deque([pawn])
    while q:
        c, r = q.popleft()
        if r == sources_row:
            return seen[(c, r)]
        for dc, dr in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            n = (c + dc, r + dr)
            if 0 <= n[0] < 9 and 0 <= n[1] < 9 and n not in seen and open_edges((c, r), n):
                seen[n] = seen[(c, r)] + 1
                q.append(n)
    return -1


def brute_force_wall_effects(planes):
    """planes: bool [C, 9, 9] of one row (canonical). Returns int [4, 9, 9]: effects, -1 for illegal, 0 off the anchor grid."""
    cur = tuple(int(x) for x in np.argwhere(planes[1])[0][::-1])  # (c, r)
    opp = tuple(int(x) for x in np.argwhere(planes[2])[0][::-1])
    hw = {(int(c), int(r)) for r, c in np.argwhere(planes[15][:8, :8])}
    vw = {(int(c), int(r)) for r, c in np.argwhere(planes[14][:8, :8])}

    def edge_open(hs, vs):
        def f(a, b):
            if b[1] == a[1] + 1 or a[1] == b[1] + 1:  # vertical step between rows min(r): blocked by horizontal walls
                r = min(a[1], b[1])
                c = a[0]
                return not ((c, r) in hs or (c - 1, r) in hs)
            c = min(a[0], b[0])
            r = a[1]
            return not ((c, r) in vs or (c, r - 1) in vs)
        return f

    out = np.zeros((4, 9, 9), dtype=np.int64)
    base = (_bfs_dist(edge_open(hw, vw), 0, cur), _bfs_dist(edge_open(hw, vw), 8, opp))
    for r in range(8):
        for c in range(8):
            for k, horizontal in enumerate((False, True)):
                if (c, r) in hw or (c, r) in vw:
                    legal = False
                elif horizontal and ((c - 1, r) in hw or (c + 1, r) in hw):
                    legal = False
                elif not horizontal and ((c, r - 1) in vw or (c, r + 1) in vw):
                    legal = False
                else:
                    hs, vs = (hw | {(c, r)}, vw) if horizontal else (hw, vw | {(c, r)})
                    d = (_bfs_dist(edge_open(hs, vs), 0, cur), _bfs_dist(edge_open(hs, vs), 8, opp))
                    legal = d[0] >= 0 and d[1] >= 0
                if not legal:
                    out[k, r, c] = out[2 + k, r, c] = -1
                else:
                    out[k, r, c] = d[0] - base[0]
                    out[2 + k, r, c] = d[1] - base[1]
    return out


def _check_against_brute_force(packed, raw, n_check):
    planes = convert.unpack_planes(packed[:n_check])
    for i in range(n_check):
        want = brute_force_wall_effects(planes[i])
        got = raw[i].astype(np.int64)
        got = np.where(got == 255, -1, got)
        assert np.array_equal(got, want), f"row {i} differs"


def test_converter_matches_writer_and_brute_force(dumped_rows):
    z = dumped_rows
    packed = z["trainBinaryInputNCHWPacked"]
    raw = convert.compute_wall_effects(packed[:, :21], _katago_bin())  # from the v3 part of the rows
    assert np.array_equal(raw, z["trainWallEffectNCHW"])
    assert np.array_equal(convert.wall_effect_globals(raw), z["trainGlobalInputNC"][:, 19:21])
    assert (raw > 0).any() and (raw == 255).any() and raw.max() >= 3
    _check_against_brute_force(packed, raw, 150)
    # The training loader rebuilds exactly the inference row.
    got = decode_binary_input(packed, z["trainSpatialDistNCHW"], 9, wallEffectNCHW=raw)
    assert np.array_equal(got, z["binaryInputNCHW"])


def _real_selfplay_file():
    files = sorted(glob.glob(os.path.expanduser("~/q1_run/run4/selfplay/*/tdata/*.npz")))
    for f in files[-30:]:
        with np.load(f) as d:
            if d["binaryInputNCHWPacked"].shape[1] == 21 and d["binaryInputNCHWPacked"].shape[0] >= 60:
                return f
    return None


def test_converter_on_real_rows():
    f = _real_selfplay_file()
    if f is None:
        pytest.skip("no run4 self-play file")
    arrs = _load(f)
    out, num_draw = convert.convert_arrays(arrs, _katago_bin())
    assert out["binaryInputNCHWPacked"].shape[1] == 25 and out["globalInputNC"].shape[1] == 21
    assert not convert.unpack_planes(out["binaryInputNCHWPacked"])[:, 21:].any()
    _check_against_brute_force(out["binaryInputNCHWPacked"], out["wallEffectNCHW"], 60)
    # every other array is carried over untouched
    for k in arrs:
        if k not in ("binaryInputNCHWPacked", "globalInputNC", "globalTargetsNC"):
            assert np.array_equal(out[k], arrs[k]), k
    assert np.array_equal(out["globalInputNC"][:, :19], arrs["globalInputNC"])


def _to_v3_form(arrays):
    """What the v3 writer produced for the same rows: no wall effects, 21 / 19 input channels, a draw as win 0.5 / loss 0.5."""
    old = {k: v for k, v in arrays.items() if k != "wallEffectNCHW"}
    old["binaryInputNCHWPacked"] = arrays["binaryInputNCHWPacked"][:, :21]
    old["globalInputNC"] = arrays["globalInputNC"][:, :19]
    g = arrays["globalTargetsNC"].copy()
    draw = (g[:, 2] == 1.0)
    g[draw, 0] = 0.5
    g[draw, 1] = 0.5
    g[draw, 2] = 0.0
    old["globalTargetsNC"] = g
    return old


def test_converter_identical_to_writer():
    with tempfile.TemporaryDirectory() as d:
        sample = os.path.join(d, "sample")
        _run([_katago_bin(), "writesampletrainquoridor", sample, "3", "200"])
        files = sorted(glob.glob(os.path.join(sample, "*.npz")))
        assert files
        total = draws = 0
        v3dir = os.path.join(d, "v3")
        os.makedirs(v3dir)
        for f in files:
            arrays = _load(f)
            assert "wallEffectNCHW" in arrays and arrays["binaryInputNCHWPacked"].shape[1] == 25
            draws += int((arrays["globalTargetsNC"][:, 2] == 1.0).sum())
            np.savez_compressed(os.path.join(v3dir, os.path.basename(f)), **_to_v3_form(arrays))
            total += arrays["globalInputNC"].shape[0]
        assert total >= 400 and draws > 0
        outdir = os.path.join(d, "out")
        _run([sys.executable, os.path.join(PYTHON_DIR, "quoridor_convert_tdata_v3_to_v4.py"), "-katago", _katago_bin(),
              "-num-processes", "2", "-out-dir", outdir, v3dir])
        for f in files:
            want = _load(f)
            got = _load(os.path.join(outdir, "v3", os.path.basename(f)))
            assert set(got) == set(want)
            for k in want:
                assert got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k]), f"{f}: {k} differs"
        # Idempotent in place: a second conversion of the converted copy skips every file.
        res = subprocess.run([sys.executable, os.path.join(PYTHON_DIR, "quoridor_convert_tdata_v3_to_v4.py"), "-katago",
                              _katago_bin(), os.path.join(outdir, "v3")], capture_output=True, text=True)
        assert res.returncode == 0 and f"0 v3 files converting, {len(files)} already v4" in res.stdout, res.stdout


def test_mirror_symmetry_of_wall_effects(dumped_rows):
    # apply_symmetry_quoridor (the training loader's mirror) of the planes of a position == the planes of the mirrored
    # position, which the brute force computes independently.
    z = dumped_rows
    packed = z["trainBinaryInputNCHWPacked"]
    ref = z["binaryInputNCHW"]
    planes = convert.unpack_planes(packed)
    mirrored_ok = 0
    for i in range(120):
        p = planes[i].copy()
        m = np.zeros_like(p)
        # mirror the board by hand: pawns c -> 8 - c, walls c -> 7 - c
        m[1] = p[1][:, ::-1]
        m[2] = p[2][:, ::-1]
        m[14, :8, :8] = p[14, :8, :8][:, ::-1]
        m[15, :8, :8] = p[15, :8, :8][:, ::-1]
        want = brute_force_wall_effects(m)
        got = apply_symmetry_quoridor(torch.from_numpy(ref[i:i + 1]), 1).numpy()[0, 21:25]
        wantf = np.where(want < 0, -1.0, np.minimum(want, 8) / 8.0).astype(np.float32)
        wantf[:, 8, :] = 0.0
        wantf[:, :, 8] = 0.0
        assert np.array_equal(got, wantf), f"row {i}"
        mirrored_ok += 1
    assert mirrored_ok == 120


def test_zero_wall_effect_inputs(dumped_rows):
    z = dumped_rows
    spatial, glob_ = z["binaryInputNCHW"].copy(), z["globalInputNC"].copy()
    assert spatial[:, 21:25].any() and glob_[:, 19:21].any()
    zero_quoridor_wall_effect_inputs(spatial, glob_)
    assert not spatial[:, 21:25].any() and not glob_[:, 19:21].any()
    assert np.array_equal(spatial[:, :21], z["binaryInputNCHW"][:, :21]) and np.array_equal(glob_[:, :19], z["globalInputNC"][:, :19])


# --- checkpoint migration ------------------------------------------------------------------------------------------------

def make_v3_checkpoint(name):
    torch.manual_seed(11)
    config = copy.deepcopy(modelconfigs.config_of_name[name])
    model = Model(config, pos_len=9)
    model.initialize()
    with torch.no_grad():
        model.value_head.linear_value.weight.mul_(5.0)
        model.value_head.linear_td_value.bias.normal_()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for p in model.parameters():
        opt.state[p]["momentum_buffer"] = torch.randn_like(p) * 1e-3
    swa = AveragedModel(model)
    return {"model": model.state_dict(), "swa_model": swa.state_dict(), "optimizer": opt.state_dict(),
            "config": config, "train_state": {"global_step_samples": 4321}}


def post_outputs(model, spatial, glob_):
    model.eval()
    with torch.no_grad():
        return model.postprocess_output(model(spatial, glob_))[0]


@pytest.mark.parametrize("name", ["b2c64_quoridor_v3", "tf2_b4c192_quoridor_v3"])
def test_upgrade_v3_to_v4(name):
    data = make_v3_checkpoint(name)
    v3 = Model(data["config"], pos_len=9)
    v3.load_state_dict(data["model"])
    upgraded = upgrade.upgrade_checkpoint(copy.deepcopy(data), log=lambda s: None)
    assert upgraded["config"]["quoridor_io_version"] == 4 and upgraded["config"]["num_value_outputs"] == 3
    assert upgraded["train_state"][upgrade.MARKER] == 4321
    v4 = Model(upgraded["config"], pos_len=9)
    v4.load_state_dict(upgraded["model"])
    assert modelconfigs.get_num_bin_input_features(v4.config) == 25

    g = torch.Generator().manual_seed(5)
    spatial = torch.randn(8, 25, 9, 9, generator=g)
    spatial[:, 0] = 1.0
    glob_ = torch.randn(8, 21, generator=g)
    a = post_outputs(v3, spatial[:, :21], glob_[:, :19])
    b = post_outputs(v4, spatial, glob_)  # the new inputs are NOT zeroed: their weights are
    # Heads without a noResult share are equal up to float summation order (the value logits [win, loss] are compared as
    # probabilities, the third logit has weight exp(-8) = 3.4e-4 against the others, i.e. a share below 5e-3 for these random logits).
    for idx, (x, y) in enumerate(zip(a, b)):
        if idx == 1:  # value
            px = torch.softmax(x, dim=1)
            py = torch.softmax(y, dim=1)
            assert torch.allclose(px, py[:, :2], atol=2e-3) and (py[:, 2] < 5e-3).all()
        elif idx == 2:  # td value [N, 4, 2 or 3]
            px = torch.softmax(x, dim=2)
            py = torch.softmax(y, dim=2)
            assert torch.allclose(px, py[:, :, :2], atol=2e-3) and (py[:, :, 2] < 5e-3).all()
        else:
            assert torch.allclose(x, y, rtol=1e-5, atol=1e-5), (idx, float((x - y).abs().max()))
    # Exactly the same logits for win / loss: the weights of the old rows are copied.
    assert torch.equal(upgraded["model"][[k for k in upgraded["model"] if k.endswith("linear_value.weight")][0]][:2],
                       data["model"][[k for k in data["model"] if k.endswith("linear_value.weight")][0]])

    swa = AveragedModel(v4)
    swa.load_state_dict(upgraded["swa_model"])
    assert (swa.module.conv_spatial.weight[:, 21:] == 0).all()
    assert swa.module.value_head.linear_value.bias[2] == upgrade.NR_BIAS_DEFAULT

    # The optimizer state loads into an optimizer over the v4 parameters and a step works; momentum of the new entries is 0.
    opt = torch.optim.SGD(v4.parameters(), lr=0.01, momentum=0.9)
    opt.load_state_dict(upgraded["optimizer"])
    buf = opt.state[v4.conv_spatial.weight]["momentum_buffer"]
    assert buf.shape == v4.conv_spatial.weight.shape and (buf[:, 21:] == 0).all()
    assert (opt.state[v4.value_head.linear_value.weight]["momentum_buffer"][2] == 0).all()
    assert (opt.state[v4.value_head.linear_td_value.bias]["momentum_buffer"].view(4, 3)[:, 2] == 0).all()
    for p in v4.parameters():
        p.grad = torch.full_like(p, 1e-3)
    opt.step()
    assert torch.isfinite(v4.conv_spatial.weight).all()

    with pytest.raises(SystemExit):  # already v4
        upgrade.upgrade_checkpoint(upgraded, log=lambda s: None)


def test_shuffler_carries_wall_effects(dumped_rows):
    z = dumped_rows
    n = 64
    with tempfile.TemporaryDirectory() as d:
        base = {
            "binaryInputNCHWPacked": z["trainBinaryInputNCHWPacked"][:n], "spatialDistNCHW": z["trainSpatialDistNCHW"][:n],
            "globalInputNC": z["trainGlobalInputNC"][:n], "policyTargetsNCMove": np.zeros((n, 2, 243), dtype=np.int16),
            "globalTargetsNC": np.zeros((n, 80), dtype=np.float32), "scoreDistrN": np.zeros((n, 282), dtype=np.int16),
            "valueTargetsNCHW": np.zeros((n, 4, 9, 9), dtype=np.int8),
        }
        v4 = dict(base, wallEffectNCHW=z["trainWallEffectNCHW"][:n])
        p1 = os.path.join(d, "a.npz")
        np.savez_compressed(p1, **v4)
        lists = [[] for _ in range(9)]
        shuffle.load_and_accumulate_input_contents(p1, lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], None, None,
                                                   lists[6], lists[7], False, False, fill_in_qvalues=False)
        arrs = shuffle.joint_concatenate([lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], None, None, lists[6], lists[7]])
        out = os.path.join(d, "out.npz")
        shuffle.save_output_npz(out, arrs, False, False, 0, n)
        got = _load(out)
        assert np.array_equal(got["wallEffectNCHW"], v4["wallEffectNCHW"])
        # I/O v3 data (no key) still passes through with no key.
        p2 = os.path.join(d, "b.npz")
        np.savez_compressed(p2, **base)
        lists = [[] for _ in range(9)]
        shuffle.load_and_accumulate_input_contents(p2, lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], None, None,
                                                   lists[6], lists[7], False, False, fill_in_qvalues=False)
        arrs = shuffle.joint_concatenate([lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], None, None, lists[6], lists[7]])
        assert arrs[9] is None
