"""Quoridor I/O v4 (docs/QuoridorIOv4.md): v4 = the v3 inputs + draws as the noResult outcome. The data converter (identical
to what the C++ writer produces, on writer rows and on real self-play rows), the version stamp that tells v3 from v4 data
(the shuffler and the loader refuse unstamped data), and the v3 -> v4 checkpoint migration (same outputs, optimizer state
loads)."""
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
from katago.train import data_processing_pytorch  # noqa: E402
import quoridor_convert_tdata_v3_to_v4 as convert  # noqa: E402
import quoridor_upgrade_v3_to_v4 as upgrade  # noqa: E402
import shuffle  # noqa: E402

CONVERT = os.path.join(PYTHON_DIR, "quoridor_convert_tdata_v3_to_v4.py")


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


def _to_v3_form(arrays):
    """What the v3 writer produced for the same rows: no stamp, a draw as win 0.5 / loss 0.5."""
    old = {k: v for k, v in arrays.items() if k != convert.KEY}
    g = arrays["globalTargetsNC"].copy()
    draw = (g[:, 2] == 1.0)
    g[draw, 0] = 0.5
    g[draw, 1] = 0.5
    g[draw, 2] = 0.0
    old["globalTargetsNC"] = g
    return old


@pytest.fixture(scope="module")
def writer_files():
    """npz files of the C++ writer (I/O v4, stamped) with a few draw rows; yields the directory."""
    with tempfile.TemporaryDirectory() as d:
        sample = os.path.join(d, "sample")
        _run([_katago_bin(), "writesampletrainquoridor", sample, "3", "200"])
        assert glob.glob(os.path.join(sample, "*.npz"))
        yield sample


def test_writer_stamps_and_v4_has_the_v3_channels(writer_files):
    for f in sorted(glob.glob(os.path.join(writer_files, "*.npz"))):
        arrays = _load(f)
        assert arrays[convert.KEY].dtype == np.int32 and arrays[convert.KEY].tolist() == [[4]]
        assert arrays["binaryInputNCHWPacked"].shape[1] == 21 and arrays["globalInputNC"].shape[1] == 19
        assert "wallEffectNCHW" not in arrays


def test_converter_identical_to_writer(writer_files):
    files = sorted(glob.glob(os.path.join(writer_files, "*.npz")))
    with tempfile.TemporaryDirectory() as d:
        total = draws = 0
        v3dir = os.path.join(d, "v3")
        os.makedirs(v3dir)
        for f in files:
            arrays = _load(f)
            draws += int((arrays["globalTargetsNC"][:, 2] == 1.0).sum())
            np.savez_compressed(os.path.join(v3dir, os.path.basename(f)), **_to_v3_form(arrays))
            total += arrays["globalInputNC"].shape[0]
        assert total >= 400 and draws > 0
        outdir = os.path.join(d, "out")
        _run([sys.executable, CONVERT, "-num-processes", "2", "-out-dir", outdir, v3dir])
        for f in files:
            want = _load(f)
            got = _load(os.path.join(outdir, "v3", os.path.basename(f)))
            assert set(got) == set(want)
            for k in want:
                assert got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k]), f"{f}: {k} differs"
        # Idempotent: a second conversion of the converted copy skips every (stamped) file, in place too.
        res = subprocess.run([sys.executable, CONVERT, os.path.join(outdir, "v3")], capture_output=True, text=True)
        assert res.returncode == 0 and f"0 v3 files converting, {len(files)} already v4" in res.stdout, res.stdout
        # Anything that is not 21 / 19 channels is an error.
        bad = os.path.join(d, "bad")
        os.makedirs(bad)
        arrays = _to_v3_form(_load(files[0]))
        arrays["binaryInputNCHWPacked"] = np.concatenate([arrays["binaryInputNCHWPacked"]] * 2, axis=1)
        np.savez_compressed(os.path.join(bad, "x.npz"), **arrays)
        res = subprocess.run([sys.executable, CONVERT, bad], capture_output=True, text=True)
        assert res.returncode != 0 and "unexpected input channels" in (res.stdout + res.stderr)


def _real_selfplay_file():
    files = sorted(glob.glob(os.path.expanduser("~/q1_run/run4/selfplay/*/tdata/*.npz")))
    for f in files[-30:]:
        with np.load(f) as d:
            if convert.KEY not in d.files and d["binaryInputNCHWPacked"].shape[1] == 21 and d["binaryInputNCHWPacked"].shape[0] >= 60:
                return f
    return None


def test_converter_on_real_rows():
    f = _real_selfplay_file()
    if f is None:
        pytest.skip("no run4 self-play file")
    arrs = _load(f)
    out, num_draw = convert.convert_arrays(arrs)
    assert out[convert.KEY].tolist() == [[4]]
    # every array but the draw targets is carried over untouched
    for k in arrs:
        if k != "globalTargetsNC":
            assert np.array_equal(out[k], arrs[k]), k
    changed = np.any(out["globalTargetsNC"] != arrs["globalTargetsNC"], axis=1)
    assert int(changed.sum()) == num_draw
    g0, g1 = arrs["globalTargetsNC"][changed], out["globalTargetsNC"][changed]
    assert (g0[:, :3] == [0.5, 0.5, 0.0]).all() and (g1[:, :3] == [0.0, 0.0, 1.0]).all()
    assert np.array_equal(np.delete(g0, [0, 1, 2], axis=1), np.delete(g1, [0, 1, 2], axis=1))


# --- the stamp tells v3 from v4 data -----------------------------------------------------------------------------------

def _read_one_batch(files, model_name):
    cfg = modelconfigs.config_of_name[model_name]
    for batch in data_processing_pytorch.read_npz_training_data(
            files, 50, world_size=1, rank=0, pos_len=9, device=torch.device("cpu"), randomize_symmetries=False,
            include_meta=False, model_config=cfg):
        return batch
    return None


def test_loader_refuses_unstamped_data(writer_files):
    f = sorted(glob.glob(os.path.join(writer_files, "*.npz")))[0]
    with tempfile.TemporaryDirectory() as d:
        v3 = os.path.join(d, "v3.npz")
        np.savez_compressed(v3, **_to_v3_form(_load(f)))
        # same channel counts, same arrays: only the stamp differs
        assert _read_one_batch([f], "b2c64_quoridor_v4") is not None
        with pytest.raises(ValueError, match="no quoridorIOVersion stamp"):
            _read_one_batch([v3], "b2c64_quoridor_v4")
        # a v3 net on v3 data keeps working (the version of the net, not the channel count, decides)
        assert _read_one_batch([v3], "b2c64_quoridor_v3") is not None
        with pytest.raises(ValueError, match="I/O v4 data"):
            _read_one_batch([f], "b2c64_quoridor_v3")


def _shuffle_one(path, out, allow=False):
    lists = [[] for _ in range(8)]
    old = shuffle.ALLOW_UNSTAMPED_V3
    shuffle.ALLOW_UNSTAMPED_V3 = allow
    try:
        shuffle.load_and_accumulate_input_contents(path, lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], None, None,
                                                   lists[6], False, False, fill_in_qvalues=False)
        arrs = shuffle.joint_concatenate([lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], None, None, lists[6]])
        shuffle.save_output_npz(out, arrs, False, False, 0, 50)
    finally:
        shuffle.ALLOW_UNSTAMPED_V3 = old
    return _load(out)


def test_shuffler_stamps_and_refuses_unstamped(writer_files):
    f = sorted(glob.glob(os.path.join(writer_files, "*.npz")))[0]
    with tempfile.TemporaryDirectory() as d:
        got = _shuffle_one(f, os.path.join(d, "out.npz"))
        assert got[convert.KEY].tolist() == [[4]]
        v3 = os.path.join(d, "v3.npz")
        np.savez_compressed(v3, **_to_v3_form(_load(f)))
        with pytest.raises(KeyError, match="no quoridorIOVersion stamp"):
            _shuffle_one(v3, os.path.join(d, "out3.npz"))
        got3 = _shuffle_one(v3, os.path.join(d, "out3.npz"), allow=True)  # -allow-unstamped-v3: stays unstamped
        assert convert.KEY not in got3
        assert np.array_equal(got3["globalTargetsNC"], _to_v3_form(_load(f))["globalTargetsNC"][:50])


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
    assert modelconfigs.get_num_bin_input_features(v4.config) == 21 and modelconfigs.get_num_global_input_features(v4.config) == 19
    assert modelconfigs.get_quoridor_io_version(v4.config) == 4 and modelconfigs.get_quoridor_io_version(v3.config) == 3

    g = torch.Generator().manual_seed(5)
    spatial = torch.randn(8, 21, 9, 9, generator=g)
    spatial[:, 0] = 1.0
    glob_ = torch.randn(8, 19, generator=g)
    a = post_outputs(v3, spatial, glob_)
    b = post_outputs(v4, spatial, glob_)
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
    assert swa.module.value_head.linear_value.bias[2] == upgrade.NR_BIAS_DEFAULT

    # The optimizer state loads into an optimizer over the v4 parameters and a step works; momentum of the new entries is 0.
    opt = torch.optim.SGD(v4.parameters(), lr=0.01, momentum=0.9)
    opt.load_state_dict(upgraded["optimizer"])
    assert (opt.state[v4.value_head.linear_value.weight]["momentum_buffer"][2] == 0).all()
    assert (opt.state[v4.value_head.linear_td_value.bias]["momentum_buffer"].view(4, 3)[:, 2] == 0).all()
    for p in v4.parameters():
        p.grad = torch.full_like(p, 1e-3)
    opt.step()
    assert torch.isfinite(v4.conv_spatial.weight).all() and v4.conv_spatial.weight.shape == v3.conv_spatial.weight.shape

    with pytest.raises(SystemExit):  # already v4
        upgrade.upgrade_checkpoint(upgraded, log=lambda s: None)


