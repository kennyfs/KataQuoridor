"""Tests for the S8-S11 fix (spatial input channels 8..11 = continuous BFS distances).

Bug being guarded against: inference fed continuous S8-S11 while training data stored them
truncated to binary. See docs/DistPlanesUpgrade.md.

1. Train/inference input parity: rows encoded exactly as the training data writer stores them
   (TrainingWriteBuffers::fillQuoridorInputRow, dumped by `katago dumpnninputs`) and decoded by the
   training loader must equal QuoridorNN::fillRow's output bit for bit.
2. Converter: fresh rows from the new C++ writer, turned back into old-format rows, must be
   upgraded by quoridor_add_dist_planes.py to exactly the new writer's arrays.
3. Checkpoint migration: a migrated random checkpoint gives identical outputs on inputs whose
   channels 8..11 are zero, and ignores channels 8..11.
"""
import copy
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(repo_root, "python"))

from katago.train import modelconfigs
from katago.train.model_pytorch import Model
from katago.train.data_processing_pytorch import decode_binary_input
import quoridor_add_dist_planes as addplanes
import quoridor_migrate_dist_planes as migrate


def _katago_bin():
    candidates = [os.environ["KATAGO_BIN"]] if os.environ.get("KATAGO_BIN") else (
        glob.glob(os.path.join(repo_root, "build", "katago")) + glob.glob(os.path.join(repo_root, "cpp", "build*", "katago")))
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        pytest.skip("KataGo binary not found (set KATAGO_BIN or build under build/ or cpp/build*/).")
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
        _run([_katago_bin(), "dumpnninputs", "-n", "600", "-seed", "distplanes", "-output", path])
        yield _load(path)


def test_train_inference_input_parity(dumped_rows):
    z = dumped_rows
    ref = z["binaryInputNCHW"]
    got = decode_binary_input(z["trainBinaryInputNCHWPacked"], z["trainSpatialDistNCHW"], 9)
    assert got.dtype == np.float32 and ref.dtype == np.float32
    assert np.array_equal(got, ref), f"mismatch in channels {sorted(set(np.nonzero(got != ref)[1]))}"
    assert np.array_equal(z["trainGlobalInputNC"], z["globalInputNC"])
    # The test must actually exercise non-binary distances and unreachable cells.
    dist = ref[:, 8:12]
    assert ((dist > 0) & (dist < 1)).any()
    assert (z["trainSpatialDistNCHW"] == 255).any() or (z["hWalls"].sum() + z["vWalls"].sum()) > 0


def _to_old_format(arrays):
    """What the pre-fix writer produced for the same rows: no spatialDistNCHW, and planes 8..11
    holding (uint8_t)(d < 0 ? 1 : min(1, d/32))."""
    old = {k: v for k, v in arrays.items() if k != "spatialDistNCHW"}
    n = arrays["binaryInputNCHWPacked"].shape[0]
    bits = np.unpackbits(arrays["binaryInputNCHWPacked"], axis=2)
    trunc = addplanes.truncated_bits(arrays["spatialDistNCHW"]).reshape(n, 4, 81).astype(np.uint8)
    bits[:, 8:12, :81] = trunc
    old["binaryInputNCHWPacked"] = np.packbits(bits, axis=2)
    return old


def test_converter_bit_identical(dumped_rows):
    with tempfile.TemporaryDirectory() as d:
        sample = os.path.join(d, "sample")
        _run([_katago_bin(), "writesampletrainquoridor", sample, "3", "200"])
        base = os.path.join(d, "selfplay")
        expected = {}
        for i, f in enumerate(sorted(glob.glob(os.path.join(sample, "*.npz")))):
            arrays = _load(f)
            assert "spatialDistNCHW" in arrays
            out = os.path.join(base, f"model{i}", "tdata", os.path.basename(f))
            os.makedirs(os.path.dirname(out))
            # First file: new-writer rows with just the key stripped. Others: fully old format.
            src = {k: v for k, v in arrays.items() if k != "spatialDistNCHW"} if i == 0 else _to_old_format(arrays)
            np.savez_compressed(out, **src)
            expected[out] = arrays
        # Rows from random games (wall-heavy middlegames, enclosed regions), also in old format.
        rows = {"binaryInputNCHWPacked": dumped_rows["trainBinaryInputNCHWPacked"],
                "spatialDistNCHW": dumped_rows["trainSpatialDistNCHW"],
                "globalInputNC": dumped_rows["trainGlobalInputNC"]}
        out = os.path.join(base, "modelR", "tdata", "rows.npz")
        os.makedirs(os.path.dirname(out))
        np.savez_compressed(out, **_to_old_format(rows))
        expected[out] = rows

        script = os.path.join(repo_root, "python", "quoridor_add_dist_planes.py")
        _run([sys.executable, script, base, "-num-processes", "2"])
        total = 0
        for path, want in expected.items():
            got = _load(path)
            assert list(got.keys())[:2] == ["binaryInputNCHWPacked", "spatialDistNCHW"]
            assert set(got.keys()) == set(want.keys())
            for k in want:
                assert got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k]), f"{path}: {k} differs"
            total += want["spatialDistNCHW"].shape[0]
        assert total == 600 + 600
        assert not glob.glob(os.path.join(base, "**", "*.tmp"), recursive=True)

        # Idempotent: a second run skips everything and leaves files untouched.
        mtimes = {p: os.path.getmtime(p) for p in expected}
        res = subprocess.run([sys.executable, script, base, "-num-processes", "2"], capture_output=True, text=True)
        assert res.returncode == 0 and "'skipped': 4" in res.stdout, res.stdout
        assert mtimes == {p: os.path.getmtime(p) for p in expected}


def _tiny_checkpoint(path):
    torch.manual_seed(123)
    cfg = modelconfigs.base_config_of_name["b2c64_quoridor"]
    model = Model(cfg, pos_len=9)
    # Make the distance-channel weights clearly nonzero.
    with torch.no_grad():
        model.conv_spatial.weight.normal_()
    swa = torch.optim.swa_utils.AveragedModel(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    spatial = torch.rand(4, 17, 9, 9)
    spatial[:, 0] = 1.0
    out = model(spatial, torch.rand(4, 15))
    sum(o.float().sum() for o in out[0] if isinstance(o, torch.Tensor)).backward()
    opt.step()
    state = {"model": model.state_dict(), "swa_model": swa.state_dict(), "optimizer": opt.state_dict(),
             "train_state": {"global_step_samples": 1}, "config": cfg}
    torch.save(state, path)
    return cfg, state


def _outputs(cfg, model_state, spatial, glob_in):
    model = Model(cfg, pos_len=9)
    model.load_state_dict(model_state)
    model.eval()
    with torch.no_grad():
        return [o for o in model(spatial, glob_in)[0] if isinstance(o, torch.Tensor)]


def test_checkpoint_migration():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "checkpoint.ckpt")
        cfg, orig = _tiny_checkpoint(path)
        orig = copy.deepcopy(orig)
        assert migrate.migrate_file(path)
        assert os.path.exists(path + migrate.BACKUP_SUFFIX)
        new = torch.load(path, map_location="cpu")
        assert new["train_state"][migrate.MARKER]

        torch.manual_seed(7)
        spatial = torch.rand(8, 17, 9, 9)
        spatial[:, 0] = 1.0
        glob_in = torch.rand(8, 15)
        zeroed = spatial.clone()
        zeroed[:, 8:12] = 0.0
        for key in ("model",):
            a = _outputs(cfg, orig[key], zeroed, glob_in)
            b = _outputs(cfg, new[key], zeroed, glob_in)
            c = _outputs(cfg, new[key], spatial, glob_in)
            for x, y, z in zip(a, b, c):
                assert torch.equal(x, y)  # same outputs when ch8-11 are zero
                assert torch.equal(y, z)  # migrated net ignores ch8-11

        w_old, w_new = orig["model"]["conv_spatial.weight"], new["model"]["conv_spatial.weight"]
        assert (w_new[:, 8:12] == 0).all() and torch.equal(w_new[:, :8], w_old[:, :8]) and torch.equal(w_new[:, 12:], w_old[:, 12:])
        s_new = new["swa_model"]["module.conv_spatial.weight"]
        assert (s_new[:, 8:12] == 0).all() and torch.equal(s_new[:, :8], orig["swa_model"]["module.conv_spatial.weight"][:, :8])
        mom = [st["momentum_buffer"] for st in new["optimizer"]["state"].values() if st["momentum_buffer"].shape == w_new.shape]
        assert len(mom) == 1 and (mom[0][:, 8:12] == 0).all() and (mom[0][:, :8] != 0).any()
        # Every other parameter and optimizer entry is untouched.
        for k in orig["model"]:
            if k != "conv_spatial.weight":
                assert torch.equal(orig["model"][k], new["model"][k]), k

        # Idempotent, including after further training changed the slice.
        new["model"]["conv_spatial.weight"][:, 8:12] = 1.0
        torch.save(new, path)
        assert not migrate.migrate_file(path)
        assert (torch.load(path, map_location="cpu")["model"]["conv_spatial.weight"][:, 8:12] == 1.0).all()


def test_missing_dist_key_fails_loudly():
    import shuffle
    keys = ["binaryInputNCHWPacked", "globalInputNC", "policyTargetsNCMove", "globalTargetsNC", "scoreDistrN", "valueTargetsNCHW"]
    with pytest.raises(KeyError, match="spatialDistNCHW"):
        shuffle.assert_keys({k: None for k in keys}, include_meta=False, include_qvalues=False)
    shuffle.assert_keys({k: None for k in keys + ["spatialDistNCHW"]}, include_meta=False, include_qvalues=False)
