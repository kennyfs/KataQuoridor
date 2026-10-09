"""Q4 I/O v2 = v1 + the metadata input (Round 7 Part C, docs/q4/Q4IO.md §10).

- configs: the "-meta" presets are I/O v2, exported as model option D = 102; v1 presets stay 101;
- data: `metadataInputNC` [N, 192] = the style features of each row's real position at slots 0..75, zeros elsewhere
  (checked against python/q4/style.py on a population self-play run); the loader hard-errors for a v2 net on data
  without the key and a v1 net ignores it; the shuffle keeps it;
- training: a v2 net trains end to end on the rows and the metadata encoder receives gradient.
(NN parity PyTorch vs Eigen / CUDA for the "-meta" nets is in test_q4_parity.py; the NN cache key in the C++ group
q4search, test S2.)
"""

import glob
import gzip
import os
import shutil
import sys

import numpy as np
import pytest
import torch

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PYTHON_DIR)

from katago.train import data_processing_pytorch as dp  # noqa: E402
from katago.train import modelconfigs  # noqa: E402
from katago.train.metrics_pytorch import Metrics  # noqa: E402
from katago.train.model_pytorch import Model  # noqa: E402
from q4.style import record_features  # noqa: E402
from tests.q4_testutil import find_katago  # noqa: E402
from tests.test_q4_selfplay import (get_model_dir, read_records, read_rows, replay_states, run_selfplay,  # noqa: E402
                                    record_game_hash)

META_CONFIGS = ["b1c32_q4-meta", "b2c64_q4-meta", "tf2_b4c192_q4-meta"]


@pytest.mark.parametrize("name", META_CONFIGS)
def test_meta_presets_are_io_v2(name):
    cfg = modelconfigs.config_of_name[name]
    assert modelconfigs.is_quoridor4(cfg) and modelconfigs.get_q4_io_version(cfg) == 2
    assert cfg["metadata_encoder"] == {"meta_encoder_version": 1, "internal_num_channels": cfg["trunk_num_channels"]}
    assert modelconfigs.get_num_bin_input_features(cfg) == 27 and modelconfigs.get_num_global_input_features(cfg) == 28
    base = modelconfigs.config_of_name[name[:-len("-meta")]]
    assert modelconfigs.get_q4_io_version(base) == 1 and "metadata_encoder" not in base
    # an inconsistent config is a hard error
    with pytest.raises(AssertionError):
        modelconfigs.get_q4_io_version(dict(base, q4_io_version=2))
    with pytest.raises(AssertionError):
        modelconfigs.get_q4_io_version(dict(cfg, q4_io_version=1))


def test_export_writes_option_d_102_and_v1_stays_101(tmp_path):
    from tests.q4_testutil import export_model, make_random_model
    for name, expected in (("b1c32_q4-meta", "102"), ("b1c32_q4", "101")):
        path = export_model(make_random_model(name, seed=3), name, str(tmp_path), name)
        with gzip.open(path, "rt", errors="ignore") as f:
            head = [next(f).strip() for _ in range(40)]
        # the header ends with the model options; option D (the I/O version) is the only 101 / 102 in it
        assert head.count(expected) >= 1, (name, head)
        assert ("102" in head) == (expected == "102")


@pytest.fixture(scope="module")
def population_run(tmp_path_factory):
    katago = find_katago("eigen")
    if katago is None:
        pytest.skip("no eigen katago build")
    out_dir = str(tmp_path_factory.mktemp("q4meta") / "selfplay")
    run_selfplay(katago, get_model_dir(), out_dir, "meta_rows_seed",
                 "numGameThreads=2,maxVisits=16,maxPlies=90,q4PopulationMixedProb=0.7,q4EliminationProb=0.3,"
                 "logToStdout=false", 30)
    return out_dir


def test_metadata_rows_are_the_style_features_of_the_position(population_run):
    records = {record_game_hash(r): r for r in read_records(population_run)}
    features = {h: record_features(r)[1] for h, r in records.items()}
    replays = {h: replay_states(r) for h, r in records.items()}
    n_main = n_side = 0
    max_style = 0.0
    for row in read_rows(population_run):
        meta = row["metadataInputNC"]
        assert meta.shape == (192,) and meta.dtype == np.float32
        assert np.all(meta[76:] == 0) and np.all(np.isfinite(meta))
        gt = row["globalTargetsNC"]
        if gt[59] == 0.0:       # side position: features of a position that is not in the record, only checked for range
            n_side += 1
            assert np.all(np.abs(meta[:76]) <= 1.0)
            continue
        h = row["hash"]
        _, _, event_of_ply = replays[h]
        expected = features[h][event_of_ply[int(gt[50])]]
        assert np.max(np.abs(meta[:76] - expected)) < 1e-6
        n_main += 1
        max_style = max(max_style, float(np.abs(meta[:76]).max()))
    assert n_main > 200 and max_style > 0.3, (n_main, max_style)
    print(f"meta: {n_main} main rows == python style features, {n_side} side rows in range")


def test_loader_requires_the_key_for_v2_nets_and_v1_ignores_it(population_run, tmp_path):
    files = sorted(glob.glob(os.path.join(population_run, "*", "tdata", "*.npz")))
    assert files
    v1, v2 = modelconfigs.config_of_name["b1c32_q4"], modelconfigs.config_of_name["b1c32_q4-meta"]
    rows_v2 = dp.load_q4_npz_rows(files[0], model_config=v2, include_meta=True)
    assert rows_v2["metadataInputNC"].shape[1:] == (192,)
    assert "metadataInputNC" not in dp.load_q4_npz_rows(files[0], model_config=v1, include_meta=False)

    stripped = str(tmp_path / "no_meta.npz")
    data = np.load(files[0])
    np.savez(stripped, **{k: data[k] for k in data.files if k != "metadataInputNC"})
    assert "metadataInputNC" not in dp.load_q4_npz_rows(stripped, model_config=v1, include_meta=False)   # v1: fine
    with pytest.raises(KeyError, match="without the style features"):
        dp.load_q4_npz_rows(stripped, model_config=v2, include_meta=True)                               # v2: refuses
    with pytest.raises(KeyError, match="without the style features"):
        list(dp.read_npz_training_data([stripped], 8, 1, 0, 11, "cpu", False, True, v2))


def test_v2_net_trains_and_the_metadata_encoder_gets_gradient(population_run):
    cfg = modelconfigs.config_of_name["b1c32_q4-meta"]
    model = Model(cfg, pos_len=11)
    model.initialize()
    files = sorted(glob.glob(os.path.join(population_run, "*", "tdata", "*.npz")))
    metrics = Metrics(1, model)
    seen = 0
    for batch in dp.read_npz_training_data(files, 16, 1, 0, 11, "cpu", True, True, cfg):
        out = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"], batch["metadataInputNC"]))
        r = metrics.metrics_dict_batchwise(model, out, None, batch, True, 1.0, False, False, 1.0, [1, 1, 1],
                                           1.0, 1.0, None, None)
        model.zero_grad()
        r["loss_sum"].backward()
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in r.values())
        g = model.metadata_encoder.linear1.weight.grad
        assert g is not None and g[:, :76].abs().sum() > 0, "the style inputs must receive gradient"
        assert g[:, 76:].abs().sum() == 0, "slots 76.. are zero in the data: no gradient there"
        seen += 1
        if seen >= 3:
            break
    assert seen > 0
