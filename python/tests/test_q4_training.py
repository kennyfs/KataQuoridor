"""Q4 training (Round 5): loader decode, 8-fold symmetry augmentation, value masking and the loss terms.

- Loader decode: decoded rows of a fresh self-play run == features.extract_features of the replayed position.
- Symmetry: for each of the 8 symmetries, the augmented inputs == extract_features(pos, symmetry=s), and the augmented
  policy / paths / future-wall targets == the targets moved by the reference cell / anchor maps (features.apply_cell,
  features.apply_anchor, the Python mirror of cpp/q4/nn/q4rawsymmetry.cpp); the played action stays legal in the
  transformed inputs.
- Value masking: an eliminated seat's value logit gets exactly zero gradient.
- Every loss term of Metrics.metrics_dict_batchwise_single_heads_output_q4 against a NumPy re-computation on a
  hand-made batch.
"""

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)

from katago.train import modelconfigs
from katago.train import data_processing_pytorch as dp
from katago.train.metrics_pytorch import Metrics
from katago.train.model_pytorch import Model
from q4.features import extract_features, apply_cell, apply_anchor
from q4.reference import Pos, eliminate, play, str_to_action
from tests.q4_testutil import find_katago
from tests.test_q4_selfplay import decode_game_hash, record_game_hash, action_to_policy_slot


@pytest.fixture(scope="module")
def selfplay_dir(tmp_path_factory):
    """A fresh self-play run with eliminations and the repetition rule, so every input channel is exercised."""
    katago = find_katago("eigen")
    if katago is None:
        pytest.skip("no eigen katago build")
    out_dir = str(tmp_path_factory.mktemp("q4train") / "selfplay")
    subprocess.check_call([
        katago, "q4selfplay",
        "-models-dir", os.path.join(REPO_DIR, "cpp", "tests", "models"),
        "-output-dir", out_dir,
        "-config", os.path.join(REPO_DIR, "cpp", "configs", "q4", "training", "q4_selfplay.cfg"),
        "-max-games-total", "8",
        "-seed", "r5_training_test",
        "-override-config",
        "numGameThreads=4,maxVisits=16,maxPlies=80,q4EliminationProb=0.5,q4RepetitionDrawProb=0.5,"
        "q4RepetitionDrawCounts=2,q4RepetitionDrawCountWeights=1.0,logToStdout=false",
    ])
    return out_dir


def replay_records(out_dir):
    """game hash -> {ply: position before the action of that ply} for every .q4.jsonl record."""
    states_by_hash = {}
    for rf in glob.glob(os.path.join(out_dir, "*", "records", "*.q4.jsonl")):
        with open(rf) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rules = rec.get("rules", {})
                pos = Pos(
                    max_plies=rules.get("maxPlies", 400),
                    repetition_draw_count=rules.get("repetitionDrawCount", 0),
                    initial_walls=rules.get("initialWalls", [7, 7, 7, 7]),
                )
                states = {}
                for ev in rec["events"]:
                    if "elim" in ev:
                        pos = eliminate(pos, ev["elim"] - 1)
                    else:
                        states[pos.plies] = pos.copy()
                        pos = play(pos, str_to_action(ev["a"]))
                states_by_hash[record_game_hash(rec)] = states
    return states_by_hash


@pytest.fixture(scope="module")
def rows_and_positions(selfplay_dir):
    """All decoded rows (symmetry 0) and, for the main rows, the replayed position."""
    states_by_hash = replay_records(selfplay_dir)
    files = sorted(glob.glob(os.path.join(selfplay_dir, "*", "tdata", "*.npz")))
    assert files
    rows = [dp.load_q4_npz_rows(f, model_config=modelconfigs.config_of_name["b1c32_q4"]) for f in files]
    rows = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    positions = []
    for i, gt in enumerate(rows["globalTargetsNC"]):
        if gt[59] == 1.0:  # main row of a finished game (side positions are not in the record)
            positions.append((i, states_by_hash[decode_game_hash(gt)][int(gt[50])]))
    return rows, positions


def test_loader_decode_matches_features(rows_and_positions):
    rows, positions = rows_and_positions
    assert len(positions) > 50
    num_elim = num_rep = num_unreachable = 0
    for i, pos in positions:
        exp_spatial, exp_glob = extract_features(pos, symmetry=0)
        np.testing.assert_array_equal(rows["binaryInputNCHW"][i], exp_spatial, err_msg=f"row {i}")
        np.testing.assert_allclose(rows["globalInputNC"][i], exp_glob, atol=1e-6)
        num_elim += sum(pos.alive) < 4
        num_rep += pos.repetition_draw_count >= 2
        num_unreachable += bool(np.any(exp_spatial[11:15] == 1.0))
    assert num_elim > 0 and num_rep > 0 and num_unreachable > 0, (num_elim, num_rep, num_unreachable)


def test_decode_distance_rule():
    """dist01 of Q4IO §3 on stored distances: 255 -> 1, d -> min(d, 64) / 64."""
    raw = np.array([0, 1, 63, 64, 65, 200, 254, 255], dtype=np.uint8).reshape(1, 1, 1, 8)
    np.testing.assert_array_equal(
        dp.decode_q4_dist_planes(raw).ravel(),
        np.array([0, 1 / 64, 63 / 64, 1, 1, 1, 1, 1], dtype=np.float32))


def expected_policy_under_symmetry(t, sym):
    """Policy targets [C, 363] moved by the reference action map (the raw output at applyAction(a, S) is action a)."""
    out = np.zeros_like(t)
    for c in range(t.shape[0]):
        for slot in np.nonzero(t[c])[0]:
            plane, cell = divmod(int(slot), 121)
            y, x = divmod(cell, 11)
            if plane == 0:
                nx, ny = apply_cell(x, y, sym)
                out[c, ny * 11 + nx] = t[c, slot]
            elif x == 10 or y == 10:
                # Padding of a wall plane: not an action. Only the uniform "unavailable" next-seat target (weight 0,
                # Q4IO §8 C2) has mass there; the augmentation drops it (the loss masks these slots anyway).
                continue
            else:
                nx, ny, is_h = apply_anchor(x, y, plane == 2, sym)
                out[c, (2 if is_h else 1) * 121 + ny * 11 + nx] = t[c, slot]
    return out


VALID_POLICY_SLOTS = np.zeros((3, 11, 11), dtype=np.float32)
VALID_POLICY_SLOTS[0] = 1.0
VALID_POLICY_SLOTS[1:, :10, :10] = 1.0
VALID_POLICY_SLOTS = VALID_POLICY_SLOTS.reshape(363)


def expected_value_targets_under_symmetry(t, sym):
    out = np.zeros_like(t)
    for y in range(11):
        for x in range(11):
            nx, ny = apply_cell(x, y, sym)
            out[0:4, ny, nx] = t[0:4, y, x]
    for k in range(4):
        for ay in range(10):
            for ax in range(10):
                for is_h in (False, True):
                    nx, ny, new_h = apply_anchor(ax, ay, is_h, sym)
                    out[4 + 2 * k + int(new_h), ny, nx] = t[4 + 2 * k + int(is_h), ay, ax]
    return out


@pytest.mark.parametrize("sym", range(8))
def test_symmetry_augmentation(rows_and_positions, sym):
    rows, positions = rows_and_positions
    batch = {k: torch.from_numpy(v) for k, v in rows.items()}
    aug = {k: v.numpy() for k, v in dp.apply_symmetry_q4_batch(batch, sym).items()}
    # Globals and the order of the relative seats do not change.
    np.testing.assert_array_equal(aug["globalInputNC"], rows["globalInputNC"])
    np.testing.assert_array_equal(aug["globalTargetsNC"], rows["globalTargetsNC"])
    for i, pos in positions:
        exp_spatial, _ = extract_features(pos, symmetry=sym)
        np.testing.assert_array_equal(aug["binaryInputNCHW"][i], exp_spatial, err_msg=f"row {i} symmetry {sym}")
    num_walls_played = 0
    for i in range(rows["globalTargetsNC"].shape[0]):
        walls = rows["policyTargetsNCMove"][i].reshape(3, 3, 11, 11)[:, 1:]
        padding_mass = walls[:, :, 10, :].sum(axis=(1, 2)) + walls[:, :, :, 10].sum(axis=(1, 2))  # per target
        assert padding_mass[0] == 0 and padding_mass[1] == 0
        assert padding_mass[2] == 0 or rows["globalTargetsNC"][i, 28] == 0.0
        np.testing.assert_array_equal(
            aug["policyTargetsNCMove"][i] * VALID_POLICY_SLOTS,
            expected_policy_under_symmetry(rows["policyTargetsNCMove"][i], sym))
        np.testing.assert_array_equal(
            aug["valueTargetsNCHW"][i], expected_value_targets_under_symmetry(rows["valueTargetsNCHW"][i], sym))
        # The played action (style target C1) is a legal action of the transformed position: pawn destinations are
        # input channel 24, legal vertical / horizontal walls channels 22 / 23 (this checks the V <-> H swap).
        played = np.nonzero(aug["policyTargetsNCMove"][i, 1])[0]
        if len(played) == 1:
            plane, cell = divmod(int(played[0]), 121)
            y, x = divmod(cell, 11)
            assert aug["binaryInputNCHW"][i, (24, 22, 23)[plane], y, x] == 1.0, (i, sym, plane, x, y)
            num_walls_played += plane > 0
        # My pawn is on my path: path channel 0 at the cell of input channel 1.
        me = np.argwhere(aug["binaryInputNCHW"][i, 1] == 1.0)
        if len(me) == 1 and rows["globalTargetsNC"][i, 27] > 0:
            assert aug["valueTargetsNCHW"][i, 0, me[0][0], me[0][1]] == 1.0
    assert num_walls_played > 0


def test_symmetry_tables_are_a_group():
    """The loader's tables: every symmetry permutes the 121 cells and the 100 anchors; directions are a permutation."""
    for sym in range(8):
        assert sorted(dp.Q4_CELL_SRC[sym]) == list(range(121))
        anchors = sorted(int(a) for a in dp.Q4_ANCHOR_SRC[sym] if a != 121)
        assert anchors == sorted(ay * 11 + ax for ay in range(10) for ax in range(10))
        assert sorted(dp.Q4_DIR_MAP[sym]) == [0, 1, 2, 3]


# ---- Loss ----------------------------------------------------------------------------------------------------------

def log_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def huber(x, y, delta):
    d = np.abs(x - y)
    return np.where(d > delta, 0.5 * delta * delta + delta * (d - delta), 0.5 * d * d)


def bce_logits(x, t):
    return np.maximum(x, 0) - x * t + np.log1p(np.exp(-np.abs(x)))


def make_hand_batch(seed=0):
    """Two hand-made rows: row 0 has all seats alive, row 1 has relative seat 2 eliminated (n = 3)."""
    rng = np.random.default_rng(seed)
    n = 2
    glob_in = np.zeros((n, 28), dtype=np.float32)
    glob_in[:, 8:12] = 1.0
    glob_in[1, 10] = 0.0
    gt = np.zeros((n, 64), dtype=np.float32)
    gt[0, 0:5] = [0, 1, 0, 0, 0]
    gt[1, 0:5] = [0, 0, 0, 0, 1]
    for c in (5, 10, 15, 20):
        v = rng.dirichlet(np.ones(5), size=n).astype(np.float32)
        v[1, 2] = 0.0
        gt[:, c:c + 5] = v / v.sum(axis=1, keepdims=True)
    gt[:, 25] = [1.0, 0.7]   # row weight
    gt[:, 26] = [1.0, 0.0]   # search policy weight (row 1: cheap search)
    gt[:, 27] = [1.0, 1.0]   # outcome weight
    gt[:, 28] = [1.0, 0.5]   # next-seat policy weight
    gt[:, 29] = 1.0          # style policy weight
    gt[:, 33] = [0.0, 0.25]  # 1 - TD weight
    gt[:, 34] = [0.0, 0.5]   # 1 - value weight
    gt[:, 35] = [37, 12]
    gt[:, 36:40] = [[0, 3, 7, 12], [4, 0, 0, 9]]
    gt[:, 40:44] = [[1, 1, 1, 1], [1, 1, 0, 1]]
    pol = np.zeros((n, 3, 363), dtype=np.float32)
    pol[:, 0, rng.choice(121, 5)] = rng.integers(1, 20, 5)
    pol[:, 0, 121 + 3 * 11 + 4] = 7
    pol[:, 1, 60] = 1
    pol[:, 2, 242 + 9 * 11 + 9] = 3
    pol[:, 2, 17] = 5
    vt = (rng.random((n, 12, 11, 11)) < 0.2).astype(np.float32)
    vt[1, 2] = 0.0
    vt[1, 8:10] = 0.0
    vt[:, 4:12, 10, :] = 0.0
    vt[:, 4:12, :, 10] = 0.0
    batch = dict(
        binaryInputNCHW=np.ones((n, 27, 11, 11), dtype=np.float32),
        globalInputNC=glob_in,
        policyTargetsNCMove=pol,
        globalTargetsNC=gt,
        scoreDistrN=np.zeros((n, 1), dtype=np.float32),
        valueTargetsNCHW=vt,
    )
    outputs = (
        rng.normal(size=(n, 2, 3, 11, 11)).astype(np.float32),
        rng.normal(size=(n, 5)).astype(np.float32),
        rng.normal(size=(n, 6)).astype(np.float32),
        rng.normal(size=(n, 1, 11, 11)).astype(np.float32),
        rng.normal(size=(n, 4, 11, 11)).astype(np.float32),
        rng.normal(size=(n, 8, 11, 11)).astype(np.float32),
        rng.normal(size=(n, 4, 5)).astype(np.float32),
        rng.normal(size=(n, 3, 3, 11, 11)).astype(np.float32),
    )
    return batch, outputs


def expected_losses(batch, outputs):
    """NumPy re-computation of every Q4 loss term (docs/q4/rounds/R5.md table)."""
    pol_out, value, misc, traj, paths, walls, td, pol_aux = [o.astype(np.float64) for o in outputs]
    gt = batch["globalTargetsNC"].astype(np.float64)
    gi = batch["globalInputNC"].astype(np.float64)
    pt = batch["policyTargetsNCMove"].astype(np.float64)
    vt = batch["valueTargetsNCHW"].astype(np.float64)
    n = gt.shape[0]
    gw = gt[:, 25]
    valid = np.zeros((3, 11, 11))
    valid[0] = 1
    valid[1:, :10, :10] = 1
    valid = valid.reshape(363)

    def policy_ce(logits, target, weight, coef):
        logits = np.where(valid > 0, logits.reshape(n, 363), -10000.0)
        target = target * valid
        target = target / np.maximum(target.sum(axis=1, keepdims=True), 1e-8)
        return np.sum(coef * gw * weight * -np.sum(target * log_softmax(logits), axis=1)), target

    def soft(target):
        s = ((target + 1e-7) * valid) ** 0.25
        return s / s.sum(axis=1, keepdims=True)

    out = {}
    out["p0loss_sum"], tp0 = policy_ce(pol_out[:, 0], pt[:, 0], gt[:, 26], 1.0)
    out["p1loss_sum"], tp1 = policy_ce(pol_aux[:, 0], pt[:, 2], gt[:, 28], 0.15)
    out["p0softloss_sum"], _ = policy_ce(pol_aux[:, 1], soft(tp0), gt[:, 26], 1.0)
    out["p1softloss_sum"], _ = policy_ce(pol_aux[:, 2], soft(tp1), gt[:, 28], 0.15)
    out["pstyleloss_sum"], _ = policy_ce(pol_out[:, 1], pt[:, 1], gt[:, 29], 1.0)

    alive = np.concatenate([gi[:, 8:12], np.ones((n, 1))], axis=1)
    vl = np.where(alive > 0, value, -10000.0)
    out["vloss_sum"] = np.sum(1.5 * gw * (1 - gt[:, 34]) * -np.sum(gt[:, 0:5] * log_softmax(vl), axis=1))
    td_t = np.stack([gt[:, 5:10], gt[:, 10:15], gt[:, 15:20], gt[:, 20:25]], axis=1)
    td_l = np.where(alive[:, None, :] > 0, td, -10000.0)
    td_ce = -np.sum(td_t * log_softmax(td_l), axis=2)
    wtd = 1 - gt[:, 33]
    for h in range(4):
        out[f"tdvloss{h + 1}_sum"] = np.sum(gw * wtd * td_ce[:, h])
    out["tdvloss_sum"] = 0.05 * sum(out[f"tdvloss{h + 1}_sum"] for h in range(4))

    wo = gt[:, 27]
    out["rtloss_sum"] = np.sum(5.0 / 9.0 * gw * wo * huber(misc[:, 0], gt[:, 35] / 100.0, 0.75))
    out["fdistloss_sum"] = np.sum(0.054 * gw[:, None] * gt[:, 40:44] * huber(32 * misc[:, 1:5], gt[:, 36:40], 3.0))
    nalive = gi[:, 8:12].sum(axis=1)
    p_short = np.exp(log_softmax(td_l[:, 2]))
    u_pred = 2 * p_short[:, 0] + 2 / nalive * p_short[:, 4] - 1
    u_real = 2 * td_t[:, 2, 0] + 2 / nalive * td_t[:, 2, 4] - 1
    pred_err = np.log1p(np.exp(0.5 * misc[:, 5])) ** 2 * 0.25
    out["evstloss_sum"] = np.sum(2.0 * gw * wtd * huber(pred_err, (u_pred - u_real) ** 2 + 1e-8, 0.4))

    out["trajloss_sum"] = np.sum(0.02 * gw * wo * bce_logits(traj[:, 0], vt[:, 0]).mean(axis=(1, 2)))
    seat = gi[:, 8:12]
    pb = bce_logits(paths, vt[:, 0:4]).mean(axis=(2, 3))
    out["pathsloss_sum"] = np.sum(0.02 * gw * wo * (pb * seat).sum(axis=1) / seat.sum(axis=1))
    wb = bce_logits(walls[:, :, :10, :10], vt[:, 4:12, :10, :10]).mean(axis=(2, 3)).reshape(n, 4, 2).mean(axis=2)
    out["wallloss_sum"] = np.sum(0.02 * gw * wo * (wb * seat).sum(axis=1) / seat.sum(axis=1))

    out["loss_sum"] = (
        out["p0loss_sum"] + out["p1loss_sum"] + 0.05 * out["p0softloss_sum"] + 0.02 * out["p1softloss_sum"]
        + 0.10 * out["pstyleloss_sum"] + out["vloss_sum"] + out["tdvloss_sum"] + out["rtloss_sum"]
        + out["fdistloss_sum"] + out["evstloss_sum"] + out["trajloss_sum"] + out["pathsloss_sum"] + out["wallloss_sum"]
    )
    return out


@pytest.fixture(scope="module")
def q4_model():
    model = Model(modelconfigs.config_of_name["b1c32_q4"], pos_len=11)
    model.initialize()
    return model


def run_q4_metrics(model, batch, outputs):
    metrics = Metrics(1, model)
    return metrics.metrics_dict_batchwise_single_heads_output_q4(
        raw_model=model, model_output_postprocessed=outputs, batch=batch, soft_policy_weight_scale=1.0,
        value_loss_scale=1.0, is_intermediate=True)


def test_each_loss_term_on_hand_made_rows(q4_model):
    batch, outputs = make_hand_batch()
    results = run_q4_metrics(
        q4_model, {k: torch.from_numpy(v) for k, v in batch.items()}, tuple(torch.from_numpy(o) for o in outputs))
    expected = expected_losses(batch, outputs)
    for key, value in expected.items():
        assert key in results, key
        np.testing.assert_allclose(float(results[key]), value, rtol=2e-5, atol=1e-6, err_msg=key)
        assert value > 0, key  # every term is exercised by the hand-made rows


def test_value_masking_eliminated_seat_gets_zero_gradient(q4_model):
    batch, outputs = make_hand_batch(seed=1)
    tbatch = {k: torch.from_numpy(v) for k, v in batch.items()}
    toutputs = [torch.from_numpy(o).requires_grad_(True) for o in outputs]
    results = run_q4_metrics(q4_model, tbatch, tuple(toutputs))
    results["loss_sum"].backward()
    value_grad, td_grad = toutputs[1].grad, toutputs[6].grad
    # Row 1 has relative seat 2 eliminated: no gradient through its value or TD logits.
    assert value_grad[1, 2] == 0.0 and torch.all(td_grad[1, :, 2] == 0.0)
    assert torch.all(value_grad[1, [0, 1, 3, 4]] != 0.0) and torch.all(value_grad[0] != 0.0)
    assert torch.all(td_grad[0] != 0.0)
    # The masked softmax gives it zero probability, so it costs nothing that the target there is 0.
    probs = torch.softmax(Metrics.q4_mask_value_logits(toutputs[1].detach(), Metrics.q4_alive_value_mask(tbatch["globalInputNC"])), dim=1)
    assert probs[1, 2] == 0.0 and abs(float(probs[1].sum()) - 1.0) < 1e-6


def test_q4_batches_train_end_to_end(selfplay_dir, q4_model):
    """read_npz_training_data -> forward -> metrics_dict_batchwise -> backward on real rows, all losses finite."""
    cfg = modelconfigs.config_of_name["b1c32_q4"]
    files = sorted(glob.glob(os.path.join(selfplay_dir, "*", "tdata", "*.npz")))
    metrics = Metrics(1, q4_model)
    seen = 0
    for batch in dp.read_npz_training_data(files, 8, 1, 0, 11, "cpu", True, False, cfg):
        out = q4_model.postprocess_output(q4_model(batch["binaryInputNCHW"], batch["globalInputNC"]))
        r = metrics.metrics_dict_batchwise(q4_model, out, None, batch, True, 1.0, False, False, 1.0, [1, 1, 1],
                                           1.0, 1.0, None, None)
        r["loss_sum"].backward()
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in r.values())
        seen += 1
    assert seen > 0
