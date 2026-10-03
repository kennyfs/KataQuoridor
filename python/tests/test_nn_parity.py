"""NN parity harness (docs/KataQuoridor_Review_and_Roadmap.md §5 Phase 0 step 4).

Pipeline, all driven from here:
  1. `katago dumpnninputs` plays random games (some with non-standard komi and maxPlies) and writes the input rows
     of each Quoridor I/O version (v1, v2, v3), the legal moves (in search space, 17x17+1 slots) and the raw board
     state and rules to an .npz.
  2. The features and legal moves are re-derived here, independently, from the raw board state and compared. (The
     v3 repetition inputs are derived from the dumped occurrence counts; the counting itself is checked against a
     brute-force recount by `runtests quoridorv3`.)
  3. For a conv net and a transformer net of each I/O version (random init with fixed seeds, output heads scaled up
     so that the policy is far from uniform), a PyTorch reference policy/value is computed at symmetries 0 and 1,
     and the model is exported to .bin.gz with export_model_pytorch.py.
  4. `katago evalnnparity` evaluates the same positions through NNEvaluator (the path the search uses) on
     whatever backend the binary was built with, and the result is compared with the reference:
     final policy over legal moves and White's win/loss/no-result probabilities, within 1e-4 (FP32), and the
     score-like outputs (score mean/lead/stdev, variance time, shortterm errors) within 1e-4 * (1 + |ref|).

The binary is taken from $KATAGO_BIN, else the most recently built cpp/build*/katago or cpp/katago; the tests
are skipped if none exists. Set $NN_PARITY_OUTDIR to keep the intermediate files, $NN_PARITY_CONFIG to pass a
-config to evalnnparity (e.g. to select a GPU backend or FP16), and $NN_PARITY_TOL to override the 1e-4 tolerance.

Run the report by hand with:  cd python && python -m pytest tests/test_nn_parity.py -rA
"""

import glob
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)

from katago.train import modelconfigs  # noqa: E402
from katago.train.model_pytorch import Model  # noqa: E402
from katago.train.data_processing_pytorch import (  # noqa: E402
    apply_symmetry_quoridor,
    apply_symmetry_policy_quoridor,
)

_spec = importlib.util.spec_from_file_location(
    "quoridor_rules_reference", os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_quoridor_perft_reference.py"))
rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rules)

NUM_ROWS = 200
SEED = "parity"
SEARCH_LEN = 17
POLICY_SIZE = SEARCH_LEN * SEARCH_LEN + 1
FP32_TOL = float(os.environ.get("NN_PARITY_TOL", "1e-4"))
# $NN_PARITY_MODELS (comma-separated) restricts the nets, e.g. conv only for CUDA NCHW, which rejects transformers.
MODEL_CONFIGS = os.environ.get(
    "NN_PARITY_MODELS",
    "b2c64_quoridor,tf2_b4c192_quoridor,b2c64_quoridor_v2,tf2_b4c192_quoridor_v2,b2c64_quoridor_v3,tf2_b4c192_quoridor_v3,tf3_b5c256_quoridor_v3",
).split(",")
IO_VERSIONS = [1, 2, 3]
NUM_SPATIAL = {1: 17, 2: 19, 3: 21}
NUM_GLOBAL = {1: 15, 2: 17, 3: 19}


# ---------------------------------------------------------------------------------------------------------
# Fixtures: binary, working dir, dumped rows

def _find_katago():
    env = os.environ.get("KATAGO_BIN")
    if env:
        return env
    candidates = glob.glob(os.path.join(REPO_DIR, "cpp", "build*", "katago")) + [os.path.join(REPO_DIR, "cpp", "katago")]
    candidates = [c for c in candidates if os.path.isfile(c) and os.access(c, os.X_OK)]
    return max(candidates, key=os.path.getmtime) if candidates else None


@pytest.fixture(scope="module")
def katago_bin():
    path = _find_katago()
    if path is None:
        pytest.skip("no katago binary found (set KATAGO_BIN or build under cpp/build*/)")
    return path


@pytest.fixture(scope="module")
def workdir():
    outdir = os.environ.get("NN_PARITY_OUTDIR")
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        yield outdir
    else:
        with tempfile.TemporaryDirectory() as d:
            yield d


def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        raise AssertionError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}\n{tail}")
    return proc


@pytest.fixture(scope="module")
def rows_by_version(katago_bin, workdir):
    out = {}
    for io_version in IO_VERSIONS:
        path = os.path.join(workdir, f"rows.v{io_version}.npz")
        _run([katago_bin, "dumpnninputs", "-n", str(NUM_ROWS), "-seed", SEED, "-io-version", str(io_version),
              "-output", path])
        with np.load(path) as d:
            out[io_version] = {k: d[k] for k in d}
    return out


# ---------------------------------------------------------------------------------------------------------
# Independent re-derivation of the features and legal moves from the raw board state and rules

def _position(rows, i):
    """Board state of row i as a rules.Pos (board coordinates)."""
    pos = rules.Pos()
    bc, br, wc, wr = (int(v) for v in rows["pawns"][i])
    pos.pawn = [(bc, br), (wc, wr)]
    pos.walls_left = [int(rows["fencesLeft"][i][0]), int(rows["fencesLeft"][i][1])]
    pos.hwalls = {(c, r) for r, c in zip(*np.nonzero(rows["hWalls"][i]))}
    pos.vwalls = {(c, r) for r, c in zip(*np.nonzero(rows["vWalls"][i]))}
    pos.to_move = rules.BLACK if int(rows["nextPlayer"][i][0]) == 1 else rules.WHITE
    return pos


def _canonical(pos):
    """Flip the board vertically when White is to move, so the side to move always heads for row 0."""
    if pos.to_move == rules.BLACK:
        return pos.pawn[0], pos.pawn[1], pos.hwalls, pos.vwalls
    flip = lambda cell: (cell[0], 8 - cell[1])  # noqa: E731
    # A wall anchored at row r spans rows r and r+1, which map to 8-r and 7-r: the new anchor is 7-r.
    return flip(pos.pawn[1]), flip(pos.pawn[0]), {(c, 7 - r) for c, r in pos.hwalls}, {(c, 7 - r) for c, r in pos.vwalls}


def _distances(h, v, sources):
    dist = -np.ones((9, 9), dtype=np.int64)  # [r][c]
    queue = list(sources)
    for c, r in sources:
        dist[r, c] = 0
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        for dc, dr in rules.DIRS:
            nxt = (cur[0] + dc, cur[1] + dr)
            if rules.can_step(h, v, cur, nxt) and dist[nxt[1], nxt[0]] < 0:
                dist[nxt[1], nxt[0]] = dist[cur[1], cur[0]] + 1
                queue.append(nxt)
    return dist


def _dist_feature(d):
    return np.where(d < 0, 1.0, np.minimum(1.0, d / 32.0)).astype(np.float32)


def python_features(pos, io_version, ply, max_plies, komi, rep_info=None, rep_moves=None):
    me, opp, h, v = _canonical(pos)
    spatial = np.zeros((NUM_SPATIAL[io_version], 9, 9), dtype=np.float32)  # [channel][row][col], canonical rows
    spatial[0] = 1.0
    spatial[1, me[1], me[0]] = 1.0
    spatial[2, opp[1], opp[0]] = 1.0
    for r in range(9):
        for c in range(9):
            spatial[3, r, c] = not rules.can_step(h, v, (c, r), (c, r - 1))  # towards own goal
            spatial[4, r, c] = not rules.can_step(h, v, (c, r), (c, r + 1))
            spatial[5, r, c] = not rules.can_step(h, v, (c, r), (c + 1, r))
            spatial[6, r, c] = not rules.can_step(h, v, (c, r), (c - 1, r))
    spatial[7, 0, :] = 1.0
    to_my_goal = _distances(h, v, [(c, 0) for c in range(9)])
    to_opp_goal = _distances(h, v, [(c, 8) for c in range(9)])
    from_me = _distances(h, v, [me])
    from_opp = _distances(h, v, [opp])
    spatial[8] = _dist_feature(to_my_goal)
    spatial[9] = _dist_feature(to_opp_goal)
    spatial[10] = _dist_feature(from_me)
    spatial[11] = _dist_feature(from_opp)
    my_short = to_my_goal[me[1], me[0]]
    opp_short = to_opp_goal[opp[1], opp[0]]
    spatial[12] = (from_me >= 0) & (to_my_goal >= 0) & (from_me + to_my_goal == my_short)
    spatial[13] = (from_opp >= 0) & (to_opp_goal >= 0) & (from_opp + to_opp_goal == opp_short)
    for c, r in v:
        spatial[14, r, c] = 1.0
    for c, r in h:
        spatial[15, r, c] = 1.0
    spatial[16, :8, :8] = 1.0
    if io_version >= 2:
        # Legal wall placements for a player with a fence (whatever the fence counts), canonical anchor rows.
        for c in range(8):
            for r in range(8):
                r_canon = 7 - r if pos.to_move == rules.WHITE else r
                spatial[17, r_canon, c] = rules.wall_ok(pos, c, r, False)
                spatial[18, r_canon, c] = rules.wall_ok(pos, c, r, True)

    glob_ = np.zeros(NUM_GLOBAL[io_version], dtype=np.float32)
    mine = pos.walls_left[pos.to_move]
    theirs = pos.walls_left[1 - pos.to_move]
    glob_[0] = 1.0 if pos.to_move == rules.WHITE else 0.0
    glob_[1] = mine / 10.0
    glob_[2] = theirs / 10.0
    if mine > 0:
        glob_[3:7] = [np.exp(-(mine - 1) / s) for s in (1.0, 2.0, 4.0, 8.0)]
    glob_[7] = 1.0 if theirs >= 1 else 0.0
    if theirs > 0:
        glob_[8:12] = [np.exp(-(theirs - 1) / s) for s in (1.0, 2.0, 4.0, 8.0)]
    manhattan = abs(pos.pawn[0][0] - pos.pawn[1][0]) + abs(pos.pawn[0][1] - pos.pawn[1][1])
    glob_[12] = 1.0 if manhattan % 2 == 1 else -1.0
    glob_[13] = _dist_feature(np.array(my_short))
    glob_[14] = _dist_feature(np.array(opp_short))
    if io_version >= 2:
        glob_[15] = max(0, max_plies - ply) / 300.0
        glob_[16] = (komi if pos.to_move == rules.WHITE else -komi) / 5.0
    if io_version >= 3:
        # rep_info = (N, current count); rep_moves[r][c] = earlier occurrences of the position after the pawn move to
        # (c, r), board coords. Only legal pawn moves can repeat.
        n, count = (int(x) for x in rep_info)
        legal_pawn = {cell for kind, cell in rules.legal_moves(pos) if kind == "p"}
        if n > 0:
            glob_[17] = 1.0
            glob_[18] = 1.0 if n == 2 else min(1.0, (count - 1) / (n - 2))
            for r, c in zip(*np.nonzero(rep_moves)):
                assert (c, r) in legal_pawn, (c, r)
                r_canon = 8 - r if pos.to_move == rules.WHITE else r
                spatial[19, r_canon, c] = 1.0
                spatial[20, r_canon, c] = 1.0 if rep_moves[r, c] + 1 >= n else 0.0
        else:
            assert count >= 1 and not rep_moves.any()
    return spatial, glob_


def search_pos(kind, cell):
    """Search-space policy index (y * 17 + x on the 17x17 grid) of a move, as in cpp/game/board.h."""
    c, r = cell
    x, y = {"p": (2 * c, 2 * r), "v": (2 * c + 1, 2 * r), "h": (2 * c + 1, 2 * r + 1)}[kind]
    return y * SEARCH_LEN + x


def python_legal_mask(pos):
    mask = np.zeros(POLICY_SIZE, dtype=np.uint8)
    for kind, cell in rules.legal_moves(pos):
        mask[search_pos(kind, cell)] = 1
    return mask


@pytest.mark.parametrize("io_version", IO_VERSIONS)
def test_dumped_features_match_python(rows_by_version, io_version):
    rows = rows_by_version[io_version]
    n = rows["binaryInputNCHW"].shape[0]
    assert rows["binaryInputNCHW"].shape == (n, NUM_SPATIAL[io_version], 9, 9)
    assert rows["globalInputNC"].shape == (n, NUM_GLOBAL[io_version])
    # The positions include games with other rules and ones close to their ply limit.
    assert (rows["komi"][:, 0] != -0.5).any() and (rows["plyInfo"][:, 1] != 300).any()
    assert ((rows["plyInfo"][:, 1] - rows["plyInfo"][:, 0]) < 10).any()
    if io_version >= 3:
        # ... and positions with the repetition rule, repeating and drawing moves.
        assert (rows["repInfo"][:, 0] > 0).any() and (rows["repMoves"] > 0).any()
        assert rows["binaryInputNCHW"][:, 20].any()
    bad = []
    for i in range(n):
        ply, max_plies = (int(x) for x in rows["plyInfo"][i])
        spatial, glob_ = python_features(_position(rows, i), io_version, ply, max_plies, float(rows["komi"][i][0]),
                                         rows["repInfo"][i], rows["repMoves"][i])
        ds = np.abs(rows["binaryInputNCHW"][i] - spatial)
        dg = np.abs(rows["globalInputNC"][i] - glob_)
        if ds.max() > 1e-6 or dg.max() > 1e-6:
            chans = sorted(set(np.nonzero(ds > 1e-6)[0].tolist()))
            gidx = np.nonzero(dg > 1e-6)[0].tolist()
            bad.append(f"row {i}: spatial channels {chans}, global indices {gidx}")
    assert not bad, f"{len(bad)}/{n} rows differ:\n" + "\n".join(bad[:20])


def test_dumped_legal_moves_match_python_rules(rows_by_version):
    rows = rows_by_version[IO_VERSIONS[-1]]
    n = rows["legalMask"].shape[0]
    bad = []
    for i in range(n):
        pos = _position(rows, i)
        assert not pos.is_terminal()
        diff = np.nonzero(rows["legalMask"][i] != python_legal_mask(pos))[0].tolist()
        if diff:
            bad.append(f"row {i}: slots {diff[:10]}")
    assert not bad, f"{len(bad)}/{n} rows differ:\n" + "\n".join(bad[:20])


# ---------------------------------------------------------------------------------------------------------
# PyTorch reference and C++ evaluation

def canonical_policy_to_search(planes, white_to_move):
    """(3, 9, 9) canonical-frame policy planes -> 290 search slots (NaN where no move maps).

    Mirrors the training-data encoding (fillPolicyTargetQuoridor in cpp/dataio/trainingwrite.cpp):
    plane 0 = pawn destination (c, r), plane 1 = vertical wall anchor, plane 2 = horizontal wall anchor,
    rows flipped (8 - r for pawns, 7 - r for wall anchors) when White is to move."""
    out = np.full(POLICY_SIZE, np.nan, dtype=np.float64)
    for r_canon in range(9):
        for c in range(9):
            r = 8 - r_canon if white_to_move else r_canon
            out[search_pos("p", (c, r))] = planes[0, r_canon, c]
    for r_canon in range(8):
        for c in range(8):
            r = 7 - r_canon if white_to_move else r_canon
            out[search_pos("v", (c, r))] = planes[1, r_canon, c]
            out[search_pos("h", (c, r))] = planes[2, r_canon, c]
    return out


def make_model(config_name):
    torch.manual_seed(12345)
    model = Model(modelconfigs.config_of_name[config_name], pos_len=9)
    model.initialize()
    # Random init gives a nearly uniform policy and a value near 50%, which would hide mapping errors.
    with torch.no_grad():
        model.policy_head.conv2p.weight.mul_(10.0)
        model.value_head.linear_value.weight.mul_(5.0)
    model.eval()
    return model


def reference_outputs(model, rows, symmetry):
    spatial = torch.from_numpy(rows["binaryInputNCHW"])
    glob_ = torch.from_numpy(rows["globalInputNC"])
    with torch.no_grad():
        out = model(apply_symmetry_quoridor(spatial, symmetry), glob_)
    planes = apply_symmetry_policy_quoridor(out[0][0][:, 0:3].contiguous(), symmetry).double().numpy()
    value_logits = out[0][1].double().numpy()  # [win, loss] for the side to move
    post = model.postprocess_output(out)[0]
    # Side-to-move utility score, its stdev, lead, variance time, shortterm squared errors (see
    # Model.postprocess_output; a v1 net's lead is its score)
    score, vtime, stdev, st_v2, st_s2, lead = (post[i].double().numpy() for i in (4, 3, 5, 6, 7, 10))

    n = planes.shape[0]
    policy = np.full((n, POLICY_SIZE), -1.0)
    value = np.zeros((n, 3))
    misc = np.zeros((n, 6))
    for i in range(n):
        white = int(rows["nextPlayer"][i][0]) == 2
        logits = canonical_policy_to_search(planes[i], white)
        legal = rows["legalMask"][i].astype(bool)
        assert not np.isnan(logits[legal]).any()
        z = logits[legal] - logits[legal].max()
        p = np.exp(z)
        policy[i, legal] = p / p.sum()
        w = np.exp(value_logits[i] - value_logits[i].max())
        win, loss = w / w.sum()
        value[i] = [win, loss, 0.0] if white else [loss, win, 0.0]
        sign = 1.0 if white else -1.0
        misc[i] = [sign * score[i], sign * lead[i], stdev[i], vtime[i], np.sqrt(st_v2[i]), np.sqrt(st_s2[i])]
    return policy, value, misc


@pytest.fixture(scope="module", params=MODEL_CONFIGS)
def exported(request, workdir):
    name = request.param
    model = make_model(name)
    ckpt = os.path.join(workdir, name + ".ckpt")
    torch.save({"model": model.state_dict(), "config": modelconfigs.config_of_name[name]}, ckpt)
    _run([sys.executable, os.path.join(PYTHON_DIR, "export_model_pytorch.py"), "-checkpoint", ckpt,
          "-export-dir", os.path.join(workdir, name), "-model-name", name + "-parity", "-filename-prefix", "model"])
    return name, model, os.path.join(workdir, name, "model.bin.gz")


MISC_NAMES = ["scoreMean", "lead", "scoreStdev", "varTimeLeft", "shorttermWinlossError", "shorttermScoreError"]


def compare(rows, ref_policy, ref_value, cpp_policy, cpp_value, ref_misc, cpp_misc):
    legal = rows["legalMask"].astype(bool)
    report = {}
    report["illegal_slots_not_minus_one"] = int(np.sum(~legal & (cpp_policy[:, :POLICY_SIZE] != -1.0)))
    report["legal_slots_negative"] = int(np.sum(legal & (cpp_policy[:, :POLICY_SIZE] < 0)))
    dp = np.where(legal, np.abs(cpp_policy[:, :POLICY_SIZE] - ref_policy), 0.0)
    dv = np.abs(cpp_value - ref_value)
    report["max_policy_diff"] = float(dp.max())
    report["mean_policy_diff"] = float(dp.sum() / legal.sum())
    report["max_value_diff"] = float(dv.max())
    report["rows_policy_over_tol"] = int(np.sum(dp.max(axis=1) > FP32_TOL))
    report["rows_value_over_tol"] = int(np.sum(dv.max(axis=1) > FP32_TOL))
    dm = np.abs(cpp_misc - ref_misc) / (1.0 + np.abs(ref_misc))
    report["max_misc_reldiff"] = float(dm.max())
    report["worst_misc_output"] = MISC_NAMES[int(np.argmax(dm.max(axis=0)))]
    top_ref = np.argmax(np.where(legal, ref_policy, -2), axis=1)
    top_cpp = np.argmax(np.where(legal, cpp_policy[:, :POLICY_SIZE], -2), axis=1)
    report["rows_top_move_differs"] = int(np.sum(top_ref != top_cpp))
    worst = int(np.argmax(dp.max(axis=1)))
    white = int(rows["nextPlayer"][worst][0]) == 2
    report["worst_policy_row"] = f"{worst} ({'White' if white else 'Black'} to move)"
    return report


@pytest.mark.parametrize("symmetry", [0, 1])
def test_nn_parity(katago_bin, workdir, rows_by_version, exported, symmetry):
    name, model, bin_gz = exported
    rows = rows_by_version[modelconfigs.get_quoridor_io_version(model.config)]
    ref_policy, ref_value, ref_misc = reference_outputs(model, rows, symmetry)
    out_path = os.path.join(workdir, f"{name}.sym{symmetry}.cpp.npz")
    np.savez(os.path.join(workdir, f"{name}.sym{symmetry}.ref.npz"), policy=ref_policy, value=ref_value)
    config_args = ["-config", os.environ["NN_PARITY_CONFIG"]] if os.environ.get("NN_PARITY_CONFIG") else []
    _run([katago_bin, "evalnnparity", "-model", bin_gz, "-n", str(NUM_ROWS), "-seed", SEED,
          "-symmetry", str(symmetry), "-output", out_path] + config_args)
    with np.load(out_path) as d:
        cpp_policy, cpp_value = d["policy"].astype(np.float64), d["value"].astype(np.float64)
        cpp_misc = d["misc"].astype(np.float64)
    report = compare(rows, ref_policy, ref_value, cpp_policy, cpp_value, ref_misc, cpp_misc)
    text = "\n".join(f"  {k}: {v}" for k, v in report.items())
    print(f"\n{name} symmetry {symmetry}:\n{text}")
    ok = (report["max_policy_diff"] <= FP32_TOL and report["max_value_diff"] <= FP32_TOL
          and report["max_misc_reldiff"] <= FP32_TOL
          and report["illegal_slots_not_minus_one"] == 0 and report["legal_slots_negative"] == 0)
    assert ok, f"{name} symmetry {symmetry} parity failure:\n{text}"
