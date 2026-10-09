"""Tests for Q4 Selfplay and Training Data Writer (Round 4).

Covers:
- C1: Writer vs Replay (replaying games from .q4.jsonl records, checking inputs with features.py,
      targets, relative-seat rotation, one-hot style policy target, paths, future walls).
- C2: TD targets (hand-built sequence and written rows vs KataGo TD formula at 1e-6).
- C3: Side positions and forks (C27=0 and C59=0 on side positions, C52=2 on fork games).
- C4: Shuffle compatibility (python/shuffle.py preserves multiset of rows by game hash and ply).
- C5: Determinism (identical seeds, numGameThreads=1, nnRandomize=false -> identical npz files).
- C6: Memory (5-minute 8-thread run, RSS change < 10%).
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
import numpy as np
import pytest

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)

from q4.features import extract_features, decode_raw_distances
from q4.reference import Pos, eliminate, play, str_to_action, compute_distances_to_center
from tests.q4_testutil import find_katago, export_model, make_random_model


def action_to_policy_slot(act):
    kind, (x, y) = act
    if kind == 'p':
        return y * 11 + x
    elif kind == 'v':
        return 121 + y * 11 + x
    elif kind == 'h':
        return 242 + y * 11 + x
    raise ValueError(f"Unknown action {act}")


_MODEL_DIR_CACHE = []


def get_model_dir():
    """A models dir holding only the Q4 test net b1c32_q4 (cpp/tests/models also holds Duel nets and empty files for
    the findLatestModel tests; population self-play lists the models dir for snapshots, so it needs a clean one)."""
    if _MODEL_DIR_CACHE:
        return _MODEL_DIR_CACHE[0]
    src = None
    for c in [os.path.join(REPO_DIR, "cpp", "tests", "models", "b1c32_q4"), os.path.join(REPO_DIR, "tests", "models", "b1c32_q4")]:
        if os.path.isfile(os.path.join(c, "model.bin.gz")):
            src = c
    if src is None:
        raise RuntimeError("b1c32_q4 test model not found (python q4/make_random_model.py b1c32_q4 cpp/tests/models --scale-heads)")
    import tempfile
    d = tempfile.mkdtemp(prefix="q4models_")
    os.makedirs(os.path.join(d, "b1c32_q4"))
    shutil.copy(os.path.join(src, "model.bin.gz"), os.path.join(d, "b1c32_q4", "model.bin.gz"))
    _MODEL_DIR_CACHE.append(d)
    return d


def get_cfg_path():
    return os.path.join(REPO_DIR, "cpp", "configs", "q4", "training", "q4_selfplay.cfg")


def decode_game_hash(row_global):
    h0 = (int(row_global[44]) & 0x3FFFFF) | \
         ((int(row_global[45]) & 0x3FFFFF) << 22) | \
         ((int(row_global[46]) & 0xFFFFF) << 44)
    h1 = (int(row_global[47]) & 0x3FFFFF) | \
         ((int(row_global[48]) & 0x3FFFFF) << 22) | \
         ((int(row_global[49]) & 0xFFFFF) << 44)
    return (h0, h1)


def record_game_hash(rec):
    """(hash0, hash1) of a .q4.jsonl record ("gameHash": 32 hex digits), as decode_game_hash returns it."""
    h = rec["gameHash"]
    assert len(h) == 32, h
    return (int(h[:16], 16), int(h[16:], 16))


def unpack_binary_channel(packed_bytes):
    bits = np.unpackbits(packed_bytes, axis=-1)[..., :121]
    return bits.reshape(*packed_bytes.shape[:-1], 11, 11).astype(np.float32)


def compute_katago_td(values_by_turn, now_factor):
    """Direct Python implementation of KataGo fillValueTDTargets for a sequence of 5-vectors.
    values_by_turn: [turn0, turn1, ..., turnN-1, finalResult]
    """
    num_turns = len(values_by_turn) - 1
    res = []
    for t in range(num_turns):
        smoothed = np.array(values_by_turn[-1], dtype=np.float64)
        for i in range(num_turns - 1, t - 1, -1):
            smoothed += now_factor * (values_by_turn[i] - smoothed)
        res.append(smoothed)
    return res


def test_c2_td_targets_direct_math():
    """C2: Direct math test on hand-built sequence of value vectors comparing KataGo TD formula at 1e-6."""
    np.random.seed(42)
    num_turns = 10
    values = [np.random.dirichlet(np.ones(5)) for _ in range(num_turns)]
    final_result = np.zeros(5)
    final_result[np.random.randint(0, 5)] = 1.0
    values.append(final_result)

    factors = [
        1.0 / (1.0 + 121.0 * 0.176),
        1.0 / (1.0 + 121.0 * 0.056),
        1.0 / (1.0 + 121.0 * 0.016),
        1.0,
    ]

    for factor in factors:
        td_res = compute_katago_td(values, factor)
        assert len(td_res) == num_turns
        for t in range(num_turns):
            assert len(td_res[t]) == 5
            assert np.abs(np.sum(td_res[t]) - 1.0) < 1e-6


def test_c5_determinism(tmp_path):
    """C5: Same seed, numGameThreads=1, nnRandomize=false -> identical npz files."""
    katago = find_katago("eigen")
    assert katago is not None, "katago binary not found"
    models_dir = get_model_dir()
    cfg_path = get_cfg_path()

    out1 = str(tmp_path / "det1")
    out2 = str(tmp_path / "det2")

    for out in [out1, out2]:
        cmd = [
            katago, "q4selfplay",
            "-models-dir", models_dir,
            "-output-dir", out,
            "-config", cfg_path,
            "-max-games-total", "2",
            "-seed", "det_pytest_seed_42",
            "-override-config",
            "numGameThreads=1,maxVisits=4,nnRandomize=false,logToStdout=false"
        ]
        subprocess.check_call(cmd)

    npz1_list = sorted(glob.glob(os.path.join(out1, "*", "tdata", "*.npz")))
    npz2_list = sorted(glob.glob(os.path.join(out2, "*", "tdata", "*.npz")))

    assert len(npz1_list) >= 1 and len(npz2_list) >= 1
    assert len(npz1_list) == len(npz2_list)

    for p1, p2 in zip(npz1_list, npz2_list):
        f1 = np.load(p1)
        f2 = np.load(p2)
        assert f1.files == f2.files
        for k in f1.files:
            assert np.array_equal(f1[k], f2[k]), f"Mismatch in key {k} between deterministic runs"


def test_c1_c2_writer_vs_replay(tmp_path):
    """C1 & C2: Run 20 short games (forced elims in 5, rep draw in 5) and verify writer against reference replay."""
    katago = find_katago("eigen")
    assert katago is not None
    models_dir = get_model_dir()
    cfg_path = get_cfg_path()
    out_dir = str(tmp_path / "c1_data")

    # Run 1: 10 normal short games
    cmd1 = [
        katago, "q4selfplay",
        "-models-dir", models_dir,
        "-output-dir", out_dir,
        "-config", cfg_path,
        "-max-games-total", "10",
        "-seed", "c1_normal_seed",
        "-override-config",
        "numGameThreads=2,maxVisits=16,maxPlies=60,q4EliminationProb=0.0,q4RepetitionDrawProb=0.0,logToStdout=false"
    ]
    subprocess.check_call(cmd1)

    # Run 2: 5 games with forced elimination
    cmd2 = [
        katago, "q4selfplay",
        "-models-dir", models_dir,
        "-output-dir", out_dir,
        "-config", cfg_path,
        "-max-games-total", "5",
        "-seed", "c1_elim_seed",
        "-override-config",
        "numGameThreads=2,maxVisits=16,maxPlies=60,q4EliminationProb=1.0,q4RepetitionDrawProb=0.0,logToStdout=false"
    ]
    subprocess.check_call(cmd2)

    # Run 3: 5 games with repetition draw count N=2
    cmd3 = [
        katago, "q4selfplay",
        "-models-dir", models_dir,
        "-output-dir", out_dir,
        "-config", cfg_path,
        "-max-games-total", "5",
        "-seed", "c1_rep_seed",
        "-override-config",
        "numGameThreads=2,maxVisits=16,maxPlies=60,q4EliminationProb=0.0,q4RepetitionDrawProb=1.0,q4RepetitionDrawCounts=2,q4RepetitionDrawCountWeights=1.0,logToStdout=false"
    ]
    subprocess.check_call(cmd3)

    # Read all records
    record_files = glob.glob(os.path.join(out_dir, "*", "records", "*.q4.jsonl"))
    assert len(record_files) >= 1

    records = []
    for rf in record_files:
        with open(rf) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    assert len(records) >= 20, f"Expected at least 20 games, got {len(records)}"

    # Replay each record to map (event sequence) -> state history
    game_replays = []
    for rec in records:
        rules = rec.get("rules", {})
        pos = Pos(
            max_plies=rules.get("maxPlies", 400),
            repetition_draw_count=rules.get("repetitionDrawCount", 0),
            initial_walls=rules.get("initialWalls", [7, 7, 7, 7]),
        )
        states_by_ply = {}
        actions_by_ply = {}
        for ev in rec.get("events", []):
            ply = pos.plies
            states_by_ply[ply] = pos.copy()
            if "elim" in ev:
                s_elim = ev["elim"] - 1
                pos = eliminate(pos, s_elim)
            else:
                act = str_to_action(ev["a"])
                actions_by_ply[ply] = act
                pos = play(pos, act)
        final_pos = pos
        game_replays.append({
            "rec": rec,
            "states_by_ply": states_by_ply,
            "actions_by_ply": actions_by_ply,
            "final_pos": final_pos,
            "final_plies": len(actions_by_ply),
        })

    # Read all training rows
    npz_files = glob.glob(os.path.join(out_dir, "*", "tdata", "*.npz"))
    assert len(npz_files) >= 1

    all_rows = []
    for nf in npz_files:
        data = np.load(nf)
        n = data["globalInputNC"].shape[0]
        for i in range(n):
            all_rows.append({
                "binaryInputPacked": data["binaryInputNCHWPacked"][i],
                "spatialDist": data["spatialDistNCHW"][i],
                "globalInput": data["globalInputNC"][i],
                "policyTargets": data["policyTargetsNCMove"][i],
                "globalTargets": data["globalTargetsNC"][i],
                "valueTargets": data["valueTargetsNCHW"][i],
            })

    assert len(all_rows) > 0

    # Group rows by decoded game hash
    rows_by_hash = {}
    for r in all_rows:
        gh = decode_game_hash(r["globalTargets"])
        rows_by_hash.setdefault(gh, []).append(r)

    # Separate side positions and group main rows by game hash
    rows_by_hash = {}
    side_pos_rows = []
    for r in all_rows:
        gt = r["globalTargets"]
        if gt[59] == 0.0 or gt[27] == 0.0:
            assert gt[27] == 0.0, "Side position row must have C27=0"
            assert gt[59] == 0.0, "Side position row must have C59=0"
            assert np.sum(r["policyTargets"][1] != 0) == 0, "Side position row must have empty style target C1"
            side_pos_rows.append(r)
        else:
            gh = decode_game_hash(gt)
            rows_by_hash.setdefault(gh, []).append(r)

    checked_rows = 0
    checked_elims = 0
    checked_reps = 0

    # Rows are bound to a replay by the game hash both carry (several games can share opening moves).
    replay_by_hash = {}
    for gr in game_replays:
        rh = record_game_hash(gr["rec"])
        assert rh not in replay_by_hash, f"two records with game hash {rh}"
        replay_by_hash[rh] = gr

    for gh, rows in rows_by_hash.items():
        matched_game = replay_by_hash.get(gh)
        assert matched_game is not None, f"Could not match game hash {gh} to any game replay"
        final_pos = matched_game["final_pos"]
        dist_to_center = compute_distances_to_center(final_pos.hwalls, final_pos.vwalls)

        for r in rows:
            gt = r["globalTargets"]
            ply = int(gt[50])
            rep_draw = int(gt[56])
            if rep_draw == 2:
                checked_reps += 1

            pos = matched_game["states_by_ply"][ply]
            act_played = matched_game["actions_by_ply"][ply]
            to_move = pos.to_move

            if sum(pos.alive) < 4:
                checked_elims += 1

            # 1. Check inputs recomputed with features.py (exact)
            exp_spatial, exp_glob = extract_features(pos, symmetry=0)
            np.testing.assert_allclose(r["globalInput"], exp_glob, atol=1e-5)

            unpacked = unpack_binary_channel(r["binaryInputPacked"])
            unpacked[10:15] = decode_raw_distances(r["spatialDist"])
            np.testing.assert_array_equal(unpacked, exp_spatial)

            # 2. Check C1 style-policy one-hot
            expected_slot = action_to_policy_slot(act_played)
            assert r["policyTargets"][1, expected_slot] == 1
            assert np.sum(r["policyTargets"][1] != 0) == 1

            # 3. Check final result (C0-C4)
            if final_pos.winner is None:
                assert gt[4] == 1.0
                assert np.all(gt[0:4] == 0.0)
            else:
                rel_winner = (final_pos.winner - to_move) % 4
                assert gt[rel_winner] == 1.0
                assert np.sum(gt[0:5] == 1.0) == 1

            # 4. Check remaining plies (C35)
            c27 = gt[27]
            assert c27 > 0
            assert gt[35] == matched_game["final_plies"] - ply

            # 5. Check final distances (C36-39) and weights (C40-43)
            for s in range(4):
                rel_s = (s - to_move) % 4
                if not final_pos.alive[s]:
                    assert gt[40 + rel_s] == 0.0
                else:
                    assert gt[40 + rel_s] == c27
                    if s == final_pos.winner:
                        assert gt[36 + rel_s] == 0.0
                    else:
                        exp_d = dist_to_center.get(final_pos.pawn[s], 255)
                        assert gt[36 + rel_s] == exp_d

            # 6. Check paths and future walls in valueTargetsNCHW
            for k in range(4):
                s = (to_move + k) % 4
                if not pos.alive[s]:
                    assert np.all(r["valueTargets"][k] == 0)
                    assert np.all(r["valueTargets"][4 + 2 * k] == 0)
                    assert np.all(r["valueTargets"][5 + 2 * k] == 0)
                else:
                    exp_visited = np.zeros((11, 11), dtype=np.int8)
                    exp_vwalls = np.zeros((11, 11), dtype=np.int8)
                    exp_hwalls = np.zeros((11, 11), dtype=np.int8)

                    # Initial pawn position of seat s at this ply
                    px, py = pos.pawn[s]
                    exp_visited[py, px] = 1

                    for fut_p in range(ply, matched_game["final_plies"]):
                        if fut_p in matched_game["actions_by_ply"]:
                            st_fut = matched_game["states_by_ply"][fut_p]
                            if st_fut.to_move == s:
                                a = matched_game["actions_by_ply"][fut_p]
                                kind, coord = a
                                if kind == 'p':
                                    exp_visited[coord[1], coord[0]] = 1
                                elif kind == 'v':
                                    exp_vwalls[coord[1], coord[0]] = 1
                                elif kind == 'h':
                                    exp_hwalls[coord[1], coord[0]] = 1

                    np.testing.assert_array_equal(r["valueTargets"][k], exp_visited)
                    np.testing.assert_array_equal(r["valueTargets"][4 + 2 * k], exp_vwalls)
                    np.testing.assert_array_equal(r["valueTargets"][5 + 2 * k], exp_hwalls)

            # 7. Check version and zeros
            assert gt[60] == 2.0
            assert gt[61] in (0, 1, 2, 3, 4, 5, 6)
            assert np.all(gt[62:64] == 0.0)

            checked_rows += 1

    assert checked_rows > 50
    assert checked_elims > 0
    assert checked_reps > 0


def test_c3_side_positions_and_forks(tmp_path):
    """C3: Side positions have C27=0 and C59=0; fork games have C52=2; inputs pass C1 check."""
    katago = find_katago("eigen")
    assert katago is not None
    models_dir = get_model_dir()
    cfg_path = get_cfg_path()
    out_dir = str(tmp_path / "c3_data")

    cmd = [
        katago, "q4selfplay",
        "-models-dir", models_dir,
        "-output-dir", out_dir,
        "-config", cfg_path,
        "-max-games-total", "8",
        "-seed", "c3_fork_seed",
        "-override-config",
        "numGameThreads=1,maxVisits=8,maxPlies=30,sidePositionProb=1.0,forkGameProb=1.0,earlyForkGameProb=0.0,logToStdout=false"
    ]
    subprocess.check_call(cmd)

    npz_files = glob.glob(os.path.join(out_dir, "*", "tdata", "*.npz"))
    assert len(npz_files) >= 1

    found_side_pos = False
    found_fork_game = False

    for nf in npz_files:
        data = np.load(nf)
        n = data["globalInputNC"].shape[0]
        for i in range(n):
            gt = data["globalTargetsNC"][i]
            c27 = gt[27]
            c52 = int(gt[52])
            c59 = gt[59]

            if c27 == 0.0:
                found_side_pos = True
                assert c59 == 0.0, f"Side position row must have C59=0, got {c59}"

            if c52 == 2:
                found_fork_game = True

    assert found_side_pos, "Expected to observe side positions with sidePositionProb=1.0"
    assert found_fork_game, "Expected to observe fork games with forkGameProb=1.0"

    # Verify every main row matches replay inputs
    record_files = glob.glob(os.path.join(out_dir, "*", "records", "*.q4.jsonl"))
    records = []
    for rf in record_files:
        with open(rf) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    game_replays = []
    for rec in records:
        rules = rec.get("rules", {})
        pos = Pos(
            max_plies=rules.get("maxPlies", 400),
            repetition_draw_count=rules.get("repetitionDrawCount", 0),
            initial_walls=rules.get("initialWalls", [7, 7, 7, 7]),
        )
        states_by_ply = {}
        actions_by_ply = {}
        for ev in rec.get("events", []):
            ply = pos.plies
            if "elim" in ev:
                s_elim = ev["elim"] - 1
                pos = eliminate(pos, s_elim)
            else:
                states_by_ply[ply] = pos.copy()
                act = str_to_action(ev["a"])
                actions_by_ply[ply] = act
                pos = play(pos, act)
        game_replays.append({
            "rec": rec,
            "states_by_ply": states_by_ply,
            "actions_by_ply": actions_by_ply,
        })

    main_rows_by_hash = {}
    for nf in npz_files:
        data = np.load(nf)
        n = data["globalInputNC"].shape[0]
        for i in range(n):
            gt = data["globalTargetsNC"][i]
            if gt[27] > 0 and gt[59] == 1.0:
                gh = decode_game_hash(gt)
                main_rows_by_hash.setdefault(gh, []).append({
                    "gt": gt,
                    "binaryInputPacked": data["binaryInputNCHWPacked"][i],
                    "spatialDist": data["spatialDistNCHW"][i],
                    "globalInput": data["globalInputNC"][i],
                    "policyTargets": data["policyTargetsNCMove"][i],
                })

    # Fork games share their opening moves with the parent game, so rows are bound to a replay by the game hash
    # both carry, not by the played actions.
    replay_by_hash = {}
    for gr in game_replays:
        rh = record_game_hash(gr["rec"])
        assert rh not in replay_by_hash, f"two records with game hash {rh}"
        replay_by_hash[rh] = gr

    checked_rows = 0
    for gh, rows in main_rows_by_hash.items():
        matched_game = replay_by_hash.get(gh)
        assert matched_game is not None, f"Could not match game hash {gh} to any game replay in C3"
        for r in rows:
            ply = int(r["gt"][50])
            act = matched_game["actions_by_ply"][ply]
            assert r["policyTargets"][1, action_to_policy_slot(act)] == 1
            st = matched_game["states_by_ply"][ply]
            exp_spatial, exp_glob = extract_features(st, symmetry=0)
            np.testing.assert_allclose(r["globalInput"], exp_glob, atol=1e-5)
            unpacked = unpack_binary_channel(r["binaryInputPacked"])
            unpacked[10:15] = decode_raw_distances(r["spatialDist"])
            np.testing.assert_array_equal(unpacked, exp_spatial)
            checked_rows += 1
    assert checked_rows > 0


def test_c4_shuffle_compatibility(tmp_path):
    """C4: shuffle.py preserves multiset of rows by game hash and ply."""
    katago = find_katago("eigen")
    assert katago is not None
    models_dir = get_model_dir()
    cfg_path = get_cfg_path()

    selfplay_dir = str(tmp_path / "selfplay")
    cmd = [
        katago, "q4selfplay",
        "-models-dir", models_dir,
        "-output-dir", selfplay_dir,
        "-config", cfg_path,
        "-max-games-total", "4",
        "-seed", "c4_shuffle_seed",
        "-override-config",
        "numGameThreads=1,maxVisits=4,maxPlies=30,logToStdout=false"
    ]
    subprocess.check_call(cmd)

    tdata_dirs = glob.glob(os.path.join(selfplay_dir, "*", "tdata"))
    assert len(tdata_dirs) >= 1
    tdata_dir = tdata_dirs[0]

    out_shuf = str(tmp_path / "shuffled")
    out_tmp = str(tmp_path / "shuf_tmp")

    shuffle_cmd = [
        sys.executable,
        os.path.join(REPO_DIR, "python", "shuffle.py"),
        tdata_dir,
        "-out-dir", out_shuf,
        "-out-tmp-dir", out_tmp,
        "-num-processes", "1",
        "-min-rows", "1",
        "-keep-target-rows", "all",
        "-approx-rows-per-out-file", "20",
        "-include-meta"
    ]
    subprocess.check_call(shuffle_cmd)

    # Collect multiset of (gameHash, ply) from original files
    orig_rows = []
    orig_meta = {}
    for f in glob.glob(os.path.join(tdata_dir, "*.npz")):
        d = np.load(f)
        assert d["metadataInputNC"].shape == (d["globalTargetsNC"].shape[0], 192)
        for gt, meta in zip(d["globalTargetsNC"], d["metadataInputNC"]):
            gh = decode_game_hash(gt)
            ply = int(gt[50])
            orig_rows.append((gh, ply))
            if gt[59] == 1.0:      # main rows (a game has one per ply; side positions can share a ply)
                orig_meta[(gh, ply)] = meta

    # Collect multiset of (gameHash, ply) from shuffled files
    shuf_rows = []
    for f in glob.glob(os.path.join(out_shuf, "*.npz")):
        d = np.load(f)
        for gt, meta in zip(d["globalTargetsNC"], d["metadataInputNC"]):
            gh = decode_game_hash(gt)
            ply = int(gt[50])
            shuf_rows.append((gh, ply))
            if gt[59] == 1.0:
                assert np.array_equal(orig_meta[(gh, ply)], meta), "metadataInputNC changed by the shuffle"

    assert len(orig_rows) > 0
    assert len(orig_rows) == len(shuf_rows)
    assert Counter(orig_rows) == Counter(shuf_rows), "Multiset of rows not preserved by shuffle.py"


def test_c7_search_policy_target_is_not_one_hot(tmp_path):
    """R6 regression: with a real search (100 visits, no cheap searches) the search policy target (channel 0) has
    several nonzero slots and the search entropy C32 is clearly positive. Before the fix every row was one-hot with
    C32 = 0 (getExploreSelectionValueInverse had swapped parameters)."""
    katago = find_katago("eigen")
    assert katago is not None
    out_dir = str(tmp_path / "c7_data")
    subprocess.check_call([
        katago, "q4selfplay",
        "-models-dir", get_model_dir(),
        "-output-dir", out_dir,
        "-config", get_cfg_path(),
        "-max-games-total", "2",
        "-seed", "c7_entropy_seed",
        "-override-config",
        "numGameThreads=2,maxVisits=100,maxPlies=40,cheapSearchProb=0.0,logToStdout=false"
    ])
    npz_files = glob.glob(os.path.join(out_dir, "*", "tdata", "*.npz"))
    assert len(npz_files) >= 1
    nonzero_slots, entropies = [], []
    for nf in npz_files:
        data = np.load(nf)
        gt = data["globalTargetsNC"]
        pol = data["policyTargetsNCMove"][:, 0].astype(np.float32)
        for i in range(gt.shape[0]):
            if gt[i, 26] > 0:  # full-search row; maxVisits = 100 so every such row has >= 50 unreduced visits
                nonzero_slots.append(int((pol[i] > 0).sum()))
                entropies.append(float(gt[i, 32]))
    assert len(nonzero_slots) >= 20, len(nonzero_slots)
    print(f"c7: {len(entropies)} rows, mean C32 {np.mean(entropies):.3f}, mean nonzero C0 slots {np.mean(nonzero_slots):.2f}")
    assert np.mean(entropies) > 0.5, np.mean(entropies)
    assert np.mean(nonzero_slots) > 1.0, np.mean(nonzero_slots)


# ---------------------------------------------------------------------------------------------------------------
# Round 7, Part A: population self-play (docs/q4/rounds/R7.md, Q4IO.md §8.1)
# ---------------------------------------------------------------------------------------------------------------

KIND_OF_TYPE = {"selfplay": 0, "weak": 1, "snapshot": 2, "greedy": 3, "randomPawn": 4, "basher": 5, "grudge": 6}
NO_FORKS = "earlyForkGameProb=0,forkGameProb=0"


def run_selfplay(katago, models_dir, out_dir, seed, overrides, games, cfg_edits=None):
    """q4selfplay with the repo config; `overrides` is the -override-config string, `cfg_edits` {key: value} replaces
    keys in a copy of the config (for values with commas)."""
    cfg_path = get_cfg_path()
    if cfg_edits:
        lines = open(cfg_path).read().splitlines()
        for key, value in cfg_edits.items():
            lines = [l for l in lines if not l.startswith(key + " ")]
            lines.append(f"{key} = {value}")
        cfg_path = out_dir + "_edited.cfg"
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        with open(cfg_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    out = subprocess.run(
        [katago, "q4selfplay", "-models-dir", models_dir, "-output-dir", out_dir, "-config", cfg_path,
         "-max-games-total", str(games), "-seed", seed, "-override-config", overrides],
        check=True, capture_output=True, text=True)
    return out.stdout + out.stderr


def read_records(out_dir):
    records = []
    for rf in sorted(glob.glob(os.path.join(out_dir, "*", "records", "*.q4.jsonl"))):
        with open(rf) as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records


def read_rows(out_dir):
    """All written rows: dicts keyed by npz key plus the decoded game hash."""
    rows = []
    for nf in sorted(glob.glob(os.path.join(out_dir, "*", "tdata", "*.npz"))):
        data = np.load(nf)
        for i in range(data["globalInputNC"].shape[0]):
            row = {k: data[k][i] for k in data.files}
            row["hash"] = decode_game_hash(row["globalTargetsNC"])
            rows.append(row)
    return rows


def replay_states(rec):
    """ply -> (position before the move, action) for the moves of a record, in event order."""
    rules = rec.get("rules", {})
    pos = Pos(max_plies=rules.get("maxPlies", 400), repetition_draw_count=rules.get("repetitionDrawCount", 0),
              initial_walls=rules.get("initialWalls", [7, 7, 7, 7]))
    states, actions, event_of_ply = {}, {}, {}
    for e, ev in enumerate(rec["events"]):
        if "elim" in ev:
            pos = eliminate(pos, ev["elim"] - 1)
        else:
            states[pos.plies] = pos.copy()
            actions[pos.plies] = str_to_action(ev["a"])
            event_of_ply[pos.plies] = e
            pos = play(pos, actions[pos.plies])
    return states, actions, event_of_ply


def rotate_to_relative(v_abs, to_move):
    """[abs seats 0..3, draw] -> [relative seats, draw]"""
    return np.array([v_abs[(to_move + k) % 4] for k in range(4)] + [v_abs[4]])


def test_p1_all_learner_games_do_not_depend_on_the_population(tmp_path):
    """P1: the composition is drawn from its own stream, so an all-learner game is bit-identical whether the run has
    q4PopulationMixedProb = 0 or 0.5 (same seed, forks off). Against the Part 0 head binary ($Q4_PART0_BIN, if set)
    the data written with q4PopulationMixedProb = 0 is bit-identical except for the column C60 (format version 1 -> 2)."""
    katago = find_katago("eigen")
    models = get_model_dir()
    common = f"numGameThreads=1,maxVisits=8,maxPlies=60,nnRandomize=false,logToStdout=false,{NO_FORKS},"
    run_selfplay(katago, models, str(tmp_path / "mixed0"), "p1_seed", common + "q4PopulationMixedProb=0", 40)
    run_selfplay(katago, models, str(tmp_path / "mixed05"), "p1_seed", common + "q4PopulationMixedProb=0.5", 40)
    rec0 = read_records(str(tmp_path / "mixed0"))
    rec05 = read_records(str(tmp_path / "mixed05"))
    assert len(rec0) == 40 and len(rec05) == 40
    for r in rec0:
        assert all(p["type"] == "selfplay" for p in r["players"])
    rows0 = read_rows(str(tmp_path / "mixed0"))
    rows05 = read_rows(str(tmp_path / "mixed05"))
    by_hash0 = {}
    for r in rows0:
        by_hash0.setdefault(r["hash"], []).append(r)
    by_hash05 = {}
    for r in rows05:
        by_hash05.setdefault(r["hash"], []).append(r)
    all_learner = [r for r in rec05 if all(p["type"] == "selfplay" for p in r["players"])]
    mixed = [r for r in rec05 if not all(p["type"] == "selfplay" for p in r["players"])]
    assert len(all_learner) >= 3 and len(mixed) >= 3, (len(all_learner), len(mixed))
    n_compared = 0
    for rec in all_learner:
        gh = record_game_hash(rec)
        assert gh in by_hash0, "an all-learner game of the mixed run is missing from the mixedProb = 0 run"
        a, b = by_hash0[gh], by_hash05[gh]
        assert len(a) == len(b)
        for ra, rb in zip(a, b):
            for k in ra:
                if k != "hash":
                    assert np.array_equal(ra[k], rb[k]), f"key {k} differs in an all-learner game"
            n_compared += 1
    assert n_compared > 50
    print(f"p1: {len(all_learner)} all-learner games ({n_compared} rows) identical; {len(mixed)} mixed games")

    part0 = os.environ.get("Q4_PART0_BIN")
    if part0:
        part0_cfg = os.environ.get("Q4_PART0_CFG", get_cfg_path())
        subprocess.run(
            [part0, "q4selfplay", "-models-dir", models, "-output-dir", str(tmp_path / "part0"), "-config", part0_cfg,
             "-max-games-total", "40", "-seed", "p1_seed", "-override-config", common.rstrip(",")],
            check=True, capture_output=True)
        rows_p0 = read_rows(str(tmp_path / "part0"))
        assert len(rows_p0) == len(rows0)
        for ra, rb in zip(rows_p0, rows0):
            for k in ra:
                if k == "hash":
                    continue
                if k == "globalTargetsNC":
                    x, y = ra[k].copy(), rb[k].copy()
                    assert x[60] == 1.0 and y[60] == 2.0
                    x[60] = y[60] = 0
                    assert np.array_equal(x, y)
                else:
                    assert np.array_equal(ra[k], rb[k]), f"key {k} differs from the Part 0 head"
        print(f"p1: {len(rows0)} rows identical to the Part 0 head binary except C60")


def test_p3_mixed_games_rows(tmp_path):
    """P3: a mixed self-play run. Non-learner rows have C26 = 0, an all-zero channel 0, C29 = 1 and the played move as
    the style target, C61 = the kind of the seat to move (as the record says); the observer's value vector is the
    row's C20-24; every row's TD columns are finite and each value vector sums to 1; grudge targets are learners."""
    katago = find_katago("eigen")
    out_dir = str(tmp_path / "p3")
    run_selfplay(katago, get_model_dir(), out_dir, "p3_seed",
                 "numGameThreads=2,maxVisits=24,maxPlies=100,logToStdout=false,q4PopulationMixedProb=1.0,"
                 "q4EliminationProb=0.2,q4PopulationObserverRowWeight=1.0", 40)
    records = read_records(out_dir)
    rows = read_rows(out_dir)
    assert len(records) == 40
    rec_by_hash = {record_game_hash(r): r for r in records}
    replays = {h: replay_states(r) for h, r in rec_by_hash.items()}

    kinds_seen = set()
    num_learners_seen = set()
    for rec in records:
        types = [p["type"] for p in rec["players"]]
        kinds_seen.update(types)
        num_learners_seen.add(types.count("selfplay"))
        assert 1 <= types.count("selfplay") <= 3, types
        for s, p in enumerate(rec["players"]):
            if p["type"] == "grudge":
                target = int(p["name"].split(">")[1])
                assert rec["players"][target]["type"] == "selfplay", f"grudge target {target} is not a learner: {types}"
            if p["type"] == "weak":
                assert p["visits"] in (1, 2, 4, 8, 16, 32)
    assert num_learners_seen == {1, 2, 3}, num_learners_seen
    assert {"selfplay", "weak", "greedy", "randomPawn", "basher", "grudge"} <= kinds_seen, kinds_seen

    n_observer = n_learner = 0
    for row in rows:
        gt = row["globalTargetsNC"]
        pol = row["policyTargetsNCMove"].astype(np.float32)
        rec = rec_by_hash[row["hash"]]
        states, actions, event_of_ply = replays[row["hash"]]
        ply = int(gt[50])
        mover = states[ply].to_move
        kind = KIND_OF_TYPE[rec["players"][mover]["type"]]
        assert gt[60] == 2.0 and np.all(gt[62:64] == 0)
        if gt[59] == 0.0:       # side position: searched by the learners' search whoever would move; C61 = 0
            assert gt[61] == 0
        else:
            assert int(gt[61]) == kind, (gt[61], kind)
        # TD columns: finite, each value vector a distribution
        for c0 in (0, 5, 10, 15, 20):
            v = gt[c0:c0 + 5]
            assert np.all(np.isfinite(v)) and np.all(v >= -1e-6) and abs(float(v.sum()) - 1.0) < 1e-5, (c0, v)
        assert np.isfinite(gt[30:33]).all()
        if gt[59] == 0.0:
            continue
        slot = action_to_policy_slot(actions[ply])
        assert pol[1].sum() == 1 and pol[1][slot] == 1 and gt[29] == 1.0, "style target = the move actually played"
        if kind == 0:
            n_learner += 1
            assert gt[26] > 0 or gt[26] == 0  # cheap-search rows are not written; full rows have C26 = 1
            continue
        n_observer += 1
        assert gt[26] == 0.0 and not np.any(pol[0]), "observer row must have no search policy target"
        assert gt[27] > 0 and gt[53] > 0  # outcome weight; C53 = the observer's visits
        assert gt[53] <= 24
        assert gt[30] == 0.0              # policy surprise 0
        # C20-24 is the observer's root value vector (printed in the record's comment, absolute seats)
        comment = rec["comments"][event_of_ply[ply]]
        v_abs = [float(x) for x in comment[comment.index("[") + 1:comment.index("]")].split(",")]
        exp = rotate_to_relative(v_abs, mover)
        assert np.max(np.abs(exp - gt[20:25])) < 2e-4, (exp, gt[20:25])
    assert n_observer > 100 and n_learner > 50, (n_observer, n_learner)
    print(f"p3: {len(records)} games, {n_observer} observer rows, {n_learner} learner rows, kinds {sorted(kinds_seen)}")


def count_rows_and_plies(out_dir):
    """Main rows and plies by whether the mover is a learner (observer rows: C61 > 0 on a main row)."""
    records = read_records(out_dir)
    rows = read_rows(out_dir)
    plies = {"learner": 0, "observer": 0}
    for rec in records:
        types = [p["type"] for p in rec["players"]]
        states, actions, _ = replay_states(rec)
        for ply in range(len(actions)):
            plies["learner" if types[states[ply].to_move] == "selfplay" else "observer"] += 1
    n = {"learner": 0, "observer": 0}
    for row in rows:
        gt = row["globalTargetsNC"]
        if gt[59] == 0.0:
            continue
        n["learner" if gt[61] == 0 else "observer"] += 1
    return n, plies


def test_p5_observer_row_weight(tmp_path):
    """P5 (round 8): observer rows are written with target weight q4PopulationObserverRowWeight, which then goes through
    KataGo's surprise weighting and stochastic integerization like any turn: the default 0.25 writes about a quarter of
    the observer rows of weight 1 (same seed), 0 writes none."""
    katago = find_katago("eigen")
    common = "numGameThreads=2,maxVisits=24,maxPlies=100,logToStdout=false,q4PopulationMixedProb=1.0"
    res = {}
    for w in ("1.0", "0.25", "0.0"):
        out = str(tmp_path / f"p5_{w}")
        run_selfplay(katago, get_model_dir(), out, "p5_seed", common + f",q4PopulationObserverRowWeight={w}", 60)
        n, plies = count_rows_and_plies(out)
        res[w] = (n, plies)
        print(f"p5 weight {w}: main rows {n}, plies {plies}, observer rows per observer ply "
              f"{n['observer'] / plies['observer']:.3f}, learner rows per learner ply {n['learner'] / plies['learner']:.3f}")
    rate = lambda w: res[w][0]["observer"] / res[w][1]["observer"]
    assert 0.15 < rate("0.25") / rate("1.0") < 0.35, (rate("0.25"), rate("1.0"))
    assert res["0.0"][0]["observer"] == 0 and res["0.0"][0]["learner"] > 0, res["0.0"]


def make_models_dir(tmp_path, names, base_time=1700000000):
    """A models dir of random b1c32_q4 nets, modified in the order given."""
    d = str(tmp_path / "models")
    for i, name in enumerate(names):
        model = make_random_model("b1c32_q4", seed=100 + i, scale_heads=True)
        path = export_model(model, "b1c32_q4", d, name)
        os.utime(path, (base_time + 100 * i, base_time + 100 * i))
    return d


def test_p4_snapshot_players_use_only_older_models(tmp_path):
    """P4: with several models in models/, snapshot players are older models only (never the current net), the ones
    q4PopulationSnapshotAges exports back (an age beyond the oldest clamps to it; duplicates collapse); with a single
    model there is no snapshot player."""
    katago = find_katago("eigen")
    # 6 nets, the newest (m6) is the current one; ages 2, 4, 32, 40 = m4, m2, m1 (clamped), m1 (collapsed)
    models = make_models_dir(tmp_path, ["m1", "m2", "m3", "m4", "m5", "m6"])
    out_dir = str(tmp_path / "p4")
    log = run_selfplay(katago, models, out_dir, "p4_seed",
                       "numGameThreads=3,maxVisits=16,maxPlies=60,q4PopulationMixedProb=1.0,"
                       "q4PopulationSnapshotVisits=8,logToStdout=true", 60,
                       cfg_edits={"q4PopulationWeights": "weak:0, snapshot:1, greedy:0, randomPawn:0, basher:0, grudge:0",
                                  "q4PopulationSnapshotAges": "2, 4, 32, 40"})
    assert "Snapshots for population games (current m6): m4 m2 m1" in log, log[-2000:]
    records = read_records(out_dir)
    nets = set()
    for rec in records:
        for p in rec["players"]:
            if p["type"] == "snapshot":
                nets.add(p["net"])
                assert p["visits"] == 8
            else:
                assert p["type"] == "selfplay" or p["type"] == "weak" or p["type"] in KIND_OF_TYPE
    assert nets == {"m4", "m2", "m1"}, nets
    # the games are recorded under the current net only
    assert [os.path.basename(d) for d in glob.glob(os.path.join(out_dir, "*"))] == ["m6"]

    # a single model: cycle 1 has no snapshots, the kind is redrawn
    out1 = str(tmp_path / "p4_cycle1")
    run_selfplay(katago, get_model_dir(), out1, "p4_seed_c1",
                 "numGameThreads=2,maxVisits=16,maxPlies=60,q4PopulationMixedProb=1.0,logToStdout=false", 30)
    types = {p["type"] for rec in read_records(out1) for p in rec["players"]}
    assert "snapshot" not in types and "weak" in types, types
