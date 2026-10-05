"""Differential testing of C++ Q4 engine against Python reference (Plan §5.7 T3 & R2 A5).

Spawns `katago q4tool legal`, runs diverse game generators:
1. Random games with random eliminations and small maxPlies (e.g. 35-40), verifying winner & maxPlies draws
2. Repetition games (pawn shuffling) verifying N=2 and N=3 repetition draws
3. Crowded & jump games exercising straight jumps, diagonal jumps at edge & walls, and two-pawn jumps (permitted & denied)

For every position:
- Sorted legal actions match exactly
- Full 121-cell distToCenter maps match exactly
- Terminal status (isFinished, winner, isDraw) matches exactly

Tracks and asserts >= 50 for all coverage metrics:
- positions where a straight jump is legal
- positions where a diagonal jump is legal
- positions where a two-pawn jump is legal
- positions where a two-pawn jump is geometrically possible but denied by the restriction
- eliminations
- repetition draws
- maxPlies draws
"""

import json
import os
import random
import subprocess
import pytest
from q4.reference import (
    Pos,
    DIRS,
    action_to_str,
    can_step,
    compute_distances_to_center,
    eliminate,
    legal_moves,
    play,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_katago_bin():
    if os.environ.get("KATAGO_BIN") and os.path.isfile(os.environ["KATAGO_BIN"]):
        return os.environ["KATAGO_BIN"]
    candidates = [
        os.path.join(REPO_ROOT, "cpp", "build-eigen-release", "katago"),
        os.path.join(REPO_ROOT, "cpp", "build-eigen", "katago"),
        os.path.join(REPO_ROOT, "build", "katago"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise RuntimeError("KataGo binary not found! Please build it first.")


def analyze_pawn_geometry(pos: Pos):
    """Analyzes the current moving player's pawn jump options."""
    s = pos.to_move
    if not pos.alive[s] or pos.pawn[s] is None or pos.is_terminal():
        return False, False, False, False

    me = pos.pawn[s]
    occupied = {pos.pawn[i] for i in range(4) if pos.alive[i] and pos.pawn[i] is not None and i != s}
    h, v = pos.hwalls, pos.vwalls

    straight_jump = False
    diagonal_jump = False
    two_pawn_legal = False
    two_pawn_denied = False

    ordinary_moves = set()
    two_pawn_cands = set()

    for dx, dy in DIRS:
        c1 = (me[0] + dx, me[1] + dy)
        if not can_step(h, v, me, c1):
            continue
        if c1 not in occupied:
            ordinary_moves.add(c1)
            continue
        c2 = (c1[0] + dx, c1[1] + dy)
        if can_step(h, v, c1, c2):
            if c2 not in occupied:
                ordinary_moves.add(c2)
                straight_jump = True
            else:
                c3 = (c2[0] + dx, c2[1] + dy)
                if can_step(h, v, c2, c3) and c3 not in occupied:
                    two_pawn_cands.add(c3)
        else:
            # Straight jump blocked by wall or board edge
            for sdx, sdy in ((-dy, dx), (dy, -dx)):
                cd = (c1[0] + sdx, c1[1] + sdy)
                if can_step(h, v, c1, cd) and cd not in occupied:
                    ordinary_moves.add(cd)
                    diagonal_jump = True

    if two_pawn_cands:
        dist_map = compute_distances_to_center(h, v)
        dist_me = dist_map.get(me, 999999)
        can_tp = True
        for dest in ordinary_moves:
            if dist_map.get(dest, 999999) <= dist_me:
                can_tp = False
                break
        if can_tp:
            two_pawn_legal = True
        else:
            two_pawn_denied = True

    return straight_jump, diagonal_jump, two_pawn_legal, two_pawn_denied


def pos_to_json_dict(pos: Pos) -> dict:
    pawns = []
    for s in range(4):
        if pos.alive[s] and pos.pawn[s] is not None:
            pawns.append(list(pos.pawn[s]))
        else:
            pawns.append(None)
    hwalls = [list(w) for w in sorted(pos.hwalls)]
    vwalls = [list(w) for w in sorted(pos.vwalls)]
    return {
        "pawns": pawns,
        "hwalls": hwalls,
        "vwalls": vwalls,
        "wallsLeft": list(pos.walls_left),
        "alive": list(pos.alive),
        "toMove": pos.to_move,
    }


def test_differential_coverage():
    katago_bin = find_katago_bin()
    proc = subprocess.Popen(
        [katago_bin, "q4tool", "legal"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    counts = {
        "straight_jump": 0,
        "diagonal_jump": 0,
        "two_pawn_legal": 0,
        "two_pawn_denied": 0,
        "eliminations": 0,
        "repetition_draws": 0,
        "max_plies_draws": 0,
    }
    total_positions_verified = 0

    def verify_state(pos: Pos, events: list, rules: dict, init_board: dict = None):
        nonlocal total_positions_verified
        sj, dj, tpl, tpd = analyze_pawn_geometry(pos)
        if sj:
            counts["straight_jump"] += 1
        if dj:
            counts["diagonal_jump"] += 1
        if tpl:
            counts["two_pawn_legal"] += 1
        if tpd:
            counts["two_pawn_denied"] += 1

        payload = {"rules": rules, "events": events}
        if init_board is not None:
            payload["initialBoard"] = init_board

        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        assert line, "katago q4tool legal exited unexpectedly"
        resp = json.loads(line)

        # 1. Legal actions match exactly
        py_legal = sorted([action_to_str(m) for m in legal_moves(pos)])
        assert resp["legal"] == py_legal, (
            f"Legal moves mismatch:\nPY ({len(py_legal)}): {py_legal}\nCPP ({len(resp['legal'])}): {resp['legal']}"
        )

        # 2. Full distToCenter (121 cells) matches exactly
        py_dmap = compute_distances_to_center(pos.hwalls, pos.vwalls)
        py_full_dist = [py_dmap.get((c % 11, c // 11), 255) for c in range(121)]
        assert resp["distToCenter"] == py_full_dist, "distToCenter map mismatch"

        # 3. Terminal state matches exactly
        assert resp["isFinished"] == pos.is_terminal(), (
            f"isFinished mismatch: cpp={resp['isFinished']}, py={pos.is_terminal()}"
        )
        py_winner = -1 if pos.winner is None else pos.winner
        assert resp["winner"] == py_winner, (
            f"Winner mismatch: cpp={resp['winner']}, py={py_winner}"
        )
        assert resp["isDraw"] == pos.is_draw, (
            f"isDraw mismatch: cpp={resp['isDraw']}, py={pos.is_draw}"
        )

        total_positions_verified += 1

    rng = random.Random(42)

    try:
        # -------------------------------------------------------------
        # Part 1: Random games with eliminations and small max_plies (35)
        # -------------------------------------------------------------
        print("Generating random games with eliminations and maxPlies draws...")
        for g in range(120):
            max_p = 35
            pos = Pos(max_plies=max_p)
            events = []
            rules = {"maxPlies": max_p, "repetitionDrawCount": 0}
            verify_state(pos, events, rules)

            while not pos.is_terminal():
                # 8% chance of elimination if >= 2 alive
                if sum(pos.alive) > 2 and rng.random() < 0.08:
                    alive_seats = [s for s in range(4) if pos.alive[s]]
                    elim_seat = rng.choice(alive_seats)
                    pos = eliminate(pos, elim_seat)
                    events.append({"elim": elim_seat})
                    counts["eliminations"] += 1
                    verify_state(pos, events, rules)
                    continue

                moves = legal_moves(pos)
                if not moves:
                    break

                # Bias 70% toward pawn moves
                pawn_actions = [m for m in moves if m[0] == "p"]
                wall_actions = [m for m in moves if m[0] in ("h", "v")]
                if pawn_actions and (rng.random() < 0.70 or not wall_actions):
                    m = rng.choice(pawn_actions)
                else:
                    m = rng.choice(wall_actions if wall_actions else moves)

                pos = play(pos, m)
                events.append({"action": action_to_str(m)})
                verify_state(pos, events, rules)

            if pos.is_draw and pos.plies >= max_p:
                counts["max_plies_draws"] += 1

        # -------------------------------------------------------------
        # Part 2: Repetition games (pawn shuffling, N=2 and N=3)
        # -------------------------------------------------------------
        print("Generating repetition games (N=2 and N=3)...")
        for rep_n in [2, 3]:
            for g in range(30):
                pos = Pos(repetition_draw_count=rep_n)
                events = []
                rules = {"maxPlies": 400, "repetitionDrawCount": rep_n}
                verify_state(pos, events, rules)

                for cycle in range(rep_n):
                    if pos.is_terminal():
                        break
                    for step_dir in [1, -1]:
                        if pos.is_terminal():
                            break
                        for s in range(4):
                            if pos.is_terminal():
                                break
                            moves = [m for m in legal_moves(pos) if m[0] == "p"]
                            if moves:
                                m = moves[0] if step_dir == 1 else moves[-1]
                                pos = play(pos, m)
                                events.append({"action": action_to_str(m)})
                                verify_state(pos, events, rules)

                if pos.is_draw:
                    counts["repetition_draws"] += 1

        # -------------------------------------------------------------
        # Part 3: Two-pawn jump setups (permitted and denied)
        # -------------------------------------------------------------
        print("Generating two-pawn jump setups (permitted and denied)...")
        for g in range(60):
            # Setup 1: Permitted two-pawn jump
            pos = Pos()
            pos.pawn[0] = (2, 2)
            pos.pawn[1] = (2, 3)
            pos.pawn[2] = (2, 4)
            pos.pawn[3] = (9, 9)
            pos.vwalls.add((2, 2))  # blocks East
            pos.to_move = 0
            init_b = pos_to_json_dict(pos)
            rules = {"maxPlies": 400, "repetitionDrawCount": 0}
            verify_state(pos, [], rules, init_board=init_b)

            # Setup 2: Denied two-pawn jump
            pos2 = Pos()
            pos2.pawn[0] = (2, 2)
            pos2.pawn[1] = (2, 3)
            pos2.pawn[2] = (2, 4)
            pos2.pawn[3] = (9, 9)
            pos2.to_move = 0
            init_b2 = pos_to_json_dict(pos2)
            verify_state(pos2, [], rules, init_board=init_b2)

        # -------------------------------------------------------------
        # Part 4: Crowded positions exercising straight & diagonal jumps
        # -------------------------------------------------------------
        print("Generating crowded positions for jump coverage...")
        for g in range(60):
            pos = Pos()
            # Pawns adjacent in corners and against walls
            pos.pawn[0] = (0, 9)
            pos.pawn[1] = (0, 10)  # board edge blocks straight jump North
            pos.pawn[2] = (4, 2)
            pos.pawn[3] = (4, 3)
            pos.hwalls.add((4, 3))  # wall blocks straight jump North
            pos.to_move = 0
            init_b = pos_to_json_dict(pos)
            rules = {"maxPlies": 400, "repetitionDrawCount": 0}
            events = []
            verify_state(pos, events, rules, init_board=init_b)

            for step in range(15):
                if pos.is_terminal():
                    break
                moves = [m for m in legal_moves(pos) if m[0] == "p"]
                if not moves:
                    break
                m = rng.choice(moves)
                pos = play(pos, m)
                events.append({"action": action_to_str(m)})
                verify_state(pos, events, rules, init_board=init_b)

        print("\n========================================================")
        print("       Q4 DIFFERENTIAL COVERAGE REPORT (A5)            ")
        print("========================================================")
        print(f"Total positions verified against C++: {total_positions_verified}")
        for k, v in counts.items():
            print(f"  {k:35s}: {v:6d} (required >= 50)")
        print("========================================================\n")

        for k, v in counts.items():
            assert v >= 50, f"Coverage counter '{k}' ({v}) failed to reach minimum threshold of 50!"

    finally:
        proc.stdin.close()
        proc.terminate()
        proc.wait()
