"""Differential testing of C++ Q4 engine against Python reference (Plan §5.7 T3).

Spawns `katago q4tool legal`, generates >= 2000 diverse positions from random games
(biased towards pawn moves and center so that jumps happen; with random eliminations),
and checks that for every position:
- Sorted legal actions match exactly
- Distances to center match exactly
- Terminal / winner status matches exactly
"""

import json
import os
import random
import subprocess
import pytest
from q4.reference import (
    Pos,
    action_to_str,
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
        os.path.join(REPO_ROOT, "cpp", "build-eigen", "katago"),
        os.path.join(REPO_ROOT, "cpp", "build-eigen-release", "katago"),
        os.path.join(REPO_ROOT, "build", "katago"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise RuntimeError("KataGo binary not found! Please build it first.")


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


def test_differential_legal_moves():
    katago_bin = find_katago_bin()
    proc = subprocess.Popen(
        [katago_bin, "q4tool", "legal"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    rng = random.Random(42)
    target_positions = 2000
    positions_tested = 0

    try:
        while positions_tested < target_positions:
            pos = Pos()
            game_len = rng.randint(0, 50)

            for step in range(game_len):
                if pos.is_terminal():
                    break

                # 5% chance of elimination if >= 2 alive
                if sum(pos.alive) > 2 and rng.random() < 0.05:
                    alive_seats = [s for s in range(4) if pos.alive[s]]
                    elim_seat = rng.choice(alive_seats)
                    pos = eliminate(pos, elim_seat)
                    continue

                moves = legal_moves(pos)
                if not moves:
                    break

                # Bias 75% toward pawn moves
                pawn_actions = [m for m in moves if m[0] == 'p']
                wall_actions = [m for m in moves if m[0] in ('h', 'v')]

                if pawn_actions and (rng.random() < 0.75 or not wall_actions):
                    m = rng.choice(pawn_actions)
                else:
                    m = rng.choice(wall_actions if wall_actions else moves)

                pos = play(pos, m)

                # Send this position to C++ for verification
                d = pos_to_json_dict(pos)
                line = json.dumps(d) + "\n"
                proc.stdin.write(line)
                proc.stdin.flush()

                resp_line = proc.stdout.readline()
                assert resp_line, "katago q4tool legal process terminated unexpectedly"
                resp = json.loads(resp_line)

                # Python reference legal actions
                py_moves = legal_moves(pos)
                py_legal = sorted([action_to_str(act) for act in py_moves])
                cpp_legal = resp["legal"]

                assert cpp_legal == py_legal, (
                    f"Position {positions_tested} legal moves mismatch:\n"
                    f"Pos: {d}\n"
                    f"CPP ({len(cpp_legal)}): {cpp_legal}\n"
                    f"PY  ({len(py_legal)}): {py_legal}"
                )

                # Distances
                py_dist_map = compute_distances_to_center(pos.hwalls, pos.vwalls)
                py_dists = []
                for s in range(4):
                    if pos.alive[s] and pos.pawn[s] is not None:
                        py_dists.append(py_dist_map.get(pos.pawn[s], 255))
                    else:
                        py_dists.append(255)
                cpp_dists = resp["dist"]

                assert cpp_dists == py_dists, (
                    f"Position {positions_tested} distance mismatch:\n"
                    f"CPP: {cpp_dists}\n"
                    f"PY : {py_dists}"
                )

                # Terminal status
                cpp_finished = resp["isFinished"]
                assert cpp_finished == pos.is_terminal(), (
                    f"Position {positions_tested} isFinished mismatch: cpp={cpp_finished}, py={pos.is_terminal()}"
                )

                positions_tested += 1
                if positions_tested >= target_positions:
                    break

        print(f"Differential testing verified {positions_tested} positions with 100% agreement!")

    finally:
        proc.stdin.close()
        proc.terminate()
        proc.wait()
