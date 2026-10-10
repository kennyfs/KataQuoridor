"""Statistics and a differential check of the diagonal-jump rule (docs/q4/Q4Rules.md 4.3-4.5): positions from biased
random playouts and from .sgfs files, compared between `katago q4tool legal` (C++) and q4.reference, and how often
the new rule changes the pawn move set against the old generator.

    python -m q4.jump_rule_stats --katago cpp/build-eigen/katago [--random 20000] [--sgfs DIR ...] [--max-sgf-positions N]
"""

import argparse
import glob
import json
import os
import random
import subprocess
from typing import Dict, Iterator, List, Optional

from q4.reference import DIRS, Pos, action_to_str, can_step, eliminate, legal_moves, pawn_moves, play, str_to_action, wall_ok
from q4.record import Q4Record

CENTER = (5, 5)


def pos_to_json(pos: Pos, to_move: int) -> dict:
    return {"pawns": [list(pos.pawn[s]) if pos.alive[s] and pos.pawn[s] is not None else None for s in range(4)],
            "hwalls": [list(w) for w in sorted(pos.hwalls)], "vwalls": [list(w) for w in sorted(pos.vwalls)],
            "wallsLeft": list(pos.walls_left), "alive": list(pos.alive), "toMove": to_move}


def has_line(pos: Pos, seat: int) -> bool:
    """True if seat has an adjacent alive pawn C1 (no wall between) with an alive pawn on C2 behind it (no wall)."""
    me = pos.pawn[seat]
    if not pos.alive[seat] or me is None:
        return False
    occ = {pos.pawn[s] for s in range(4) if s != seat and pos.alive[s] and pos.pawn[s] is not None}
    for dx, dy in DIRS:
        c1 = (me[0] + dx, me[1] + dy)
        c2 = (c1[0] + dx, c1[1] + dy)
        if c1 in occ and can_step(pos.hwalls, pos.vwalls, me, c1) and can_step(pos.hwalls, pos.vwalls, c1, c2) and c2 in occ:
            return True
    return False


def random_crowded_position(rng: random.Random) -> Pos:
    """3 or 4 alive pawns near the centre, often in a line, with random walls."""
    pos = Pos()
    cells: List = []
    n_alive = rng.choice([3, 4, 4])
    dead = rng.sample(range(4), 4 - n_alive)
    x, y = rng.randint(2, 8), rng.randint(2, 8)
    dx, dy = rng.choice(DIRS)
    for _ in range(n_alive):
        if (x, y) == CENTER or (x, y) in cells or not (0 <= x <= 10 and 0 <= y <= 10) or rng.random() < 0.25:
            x, y = rng.randint(2, 8), rng.randint(2, 8)
            while (x, y) == CENTER or (x, y) in cells:
                x, y = rng.randint(2, 8), rng.randint(2, 8)
        cells.append((x, y))
        x, y = x + dx, y + dy
    alive_seats = [s for s in range(4) if s not in dead]
    rng.shuffle(alive_seats)
    for s, c in zip(alive_seats, cells):
        pos.pawn[s] = c
    for s in dead:
        pos.alive[s] = False
        pos.pawn[s] = None
    pos.to_move = alive_seats[0]
    for _ in range(rng.randint(0, 14)):
        ax, ay, orient = rng.randint(0, 9), rng.randint(0, 9), rng.choice("hv")
        if wall_ok(pos, ax, ay, orient):
            (pos.hwalls if orient == "h" else pos.vwalls).add((ax, ay))
    return pos


def random_positions(n: int, seed: int = 1) -> Iterator[Pos]:
    """About n positions: crowded starts and a few random plies after each (pawn moves biased 80%)."""
    rng = random.Random(seed)
    made = 0
    while made < n:
        pos = random_crowded_position(rng)
        for _ in range(rng.randint(1, 8)):
            if pos.is_terminal() or not pos.alive[pos.to_move]:
                break
            yield pos
            made += 1
            moves = legal_moves(pos)
            if not moves:
                break
            pawn = [m for m in moves if m[0] == "p"]
            m = rng.choice(pawn) if pawn and rng.random() < 0.8 else rng.choice(moves)
            pos = play(pos, m)


def sgf_positions(dirs: List[str], max_positions: int, game_stride: int = 1) -> Iterator[Pos]:
    """Positions of the .sgfs under dirs, replayed with the generator of their rule version (old for files without
    jump=diag), returned as plain positions of the new generator."""
    n = 0
    game = -1
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "**", "*.sgfs"), recursive=True)):
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    game += 1
                    if game % game_stride:
                        continue
                    rec = Q4Record.from_sgf_line(line)
                    pos = Pos(max_plies=rec.rules.get("maxPlies", 400),
                              repetition_draw_count=rec.rules.get("repetitionDrawCount", 0),
                              initial_walls=rec.rules.get("initialWalls", [7, 7, 7, 7]))
                    pos.old_jump_rule = not rec.jump_diag
                    for ev in rec.events:
                        if "elim" in ev:
                            pos = eliminate(pos, ev["elim"] - 1)
                        else:
                            pos = play(pos, str_to_action(ev["a"]))
                        if pos.is_terminal():
                            break
                        q = pos.copy()
                        q.old_jump_rule = False
                        yield q
                        n += 1
                        if n >= max_positions:
                            return


class Cpp:
    def __init__(self, katago: str):
        self.proc = subprocess.Popen([katago, "q4tool", "legal"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)

    def legal(self, pos: Pos, seat: int, old: bool = False) -> List[str]:
        d = pos_to_json(pos, seat)
        d["oldJump"] = old
        self.proc.stdin.write(json.dumps(d) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, "q4tool legal exited"
        return [a for a in json.loads(line)["legal"] if a[-1].isdigit()]  # the pawn moves

    def close(self):
        self.proc.stdin.close()
        self.proc.terminate()
        self.proc.wait()


def compare(cpp: Cpp, positions: Iterator[Pos]) -> Dict[str, int]:
    """Compares the pawn moves of C++ and the reference for every alive seat (new rule, and the old generator on a
    sample), and counts the positions with a line for the mover and those where the new rule changes its move set."""
    stats = {"positions": 0, "seat_checks": 0, "mismatches": 0, "mover_line": 0, "mover_changed": 0,
             "any_seat_line": 0, "any_seat_changed": 0}
    for pos in positions:
        stats["positions"] += 1
        any_line = any_changed = False
        for seat in range(4):
            if not pos.alive[seat] or pos.pawn[seat] is None:
                continue
            new = pawn_moves(pos, seat)
            q = pos.copy()
            q.old_jump_rule = True
            old = pawn_moves(q, seat)
            got = cpp.legal(pos, seat)
            want_all = sorted(action_to_str(('p', c)) for c in new)
            stats["seat_checks"] += 1
            if got != want_all:
                stats["mismatches"] += 1
            got_old = cpp.legal(pos, seat, old=True)
            want_old = sorted(action_to_str(('p', c)) for c in old)
            if got_old != want_old:
                stats["mismatches"] += 1
            line = has_line(pos, seat)
            any_line = any_line or line
            changed = new != old
            any_changed = any_changed or changed
            if seat == pos.to_move:
                stats["mover_line"] += line
                stats["mover_changed"] += changed
        stats["any_seat_line"] += any_line
        stats["any_seat_changed"] += any_changed
    return stats


def _with_to_move(pos: Pos, seat: int) -> Pos:
    q = pos.copy()
    q.to_move = seat
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--random", type=int, default=20000)
    ap.add_argument("--sgfs", nargs="*", default=[])
    ap.add_argument("--max-sgf-positions", type=int, default=20000)
    ap.add_argument("--sgf-game-stride", type=int, default=1, help="use every n-th game")
    a = ap.parse_args()
    cpp = Cpp(a.katago)
    try:
        if a.random:
            print("random crowded:", compare(cpp, random_positions(a.random)))
        if a.sgfs:
            print("sgfs:", compare(cpp, sgf_positions(a.sgfs, a.max_sgf_positions, a.sgf_game_stride)))
    finally:
        cpp.close()


if __name__ == "__main__":
    main()
