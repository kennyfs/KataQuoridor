"""Independent Quoridor (Duel, 9x9, 10 walls each) move generator and perft reference.

This deliberately shares no code or representation with the C++ engine: it works on plain 9x9 cell
coordinates and a set of wall anchors, and checks wall legality with a full BFS for both players on
every candidate wall (no cached paths). Its perft counts are pinned in `cpp/tests/testquoridorperft.cpp`
and re-checked here at shallow depth.

Conventions (matching GameRules.md and cpp/game/board.h):
  - Cells are (c, r), 0 <= c, r <= 8, printed as "a1".."i9" (column letter, row digit).
  - Black starts at e9 and moves first, goal row r == 0. White starts at e1, goal row r == 8.
  - Wall anchors are (c, r), 0 <= c, r <= 7, printed as "e2h" / "e2v".
    H wall (c, r) blocks (c, r)-(c, r+1) and (c+1, r)-(c+1, r+1).
    V wall (c, r) blocks (c, r)-(c+1, r) and (c, r+1)-(c+1, r+1).
  - A position where a pawn stands on its goal row is terminal and is not expanded by perft.

Regenerate every pinned number (slow, a few minutes):
    python python/tests/test_quoridor_perft_reference.py
"""

import sys
from collections import deque

N = 9
NUM_WALLS = 10
BLACK, WHITE = 0, 1
GOAL_ROW = (0, N - 1)


class Pos:
    def __init__(self):
        self.pawn = [(4, 8), (4, 0)]
        self.walls_left = [NUM_WALLS, NUM_WALLS]
        self.hwalls = set()
        self.vwalls = set()
        self.to_move = BLACK

    def copy(self):
        p = Pos()
        p.pawn = list(self.pawn)
        p.walls_left = list(self.walls_left)
        p.hwalls = set(self.hwalls)
        p.vwalls = set(self.vwalls)
        p.to_move = self.to_move
        return p

    def is_terminal(self):
        return self.pawn[BLACK][1] == GOAL_ROW[BLACK] or self.pawn[WHITE][1] == GOAL_ROW[WHITE]


def blocked(hwalls, vwalls, a, b):
    """True if the edge between orthogonally adjacent cells a and b is blocked by a wall."""
    (c0, r0), (c1, r1) = a, b
    if c0 == c1:
        r = min(r0, r1)
        return (c0, r) in hwalls or (c0 - 1, r) in hwalls
    c = min(c0, c1)
    return (c, r0) in vwalls or (c, r0 - 1) in vwalls


def on_board(cell):
    return 0 <= cell[0] < N and 0 <= cell[1] < N


DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def can_step(hwalls, vwalls, a, b):
    return on_board(b) and not blocked(hwalls, vwalls, a, b)


def reaches_goal(hwalls, vwalls, start, goal_row):
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur[1] == goal_row:
            return True
        for dc, dr in DIRS:
            nxt = (cur[0] + dc, cur[1] + dr)
            if nxt not in seen and can_step(hwalls, vwalls, cur, nxt):
                seen.add(nxt)
                q.append(nxt)
    return False


def pawn_moves(pos):
    me = pos.pawn[pos.to_move]
    opp = pos.pawn[1 - pos.to_move]
    h, v = pos.hwalls, pos.vwalls
    out = set()
    for dc, dr in DIRS:
        adj = (me[0] + dc, me[1] + dr)
        if not can_step(h, v, me, adj):
            continue
        if adj != opp:
            out.add(adj)
            continue
        behind = (adj[0] + dc, adj[1] + dr)
        if can_step(h, v, adj, behind):
            out.add(behind)
            continue
        for sc, sr in ((dr, dc), (-dr, -dc)):
            side = (adj[0] + sc, adj[1] + sr)
            if can_step(h, v, adj, side) and side != me:
                out.add(side)
    return sorted(out)


def wall_ok(pos, c, r, horizontal):
    h, v = pos.hwalls, pos.vwalls
    if (c, r) in h or (c, r) in v:
        return False
    if horizontal:
        if (c - 1, r) in h or (c + 1, r) in h:
            return False
        h = h | {(c, r)}
    else:
        if (c, r - 1) in v or (c, r + 1) in v:
            return False
        v = v | {(c, r)}
    return (reaches_goal(h, v, pos.pawn[BLACK], GOAL_ROW[BLACK]) and
            reaches_goal(h, v, pos.pawn[WHITE], GOAL_ROW[WHITE]))


def legal_moves(pos):
    """Moves as ('p', (c, r)) or ('h'/'v', (c, r))."""
    moves = [('p', cell) for cell in pawn_moves(pos)]
    if pos.walls_left[pos.to_move] > 0:
        for c in range(N - 1):
            for r in range(N - 1):
                if wall_ok(pos, c, r, True):
                    moves.append(('h', (c, r)))
                if wall_ok(pos, c, r, False):
                    moves.append(('v', (c, r)))
    return moves


def play(pos, move):
    p = pos.copy()
    kind, cell = move
    if kind == 'p':
        p.pawn[p.to_move] = cell
    else:
        (p.hwalls if kind == 'h' else p.vwalls).add(cell)
        p.walls_left[p.to_move] -= 1
    p.to_move = 1 - p.to_move
    return p


def parse_move(s):
    c = ord(s[0]) - ord('a')
    r = int(s[1]) - 1
    if len(s) == 2:
        return ('p', (c, r))
    assert len(s) == 3 and s[2] in 'hv'
    return (s[2], (c, r))


def position_from(moves_str):
    pos = Pos()
    for s in moves_str.split():
        m = parse_move(s)
        assert m in legal_moves(pos), "illegal move in setup: " + s
        assert not pos.is_terminal()
        pos = play(pos, m)
    return pos


def perft(pos, depth):
    if pos.is_terminal():
        return 1
    if depth == 0:
        return 1
    moves = legal_moves(pos)
    assert len(moves) > 0, "non-terminal position without legal moves"
    if depth == 1:
        return len(moves)
    return sum(perft(play(pos, m), depth - 1) for m in moves)


# Keep in sync with cpp/tests/testquoridorperft.cpp.
PERFT_POSITIONS = [
    # name, setup moves, [perft(1), perft(2), perft(3)]
    ("start", "", [131, 16677, 2062264]),
    ("face-to-face, white can jump straight",
     "e8 e2 e7 e3 e6 e4 e5", [132, 16938, 2111842]),
    ("wall behind white, black gets diagonal jumps",
     "e8 e2 e7 e3 e6 e4 e5 e3h", [129, 15922, 1936376]),
    ("white at e8 under black at e9 on the edge, diagonal jumps win",
     "a2h e2 c2h e3 a4h e4 c4h e5 a6h e6 c6h e7 a8h e8 c8h", [109, 11083, 1129546]),
    ("black out of walls, white has all ten",
     "a2h e2 c2h e1 a4h e2 c4h e1 a6h e2 c6h e1 a8h e2 c8h e1 g2h e2 g4h e1", [3, 296, 878]),
]


def test_shallow_perft_matches_pinned():
    for name, setup, counts in PERFT_POSITIONS:
        pos = position_from(setup)
        for depth, expected in enumerate(counts[:2], start=1):
            assert perft(pos, depth) == expected, (name, depth)


def test_basic_rules():
    pos = Pos()
    assert pawn_moves(pos) == [(3, 8), (4, 7), (5, 8)]
    # Same-orientation neighbours and crossing walls conflict; perpendicular touching is fine.
    p = play(pos, parse_move("a1h"))
    p.to_move = BLACK
    assert not wall_ok(p, 1, 0, True)
    assert not wall_ok(p, 0, 0, False)
    assert wall_ok(p, 1, 0, False)
    assert wall_ok(p, 2, 0, True)


def _perft_after(args):
    setup, move, depth = args
    return perft(play(position_from(setup), move), depth)


if __name__ == "__main__":
    import multiprocessing
    depth_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with multiprocessing.Pool() as pool:
        for name, setup, _ in PERFT_POSITIONS:
            pos = position_from(setup)
            counts = []
            for d in range(1, depth_limit + 1):
                moves = legal_moves(pos)
                counts.append(sum(pool.map(_perft_after, [(setup, m, d - 1) for m in moves])))
            print(name, counts, flush=True)
