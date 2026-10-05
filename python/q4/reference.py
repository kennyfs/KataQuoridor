"""Independent Quoridor Four at a Table (Q4, 11x11, 4 players) reference implementation.

Shares no code or representation with the C++ engine.
Uses plain (x, y) coordinates and wall sets, with full BFS for distance and wall legality.
Follows GameRules.md and docs/q4/Q4Rules.md strictly.
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

BOARD_SIZE = 11
NUM_ANCHORS = 10
CENTER = (5, 5)
NUM_SEATS = 4
DEFAULT_WALLS = 7
MAX_PLIES_DEFAULT = 400

START_POSITIONS = (
    (5, 0),   # Seat 0: South (f1)
    (0, 5),   # Seat 1: West (a6)
    (5, 10),  # Seat 2: North (f11)
    (10, 5),  # Seat 3: East (k6)
)

DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # N, E, S, W


def on_board(x: int, y: int) -> bool:
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def cell_to_str(cell: Tuple[int, int]) -> str:
    x, y = cell
    return f"{chr(ord('a') + x)}{y + 1}"


def str_to_cell(s: str) -> Tuple[int, int]:
    if len(s) < 2 or len(s) > 3:
        raise ValueError(f"Invalid cell string: {s}")
    col = s[0].lower()
    x = ord(col) - ord('a')
    y = int(s[1:]) - 1
    if not on_board(x, y):
        raise ValueError(f"Cell coordinates out of range: {s}")
    return (x, y)


def wall_to_str(anchor: Tuple[int, int], orient: str) -> str:
    ax, ay = anchor
    return f"{chr(ord('a') + ax)}{ay + 1}{orient}"


def str_to_wall(s: str) -> Tuple[Tuple[int, int], str]:
    if len(s) < 3 or len(s) > 4:
        raise ValueError(f"Invalid wall string: {s}")
    orient = s[-1].lower()
    if orient not in ('h', 'v'):
        raise ValueError(f"Invalid wall orientation: {s}")
    col = s[0].lower()
    ax = ord(col) - ord('a')
    ay = int(s[1:-1]) - 1
    if not (0 <= ax < NUM_ANCHORS and 0 <= ay < NUM_ANCHORS):
        raise ValueError(f"Wall anchor out of range: {s}")
    return ((ax, ay), orient)


def action_to_str(action: Tuple[str, Tuple[int, int]]) -> str:
    kind, coord = action
    if kind == 'p':
        return cell_to_str(coord)
    return wall_to_str(coord, kind)


def str_to_action(s: str) -> Tuple[str, Tuple[int, int]]:
    s = s.strip()
    if len(s) >= 3 and s[-1].lower() in ('h', 'v'):
        anchor, orient = str_to_wall(s)
        return (orient, anchor)
    return ('p', str_to_cell(s))


def action_to_index(action: Tuple[str, Tuple[int, int]]) -> int:
    kind, (x, y) = action
    if kind == 'p':
        return y * BOARD_SIZE + x
    anchor_idx = y * NUM_ANCHORS + x
    if kind == 'v':
        return 121 + anchor_idx
    if kind == 'h':
        return 221 + anchor_idx
    raise ValueError(f"Unknown action kind: {kind}")


def index_to_action(idx: int) -> Tuple[str, Tuple[int, int]]:
    if 0 <= idx <= 120:
        x = idx % BOARD_SIZE
        y = idx // BOARD_SIZE
        return ('p', (x, y))
    if 121 <= idx <= 220:
        a = idx - 121
        return ('v', (a % NUM_ANCHORS, a // NUM_ANCHORS))
    if 221 <= idx <= 320:
        a = idx - 221
        return ('h', (a % NUM_ANCHORS, a // NUM_ANCHORS))
    raise ValueError(f"Action index out of range: {idx}")


def is_edge_blocked(hwalls: Set[Tuple[int, int]], vwalls: Set[Tuple[int, int]],
                    c0: Tuple[int, int], c1: Tuple[int, int]) -> bool:
    """True if passage between orthogonally adjacent cells c0 and c1 is blocked by a wall."""
    x0, y0 = c0
    x1, y1 = c1
    if x0 == x1:
        # Vertical movement (crosses horizontal line)
        y = min(y0, y1)
        return (x0, y) in hwalls or (x0 - 1, y) in hwalls
    # Horizontal movement (crosses vertical line)
    x = min(x0, x1)
    return (x, y0) in vwalls or (x, y0 - 1) in vwalls


def can_step(hwalls: Set[Tuple[int, int]], vwalls: Set[Tuple[int, int]],
             c0: Tuple[int, int], c1: Tuple[int, int]) -> bool:
    """True if a pawn can step directly from c0 to adjacent c1 (boundary and walls only)."""
    return on_board(c1[0], c1[1]) and not is_edge_blocked(hwalls, vwalls, c0, c1)


def compute_distances_to_center(hwalls: Set[Tuple[int, int]],
                                vwalls: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    """Computes shortest-path walls-only BFS distance from each cell to CENTER."""
    dist: Dict[Tuple[int, int], int] = {}
    dist[CENTER] = 0
    q = deque([CENTER])
    while q:
        curr = q.popleft()
        d = dist[curr]
        for dx, dy in DIRS:
            nxt = (curr[0] + dx, curr[1] + dy)
            if nxt not in dist and can_step(hwalls, vwalls, curr, nxt):
                dist[nxt] = d + 1
                q.append(nxt)
    return dist


class Pos:
    def __init__(self,
                 max_plies: int = MAX_PLIES_DEFAULT,
                 repetition_draw_count: int = 0,
                 initial_walls: Optional[List[int]] = None):
        self.pawn: List[Optional[Tuple[int, int]]] = list(START_POSITIONS)
        if initial_walls is None:
            self.walls_left = [DEFAULT_WALLS] * NUM_SEATS
        else:
            self.walls_left = list(initial_walls)
        self.hwalls: Set[Tuple[int, int]] = set()
        self.vwalls: Set[Tuple[int, int]] = set()
        self.alive: List[bool] = [True] * NUM_SEATS
        self.to_move: int = 0
        self.plies: int = 0
        self.max_plies: int = max_plies
        self.repetition_draw_count: int = repetition_draw_count
        self.winner: Optional[int] = None  # 0..3 for seat win
        self.is_draw: bool = False
        # Repetition history: list of position keys since last wall or elimination
        self.position_history: List[Tuple] = []
        self._record_position()

    def copy(self) -> 'Pos':
        p = Pos(self.max_plies, self.repetition_draw_count)
        p.pawn = list(self.pawn)
        p.walls_left = list(self.walls_left)
        p.hwalls = set(self.hwalls)
        p.vwalls = set(self.vwalls)
        p.alive = list(self.alive)
        p.to_move = self.to_move
        p.plies = self.plies
        p.winner = self.winner
        p.is_draw = self.is_draw
        p.position_history = list(self.position_history)
        return p

    def is_terminal(self) -> bool:
        return self.winner is not None or self.is_draw

    def result_str(self) -> str:
        if self.winner is not None:
            return f"{self.winner + 1}+"
        if self.is_draw:
            return "Draw"
        return "none"

    def position_key(self) -> Tuple:
        return (
            tuple(self.pawn),
            frozenset(self.hwalls),
            frozenset(self.vwalls),
            tuple(self.walls_left),
            tuple(self.alive),
            self.to_move,
        )

    def _record_position(self) -> None:
        if self.repetition_draw_count >= 2:
            key = self.position_key()
            self.position_history.append(key)
            if self.position_history.count(key) >= self.repetition_draw_count:
                self.is_draw = True


def next_alive_seat(curr_seat: int, alive: List[bool]) -> int:
    for step in range(1, NUM_SEATS + 1):
        s = (curr_seat + step) % NUM_SEATS
        if alive[s]:
            return s
    return curr_seat


def wall_conflicts(hwalls: Set[Tuple[int, int]], vwalls: Set[Tuple[int, int]],
                   ax: int, ay: int, orient: str) -> bool:
    if not (0 <= ax < NUM_ANCHORS and 0 <= ay < NUM_ANCHORS):
        return True
    if orient == 'h':
        if (ax, ay) in hwalls or (ax - 1, ay) in hwalls or (ax + 1, ay) in hwalls:
            return True
        if (ax, ay) in vwalls:
            return True
    elif orient == 'v':
        if (ax, ay) in vwalls or (ax, ay - 1) in vwalls or (ax, ay + 1) in vwalls:
            return True
        if (ax, ay) in hwalls:
            return True
    else:
        return True
    return False


def wall_ok(pos: Pos, ax: int, ay: int, orient: str) -> bool:
    """Checks whether placing a wall at (ax, ay) with orient is legal."""
    if wall_conflicts(pos.hwalls, pos.vwalls, ax, ay, orient):
        return False

    # Tentative wall addition
    if orient == 'h':
        new_h = pos.hwalls | {(ax, ay)}
        new_v = pos.vwalls
    else:
        new_h = pos.hwalls
        new_v = pos.vwalls | {(ax, ay)}

    # No-full-block: full BFS from center must reach all alive players
    # Using compute_distances_to_center
    dist = compute_distances_to_center(new_h, new_v)
    for s in range(NUM_SEATS):
        if pos.alive[s]:
            p = pos.pawn[s]
            if p is not None and p not in dist:
                return False
    return True


def pawn_moves(pos: Pos, seat: Optional[int] = None) -> List[Tuple[int, int]]:
    """Returns sorted list of legal pawn destination cells for the given seat (default to_move)."""
    if seat is None:
        seat = pos.to_move
    if not pos.alive[seat]:
        return []

    me = pos.pawn[seat]
    if me is None:
        return []

    # Occupied cells by alive pawns
    occupied = {pos.pawn[s] for s in range(NUM_SEATS) if pos.alive[s] and pos.pawn[s] is not None and s != seat}

    h, v = pos.hwalls, pos.vwalls
    ordinary_moves: Set[Tuple[int, int]] = set()
    two_pawn_candidates: Set[Tuple[int, int]] = set()

    for dx, dy in DIRS:
        c1 = (me[0] + dx, me[1] + dy)
        if not can_step(h, v, me, c1):
            continue

        if c1 not in occupied:
            ordinary_moves.add(c1)
            continue

        # c1 is occupied by an alive pawn.
        c2 = (c1[0] + dx, c1[1] + dy)
        if can_step(h, v, c1, c2):
            # Straight step from c1 to c2 is unobstructed by wall or board edge
            if c2 not in occupied:
                # Direct straight jump
                ordinary_moves.add(c2)
                # Diagonal jump is forbidden when straight jump is clear
            else:
                # c2 is occupied by another pawn!
                # Diagonal jump is forbidden (blocked by pawn, not wall/edge).
                # Check two-pawn jump candidate C3
                c3 = (c2[0] + dx, c2[1] + dy)
                if can_step(h, v, c2, c3) and c3 not in occupied:
                    two_pawn_candidates.add(c3)
        else:
            # Straight step from c1 to c2 is blocked by wall or board edge.
            # Diagonal jumps are permitted!
            for sdx, sdy in ((-dy, dx), (dy, -dx)):
                c_diag = (c1[0] + sdx, c1[1] + sdy)
                if can_step(h, v, c1, c_diag) and c_diag not in occupied:
                    ordinary_moves.add(c_diag)

    final_moves = set(ordinary_moves)

    # Evaluate two-pawn jump restriction
    if two_pawn_candidates:
        dist_to_center = compute_distances_to_center(h, v)
        dist_me = dist_to_center.get(me, 999999)
        # Allowed only if EVERY ordinary legal move strictly increases distance to center
        # (or if ordinary_moves is empty)
        can_two_pawn = True
        for dest in ordinary_moves:
            if dist_to_center.get(dest, 999999) <= dist_me:
                can_two_pawn = False
                break
        if can_two_pawn:
            final_moves.update(two_pawn_candidates)

    return sorted(final_moves)


def legal_moves(pos: Pos, seat: Optional[int] = None) -> List[Tuple[str, Tuple[int, int]]]:
    """Returns all legal actions for the given seat (default to_move)."""
    if pos.is_terminal():
        return []
    if seat is None:
        seat = pos.to_move
    if not pos.alive[seat]:
        return []

    moves: List[Tuple[str, Tuple[int, int]]] = [('p', cell) for cell in pawn_moves(pos, seat)]

    if pos.walls_left[seat] > 0:
        for ay in range(NUM_ANCHORS):
            for ax in range(NUM_ANCHORS):
                if wall_ok(pos, ax, ay, 'v'):
                    moves.append(('v', (ax, ay)))
                if wall_ok(pos, ax, ay, 'h'):
                    moves.append(('h', (ax, ay)))

    return moves


def play(pos: Pos, move: Tuple[str, Tuple[int, int]]) -> Pos:
    """Plays an action and returns the new board state."""
    if pos.is_terminal():
        raise ValueError("Cannot play in terminal state")

    p = pos.copy()
    kind, coord = move
    seat = p.to_move

    if kind == 'p':
        p.pawn[seat] = coord
        if coord == CENTER:
            p.winner = seat
            return p
    else:
        if p.walls_left[seat] <= 0:
            raise ValueError(f"Seat {seat} has no walls left")
        if kind == 'h':
            p.hwalls.add(coord)
        elif kind == 'v':
            p.vwalls.add(coord)
        else:
            raise ValueError(f"Invalid move kind {kind}")
        p.walls_left[seat] -= 1
        # Reset position repetition history after a wall placement
        p.position_history = []

    p.plies += 1
    if p.plies >= p.max_plies:
        p.is_draw = True
        return p

    p.to_move = next_alive_seat(seat, p.alive)
    p._record_position()
    return p


def eliminate(pos: Pos, seat: int) -> Pos:
    """Applies an external elimination event for seat, returning new state."""
    if not (0 <= seat < NUM_SEATS):
        raise ValueError(f"Invalid seat: {seat}")
    if not pos.alive[seat]:
        return pos  # Already eliminated

    p = pos.copy()
    p.alive[seat] = False
    p.pawn[seat] = None
    p.walls_left[seat] = 0
    # Reset repetition history after an elimination
    p.position_history = []

    # Check remaining alive seats
    alive_seats = [s for s in range(NUM_SEATS) if p.alive[s]]
    if len(alive_seats) <= 1:
        if len(alive_seats) == 1:
            p.winner = alive_seats[0]
        else:
            p.is_draw = True
        return p

    # If eliminated seat was to move, advance turn
    if p.to_move == seat:
        p.to_move = next_alive_seat(seat, p.alive)

    p._record_position()
    return p


def perft(pos: Pos, depth: int) -> int:
    """Standard perft enumeration for Q4."""
    if pos.is_terminal() or depth == 0:
        return 1
    moves = legal_moves(pos)
    if depth == 1:
        return len(moves)
    return sum(perft(play(pos, m), depth - 1) for m in moves)
