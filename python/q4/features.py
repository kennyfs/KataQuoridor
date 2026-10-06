"""Independent Python reference implementation of Q4 Neural Network Input Features (Q4 I/O v1).

Implements docs/q4/Q4IO.md independently on top of python/q4/reference.py (no C++ involved).
Computes all 27 spatial channels and 28 global features for any Q4 state, across all 8 symmetries.
"""

from collections import deque
from typing import Dict, Set, Tuple
import numpy as np

from q4.reference import (
    Pos, NUM_ANCHORS, DIRS,
    can_step, compute_distances_to_center, wall_ok,
    pawn_moves, play
)

NUM_SPATIAL_CHANNELS = 27
NUM_GLOBAL_FEATURES = 28
NUM_RAW_DIST_CHANNELS = 5  # spatial channels 10..14, stored as uint8 in training rows
DIST_SCALE = 64.0
MAX_WALLS_NORM = 7.0
POS_LEN = 11
NUM_ACTIONS = 321


def dist01(d: int) -> float:
    """Float feature of a raw walls-only distance (255 = unreachable): 1.0 if unreachable, else min(d, 64) / 64."""
    return 1.0 if d == 255 else min(d, 64) / DIST_SCALE


def decode_raw_distances(raw: np.ndarray) -> np.ndarray:
    """uint8 raw distances [..., 11, 11] (training-row storage) -> the float feature planes."""
    raw = np.asarray(raw)
    return np.where(raw == 255, 1.0, np.minimum(raw, 64) / DIST_SCALE).astype(np.float32)


def transform_coords(x: int, y: int, sym: int) -> Tuple[int, int]:
    """Applies symmetry sym (0..7) to centered coordinates (x, y)."""
    if sym == 0: return x, y
    if sym == 1: return -y, x   # Rot90 CCW
    if sym == 2: return -x, -y  # Rot180
    if sym == 3: return y, -x   # Rot270 CCW
    if sym == 4: return -x, y   # Flip X
    if sym == 5: return x, -y   # Flip Y
    if sym == 6: return y, x    # Transpose (main diagonal)
    if sym == 7: return -y, -x  # Flip anti-diagonal
    return x, y


def apply_cell(x: int, y: int, sym: int) -> Tuple[int, int]:
    ox, oy = transform_coords(x - 5, y - 5, sym)
    return ox + 5, oy + 5


def apply_anchor(ax: int, ay: int, is_h: bool, sym: int) -> Tuple[int, int, bool]:
    oax, oay = transform_coords(2 * ax - 9, 2 * ay - 9, sym)
    nax = (oax + 9) // 2
    nay = (oay + 9) // 2
    swaps_axes = (sym in (1, 3, 6, 7))
    new_is_h = not is_h if swaps_axes else is_h
    return nax, nay, new_is_h


def apply_direction(d: int, sym: int) -> int:
    dx, dy = DIRS[d]
    odx, ody = transform_coords(dx, dy, sym)
    for nd, (ndx, ndy) in enumerate(DIRS):
        if ndx == odx and ndy == ody:
            return nd
    return d


def compute_distances_from_cell(hwalls: Set[Tuple[int, int]],
                                vwalls: Set[Tuple[int, int]],
                                start_cell: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
    """Computes shortest-path walls-only BFS distance from start_cell to all cells."""
    dist: Dict[Tuple[int, int], int] = {start_cell: 0}
    q = deque([start_cell])
    while q:
        curr = q.popleft()
        d = dist[curr]
        for dx, dy in DIRS:
            nxt = (curr[0] + dx, curr[1] + dy)
            if nxt not in dist and can_step(hwalls, vwalls, curr, nxt):
                dist[nxt] = d + 1
                q.append(nxt)
    return dist


def raw_distances(pos: Pos) -> np.ndarray:
    """Raw walls-only distances of the 5 distance channels, uint8 [5, 11, 11] ([y][x]); 255 = unreachable.

    [0] = distance to the center, [1 + k] = distance from the pawn of relative seat k (all 255 if eliminated)."""
    to_move = pos.to_move
    raw = np.full((NUM_RAW_DIST_CHANNELS, POS_LEN, POS_LEN), 255, dtype=np.uint8)
    dist_to_center = compute_distances_to_center(pos.hwalls, pos.vwalls)
    for (x, y), d in dist_to_center.items():
        raw[0, y, x] = d
    for k in range(4):
        s = (to_move + k) % 4
        if pos.alive[s] and pos.pawn[s] is not None:
            for (x, y), d in compute_distances_from_cell(pos.hwalls, pos.vwalls, pos.pawn[s]).items():
                raw[1 + k, y, x] = d
    return raw


def extract_features(pos: Pos, symmetry: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts spatial (27, 11, 11) and global (28,) features from pos under symmetry (0..7)."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, POS_LEN, POS_LEN), dtype=np.float32)
    g = np.zeros(NUM_GLOBAL_FEATURES, dtype=np.float32)

    to_move = pos.to_move
    seats = [(to_move + k) % 4 for k in range(4)]  # absolute seat of relative slot k

    spatial[0, :, :] = 1.0  # on-board
    for k, s in enumerate(seats):
        if pos.alive[s] and pos.pawn[s] is not None:
            x, y = pos.pawn[s]
            spatial[1 + k, y, x] = 1.0
    spatial[5, 5, 5] = 1.0  # goal

    # Blocked to N / E / S / W (wall or board edge)
    for y in range(POS_LEN):
        for x in range(POS_LEN):
            for d, (dx, dy) in enumerate(DIRS):
                if not can_step(pos.hwalls, pos.vwalls, (x, y), (x + dx, y + dy)):
                    spatial[6 + d, y, x] = 1.0

    # Distances (channels 10..14) and the cells on some shortest path to the center (15..18)
    raw = raw_distances(pos)
    for ch in range(NUM_RAW_DIST_CHANNELS):
        spatial[10 + ch] = decode_raw_distances(raw[ch])
    dist_to_center = compute_distances_to_center(pos.hwalls, pos.vwalls)
    for k, s in enumerate(seats):
        if not pos.alive[s] or pos.pawn[s] is None:
            continue
        p_center = dist_to_center.get(pos.pawn[s], 255)
        if p_center == 255:
            continue
        for y in range(POS_LEN):
            for x in range(POS_LEN):
                df, dc = int(raw[1 + k, y, x]), int(raw[0, y, x])
                if df != 255 and dc != 255 and df + dc == p_center:
                    spatial[15 + k, y, x] = 1.0

    for ax, ay in pos.vwalls:
        spatial[19, ay, ax] = 1.0
    for ax, ay in pos.hwalls:
        spatial[20, ay, ax] = 1.0
    spatial[21, :NUM_ANCHORS, :NUM_ANCHORS] = 1.0
    for ay in range(NUM_ANCHORS):
        for ax in range(NUM_ANCHORS):
            if wall_ok(pos, ax, ay, 'v'):
                spatial[22, ay, ax] = 1.0
            if wall_ok(pos, ax, ay, 'h'):
                spatial[23, ay, ax] = 1.0

    # Legal pawn destinations, and (rule on) the ones that repeat a position / end the game by repetition
    rep_on = pos.repetition_draw_count >= 2
    for mx, my in pawn_moves(pos, to_move):
        spatial[24, my, mx] = 1.0
        if rep_on:
            occ = pos.position_history.count(play(pos, ('p', (mx, my))).position_key())
            if occ > 0:
                spatial[25, my, mx] = 1.0
            if occ + 1 >= pos.repetition_draw_count:
                spatial[26, my, mx] = 1.0

    # Global features
    n_alive = sum(pos.alive)
    order, nxt = {}, 0
    for s in seats:
        if pos.alive[s]:
            order[s] = nxt
            nxt += 1
    leader, leader_estimate = -1, 0.0
    for k, s in enumerate(seats):
        g[0 + k] = pos.walls_left[s] / MAX_WALLS_NORM
        g[4 + k] = 1.0 if pos.walls_left[s] > 0 else 0.0
        g[8 + k] = 1.0 if pos.alive[s] else 0.0
        if not pos.alive[s] or pos.pawn[s] is None:
            continue
        d = dist_to_center.get(pos.pawn[s], 255)
        g[12 + k] = dist01(d)
        if d == 255:
            g[16 + k] = 1.0
        else:
            estimate = ((d - 1) * n_alive + order[s] + 1) / 64.0
            g[16 + k] = min(1.0, estimate)
            if leader < 0 or estimate < leader_estimate:
                leader, leader_estimate = k, estimate
    if leader >= 0:
        g[20 + leader] = 1.0
    g[24] = n_alive / 4.0
    g[25] = max(0, pos.max_plies - pos.plies) / 400.0
    g[26] = 1.0 if rep_on else 0.0
    if rep_on:
        count = pos.position_history.count(pos.position_key())
        n = pos.repetition_draw_count
        g[27] = 1.0 if n == 2 else min(1.0, max(0, count - 1) / (n - 2))

    if symmetry != 0:
        spatial = apply_symmetry_to_spatial(spatial, symmetry)
    return spatial, g


def apply_symmetry_to_spatial(spatial: np.ndarray, symmetry: int) -> np.ndarray:
    """Applies symmetry (0..7) to a spatial feature tensor [27, 11, 11]."""
    out = np.zeros_like(spatial)
    cell_channels = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 25, 26]
    for y in range(POS_LEN):
        for x in range(POS_LEN):
            nx, ny = apply_cell(x, y, symmetry)
            for ch in cell_channels:
                out[ch, ny, nx] = spatial[ch, y, x]
            for d in range(4):
                out[6 + apply_direction(d, symmetry), ny, nx] = spatial[6 + d, y, x]
    for ay in range(NUM_ANCHORS):
        for ax in range(NUM_ANCHORS):
            out[21, ay, ax] = spatial[21, ay, ax]
            for orient, is_h in ((0, False), (1, True)):
                nax, nay, new_is_h = apply_anchor(ax, ay, is_h, symmetry)
                out[19 + (1 if new_is_h else 0), nay, nax] = spatial[19 + orient, ay, ax]
                out[22 + (1 if new_is_h else 0), nay, nax] = spatial[22 + orient, ay, ax]
    return out


def map_policy_to_game(raw_policy: np.ndarray, sym: int) -> np.ndarray:
    """Maps raw policy logits (2, 3, 11, 11) or (6, 11, 11) back to game action logits (2, 321), undoing symmetry sym."""
    if raw_policy.shape == (6, POS_LEN, POS_LEN):
        raw_policy = raw_policy.reshape(2, 3, POS_LEN, POS_LEN)
    out = np.zeros((2, NUM_ACTIONS), dtype=np.float32)

    for ch in range(2):
        for act in range(NUM_ACTIONS):
            if act < 121:
                # Pawn move
                x, y = act % POS_LEN, act // POS_LEN
                tx, ty = apply_cell(x, y, sym)
                out[ch, act] = raw_policy[ch, 0, ty, tx]
            elif act < 221:
                # Vertical wall
                anchor = act - 121
                ax, ay = anchor % NUM_ANCHORS, anchor // NUM_ANCHORS
                tax, tay, is_h = apply_anchor(ax, ay, False, sym)
                plane = 2 if is_h else 1
                out[ch, act] = raw_policy[ch, plane, tay, tax]
            else:
                # Horizontal wall
                anchor = act - 221
                ax, ay = anchor % NUM_ANCHORS, anchor // NUM_ANCHORS
                tax, tay, is_h = apply_anchor(ax, ay, True, sym)
                plane = 2 if is_h else 1
                out[ch, act] = raw_policy[ch, plane, tay, tax]

    return out


def decode_trajectory(raw_trajectory: np.ndarray, sym: int) -> np.ndarray:
    """Decodes raw trajectory (1, 11, 11) or (11, 11) logits back to game cell probabilities (11, 11) via sigmoid and inverse symmetry."""
    raw = raw_trajectory.reshape(POS_LEN, POS_LEN)
    out = np.zeros((POS_LEN, POS_LEN), dtype=np.float32)
    for y in range(POS_LEN):
        for x in range(POS_LEN):
            tx, ty = apply_cell(x, y, sym)
            logit = raw[ty, tx]
            out[y, x] = 1.0 / (1.0 + np.exp(-logit))
    return out


def decode_shortterm_value_error(raw_misc_5: float, multiplier: float = 0.25) -> float:
    """Decodes short-term value error from misc slot 5 (Plan §7 item 4)."""
    x = 0.5 * float(raw_misc_5)
    s = x if x > 40.0 else np.log1p(np.exp(x))
    return float(np.sqrt(s * s * multiplier))


def unapply_raw_policy(raw_policy: np.ndarray, sym: int) -> np.ndarray:
    """Turns raw policy logits [6, 11, 11] under symmetry sym into symmetry 0 orientation."""
    if sym == 0:
        return raw_policy.copy()
    was_4d = False
    if raw_policy.shape == (2, 3, POS_LEN, POS_LEN):
        was_4d = True
        src = raw_policy
    else:
        src = raw_policy.reshape(2, 3, POS_LEN, POS_LEN)
    dst = np.zeros_like(src)
    for ch in range(2):
        # Plane 0: pawn moves
        for y in range(POS_LEN):
            for x in range(POS_LEN):
                tx, ty = apply_cell(x, y, sym)
                dst[ch, 0, y, x] = src[ch, 0, ty, tx]
        # Planes 1 & 2: V and H walls
        for ay in range(NUM_ANCHORS):
            for ax in range(NUM_ANCHORS):
                tax_v, tay_v, is_h_v = apply_anchor(ax, ay, False, sym)
                dst[ch, 1, ay, ax] = src[ch, 2 if is_h_v else 1, tay_v, tax_v]
                tax_h, tay_h, is_h_h = apply_anchor(ax, ay, True, sym)
                dst[ch, 2, ay, ax] = src[ch, 2 if is_h_h else 1, tay_h, tax_h]
    return dst if was_4d else dst.reshape(6, POS_LEN, POS_LEN)


def unapply_raw_trajectory(raw_trajectory: np.ndarray, sym: int) -> np.ndarray:
    """Turns raw trajectory logits [11, 11] under symmetry sym into symmetry 0 orientation."""
    if sym == 0:
        return raw_trajectory.copy()
    src = raw_trajectory.reshape(POS_LEN, POS_LEN)
    dst = np.zeros_like(src)
    for y in range(POS_LEN):
        for x in range(POS_LEN):
            tx, ty = apply_cell(x, y, sym)
            dst[y, x] = src[ty, tx]
    return dst.reshape(raw_trajectory.shape)


