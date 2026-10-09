"""Independent Python implementation of the Q4 style features (docs/q4/Q4IO.md §9), computed from a game record with
the reference engine (q4.reference). `cpp/q4/q4style.cpp` is checked against it (`q4tool style`)."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from q4.reference import Pos, compute_distances_to_center, eliminate, pawn_moves, play, str_to_action

NUM_DESCRIPTORS = 9
HALFLIVES = (4.0, 16.0)
PER_SEAT = 2 * NUM_DESCRIPTORS + 1      # 19
NUM_FEATURES = 4 * PER_SEAT             # 76
UNREACHABLE = 999999


def seat_distance(pos: Pos, dist: Dict[Tuple[int, int], int], seat: int) -> Optional[int]:
    """Walls-only distance of the pawn of `seat` to the center; None for an eliminated seat or an unreachable pawn."""
    if not pos.alive[seat] or pos.pawn[seat] is None:
        return None
    return dist.get(pos.pawn[seat])


def arrival_estimate(pos: Pos, dist: Dict[Tuple[int, int], int], seat: int) -> int:
    """Q4IO §4 slots 16-19: ((d - 1) * n + order + 1), `order` = position of the seat in the alive turn order from
    pos.to_move (0 = to move); 0 at the center, UNREACHABLE if unreachable or eliminated."""
    d = seat_distance(pos, dist, seat)
    if d is None:
        return UNREACHABLE
    if d == 0:
        return 0
    n_alive = sum(pos.alive)
    order, cur = 0, pos.to_move
    while cur != seat:
        cur = (cur + 1) % 4
        while not pos.alive[cur]:
            cur = (cur + 1) % 4
        order += 1
    return (d - 1) * n_alive + order + 1


def descriptors(before: Pos, after: Pos, action: Tuple[str, Tuple[int, int]]) -> np.ndarray:
    """The 9 per-move descriptors of `action`, played by before.to_move on `before` giving `after`."""
    x = np.zeros(NUM_DESCRIPTORS)
    m = before.to_move
    dist_b = compute_distances_to_center(before.hwalls, before.vwalls)
    dist_a = compute_distances_to_center(after.hwalls, after.vwalls)
    kind, coord = action
    if kind == 'p':
        d0 = seat_distance(before, dist_b, m)
        d1 = seat_distance(after, dist_a, m)
        x[1] = max(-1.0, min(1.0, (d0 - d1) / 2.0))
        dests = pawn_moves(before, m)
        best = min(dist_b[c] for c in dests)
        x[2] = 1.0 if dist_b[coord] == best else 0.0
        return x

    x[0] = 1.0
    lead, lead_est = None, None
    for s in range(4):
        if s == m or not before.alive[s]:
            continue
        est = arrival_estimate(before, dist_b, s)
        if lead is None or est < lead_est:
            lead, lead_est = s, est
    delta = [0] * 4
    max_opp = 0
    for s in range(4):
        d0 = seat_distance(before, dist_b, s)
        d1 = seat_distance(after, dist_a, s)
        if d0 is None or d1 is None:
            continue
        delta[s] = d1 - d0
        x[3 + s] = min(delta[s] / 8.0, 1.0)
        if s != m:
            max_opp = max(max_opp, delta[s])
    x[7] = 1.0 if (lead is not None and max_opp > 0 and delta[lead] == max_opp) else 0.0
    x[8] = 1.0 if max_opp == 0 else 0.0
    return x


class StyleTracker:
    def __init__(self):
        self.mean = np.zeros((4, 2, NUM_DESCRIPTORS))
        self.n = [0, 0, 0, 0]

    def copy(self) -> 'StyleTracker':
        t = StyleTracker()
        t.mean = self.mean.copy()
        t.n = list(self.n)
        return t

    def observe_move(self, before: Pos, after: Pos, action) -> None:
        m = before.to_move
        x = descriptors(before, after, action)
        self.n[m] += 1
        for h, half in enumerate(HALFLIVES):
            rate = max(1.0 / self.n[m], 1.0 - 2.0 ** (-1.0 / half))
            self.mean[m, h] += (x - self.mean[m, h]) * rate

    def observe_elimination(self, seat: int) -> None:
        self.n[seat] = 0
        self.mean[seat] = 0.0

    def encode(self, perspective: int) -> np.ndarray:
        out = np.zeros(NUM_FEATURES)
        for k in range(4):
            s = (perspective + k) % 4
            for h in range(2):
                for d in range(NUM_DESCRIPTORS):
                    col = d
                    if 3 <= d <= 6:
                        col = 3 + ((d - 3) - perspective) % 4
                    out[k * PER_SEAT + h * NUM_DESCRIPTORS + col] = self.mean[s, h, d]
            out[k * PER_SEAT + 2 * NUM_DESCRIPTORS] = min(self.n[s] / 64.0, 1.0)
        return out


def record_features(record: dict) -> Tuple[List[int], np.ndarray]:
    """Replays a record (dict of a .jsonl line); returns the seat to move and the 76 features at the start position
    and after every event."""
    rules = record.get("rules", {})
    pos = Pos(max_plies=rules.get("maxPlies", 400), repetition_draw_count=rules.get("repetitionDrawCount", 0),
              initial_walls=rules.get("initialWalls", [7, 7, 7, 7]))
    tracker = StyleTracker()
    perspectives = [pos.to_move]
    feats = [tracker.encode(pos.to_move)]
    for ev in record["events"]:
        if "elim" in ev:
            seat = ev["elim"] - 1
            pos = eliminate(pos, seat)
            tracker.observe_elimination(seat)
        else:
            action = str_to_action(ev["a"])
            nxt = play(pos, action)
            tracker.observe_move(pos, nxt, action)
            pos = nxt
        perspectives.append(pos.to_move)
        feats.append(tracker.encode(pos.to_move))
    return perspectives, np.array(feats)
