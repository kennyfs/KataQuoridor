"""Tests of the Q4 style features (Round 7 Part B, docs/q4/Q4IO.md §9).

S1: scripted histories with hand-derived values (Python and C++), bots that only wall the leader / a fixed seat / never
wall, invariance under the 8 board symmetries, C++ (`q4tool style`) == python/q4/style.py on 2,000 random
population-like games, and (C++ test group q4board) undo restores the tracker.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PYTHON_DIR)

from q4.features import apply_anchor, apply_cell  # noqa: E402
from q4.reference import START_POSITIONS, action_to_str, str_to_action  # noqa: E402
from q4.style import NUM_FEATURES, PER_SEAT, StyleTracker, record_features  # noqa: E402
from tests.q4_testutil import find_katago  # noqa: E402


def katago():
    k = find_katago("eigen")
    assert k is not None, "katago binary not found"
    return k


def cpp_style(lines, stdin_is_records=True):
    """`q4tool style` on JSON lines: [(perspectives, features[P, 76])] per game."""
    proc = subprocess.run([katago(), "q4tool", "style"], input="\n".join(lines) + "\n", capture_output=True,
                          text=True, check=True)
    out = []
    for line in proc.stdout.splitlines():
        d = json.loads(line)
        out.append((d["perspectives"], np.array(d["features"]).reshape(d["numPositions"], NUM_FEATURES)))
    return out


def popgames(n, seed, extra=()):
    proc = subprocess.run([katago(), "q4tool", "popgames", "-n", str(n), "-seed", str(seed), *extra],
                          capture_output=True, text=True, check=True)
    return proc.stdout.strip().splitlines()


def nonterminal(rec, persp, feats):
    """Drops the position after the last event of a finished game: nobody moves there, so the perspective (the seat
    'to move') is a convention that the two engines do not share."""
    if rec.get("result", "none") != "none":
        return persp[:-1], feats[:-1]
    return persp, feats


def record_of(events, **rules):
    return {"rules": {"maxPlies": 400, "repetitionDrawCount": 0, **rules}, "events": events}


def seat_block(features, k):
    return features[k * PER_SEAT:(k + 1) * PER_SEAT]


def both(record):
    """Python and C++ features of a record (checked equal), as (perspectives, [P, 76])."""
    persp, py = record_features(record)
    (cpersp, cpp), = cpp_style([json.dumps(record)])
    assert cpersp == persp
    assert np.max(np.abs(cpp - py)) < 1e-6, np.max(np.abs(cpp - py))
    return persp, py


def test_s1_scripted_values():
    # f2 (seat 0: progress 0.5, greedy step), e2h by seat 1 (hits the leader seat 0: its distance 4 -> 6, victim
    # column 6 / 8... see below), f10 (seat 2), j6 (seat 3), e2 by seat 0 (a step away: progress -0.5, not greedy)
    ev = [{"a": "f2"}, {"a": "e2h"}, {"a": "f10"}, {"a": "j6"}, {"a": "e2"}]
    persp, f = both(record_of(ev))
    assert persp == [0, 1, 2, 3, 0, 1]

    # after f2 (perspective seat 1): seat 0 is "previous" (relative 3)
    b = seat_block(f[1], 3)
    expected = [0, 0.5, 1, 0, 0, 0, 0, 0, 0]
    assert np.allclose(b[0:9], expected) and np.allclose(b[9:18], expected) and b[18] == pytest.approx(1 / 64)
    assert np.all(f[1][:3 * PER_SEAT] == 0)

    # after e2h (perspective seat 2: me 2, next 3, across 0, previous 1): seat 1's first move = a wall that hits the
    # leader (seat 0, arrival estimate 16 vs 18 and 19); the only victim is seat 0 (distance 4 -> 6: 2 / 8), which is
    # relative column 3 + (0 - 2) mod 4 = 5 in this perspective
    b = seat_block(f[2], 3)
    expected = [1, 0, 0, 0, 0, 0.25, 0, 1, 0]
    assert np.allclose(b[0:9], expected) and np.allclose(b[9:18], expected) and b[18] == pytest.approx(1 / 64)
    assert np.allclose(seat_block(f[2], 2)[0:9], [0, 0.5, 1, 0, 0, 0, 0, 0, 0])   # seat 0 unchanged

    # after e2 (perspective seat 1): seat 0 has two moves; second x = [0, -0.5, 0, ...] with rate max(1/2, ...) = 0.5:
    # mean_1 = 0.5 + (-0.5 - 0.5) / 2 = 0, mean_2 = 1 + (0 - 1) / 2 = 0.5; count 2 / 64. Seat 1's wall is the same
    # in perspective seat 1 (victim of absolute seat 0 is relative column 3 + (0 - 1) mod 4 = 6)
    assert persp[5] == 1
    b0 = seat_block(f[5], 3)
    assert np.allclose(b0[0:9], [0, 0, 0.5, 0, 0, 0, 0, 0, 0]) and np.allclose(b0[9:18], b0[0:9])
    assert b0[18] == pytest.approx(2 / 64)
    b1 = seat_block(f[5], 0)
    assert np.allclose(b1[0:9], [1, 0, 0, 0, 0, 0, 0.25, 1, 0])
    # seats 2 and 3 stepped toward the center once each: progress 0.5, greedy
    assert np.allclose(seat_block(f[5], 1)[0:9], [0, 0.5, 1, 0, 0, 0, 0, 0, 0])
    assert np.allclose(seat_block(f[5], 2)[0:9], [0, 0.5, 1, 0, 0, 0, 0, 0, 0])

    # an elimination zeroes the seat for good; later features keep it at zero
    ev2 = ev + [{"elim": 2}]            # records count seats from 1: seat index 1
    persp2, f2 = both(record_of(ev2))
    after_elim = f2[-1]
    p = persp2[-1]
    assert np.all(seat_block(after_elim, (1 - p) % 4) == 0)
    assert np.any(seat_block(after_elim, (0 - p) % 4) != 0)


def test_s1_bots_styles():
    """A greedy seat never walls; a basher only walls the leader (when it walls); a grudge seat's victim is its
    target. The bots take a step (greedy fallback) or a wall, so x[0] + x[2] = 1 for each of their moves."""
    # seats: 0 greedy, 1 basher, 2 grudge -> target seat 0, 3 randomPawn
    lines = popgames(60, 11, ["-bots", "greedy,basher,grudge,randomPawn", "-targets", "0,0,0,0", "-elimprob", "0",
                              "-maxplies", "200"])
    walls_seen = {1: 0, 2: 0}
    basher_ratio = []
    for line in lines:
        rec = json.loads(line)
        persp, f = nonterminal(rec, *record_features(rec))
        # the features with seat to move 0 at the end of the game are in absolute order (perspective 0)
        last = len(persp) - 1
        while persp[last] != 0:
            last -= 1
        feats = f[last]
        g = seat_block(feats, 0)
        if g[18] > 0:      # the greedy seat has moved: never a wall, always a shortest step
            assert g[0] == 0 and g[2] == pytest.approx(1.0) and np.all(g[3:9] == 0) and -1 <= g[1] <= 1
        for seat, name in ((1, "basher"), (2, "grudge")):
            blk = seat_block(feats, seat)
            for h in range(2):
                m = blk[h * 9:(h + 1) * 9]
                assert m[0] + m[2] == pytest.approx(1.0, abs=1e-9) or blk[18] == 0
            m = blk[0:9]
            if m[0] > 0:
                walls_seen[seat] += 1
                assert m[8] == 0, "a basher / grudge wall always lengthens an opponent's path"
                if seat == 1:
                    # the basher's wall maximizes the leader's increase (another opponent's may be larger)
                    basher_ratio.append(m[7] / m[0])
                else:
                    assert m[3 + 0] > 0, "the grudge seat's walls hit its target (seat 0)"
    assert walls_seen[1] > 5 and walls_seen[2] > 5, walls_seen
    assert np.mean(basher_ratio) > 0.8, np.mean(basher_ratio)


def transform_record(rec, sym):
    """The same game on the symmetric board: seat identities stay, cells and walls are mapped. Returns the dict for
    `q4tool style` (initialBoard + events, eliminations 0-based) and the python record equivalent."""
    pawns = [list(apply_cell(x, y, sym)) for (x, y) in START_POSITIONS]
    events = []
    for ev in rec["events"]:
        if "elim" in ev:
            events.append({"elim": ev["elim"] - 1})
            continue
        kind, (x, y) = str_to_action(ev["a"])
        if kind == 'p':
            nx, ny = apply_cell(x, y, sym)
            events.append({"a": action_to_str(('p', (nx, ny)))})
        else:
            nax, nay, is_h = apply_anchor(x, y, kind == 'h', sym)
            events.append({"a": action_to_str(('h' if is_h else 'v', (nax, nay)))})
    return {"rules": rec["rules"], "initialBoard": {"pawns": pawns}, "events": events}


def python_features_of_transformed(t):
    from q4.reference import Pos, eliminate, play
    rules = t["rules"]
    pos = Pos(max_plies=rules["maxPlies"], repetition_draw_count=rules["repetitionDrawCount"])
    pos.pawn = [tuple(p) for p in t["initialBoard"]["pawns"]]
    tracker = StyleTracker()
    feats = [tracker.encode(pos.to_move)]
    for ev in t["events"]:
        if "elim" in ev:
            pos = eliminate(pos, ev["elim"])
            tracker.observe_elimination(ev["elim"])
        else:
            a = str_to_action(ev["a"])
            nxt = play(pos, a)
            tracker.observe_move(pos, nxt, a)
            pos = nxt
        feats.append(tracker.encode(pos.to_move))
    return np.array(feats)


def test_s1_invariant_under_the_8_symmetries():
    recs = [json.loads(l) for l in popgames(60, 21, ["-maxplies", "100"])]
    n_checked = 0
    for rec in recs:
        base_persp, base = record_features(rec)
        _, base = nonterminal(rec, base_persp, base)
        for sym in range(1, 8):
            t = transform_record(rec, sym)
            _, py = nonterminal(rec, base_persp, python_features_of_transformed(t))
            assert py.shape == base.shape and np.max(np.abs(py - base)) < 1e-12, f"python, symmetry {sym}"
            (cp, cpp), = cpp_style([json.dumps(t)])
            _, cpp = nonterminal(rec, cp, cpp)
            assert np.max(np.abs(cpp - base)) < 1e-6, f"C++, symmetry {sym}"
            n_checked += 1
    assert n_checked == 60 * 7


def test_s1_cpp_equals_python_on_2000_population_games():
    n = int(os.environ.get("Q4_STYLE_GAMES", "2000"))
    lines = popgames(n, 31, ["-maxplies", "100"])
    assert len(lines) == n
    cpp = cpp_style(lines)
    max_diff, positions, with_elim, walls = 0.0, 0, 0, 0
    for line, (cpersp, cfeat) in zip(lines, cpp):
        rec = json.loads(line)
        persp, py = nonterminal(rec, *record_features(rec))
        cpersp, cfeat = nonterminal(rec, cpersp, cfeat)
        assert cpersp == persp
        assert cfeat.shape == py.shape
        max_diff = max(max_diff, float(np.max(np.abs(cfeat - py))))
        positions += len(persp)
        with_elim += any("elim" in e for e in rec["events"])
        walls += sum(1 for e in rec["events"] if "a" in e and e["a"][-1] in "hv")
    print(f"s1: {n} games, {positions} positions, {with_elim} with eliminations, {walls} walls, max |C++ - python| = {max_diff:.3g}")
    assert max_diff < 1e-6
    assert with_elim > n // 10 and walls > n
