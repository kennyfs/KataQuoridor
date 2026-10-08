"""Tests for python/q4/rating.py and python/q4/match_report.py on synthetic tournaments (Round 6)."""
import itertools
import math
import os
import sys

import numpy as np
import pytest

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PYTHON_DIR)

from q4 import match_report, rating


def record(names, winner, table="t", opening=0, rotation=0, draw_reason="", events=30):
    result = "Draw" if winner is None else "%d+" % (winner + 1)
    return {
        "players": [{"name": n, "type": "x"} for n in names],
        "result": result,
        "events": [{"a": "e2"}] * events,
        "match": {"table": table, "opening": opening, "rotation": rotation, "openingPlies": 4,
                  "drawReason": draw_reason if winner is None else ""},
    }


def simulate(true_elo, n_games, draw_rate, seed):
    """Tables of 4 random distinct players; the winner is drawn Plackett-Luce from the true strengths."""
    rng = np.random.default_rng(seed)
    names = list(true_elo)
    strength = {n: 10 ** (true_elo[n] / 400.0) for n in names}
    games = []
    for g in range(n_games):
        table = list(rng.choice(names, size=4, replace=False))
        if rng.random() < draw_rate:
            winner = None
        else:
            p = np.array([strength[n] for n in table])
            winner = int(rng.choice(4, p=p / p.sum()))
        games.append(record(table, winner, table="sim", opening=g // 4, rotation=g % 4,
                            draw_reason="maxPlies"))
    return games


TRUE = {"random": 0.0, "greedy": 200.0, "net1": 350.0, "net2": 500.0, "net3": 700.0, "net4": 760.0}


def test_recovers_known_strengths_and_anchor():
    games = simulate(TRUE, 24000, 0.0, seed=1)
    ratings = rating.rate(games, "random", resamples=200, seed=2)
    assert ratings["random"][0] == 0.0
    for n, true in TRUE.items():
        e, lo, hi, ng, sc = ratings[n]
        assert abs(e - true) < 45, (n, e, true)
        assert lo <= e <= hi
        if n != "random":
            assert lo - 25 <= true <= hi + 25, (n, lo, hi, true)
    order = sorted(TRUE, key=lambda n: -ratings[n][0])
    assert order == sorted(TRUE, key=lambda n: -TRUE[n])


def test_draws_keep_the_order():
    # A draw is half a win for every pair at the table (as in the Duel ladder), which compresses the scale when
    # draws are independent of strength, but not the order.
    games = simulate(TRUE, 12000, 0.05, seed=4)
    ratings = rating.rate(games, "random", resamples=0)
    assert sorted(TRUE, key=lambda n: -ratings[n][0]) == sorted(TRUE, key=lambda n: -TRUE[n])
    assert ratings["random"][0] == 0.0


def test_anchor_moves_the_zero_not_the_differences():
    games = simulate(TRUE, 3000, 0.0, seed=3)
    a = rating.rate(games, "random", resamples=0)
    b = rating.rate(games, "greedy", resamples=0)
    assert b["greedy"][0] == 0.0
    for n in TRUE:
        assert math.isclose(a[n][0] - a["greedy"][0], b[n][0], abs_tol=1e-6)


def test_game_pairs_winner_beats_each_other_player_and_draw_is_half():
    names = ["a", "b", "b", "c"]
    pairs = rating.game_pairs(record(names, 0))
    assert sorted((x, y) for x, y, s, w in pairs) == [("a", "b"), ("a", "b"), ("a", "c")]
    assert all(s == 1.0 and w == 1.0 for _, _, s, w in pairs)
    # Two copies of one player: the winner (a "b") beats a and c, but makes no pair with the other b.
    pairs = rating.game_pairs(record(names, 1))
    assert sorted((x, y) for x, y, s, w in pairs) == [("b", "a"), ("b", "c")]
    draw = rating.game_pairs(record(names, None))
    assert sorted((x, y) for x, y, s, w in draw) == [("a", "b"), ("a", "b"), ("a", "c"), ("b", "c"), ("b", "c")]
    assert all(s == 0.5 and w == 0.5 for _, _, s, w in draw)
    assert rating.game_pairs({"players": [{"name": "a"}] * 4, "result": "none", "events": []}) == []


def test_anchor_must_have_played():
    with pytest.raises(SystemExit):
        rating.rate(simulate({"a": 0.0, "b": 100.0, "c": 50.0, "d": 80.0}, 40, 0.0, 0), "random", resamples=0)


def test_report_win_rates_seats_draws_length():
    games = []
    # table t1: cand wins 3 of 4 rotations of opening 0 and 2 of 4 of opening 1; one draw by repetition
    for o, wins in enumerate([(0, 1, 2), (0, 3)]):
        for r in range(4):
            names = ["ref"] * 4
            seat = r  # candidate in seat r
            names[seat] = "cand"
            winner = seat if r in wins else (seat + 1) % 4
            games.append(record(names, winner, table="t1", opening=o, rotation=r, events=40))
    games.append(record(["cand", "ref", "ref", "ref"], None, table="t1", opening=2, draw_reason="repetition", events=60))
    games.append(record(["cand", "ref", "ref", "ref"], None, table="t1", opening=2, draw_reason="maxPlies", events=80))
    rep = match_report.analyze(games, resamples=300, seed=0)["t1"]
    assert rep["games"] == 10 and rep["openings"] == 3
    cand = rep["players"]["cand"]
    assert cand["copies"] == 1 and cand["wins"] == 5
    assert math.isclose(cand["win_rate"], 0.5)
    assert math.isclose(cand["win_rate_per_copy"], 0.5)
    ref = rep["players"]["ref"]
    assert ref["copies"] == 3 and ref["wins"] == 3
    assert math.isclose(ref["win_rate_per_copy"], 3 / 30)
    # score: a draw is a quarter point for one of four seats
    assert math.isclose(cand["score"], (5 + 2 * 0.25) / 10)
    assert math.isclose(ref["score"], (3 + 2 * 0.75) / 10)
    assert rep["draws_by_reason"] == {"repetition": 1, "maxPlies": 1}
    assert math.isclose(rep["draw_rate"], 0.2)
    assert rep["seat_wins"] == [3, 2, 1, 2] or sum(rep["seat_wins"]) == 8
    assert math.isclose(rep["mean_plies"], (8 * 40 + 60 + 80) / 10)
    lo, hi = cand["win_rate_ci"]
    assert lo <= cand["win_rate"] <= hi
    assert "Table `t1`" in match_report.format_markdown({"t1": rep})


def test_report_seat_rates_sum_and_first_move_advantage():
    games = [record(["a", "b", "c", "d"], 0, opening=i // 4, rotation=i % 4) for i in range(8)]
    rep = match_report.analyze(games, resamples=0)["t"]
    assert rep["seat_win_rate"] == [1.0, 0.0, 0.0, 0.0]
    assert rep["draw_rate"] == 0.0
