import math

import numpy as np

from quoridor_arena import elo


def simulate(true, pairs, openings_per_pair, draw_rate, rng):
    names = list(true)
    games = []
    for a, b in pairs:
        p = 1.0 / (1.0 + 10 ** ((true[b] - true[a]) / 400.0))
        # Draws must keep the expected score equal to p (a draw is half a win), else ratings compress.
        d = min(draw_rate, 2 * min(p, 1 - p))
        for k in range(openings_per_pair):
            for swap in (0, 1):
                r = rng.random()
                if r < d:
                    w = None
                elif r < d + (p - d / 2):
                    w = a
                else:
                    w = b
                games.append({"id": "%s_%s_%d_%d" % (a, b, k, swap), "pair": [a, b], "opening": k,
                              "winner_name": w, "winner": None if w is None else "b"})
    return names, games


def test_recovers_known_ratings():
    true = {"r": 0.0, "a": 150.0, "b": 300.0, "c": 450.0, "d": 600.0, "e": 800.0}
    names = list(true)
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    rng = np.random.default_rng(123)
    names, games = simulate(true, pairs, 100, 0.05, rng)
    ratings = elo.rate(names, games, "r", resamples=300, seed=1)
    inside = 0
    for n in names:
        e, lo, hi, ng, sc = ratings[n]
        assert abs(e - true[n]) < 60, (n, e, true[n])
        assert lo <= e <= hi
        inside += lo <= true[n] <= hi
    assert ratings["r"][0] == 0.0
    assert inside >= len(names) - 1  # 95% intervals: allow one miss


def test_prior_keeps_perfect_scores_finite():
    games = []
    for k in range(10):
        games.append({"id": str(k), "pair": ["x", "y"], "opening": k, "winner_name": "y", "winner": "w"})
    r = elo.rate(["x", "y"], games, "x", resamples=0)
    assert math.isfinite(r["y"][0]) and r["y"][0] > 300
    # 10 wins + 1 virtual draw => score 10.5/11: Elo = 400*log10(10.5/0.5)
    assert abs(r["y"][0] - 400 * math.log10(21)) < 1e-6


def test_disconnected_player_is_nan():
    games = [{"id": "1", "pair": ["x", "y"], "opening": 0, "winner_name": "y", "winner": "w"},
             {"id": "2", "pair": ["z", "q"], "opening": 0, "winner_name": "z", "winner": "b"}]
    r = elo.rate(["x", "y", "z", "q"], games, "x", resamples=0)
    assert math.isnan(r["z"][0]) and math.isfinite(r["y"][0])


def test_report_renders(tmp_path):
    true = {"sq-random": 0.0, "a": 200.0, "b": 400.0}
    names = list(true)
    names, games = simulate(true, [("sq-random", "a"), ("sq-random", "b"), ("a", "b")], 10, 0.1,
                            np.random.default_rng(0))
    import json
    with open(tmp_path / "results.jsonl", "w") as f:
        for g in games:
            g.update({"plies": 50, "reason": "goal" if g["winner_name"] else "draw300",
                      "margin": 3 if g["winner_name"] else 0, "seconds": 1.0})
            f.write(json.dumps(g) + "\n")
    text = elo.make_report(str(tmp_path), names, "sq-random", resamples=50)
    assert "## Ratings" in text and "## Crosstable" in text and "## Pairs" in text
