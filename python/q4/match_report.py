#!/usr/bin/env python3
"""Report of Q4 match results (katago q4match -output games.jsonl, docs/q4/rounds/R6.md).

Per table: the win rate of every player (wins / games, and per copy at the table: wins / (games x copies), 25% =
even), its score (a draw counts half), both with a 95% bootstrap CI over openings (all rotations of an opening share
the opening and are not independent); the win rate of every seat (the first-move advantage); the draw rate split by
reason; the mean game length.

    python q4/match_report.py games.jsonl [more.jsonl ...] [--json report.json] [--bootstrap 2000]
"""
import argparse
import collections
import json
import sys

import numpy as np


def load_games(paths):
    games = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    games.append(json.loads(line))
    return games


def game_table(g):
    m = g.get("match")
    if m is not None:
        return m["table"]
    return "+".join(sorted(p["name"] for p in g["players"]))


def game_opening(g, index):
    m = g.get("match")
    return m["opening"] if m is not None else index


def winner_seat(g):
    r = g["result"]
    if len(r) == 2 and r[1] == "+" and r[0] in "1234":
        return int(r[0]) - 1
    return None


def draw_reason(g):
    if winner_seat(g) is not None:
        return None
    m = g.get("match")
    if m is not None and m.get("drawReason"):
        return m["drawReason"]
    return "draw" if g["result"] == "Draw" else "unfinished"


def cluster_ci(unit_values, resamples, rng):
    """Mean and 95% percentile bootstrap CI of per-unit (num, den) pairs: sum(num) / sum(den)."""
    arr = np.asarray(unit_values, float)
    if len(arr) == 0 or arr[:, 1].sum() == 0:
        return float("nan"), float("nan"), float("nan")
    point = arr[:, 0].sum() / arr[:, 1].sum()
    if len(arr) < 2 or resamples <= 0:
        return point, float("nan"), float("nan")
    idx = rng.integers(0, len(arr), size=(resamples, len(arr)))
    num = arr[idx, 0].sum(axis=1)
    den = arr[idx, 1].sum(axis=1)
    ok = den > 0
    rates = num[ok] / den[ok]
    return point, float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def analyze(games, resamples=2000, seed=0):
    rng = np.random.default_rng(seed)
    by_table = collections.OrderedDict()
    for i, g in enumerate(games):
        by_table.setdefault(game_table(g), []).append((game_opening(g, i), g))
    report = collections.OrderedDict()
    for table, items in by_table.items():
        n = len(items)
        names = sorted({p["name"] for _, g in items for p in g["players"]})
        copies = {nm: collections.Counter(p["name"] for p in items[0][1]["players"])[nm] for nm in names}
        # per opening: games, wins/score per player
        units = collections.OrderedDict()
        for o, g in items:
            u = units.setdefault(o, {"games": 0, "wins": collections.Counter(), "draws": 0})
            u["games"] += 1
            w = winner_seat(g)
            if w is None:
                u["draws"] += 1
            else:
                u["wins"][g["players"][w]["name"]] += 1
        players = collections.OrderedDict()
        for nm in names:
            win = [(u["wins"][nm], u["games"]) for u in units.values()]
            win_copy = [(u["wins"][nm], u["games"] * copies[nm]) for u in units.values()]
            score = [(u["wins"][nm] + u["draws"] * copies[nm] / 4.0, u["games"]) for u in units.values()]
            wr, wlo, whi = cluster_ci(win, resamples, rng)
            wc, clo, chi = cluster_ci(win_copy, resamples, rng)
            sc, slo, shi = cluster_ci(score, resamples, rng)
            players[nm] = {"copies": copies[nm], "wins": int(sum(u["wins"][nm] for u in units.values())),
                           "win_rate": wr, "win_rate_ci": [wlo, whi], "win_rate_per_copy": wc,
                           "win_rate_per_copy_ci": [clo, chi], "score": sc, "score_ci": [slo, shi]}
        seat_wins = [0, 0, 0, 0]
        reasons = collections.Counter()
        for _, g in items:
            w = winner_seat(g)
            if w is not None:
                seat_wins[w] += 1
            else:
                reasons[draw_reason(g)] += 1
        plies = [len(g["events"]) for _, g in items]
        report[table] = {
            "games": n, "openings": len(units), "players": players,
            "seat_win_rate": [s / n for s in seat_wins], "seat_wins": seat_wins,
            "draw_rate": sum(reasons.values()) / n, "draws_by_reason": dict(reasons),
            "mean_plies": float(np.mean(plies)),
        }
    return report


def format_markdown(report):
    def pct(x):
        return "—" if x is None or x != x else "%.1f%%" % (100 * x)

    def ci(lohi):
        return "[%s, %s]" % (pct(lohi[0]), pct(lohi[1]))

    lines = []
    for table, t in report.items():
        lines.append("### Table `%s`: %d games, %d openings" % (table, t["games"], t["openings"]))
        lines.append("")
        lines.append("| Player | Copies | Wins | Win rate | 95% CI | Win rate per copy | 95% CI | Score | 95% CI |")
        lines.append("|---|---:|---:|---:|---|---:|---|---:|---|")
        for nm, p in t["players"].items():
            lines.append("| %s | %d | %d | %s | %s | %s | %s | %s | %s |" % (
                nm, p["copies"], p["wins"], pct(p["win_rate"]), ci(p["win_rate_ci"]), pct(p["win_rate_per_copy"]),
                ci(p["win_rate_per_copy_ci"]), pct(p["score"]), ci(p["score_ci"])))
        lines.append("")
        lines.append("Seat win rates (seat 1..4): " + ", ".join(pct(x) for x in t["seat_win_rate"]) +
                     ". Draws: %s (%s). Mean length: %.1f plies." % (
                         pct(t["draw_rate"]),
                         ", ".join("%s %d" % kv for kv in sorted(t["draws_by_reason"].items())) or "none",
                         t["mean_plies"]))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    report = analyze(load_games(args.games), args.bootstrap, args.seed)
    sys.stdout.write(format_markdown(report) + "\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
