#!/usr/bin/env python3
"""Ratings of Q4 players from match games (katago q4match -output games.sgfs, docs/q4/rounds/R6.md).

Bradley-Terry, fitted by the Duel ladder's code (quoridor_arena.elo.fit_bt: Elo scale, one virtual draw between every
pair that met, maximum likelihood), on "the winner beats every other player at the table": a game with winner W is
one win of W over each other seat (3 pairs; two seats of the same player name make no pair). A game without a winner
(a draw, as the Duel ladder counts a draw: half a win for each side) gives every pair of different players at the
table half a point each, with weight 1/2 per pair so a draw carries the same total weight as a decided game.
Ratings are anchored at `random` = 0. Intervals: 95% bootstrap over openings (all rotations of an opening of a table
are one unit), as the Duel ladder resamples paired-game units.

    python q4/rating.py games.sgfs [more.sgfs ...] [--anchor random] [--bootstrap 1000] [--json out.json]
"""
import argparse
import collections
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q4.record import load_records  # noqa: E402
from quoridor_arena import elo  # noqa: E402  (the Duel ladder's Bradley-Terry fit)


def game_pairs(g):
    """[(name_i, name_j, score_i, weight)] for one game record."""
    names = [p["name"] for p in g["players"]]
    r = g["result"]
    if len(r) == 2 and r[1] == "+" and r[0] in "1234":
        w = names[int(r[0]) - 1]
        return [(w, other, 1.0, 1.0) for other in names if other != w]
    if r == "Draw":
        return [(a, b, 0.5, 0.5) for a, b in itertools.combinations(names, 2) if a != b]
    return []  # unfinished


def build_units(games):
    """Games grouped by (table, opening); returns player names and a list of (S, N) matrices per unit."""
    names = sorted({p["name"] for g in games for p in g["players"]})
    index = {n: k for k, n in enumerate(names)}
    units = collections.OrderedDict()
    for i, g in enumerate(games):
        m = g.get("match")
        key = (m["table"], m["opening"]) if m is not None else ("game", i)
        u = units.setdefault(key, [np.zeros((len(names), len(names))), np.zeros((len(names), len(names)))])
        for a, b, s, w in game_pairs(g):
            ia, ib = index[a], index[b]
            u[0][ia, ib] += w * s
            u[0][ib, ia] += w * (1.0 - s)
            u[1][ia, ib] += w
            u[1][ib, ia] += w
    return names, list(units.values())


def rate(games, anchor="random", resamples=1000, seed=0):
    """Returns {name: (elo, lo, hi, seat_games, score)}; the CI is nan with resamples = 0."""
    names, units = build_units(games)
    if anchor not in names:
        raise SystemExit("anchor %r did not play (players: %s)" % (anchor, ", ".join(names)))
    a = names.index(anchor)
    S = sum(u[0] for u in units)
    N = sum(u[1] for u in units)
    point = elo.fit_bt(S, N, a)
    lo = hi = np.full(len(names), np.nan)
    if resamples > 0 and len(units) > 1:
        rng = np.random.default_rng(seed)
        S_u = np.stack([u[0] for u in units])
        N_u = np.stack([u[1] for u in units])
        prior = N > 0
        boot = np.zeros((resamples, len(names)))
        for r in range(resamples):
            counts = np.bincount(rng.integers(0, len(units), size=len(units)), minlength=len(units)).astype(float)
            boot[r] = elo.fit_bt(np.tensordot(counts, S_u, 1), np.tensordot(counts, N_u, 1), a, prior_pairs=prior, init=point)
        lo = np.nanpercentile(boot, 2.5, axis=0)
        hi = np.nanpercentile(boot, 97.5, axis=0)
    ngames = N.sum(axis=1)
    score = S.sum(axis=1)
    return {n: (float(point[k]), float(lo[k]), float(hi[k]), float(ngames[k]), float(score[k])) for k, n in enumerate(names)}


def format_markdown(ratings):
    def num(x):
        return "—" if x != x else "%.0f" % x

    order = sorted(ratings, key=lambda n: -np.nan_to_num(ratings[n][0], nan=-1e9))
    lines = ["| # | Player | Elo | 95% CI | Pairs | Pair score |", "|---|---|---:|---|---:|---:|"]
    for k, n in enumerate(order, 1):
        e, lo, hi, ng, sc = ratings[n]
        lines.append("| %d | %s | %s | [%s, %s] | %.0f | %s |" % (
            k, n, num(e), num(lo), num(hi), ng, "—" if ng == 0 else "%.0f%%" % (100 * sc / ng)))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="+")
    ap.add_argument("--anchor", default="random")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    games = []
    for path in args.games:
        games += list(load_records(path))
    ratings = rate(games, args.anchor, args.bootstrap, args.seed)
    sys.stdout.write("Anchor: %s = 0 Elo; %d games.\n\n" % (args.anchor, len(games)) + format_markdown(ratings) + "\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(ratings, f, indent=1)


if __name__ == "__main__":
    main()
