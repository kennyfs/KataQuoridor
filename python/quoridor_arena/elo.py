"""Bradley-Terry Elo ratings from arena results, with bootstrap confidence intervals and a report.

Model: P(i beats j) = 1 / (1 + 10^((R_j - R_i) / 400)). A draw counts as half a win for each side.
A weak prior adds one virtual draw between every pair that actually played, so 100% / 0% pairs
stay finite. Ratings are anchored so that the anchor player (default: sq-random) is 0 Elo.
Confidence intervals come from a bootstrap over paired-game units (same pair, same opening).

Usage:
    python -m quoridor_arena.elo <out_dir> [--anchor sq-random] [--bootstrap 1000]
"""
import argparse
import collections
import json
import math
import os
import sys

import numpy as np

ELO_PER_NAT = 400.0 / math.log(10.0)


def load_results(path):
    games = []
    if not os.path.exists(path):
        return games
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


def game_score(g, name):
    """Score of `name` in game g: 1 win, 0.5 draw, 0 loss."""
    if g["winner_name"] is None:
        return 0.5
    return 1.0 if g["winner_name"] == name else 0.0


def build_units(games, index):
    """Group games into bootstrap units (pair + opening). Returns arrays (i, j, score_i, n) per unit."""
    units = collections.OrderedDict()
    for g in games:
        a, b = g["pair"]
        key = (a, b, g["opening"])
        u = units.setdefault(key, [index[a], index[b], 0.0, 0])
        u[2] += game_score(g, a)
        u[3] += 1
    if not units:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0)
    arr = np.array(list(units.values()), dtype=float)
    return arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2], arr[:, 3]


def aggregate(n_players, ui, uj, us, un, weights=None):
    """Score matrix S[i,j] (score of i against j) and game-count matrix N."""
    w = np.ones(len(ui)) if weights is None else weights
    S = np.zeros((n_players, n_players))
    N = np.zeros((n_players, n_players))
    np.add.at(S, (ui, uj), w * us)
    np.add.at(S, (uj, ui), w * (un - us))
    np.add.at(N, (ui, uj), w * un)
    np.add.at(N, (uj, ui), w * un)
    return S, N


def components(N):
    n = N.shape[0]
    comp = -np.ones(n, int)
    c = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack = [s]
        comp[s] = c
        while stack:
            v = stack.pop()
            for u in np.nonzero(N[v] > 0)[0]:
                if comp[u] < 0:
                    comp[u] = c
                    stack.append(u)
        c += 1
    return comp


def fit_bt(S, N, anchor, prior_pairs=None, prior_draws=1.0, init=None, iters=100, tol=1e-9):
    """Maximum-likelihood Bradley-Terry fit. Returns Elo ratings (anchor = 0); players not
    connected to the anchor get NaN.

    prior_pairs: boolean matrix of pairs that get `prior_draws` virtual draws (default: pairs with N>0).
    """
    n = S.shape[0]
    if prior_pairs is None:
        prior_pairs = N > 0
    S = S + 0.5 * prior_draws * prior_pairs
    N = N + prior_draws * prior_pairs
    comp = components(N)
    active = np.nonzero(comp == comp[anchor])[0]
    free = np.array([k for k in active if k != anchor], int)
    theta = np.zeros(n) if init is None else np.nan_to_num(np.asarray(init, float) / ELO_PER_NAT)
    theta[anchor] = 0.0
    if len(free) > 0:
        for _ in range(iters):
            d = theta[:, None] - theta[None, :]
            p = 1.0 / (1.0 + np.exp(-d))
            grad = (S - N * p).sum(axis=1)
            W = N * p * (1.0 - p)
            H = W.copy()
            H[np.diag_indices(n)] = -W.sum(axis=1)
            g = grad[free]
            Hf = H[np.ix_(free, free)]
            step = np.linalg.solve(Hf, -g)
            # Damp huge steps (only matters from a poor start).
            m = np.max(np.abs(step))
            if m > 2.0:
                step *= 2.0 / m
            theta[free] += step
            if m < tol:
                break
    elo = theta * ELO_PER_NAT
    elo[comp != comp[anchor]] = np.nan
    return elo


def bootstrap(n_players, ui, uj, us, un, anchor, resamples=1000, seed=0, point=None):
    """Resample paired-game units with replacement; returns an array (resamples, n_players)."""
    rng = np.random.default_rng(seed)
    S0, N0 = aggregate(n_players, ui, uj, us, un)
    prior_pairs = N0 > 0
    out = np.zeros((resamples, n_players))
    m = len(ui)
    for r in range(resamples):
        counts = np.bincount(rng.integers(0, m, size=m), minlength=m).astype(float)
        S, N = aggregate(n_players, ui, uj, us, un, counts)
        out[r] = fit_bt(S, N, anchor, prior_pairs=prior_pairs, init=point)
    return out


def rate(names, games, anchor, resamples=1000, seed=0):
    """Returns dict name -> (elo, lo, hi, games, score)."""
    index = {n: k for k, n in enumerate(names)}
    if anchor not in index:
        raise SystemExit("anchor %r is not among the players %s" % (anchor, names))
    a = index[anchor]
    ui, uj, us, un = build_units(games, index)
    S, N = aggregate(len(names), ui, uj, us, un)
    elo = fit_bt(S, N, a)
    if resamples > 0 and len(ui) > 0:
        boot = bootstrap(len(names), ui, uj, us, un, a, resamples, seed, point=elo)
        lo = np.nanpercentile(boot, 2.5, axis=0)
        hi = np.nanpercentile(boot, 97.5, axis=0)
    else:
        lo = hi = np.full(len(names), np.nan)
    ngames = N.sum(axis=1)
    score = S.sum(axis=1)
    return {n: (elo[k], lo[k], hi[k], int(ngames[k]), score[k]) for n, k in index.items()}


# -- report -------------------------------------------------------------------

def _pct(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else "%.0f%%" % (100 * x)


def _num(x, fmt="%.0f"):
    return "—" if x is None or math.isnan(x) else fmt % x


def make_report(out_dir, names=None, anchor="sq-random", resamples=1000, seed=0, run_info=None):
    games = load_results(os.path.join(out_dir, "results.jsonl"))
    if names is None:
        names = []
        for g in games:
            for n in g["pair"]:
                if n not in names:
                    names.append(n)
    played = {n for g in games for n in g["pair"]}
    names = [n for n in names if n in played]
    lines = ["# Quoridor arena report", ""]
    if not games:
        lines.append("No games yet.")
        return "\n".join(lines) + "\n"
    if anchor not in played:
        new_anchor = names[0]
        lines.append("_Anchor %s did not play; anchoring %s at 0 instead._" % (anchor, new_anchor))
        lines.append("")
        anchor = new_anchor
    ratings = rate(names, games, anchor, resamples, seed)
    order = sorted(names, key=lambda n: -np.nan_to_num(ratings[n][0], nan=-1e9))

    # Summary
    n = len(games)
    draws = sum(1 for g in games if g["winner"] is None)
    bwins = sum(1 for g in games if g["winner"] == "b")
    reasons = collections.Counter(g["reason"] for g in games)
    secs = sum(g.get("seconds", 0) for g in games)
    lines += ["Games: %d, players: %d, pairs: %d. Anchor: %s = 0 Elo. CI: 95%% bootstrap over "
              "paired-game units (%d resamples)." % (n, len(names), len({tuple(g["pair"]) for g in games}),
                                                     anchor, resamples),
              "",
              "Overall: Black (first player) score %.1f%%, draw rate %.1f%%, average plies %.1f, "
              "summed game time %.1f h." % (100 * (bwins + 0.5 * draws) / n, 100 * draws / n,
                                            sum(g["plies"] for g in games) / n, secs / 3600),
              "",
              "End reasons: " + ", ".join("%s %d" % kv for kv in sorted(reasons.items())),
              ""]
    if run_info:
        lines += [run_info, ""]
    bad = [g for g in games if g["reason"] not in ("goal",) and not g["reason"].startswith("draw")]
    if bad:
        lines += ["## Anomalies", ""]
        for g in bad:
            lines.append("- `%s`: %s, %s wins; %s" % (g["id"], g["reason"], g["winner_name"], g.get("detail", "")))
        lines.append("")

    # Rating table
    lines += ["## Ratings", "", "| # | Player | Elo | 95% CI | Games | Score |", "|---|---|---:|---|---:|---:|"]
    for r, name in enumerate(order, 1):
        e, lo, hi, ng, sc = ratings[name]
        lines.append("| %d | %s | %s | [%s, %s] | %d | %s |" % (
            r, name, _num(e), _num(lo), _num(hi), ng, _pct(sc / ng if ng else None)))
    lines.append("")

    # Crosstable
    pair_games = collections.defaultdict(list)
    for g in games:
        pair_games[tuple(g["pair"])].append(g)

    def vs(a, b):
        gs = pair_games.get((a, b), []) + pair_games.get((b, a), [])
        if not gs:
            return None, 0
        return sum(game_score(g, a) for g in gs) / len(gs), len(gs)

    lines += ["## Crosstable", "", "Row player's score against the column player (games in parentheses). "
              "Columns are numbered by rank.", ""]
    lines.append("| # | Player | " + " | ".join(str(k) for k in range(1, len(order) + 1)) + " |")
    lines.append("|---|---|" + "---:|" * len(order))
    for r, a in enumerate(order, 1):
        cells = []
        for b in order:
            if a == b:
                cells.append("·")
                continue
            s, c = vs(a, b)
            cells.append("" if c == 0 else "%s (%d)" % (_pct(s), c))
        lines.append("| %d | %s | %s |" % (r, a, " | ".join(cells)))
    lines.append("")

    # Per pair
    lines += ["## Pairs", "", "| Pair | Games | Score (first) | Black wins | Draws | Avg plies | Avg margin | Other ends |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for (a, b), gs in sorted(pair_games.items(), key=lambda kv: (order.index(kv[0][0]), order.index(kv[0][1]))):
        k = len(gs)
        sc = sum(game_score(g, a) for g in gs) / k
        bw = sum(1 for g in gs if g["winner"] == "b") / k
        dr = sum(1 for g in gs if g["winner"] is None) / k
        pl = sum(g["plies"] for g in gs) / k
        decided = [g["margin"] for g in gs if g["winner"] is not None]
        mg = sum(decided) / len(decided) if decided else float("nan")
        other_ends = collections.Counter(g["reason"] for g in gs if g["reason"] != "goal")
        lines.append("| %s vs %s | %d | %s | %s | %s | %.1f | %s | %s |" % (
            a, b, k, _pct(sc), _pct(bw), _pct(dr), pl, _num(mg, "%.1f"),
            ", ".join("%s %d" % kv for kv in sorted(other_ends.items()))))
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--anchor", default="sq-random")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    names = None
    roster_path = os.path.join(args.out_dir, "roster_used.json")
    if os.path.exists(roster_path):
        with open(roster_path) as f:
            names = [e["name"] for e in json.load(f)["engines"]]
    text = make_report(args.out_dir, names, args.anchor, args.bootstrap, args.seed)
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
