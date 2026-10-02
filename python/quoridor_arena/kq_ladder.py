"""KataQuoridor-only Elo ladder over many nets, played by one `katago match` process.

The QTP arena (arena.py) runs every player as its own process with one search thread, so NN batches are tiny.
Here all bots live in one `katago match` process: one NN evaluator per model file, many game threads, so the GPU
sees large batches. Games come from a fixed game list (match's gameListFile): sparse pairs between nets that are
close in training (not a round robin), each opening of the arena's kind played twice with colours swapped.

Usage (from the python/ directory):
    python -m quoridor_arena.kq_ladder run    --config quoridor_arena/kq_ladder_example.json --out ~/arena/kq_ladder1
    python -m quoridor_arena.kq_ladder plan   --config C --out DIR      # schedule only, no games
    python -m quoridor_arena.kq_ladder adapt  --config C --out DIR --pairs 10 --games 40   # add games, then `run`
    python -m quoridor_arena.kq_ladder report --config C --out DIR

`run` plays every scheduled game whose id is not yet in DIR/match/results.jsonl (so it resumes), then writes
DIR/results.jsonl (arena format, for elo.py), DIR/report.md and DIR/elo_vs_samples.png. See docs/Evaluation.md.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

from . import elo, openings
from .qtp import QTPEngine
from .referee import Arbiter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

_SAMPLES_RE = re.compile(r"-s(\d+)(?:-|$)")


# -- config and roster ----------------------------------------------------------

def expand(path):
    path = path.replace("{repo}", REPO)
    path = re.sub(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), ""), path)
    return os.path.abspath(os.path.expanduser(path))


def load_config(path):
    with open(path) as f:
        return json.load(f)


def model_samples(model_dir):
    m = _SAMPLES_RE.search(os.path.basename(model_dir.rstrip("/")))
    if not m:
        raise SystemExit("cannot read the training samples (-s<N>-) from model directory %s" % model_dir)
    return int(m.group(1))


def list_models(spec):
    """spec: a glob string or a list of directories. Returns directories sorted by training samples."""
    dirs = []
    for item in ([spec] if isinstance(spec, str) else spec):
        hits = sorted(glob.glob(expand(item)))
        if not hits:
            raise SystemExit("no model directory matches %s" % item)
        dirs += hits
    dirs = [d for d in dirs if os.path.isdir(d)]
    return sorted(set(dirs), key=model_samples)


def bot_name(prefix, samples, visits):
    return "%s-s%d-v%d" % (prefix, samples, visits)


def make_roster(cfg, model_lister=list_models):
    """Returns (nets, bots). nets: per series, the list of {series, prefix, samples, model, gen} in sample order.
    bots: ordered dict name -> {name, series, samples, visits, model, gen}. Every net gets cfg["visits"]; the
    visit-scaling net also gets the extra visit counts."""
    visits = cfg["visits"]
    nets = collections.OrderedDict()
    bots = collections.OrderedDict()
    for s in cfg["series"]:
        lst = []
        for gen, d in enumerate(model_lister(s["models"])):
            lst.append({"series": s["name"], "prefix": s["prefix"], "samples": model_samples(d),
                        "model": os.path.join(d, "model.bin.gz"), "gen": gen})
        nets[s["name"]] = lst
        for n in lst:
            name = bot_name(n["prefix"], n["samples"], visits)
            bots[name] = dict(n, name=name, visits=visits)
    vs = cfg.get("visit_scaling")
    if vs:
        n = pick_net(nets, vs["series"], vs.get("net", "latest"))
        for v in vs["visits"]:
            name = bot_name(n["prefix"], n["samples"], v)
            if name not in bots:
                bots[name] = dict(n, name=name, visits=v)
    return nets, bots


def pick_net(nets, series, which):
    lst = nets[series]
    if which == "latest":
        return lst[-1]
    if which == "oldest":
        return lst[0]
    for n in lst:
        if n["samples"] == int(which):
            return n
    raise SystemExit("no net with %s samples in series %s" % (which, series))


# -- schedule -------------------------------------------------------------------

def games_for_distance(s_a, s_b, g):
    """Games for a pair, from the distance in log training samples: a Gaussian
    max * exp(-d^2 / (2 sigma^2)), d = |ln s_a - ln s_b|, floored at min, rounded up to an even number (every
    opening is played with both colours)."""
    d = abs(math.log(s_a) - math.log(s_b))
    n = g["max"] * math.exp(-0.5 * (d / g["sigma_log_samples"]) ** 2)
    n = max(g["min"], int(math.ceil(n)))
    return n + (n % 2)


def make_pairs(cfg, nets, bots):
    """Sparse pairs at cfg["visits"]: within each series, generation distances in links.generation_distances;
    across series (links.cross), each net of the other series against the nearest net (in log samples) of the
    first series, plus oldest-oldest and latest-latest; plus the visit-scaling links (extra visits vs the same
    net at the base visits). Returns an ordered list of (a, b, games, kind); a is the older net (or the base-visits
    bot)."""
    visits = cfg["visits"]
    g = cfg["games"]
    name = lambda n, v=visits: bot_name(n["prefix"], n["samples"], v)
    pairs = collections.OrderedDict()

    def add(na, nb, kind, games=None):
        if na["samples"] > nb["samples"]:
            na, nb = nb, na
        a, b = name(na), name(nb)
        if a == b or (a, b) in pairs or (b, a) in pairs:
            return
        pairs[(a, b)] = (games or games_for_distance(na["samples"], nb["samples"], g), kind)

    for series, lst in nets.items():
        for k in cfg["links"]["generation_distances"]:
            for i in range(len(lst) - k):
                add(lst[i], lst[i + k], "gen%d" % k)
    series_names = list(nets)
    if cfg["links"].get("cross", True) and len(series_names) > 1:
        main = nets[series_names[0]]
        for other in series_names[1:]:
            for n in nets[other]:
                nearest = min(main, key=lambda m: abs(math.log(m["samples"]) - math.log(n["samples"])))
                add(nearest, n, "cross")
            add(main[0], nets[other][0], "cross-ends")
            add(main[-1], nets[other][-1], "cross-ends")
    vs = cfg.get("visit_scaling")
    if vs:
        n = pick_net(nets, vs["series"], vs.get("net", "latest"))
        for v in vs["visits"]:
            if v != visits:
                pairs[(name(n), name(n, v))] = (vs["games"], "visits")
    out = [(a, b, n, kind) for (a, b), (n, kind) in pairs.items()]
    for a, b, _, _ in out:
        assert a in bots and b in bots, (a, b)
    return out


def game_id(a, b, k, swap):
    return "%s_vs_%s_o%03d_%s" % (a, b, k, "ba" if swap else "ab")


def make_schedule(pairs, extra=None):
    """Ordered list of games (id, a, b, opening k, black, white), grouped by pair, so that the game threads work on
    few models at a time and each model's NN batches are large. extra: {"a:b": n} games added by `adapt`."""
    extra = extra or {}
    sched = []
    for a, b, n, _ in pairs:
        n += extra.get("%s:%s" % (a, b), 0)
        for gi in range(n):
            k, swap = divmod(gi, 2)
            black, white = (b, a) if swap else (a, b)
            sched.append((game_id(a, b, k, swap), a, b, k, black, white))
    return sched


# -- match config and game list -------------------------------------------------

def match_config(cfg, bots, game_list, results_file):
    rules = dict({"komi": -0.5, "maxPlies": 300, "timeBonusPerPly": 0.0, "repetitionDrawCount": 0,
                  "blackInitialWalls": 10, "whiteInitialWalls": 10}, **cfg.get("rules", {}))
    lines = ["# Written by quoridor_arena/kq_ladder.py", "logSearchInfo = false", "logMoves = false",
             "logGamesEvery = 100000", "logToStdout = true", "numBots = %d" % len(bots)]
    for i, b in enumerate(bots.values()):
        lines += ["botName%d = %s" % (i, b["name"]), "nnModelFile%d = %s" % (i, b["model"]),
                  "maxVisits%d = %d" % (i, b["visits"])]
    lines += ["gameListFile = %s" % game_list, "gameResultsFile = %s" % results_file,
              "bSizes = 17", "bSizeRelProbs = 1",
              # Standard match settings: no resignation, no komi changes, no handicap.
              "allowResignation = false", "resignThreshold = -0.95", "resignConsecTurns = 6",
              "komiMean = %s" % rules["komi"], "handicapProb = 0.0", "handicapCompensateKomiProb = 0.0",
              "maxPlies = %d" % rules["maxPlies"], "timeBonusPerPly = %s" % rules["timeBonusPerPly"],
              "repetitionDrawCount = %d" % rules["repetitionDrawCount"],
              "blackInitialWalls = %d" % rules["blackInitialWalls"],
              "whiteInitialWalls = %d" % rules["whiteInitialWalls"],
              "numSearchThreads = 1"]
    for k, v in cfg.get("search", {}).items():
        lines.append("%s = %s" % (k, json.dumps(v) if isinstance(v, bool) else v))
    for k, v in cfg["match"].items():
        lines.append("%s = %s" % (k, json.dumps(v) if isinstance(v, bool) else v))
    return "\n".join(lines) + "\n"


def game_list_text(games, bot_index, opening_list):
    return "".join("%s %d %d %s\n" % (gid, bot_index[black], bot_index[white], " ".join(opening_list[k]))
                   for gid, a, b, k, black, white in games)


# -- results --------------------------------------------------------------------

def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print("warning: dropping unparseable line in %s: %r" % (path, line[:80]), file=sys.stderr)
    return out


def parse_lead(result):
    """White's lead from a result string: W+x -> x, B+x -> -x, 0 -> 0; None for anything else."""
    if result == "0":
        return 0.0
    m = re.match(r"^([BW])\+([0-9.]+)$", result or "")
    if not m:
        return None
    return float(m.group(2)) * (1 if m.group(1) == "W" else -1)


def convert_result(r, game_info, opening_moves, komi=-0.5):
    """A `katago match` gameResultsFile record -> an arena results.jsonl record (see docs/Evaluation.md)."""
    gid, a, b, k, black, white = game_info
    if (r["black"], r["white"]) != (black, white):
        raise ValueError("game %s: match played %s vs %s, schedule says %s vs %s" % (
            gid, r["black"], r["white"], black, white))
    winner = r["winner"]
    lead_w = parse_lead(r["result"])
    if winner is None:
        dr = r.get("draw_reason", "")
        reason = "repetition" if dr == "repetition" else "draw%d" % r["plies"]
        margin = 0
    else:
        reason = "goal"
        # The margin (distance of the pawn that did not arrive) is lead + 0.5 in the standard game (Evaluation.md).
        margin = abs(lead_w) + 0.5 if lead_w is not None else None
    return {"id": gid, "pair": [a, b], "opening": k, "opening_moves": opening_moves, "black": black,
            "white": white, "winner": winner,
            "winner_name": None if winner is None else (black if winner == "b" else white),
            "margin": margin, "lead": None if lead_w is None else abs(lead_w), "result": r["result"],
            "komi": komi, "black_walls": r.get("black_walls", 10), "white_walls": r.get("white_walls", 10),
            "repetition_draw_count": 0, "plies": r["plies"], "reason": reason,
            "seconds": r.get("b_time", 0.0) + r.get("w_time", 0.0), "rules": r.get("rules")}


def convert_all(raw, schedule, opening_list, komi=-0.5):
    """Converts the raw results of scheduled games (first record per id; unknown ids are ignored)."""
    info = {g[0]: g for g in schedule}
    out, seen = [], set()
    for r in raw:
        gid = r.get("id")
        if gid in seen or gid not in info:
            continue
        seen.add(gid)
        out.append(convert_result(r, info[gid], opening_list[info[gid][3]], komi))
    return out


# -- ratings --------------------------------------------------------------------

def rate_with_boot(names, games, anchor, resamples, seed=0):
    """Point Elo (anchor = 0) and bootstrap samples (resamples x players), both anchored at `anchor`."""
    index = {n: k for k, n in enumerate(names)}
    ui, uj, us, un = elo.build_units(games, index)
    S, N = elo.aggregate(len(names), ui, uj, us, un)
    point = elo.fit_bt(S, N, index[anchor])
    boot = elo.bootstrap(len(names), ui, uj, us, un, index[anchor], resamples, seed, point) if resamples else None
    return point, boot, N, S


def diff_ci(point, boot, i, j):
    """Elo(i) - Elo(j) with a 95% bootstrap interval."""
    d = point[i] - point[j]
    if boot is None:
        return d, float("nan"), float("nan")
    bd = boot[:, i] - boot[:, j]
    return d, float(np.nanpercentile(bd, 2.5)), float(np.nanpercentile(bd, 97.5))


def widest_pairs(pairs, names, point, boot, count):
    """The scheduled pairs whose Elo difference has the widest bootstrap CI."""
    index = {n: k for k, n in enumerate(names)}
    width = []
    for a, b, _, _ in pairs:
        _, lo, hi = diff_ci(point, boot, index[a], index[b])
        width.append((hi - lo, a, b))
    width.sort(reverse=True)
    return width[:count]


def ladder_reference(ladder_dir, ref_name, prefix):
    """Elo relative to ref_name from an arena ladder (e.g. ~/arena/ladder1), point estimates only, with the arena's
    kq-s<N>-v<V> names mapped to <prefix>-s<N>-v<V>."""
    games = load_jsonl(os.path.join(expand(ladder_dir), "results.jsonl"))
    if not games:
        return {}
    names = sorted({n for g in games for n in g["pair"]})
    old_ref = "kq-" + ref_name.split("-", 1)[1]
    if old_ref not in names:
        return {}
    r = elo.rate(names, games, old_ref, resamples=0)
    return {(prefix + n[2:] if n.startswith("kq-") else n): v[0] for n, v in r.items()}


# -- report ---------------------------------------------------------------------

def _f(x, fmt="%.0f"):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt % x


def make_report(cfg, out_dir, bots, pairs, games, resamples=1000, runs=None):
    names = [n for n in bots if any(n in g["pair"] for g in games)]
    if not games:
        return "# KataQuoridor ladder\n\nNo games yet.\n", None
    first_series = cfg["series"][0]["name"]
    anchor = cfg.get("anchor") or next(n for n in names if bots[n]["series"] == first_series)
    ref = cfg.get("reference")
    point, boot, N, S = rate_with_boot(names, games, anchor, resamples)
    idx = {n: k for k, n in enumerate(names)}
    lo = np.nanpercentile(boot, 2.5, axis=0) if boot is not None else np.full(len(names), np.nan)
    hi = np.nanpercentile(boot, 97.5, axis=0) if boot is not None else np.full(len(names), np.nan)
    has_ref = ref in idx
    ladder = {}
    if has_ref and cfg.get("compare_ladder"):
        ladder = ladder_reference(cfg["compare_ladder"], ref, bots[ref]["prefix"])

    n = len(games)
    draws = sum(1 for g in games if g["winner"] is None)
    bwins = sum(1 for g in games if g["winner"] == "b")
    reasons = collections.Counter(g["reason"] if not g["reason"].startswith("draw") else "draw (maxPlies)"
                                  for g in games)
    rules = cfg.get("rules", {})
    L = ["# KataQuoridor ladder", ""]
    L += ["Games: %d, bots: %d, pairs: %d (sparse, see Schedule). Engine: one `katago match` process, %d visits "
          "(1 search thread per bot), rules: komi %s, walls 10/10, maxPlies %s, timeBonusPerPly (λ) %s, repetition "
          "rule off." % (n, len(names), len(pairs), cfg["visits"], rules.get("komi", -0.5),
                         rules.get("maxPlies", 300), rules.get("timeBonusPerPly", 0.0)),
          "",
          "Anchor: %s = 0 Elo%s. CI: 95%% bootstrap over paired-game units (both colours of one opening in one pair), "
          "%d resamples." % (anchor, (" ; also relative to %s" % ref) if has_ref else "", resamples),
          "",
          "Overall: Black (first player) won %.1f%% of decided games, score %.1f%%; draw rate %.1f%%; average plies "
          "%.1f. End reasons: %s." % (
              100 * bwins / max(1, n - draws), 100 * (bwins + 0.5 * draws) / n, 100 * draws / n,
              sum(g["plies"] for g in games) / n, ", ".join("%s %d" % kv for kv in sorted(reasons.items()))),
          ""]
    if runs:
        L += ["Runs: " + "; ".join("%d games in %.1f min (%.2f games/s, %s game threads, batch %s)" % (
            r["games"], r["seconds"] / 60, r["games"] / max(1e-9, r["seconds"]), r.get("numGameThreads"),
            r.get("nnMaxBatchSize")) for r in runs if r["games"] > 0), ""]
    L += ["![Elo vs training samples](elo_vs_samples.png)", ""]

    # Rating table
    hdr = "| Bot | Series | Samples | Visits | Elo | 95% CI |"
    sep = "|---|---|---:|---:|---:|---|"
    if has_ref:
        hdr += " Elo vs %s | 95%% CI |" % ref
        sep += "---:|---|"
    if ladder:
        hdr += " ladder1 (vs same ref) |"
        sep += "---:|"
    hdr += " Games | Score | Black win | Draws | Avg plies |"
    sep += "---:|---:|---:|---:|---:|"
    L += ["## Ratings", "", hdr, sep]
    by_bot = collections.defaultdict(list)
    for g in games:
        for p in g["pair"]:
            by_bot[p].append(g)
    order = sorted(names, key=lambda x: ([s["name"] for s in cfg["series"]].index(bots[x]["series"]),
                                         bots[x]["samples"], bots[x]["visits"]))
    for name in order:
        k = idx[name]
        gs = by_bot[name]
        ng = len(gs)
        sc = sum(elo.game_score(g, name) for g in gs) / ng
        as_black = [g for g in gs if g["black"] == name]
        bw = sum(1 for g in as_black if g["winner"] == "b") / max(1, len(as_black))
        dr = sum(1 for g in gs if g["winner"] is None) / ng
        pl = sum(g["plies"] for g in gs) / ng
        row = "| %s | %s | %d | %d | %s | [%s, %s] |" % (name, bots[name]["series"], bots[name]["samples"],
                                                       bots[name]["visits"], _f(point[k]), _f(lo[k]), _f(hi[k]))
        if has_ref:
            d, dlo, dhi = diff_ci(point, boot, k, idx[ref])
            row += " %s | [%s, %s] |" % (_f(d), _f(dlo), _f(dhi))
        if ladder:
            row += " %s |" % _f(ladder.get(name))
        row += " %d | %.0f%% | %.0f%% | %.0f%% | %.1f |" % (ng, 100 * sc, 100 * bw, 100 * dr, pl)
        L.append(row)
    L.append("")
    L += ["Black win = the bot's win rate when it played Black. ladder1 = the QTP arena ladder (other opponents, "
          "repetition rule off, no λ), relative to the same reference net, point estimates.", ""] if ladder else \
         ["Black win = the bot's win rate when it played Black.", ""]

    # Visit scaling
    vs = cfg.get("visit_scaling")
    if vs:
        L += ["## Visit scaling", "", "| Bot | vs the same net at %d visits | 95%% CI | Games | Score |" % cfg["visits"],
              "|---|---:|---|---:|---:|"]
        for a, b, _, kind in pairs:
            if kind != "visits" or a not in idx or b not in idx:
                continue
            d, dlo, dhi = diff_ci(point, boot, idx[b], idx[a])
            gs = [g for g in games if g["pair"] == [a, b]]
            sc = sum(elo.game_score(g, b) for g in gs) / max(1, len(gs))
            L.append("| %s | %s | [%s, %s] | %d | %.0f%% |" % (b, _f(d, "%+.0f"), _f(dlo, "%+.0f"), _f(dhi, "%+.0f"),
                                                              len(gs), 100 * sc))
        L.append("")

    # Elo per training sample
    L += ["## Elo per training sample", "",
          "Least-squares slope of Elo against log2(samples) at %d visits (Elo per doubling of the training samples), "
          "over all nets of a series and over the nets with at least 3M samples:" % cfg["visits"], ""]
    for s in cfg["series"]:
        pts = [(bots[x]["samples"], point[idx[x]]) for x in order
               if bots[x]["series"] == s["name"] and bots[x]["visits"] == cfg["visits"]]
        parts = []
        for label, sel in (("all", pts), (">= 3M", [p for p in pts if p[0] >= 3e6])):
            if len(sel) >= 2:
                xs = np.log2([p[0] for p in sel])
                ys = np.array([p[1] for p in sel])
                parts.append("%s: %.0f Elo/doubling (%d nets)" % (label, np.polyfit(xs, ys, 1)[0], len(sel)))
        L.append("- %s: %s" % (s["name"], "; ".join(parts) or "—"))
    L.append("")

    # Schedule
    kinds = collections.Counter()
    kind_games = collections.Counter()
    played = collections.Counter(tuple(g["pair"]) for g in games)
    for a, b, _, kind in pairs:
        kinds[kind] += 1
        kind_games[kind] += played[(a, b)]
    L += ["## Schedule", "", "| Link | Pairs | Games played |", "|---|---:|---:|"]
    for kind in kinds:
        L.append("| %s | %d | %d |" % (kind, kinds[kind], kind_games[kind]))
    L += ["", "Games per pair: max(%d, %d · exp(−d²/(2·%.2f²))), d = |ln samples_a − ln samples_b|, rounded up to "
          "even; visit links %s games%s." % (
              cfg["games"]["min"], cfg["games"]["max"], cfg["games"]["sigma_log_samples"],
              vs["games"] if vs else "—",
              "; plus `adapt` games on the pairs with the widest CI" if any(
                  played[(a, b)] > nn for a, b, nn, _ in pairs) else ""), ""]

    # Per pair, from the arena report.
    arena = elo.make_report(out_dir, names=order, anchor=anchor, resamples=0)
    if "## Pairs" in arena:
        L.append(arena[arena.index("## Pairs"):])
    return "\n".join(L) + "\n", (names, point, lo, hi, idx, anchor)


def make_plot(cfg, bots, rated, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names, point, lo, hi, idx, anchor = rated
    ref = cfg.get("reference")
    shift = point[idx[ref]] if ref in idx else 0.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    for ax, logx in zip(axes, (False, True)):
        for si, s in enumerate(cfg["series"]):
            sel = sorted([n for n in names if bots[n]["series"] == s["name"] and bots[n]["visits"] == cfg["visits"]],
                         key=lambda n: bots[n]["samples"])
            if not sel:
                continue
            x = np.array([bots[n]["samples"] for n in sel]) / 1e6
            y = np.array([point[idx[n]] for n in sel]) - shift
            yl = y - (np.array([lo[idx[n]] for n in sel]) - shift)
            yh = (np.array([hi[idx[n]] for n in sel]) - shift) - y
            ax.errorbar(x, y, yerr=[yl, yh], marker="o", ms=4, capsize=2, lw=1.5, color=colors[si % len(colors)],
                        label="%s (v%d)" % (s["name"], cfg["visits"]))
        for n in names:
            if bots[n]["visits"] != cfg["visits"]:
                ax.plot(bots[n]["samples"] / 1e6, point[idx[n]] - shift, marker="^", ms=7, ls="none",
                        color="#6b7280")
                ax.annotate("v%d" % bots[n]["visits"], (bots[n]["samples"] / 1e6, point[idx[n]] - shift),
                            textcoords="offset points", xytext=(5, -3), fontsize=8, color="#6b7280")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel("training samples (millions)" + (", log scale" if logx else ""))
        ax.set_ylabel("Elo (%s = 0)" % (ref if ref in idx else anchor))
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    fig.suptitle("KataQuoridor Elo vs training samples (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# -- commands -------------------------------------------------------------------

class Ladder:
    def __init__(self, cfg, out_dir):
        self.cfg = cfg
        self.out = os.path.abspath(os.path.expanduser(out_dir))
        os.makedirs(os.path.join(self.out, "match", "sgfs"), exist_ok=True)
        self.nets, self.bots = make_roster(cfg)
        self.pairs = make_pairs(cfg, self.nets, self.bots)
        self.extra_path = os.path.join(self.out, "extra_games.json")
        self.extra = json.load(open(self.extra_path)) if os.path.exists(self.extra_path) else {}
        self.schedule = make_schedule(self.pairs, self.extra)
        self.raw_path = os.path.join(self.out, "match", "results.jsonl")

    def katago(self):
        return expand(self.cfg["katago"])

    def openings(self):
        o = self.cfg["openings"]
        count = max(g[3] for g in self.schedule) + 1
        arbiter_argv = [self.katago(), "gtp", "-model", "/dev/null", "-config", expand(self.cfg["gtp_config"]),
                        "-override-config", "debugSkipNeuralNet=true,logAllGTPCommunication=false,logSearchInfo=false,"
                        "logDir=%s" % os.path.join(self.out, "gtp_logs")]

        def factory():
            arb = Arbiter(QTPEngine("arbiter", arbiter_argv, os.path.join(self.out, "match", "arbiter.log"),
                                    cwd=REPO, default_timeout=120))
            arb.start()
            return arb
        return openings.load_or_create(os.path.join(self.out, "openings.json"), factory, count, o["plies"],
                                       o["wall_frac"], o["seed"])

    def write_roster(self):
        with open(os.path.join(self.out, "roster_used.json"), "w") as f:
            json.dump({"engines": [{"name": b["name"], "series": b["series"], "samples": b["samples"],
                                    "visits": b["visits"], "model": b["model"]} for b in self.bots.values()],
                       "pairs": [{"a": a, "b": b, "games": n, "kind": k} for a, b, n, k in self.pairs],
                       "extra_games": self.extra, "config": self.cfg}, f, indent=1)

    def describe(self):
        kinds = collections.Counter(k for _, _, _, k in self.pairs)
        return "%d bots, %d pairs (%s), %d games" % (
            len(self.bots), len(self.pairs), ", ".join("%s %d" % kv for kv in kinds.items()), len(self.schedule))

    def run(self, limit=None):
        opening_list = self.openings()
        self.write_roster()
        done = {r.get("id") for r in load_jsonl(self.raw_path)}
        todo = [g for g in self.schedule if g[0] not in done]
        print("%s; %d done, %d to play%s" % (self.describe(), len(self.schedule) - len(todo), len(todo),
                                             (", this run at most %d" % limit) if limit else ""), flush=True)
        if limit:
            todo = todo[:limit]
        if todo:
            index = {n: k for k, n in enumerate(self.bots)}
            gl = os.path.join(self.out, "match", "games.txt")
            with open(gl, "w") as f:
                f.write(game_list_text(todo, index, opening_list))
            cfg_path = os.path.join(self.out, "match", "match.cfg")
            with open(cfg_path, "w") as f:
                f.write(match_config(self.cfg, self.bots, gl, self.raw_path))
            n0 = len(load_jsonl(self.raw_path))
            t0 = time.time()
            with open(os.path.join(self.out, "match", "match.log"), "a") as log:
                rc = subprocess.call([self.katago(), "match", "-config", cfg_path, "-sgf-output-dir",
                                      os.path.join(self.out, "match", "sgfs")], stdout=log, stderr=subprocess.STDOUT)
            secs = time.time() - t0
            n1 = len(load_jsonl(self.raw_path))
            with open(os.path.join(self.out, "runs.jsonl"), "a") as f:
                f.write(json.dumps({"time": time.strftime("%FT%T"), "games": n1 - n0, "seconds": secs,
                                    "numGameThreads": self.cfg["match"].get("numGameThreads"),
                                    "nnMaxBatchSize": self.cfg["match"].get("nnMaxBatchSize"), "rc": rc}) + "\n")
            print("katago match exited %d: %d games in %.1f min (%.2f games/s)" % (
                rc, n1 - n0, secs / 60, (n1 - n0) / max(secs, 1e-9)), flush=True)
            if rc != 0:
                print("!!! see %s" % os.path.join(self.out, "match", "match.log"), flush=True)
        return self.report()

    def report(self, resamples=1000):
        opening_list = self.openings()
        games = convert_all(load_jsonl(self.raw_path), self.schedule, opening_list,
                            self.cfg.get("rules", {}).get("komi", -0.5))
        bad = [g for g in games if g["rules"] and any(t in g["rules"] for t in ("repetition", "Walls", "komi"))]
        if bad:
            raise SystemExit("games with non-standard rules in the results, e.g. %s: %s" % (bad[0]["id"], bad[0]["rules"]))
        with open(os.path.join(self.out, "results.jsonl"), "w") as f:
            for g in games:
                f.write(json.dumps(g) + "\n")
        runs = load_jsonl(os.path.join(self.out, "runs.jsonl"))
        text, rated = make_report(self.cfg, self.out, self.bots, self.pairs, games, resamples, runs)
        with open(os.path.join(self.out, "report.md"), "w") as f:
            f.write(text)
        if rated:
            make_plot(self.cfg, self.bots, rated, os.path.join(self.out, "elo_vs_samples.png"))
        print("report written to %s" % os.path.join(self.out, "report.md"))
        return 0

    def adapt(self, count, games_each, resamples=300):
        """Adds games_each games to each of the `count` scheduled pairs whose Elo difference has the widest CI."""
        opening_list = self.openings()
        games = convert_all(load_jsonl(self.raw_path), self.schedule, opening_list)
        names = [n for n in self.bots if any(n in g["pair"] for g in games)]
        first = next(n for n in names if self.bots[n]["series"] == self.cfg["series"][0]["name"])
        point, boot, _, _ = rate_with_boot(names, games, first, resamples)
        played = [p for p in self.pairs if p[0] in names and p[1] in names]
        for width, a, b in widest_pairs(played, names, point, boot, count):
            key = "%s:%s" % (a, b)
            self.extra[key] = self.extra.get(key, 0) + games_each + (games_each % 2)
            print("adding %d games to %s (CI width %.0f)" % (games_each, key, width))
        with open(self.extra_path, "w") as f:
            json.dump(self.extra, f, indent=1)
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "run", "report", "adapt"])
    ap.add_argument("--config", default=os.path.join(HERE, "kq_ladder_example.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="run: play at most this many of the remaining games (pilot)")
    ap.add_argument("--pairs", type=int, default=10, help="adapt: number of pairs to extend")
    ap.add_argument("--games", type=int, default=40, help="adapt: games added per pair")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)
    lad = Ladder(load_config(args.config), args.out)
    if args.command == "plan":
        print(lad.describe())
        for a, b, n, kind in lad.pairs:
            print("%-6s %-26s %-26s %3d" % (kind, a, b, n + lad.extra.get("%s:%s" % (a, b), 0)))
        return 0
    if args.command == "run":
        return lad.run(args.limit)
    if args.command == "report":
        return lad.report(args.bootstrap)
    return lad.adapt(args.pairs, args.games)


if __name__ == "__main__":
    sys.exit(main())
