#!/usr/bin/env python3
"""Plots for watching a Quoridor Duel training run (docs/RunPlots.md).

    python runplots.py extract  --dirs RUN/selfplay/NAME-s* --out openings.jsonl
    python runplots.py openings --data openings.jsonl --out openings.png [--line "TITLE=MOVES"]... [--event ROWS=LABEL]
    python runplots.py score    --data openings.jsonl --out black_score.png [--line "TITLE=MOVES"]...
    python runplots.py selfplay --dirs RUN/selfplay/NAME-s* --out selfplay.png [--compare OTHER-s*] [--json out.json]
    python runplots.py loss     --traindir RUN/train/NAME --out loss.png [--ignore-first 1e6]

extract:  per self-play dir, the standard games (komi -0.5, 10 / 10 walls, gtype=normal, no compensated or random
          komi): startTurnIdx (policy-init plies), result and the first 16 plies in 9x9 notation, mirror-canonical.
openings: share of the games without policy-init plies that start with each line, against the self-play rows the
          generating net had seen (-d<rows> in the dir name); neighbouring nets are pooled to >= --pool games per
          point. A board diagram of each line is drawn beside its panel.
score:    Black's score in the same games, per line (points with >= --min-games games).
selfplay: per generating net: komi -0.5 Black score by policy-init length, compensated komi (mean komi, Black score),
          random-komi Black score, draw rate, game length and the plies after init until the game is decided.
          --compare adds another run's nets as dotted reference levels (e.g. the end of an earlier run).
loss:     train (moving average) vs val: policy KL to the target (p0loss - ptentr), p0loss, top-1, value loss, soft
          policy KL, total loss; the y range ignores the first --ignore-first samples.

Lines are in 9x9 notation (e8 pawn, d8v / c7h walls), mirror-canonical: the first move that differs from its
left-right mirror is the smaller one (a6h, not h6h). --event marks a change (e.g. a net size switch) with a dashed
line: rows for the self-play plots, samples for the loss plot.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.expanduser("~/sync/HW/Quoridor/KataQuoridor/python"))
from quoridor_arena.openings_from_selfplay import MOVE, mirror, to_9x9  # noqa: E402

MOVE_C = re.compile(r";([BW])\[([a-z]*)\](?:C\[([^\]]*)\])?")
INIT_BINS = [(0, 0), (1, 6), (7, 16), (17, 999)]
INK, MUTED, GRID, DOT, BG = "#1f2328", "#6e7781", "#d8dee4", "#2f6fdb", "#f6f8fa"
COLORS = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#d55e00", "#56b4e9"]
M6 = "e8 e2 e7 e3 e6 e4"
DEFAULT_LINES = [("Race to the centre (6 pawn moves)", M6), ("First move: wall d8v", "d8v"),
                 ("Straight race", "e8 e2")]


def rows_of(name):
    m = re.search(r"-d(\d+)", name)
    return int(m.group(1)) if m else 0


def samples_of(name):
    return int(re.search(r"-s(\d+)", name).group(1))


def result_of(line):
    r = re.search(r"RE\[([^\]]*)\]", line).group(1)[:1]
    return "D" if r in "0V" else r


def is_standard(line):
    return ("KM[-0.5]" in line and "WB[10]" in line and "WW[10]" in line and "gtype=normal" in line
            and "komiComp=" not in line and "komiRandom=1" not in line)


def start_turn(line):
    m = re.search(r"startTurnIdx=(\d+)", line)
    return int(m.group(1)) if m else 0


def game_lines(d):
    for f in glob.glob(os.path.join(d, "sgfs", "*.sgfs")):
        for line in open(f):
            if line.startswith("(;"):
                yield line


# --- extract ----------------------------------------------------------------------------------------------------------
def canon(moves):
    for m in moves:
        mm = mirror(m)
        if mm != m:
            return moves if m < mm else [mirror(x) for x in moves]
    return moves


def extract_one(d):
    games = []
    for line in game_lines(d):
        if not is_standard(line):
            continue
        mv = [to_9x9(x[1]) for x in MOVE.findall(line)[:16]]
        games.append([start_turn(line), "repetitionDrawCount" in line, result_of(line), " ".join(canon(mv))])
    name = os.path.basename(d.rstrip("/"))
    return {"dir": name, "samples": samples_of(name), "rows": rows_of(name), "games": games}


def cmd_extract(a):
    dirs = sorted(a.dirs, key=lambda d: rows_of(os.path.basename(d.rstrip("/"))))
    with ProcessPoolExecutor(a.procs) as ex, open(a.out, "w") as out:
        for r in ex.map(extract_one, dirs):
            out.write(json.dumps(r) + "\n")


# --- openings / score -------------------------------------------------------------------------------------------------
def pooled(data, pool):
    """Games without policy-init plies, nets pooled (in rows order) to >= pool games per point -> (x, [[(moves, result)]])."""
    rs = [json.loads(l) for l in open(data)]
    rs = [r for r in rs if sum(1 for g in r["games"] if g[0] == 0) >= 50]
    rs.sort(key=lambda r: rows_of(r["dir"]))
    bins, cur = [], []
    for r in rs:
        cur.append(r)
        if sum(sum(1 for g in x["games"] if g[0] == 0) for x in cur) >= pool:
            bins.append(cur)
            cur = []
    if cur:
        bins.append(cur)
    xs = [sum(rows_of(r["dir"]) for r in b) / len(b) / 1e6 for b in bins]
    sets = [[(g[3].split(), g[2]) for r in b for g in r["games"] if g[0] == 0] for b in bins]
    return xs, sets


def parse_lines(specs):
    if not specs:
        return DEFAULT_LINES
    out = []
    for s in specs:
        t, _, m = s.partition("=")
        out.append((t.strip(), m.strip()) if m else (s.strip(), s.strip()))
    return out


def parse_events(specs):
    out = []
    for s in specs or []:
        x, _, lab = s.partition("=")
        out.append((float(x) / 1e6 if float(x) > 1000 else float(x), lab))
    return out


def draw_board(ax, line):
    from matplotlib.patches import Circle, Rectangle
    mv = line.split()
    ax.set_xlim(-0.6, 9.6); ax.set_ylim(-0.6, 9.6); ax.set_aspect("equal"); ax.axis("off")
    for x in range(9):
        for y in range(9):
            ax.add_patch(Rectangle((x + 0.06, y + 0.06), 0.88, 0.88, color="#eef1f4", lw=0))
    for i, c in enumerate("abcdefghi"):
        ax.text(i + 0.5, -0.35, c, ha="center", va="center", fontsize=7, color=MUTED)
        ax.text(-0.35, i + 0.5, str(i + 1), ha="center", va="center", fontsize=7, color=MUTED)
    ax.add_patch(Circle((4.5, 8.5), 0.3, fill=False, ec="#9aa4ae", lw=1, ls=":"))
    ax.add_patch(Circle((4.5, 0.5), 0.3, fill=False, ec="#9aa4ae", lw=1, ls=":"))
    for n, m in enumerate(mv, 1):
        fc, tc = ("#1f2328", "white") if n % 2 == 1 else ("white", "#1f2328")
        c, r = ord(m[0]) - 97, int(m[1]) - 1
        if len(m) == 2:
            ax.add_patch(Circle((c + 0.5, r + 0.5), 0.36, fc=fc, ec="#1f2328", lw=1.2, zorder=3))
            ax.text(c + 0.5, r + 0.5, str(n), ha="center", va="center", fontsize=8, color=tc, zorder=4, weight="bold")
            continue
        if m[2] == "h":
            ax.add_patch(Rectangle((c + 0.06, r + 0.92), 1.88, 0.16, fc=fc, ec="#1f2328", lw=1, zorder=3))
        else:
            ax.add_patch(Rectangle((c + 0.92, r + 0.06), 0.16, 1.88, fc=fc, ec="#1f2328", lw=1, zorder=3))
        ax.add_patch(Circle((c + 1, r + 1), 0.27, fc=fc, ec="#1f2328", lw=1, zorder=4))
        ax.text(c + 1, r + 1, str(n), ha="center", va="center", fontsize=7, color=tc, zorder=5, weight="bold")


def mark_events(ax, events):
    for x0, lab in events:
        ax.axvline(x0, color="#afb8c1", lw=1, ls="--", zorder=1)
        ax.text(x0, 0.02, " " + lab, transform=ax.get_xaxis_transform(), fontsize=7, color=MUTED, va="bottom")


def cmd_openings(a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, sets = pooled(a.data, a.pool)
    lines = parse_lines(a.line)
    events = parse_events(a.event)
    cols = 3
    rows = (len(lines) + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 6.2, rows * 3.0))
    gs = fig.add_gridspec(rows, cols * 2, width_ratios=[1.35, 1] * cols, wspace=0.15, hspace=0.55)
    xmax = max(xs) * 1.05
    for i, (title, line) in enumerate(lines):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, 2 * c]); bx = fig.add_subplot(gs[r, 2 * c + 1])
        mv = line.split(); k = len(mv)
        f = [sum(1 for g, _ in s if g[:k] == mv) / len(s) for s in sets]
        ax.set_facecolor(BG)
        mark_events(ax, events)
        ax.plot(xs, [100 * v for v in f], "-o", ms=3.5, lw=1.2, color=DOT, zorder=3)
        ax.set_ylim(0, max(5, 105 * max(f))); ax.set_xlim(0, xmax)
        ax.set_title(title, fontsize=10.5, color=INK, weight="bold", loc="left")
        ax.set_ylabel("% of games", fontsize=8, color=MUTED)
        ax.set_xlabel("self-play data rows (M)", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        draw_board(bx, line)
    fig.suptitle("%s standard-game openings (komi -0.5, 10/10 walls, no policy-init plies; mirror images merged)\n"
                 "share of the self-play games that start with the line, nets pooled to >= %d games per point (%d points)"
                 % (a.title, a.pool, len(xs)), fontsize=10, color=INK, y=0.995)
    fig.savefig(a.out, dpi=130, bbox_inches="tight", facecolor="white")


def cmd_score(a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, sets = pooled(a.data, a.pool)
    sc = lambda rs: sum(1 if r == "B" else 0.5 if r == "D" else 0 for r in rs) / len(rs)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, [100 * sc([r for _, r in s]) for s in sets], "-o", ms=3, lw=2.2, c=MUTED, label="all no-init games")
    for (title, line), col in zip(parse_lines(a.line), COLORS):
        mv = line.split(); k = len(mv); ys = []
        for s in sets:
            m = [r for g, r in s if g[:k] == mv]
            ys.append(100 * sc(m) if len(m) >= a.min_games else float("nan"))
        ax.plot(xs, ys, "-o", ms=3, lw=1.6, c=col, label=title)
    ax.set_ylim(30, 102); ax.set_xlim(0, max(xs) * 1.05)
    mark_events(ax, parse_events(a.event))
    ax.axhline(50, c="#afb8c1", lw=0.8)
    ax.set_ylabel("Black score (%)"); ax.set_xlabel("self-play data rows (M)"); ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("%s self-play, komi -0.5, no policy init: Black score by opening line (points with >= %d games)"
                 % (a.title, a.min_games), fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(a.out, dpi=120)


# --- selfplay ---------------------------------------------------------------------------------------------------------
def selfplay_one(d):
    name = os.path.basename(d.rstrip("/"))
    res = Counter(); std = {b: Counter() for b in INIT_BINS}; comp = Counter(); rnd = Counter(); compk = []
    plies = []; decided = []; n = 0
    for line in game_lines(d):
        n += 1
        r = result_of(line); res[r] += 1
        mv = MOVE_C.findall(line); plies.append(len(mv))
        si = start_turn(line)
        if not ("gtype=normal" in line and "WB[10]" in line and "WW[10]" in line):
            continue
        km = re.search(r"KM\[([-\d.]+)\]", line); km = float(km.group(1)) if km else None
        if "komiComp=" in line:
            comp[r] += 1; compk.append(km); continue
        if "komiRandom=1" in line:
            rnd[r] += 1; continue
        if km != -0.5:
            continue
        for b in INIT_BINS:
            if b[0] <= si <= b[1]:
                std[b][r] += 1
        # first ply after init from which every searched move has training weight <= 0.1: the game is decided
        d0 = None
        for i, (_, _, cm) in enumerate(mv):
            if i < si or not cm:
                continue
            w = re.search(r"weight=([\d.]+)", cm)
            if w and float(w.group(1)) <= 0.1:
                d0 = i if d0 is None else d0
            else:
                d0 = None
        if d0 is not None:
            decided.append(d0 - si)
    sc = lambda x: (x["B"] + 0.5 * x["D"]) / max(1, sum(x.values()))
    out = {"dir": name, "samples": samples_of(name), "rows": rows_of(name), "games": n,
           "plies": st.mean(plies) if plies else None, "draw": res["D"] / max(1, n),
           "comp_n": sum(comp.values()), "comp_black": sc(comp), "comp_komi": st.mean(compk) if compk else None,
           "rnd_n": sum(rnd.values()), "rnd_black": sc(rnd), "decided_after": st.median(decided) if decided else None}
    for b in INIT_BINS:
        out["std%d_%d" % b] = sc(std[b]); out["n%d_%d" % b] = sum(std[b].values())
    return out


def cmd_selfplay(a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with ProcessPoolExecutor(a.procs) as ex:
        rows = [r for r in ex.map(selfplay_one, a.dirs) if r["games"] >= a.min_games]
        ref = [r for r in ex.map(selfplay_one, a.compare or []) if r["games"] >= a.min_games]
    rows.sort(key=lambda r: r["rows"])
    if a.json:
        json.dump({"run": rows, "compare": ref}, open(a.json, "w"), indent=0)
    x = [r["rows"] / 1e6 for r in rows]
    events = parse_events(a.event)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))

    def deco(p, title):
        mark_events(p, events)
        p.set_title(title, fontsize=10, loc="left"); p.grid(alpha=0.25)
        p.set_xlabel("self-play data rows (M)", fontsize=8)

    p = ax[0, 0]
    for (lo, hi), lab, col in zip(INIT_BINS, ["no policy init", "init 1-6 plies", "init 7-16", "init 17+"], COLORS):
        k = "std%d_%d" % (lo, hi)
        p.plot(x, [r[k] for r in rows], "-o", ms=3, c=col, label=lab)
        if ref:
            p.axhline(st.mean(r[k] for r in ref), c=col, ls=":", lw=1.2)
    p.set_ylim(0.35, 1.02); p.legend(fontsize=7, loc="upper left")
    deco(p, "komi -0.5: Black score by policy-init length" + ("\n(dotted: --compare nets)" if ref else ""))
    p = ax[0, 1]
    p.plot(x, [r["comp_komi"] for r in rows], "-o", ms=3, c=COLORS[0])
    deco(p, "compensated komi (after policy init): mean komi")
    p = ax[0, 2]
    p.plot(x, [r["comp_black"] for r in rows], "-o", ms=3, c=COLORS[0], label="compensated komi")
    p.plot(x, [r["rnd_black"] for r in rows], "-o", ms=3, c=COLORS[1], label="random komi")
    p.axhline(0.5, c="#999", lw=0.8); p.legend(fontsize=7)
    deco(p, "Black score, compensated / random komi")
    p = ax[1, 0]
    p.plot(x, [r["draw"] for r in rows], "-o", ms=3, c=COLORS[3])
    deco(p, "draw rate (all games)")
    p = ax[1, 1]
    p.plot(x, [r["plies"] for r in rows], "-o", ms=3, c=COLORS[2], label="game length (all games)")
    p.plot(x, [r["decided_after"] for r in rows], "-o", ms=3, c=COLORS[0],
           label="plies after init until decided (komi -0.5; weight <= 0.1 from there on)")
    p.legend(fontsize=7)
    deco(p, "game length (plies)")
    p = ax[1, 2]
    p.plot(x, [r["games"] for r in rows], "-o", ms=3, c=MUTED)
    deco(p, "games per generating net")
    fig.suptitle("%s self-play per generating net" % a.title, fontsize=12)
    fig.tight_layout(); fig.savefig(a.out, dpi=105)


# --- loss -------------------------------------------------------------------------------------------------------------
def cmd_loss(a):
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V = [json.loads(l) for l in open(os.path.join(a.traindir, "metrics_val.json"))]
    T = [json.loads(l) for l in open(os.path.join(a.traindir, "metrics_train.json"))]
    vx = np.array([d.get("nsamp_train", d["nsamp"]) for d in V]) / 1e6
    tx = np.array([d["nsamp"] for d in T]) / 1e6
    panels = [("policy KL to target (p0loss - ptentr)", lambda d: d["p0loss"] - d["ptentr"]),
              ("policy loss p0", lambda d: d["p0loss"]),
              ("policy top-1 accuracy", lambda d: d["pacc1"]),
              ("value loss (win/loss)", lambda d: d["vloss"]),
              ("soft policy KL (p0softloss - ptsoftentr)", lambda d: d["p0softloss"] - d["ptsoftentr"]),
              ("total loss", lambda d: d["loss"])]
    k = a.smooth
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    lo_x = a.ignore_first / 1e6
    for p, (title, f) in zip(ax.flat, panels):
        ty = np.array([f(d) for d in T]); vy = np.array([f(d) for d in V])
        sm = np.convolve(ty, np.ones(k) / k, "valid")
        p.plot(tx[k - 1:], sm, c="#9aa", lw=1.4, label="train (moving average)")
        p.plot(vx, vy, "o-", c="#d55e00", ms=4, lw=1.6, label="val")
        for x0, _ in parse_events(a.event):
            p.axvline(x0, c="#888", ls="--", lw=1)
        ys = list(vy[vx >= lo_x]) + list(sm[tx[k - 1:] >= lo_x])
        if ys:
            lo, hi = min(ys), max(ys); pad = (hi - lo) * 0.08 + 1e-6
            p.set_ylim(lo - pad, hi + pad)
        p.set_xlim(0, max(vx.max() if len(vx) else 0, tx.max()) * 1.02)
        p.set_title(title, fontsize=10); p.set_xlabel("training samples (M)"); p.grid(alpha=0.25)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("%s: training vs val (y range ignores the first %.1fM samples)" % (a.title, lo_x))
    fig.tight_layout(); fig.savefig(a.out, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--dirs", nargs="+", required=True); e.add_argument("--out", required=True)
    e.add_argument("--procs", type=int, default=8)
    for name in ("openings", "score"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True); p.add_argument("--out", required=True)
        p.add_argument("--line", action="append", help='"TITLE=MOVES" (9x9 notation, mirror-canonical), repeatable')
        p.add_argument("--event", action="append", help='"ROWS=LABEL", e.g. "4903432=b5"')
        p.add_argument("--pool", type=int, default=600); p.add_argument("--title", default="")
        p.add_argument("--min-games", type=int, default=40)
    s = sub.add_parser("selfplay")
    s.add_argument("--dirs", nargs="+", required=True); s.add_argument("--compare", nargs="*")
    s.add_argument("--out", required=True); s.add_argument("--json"); s.add_argument("--event", action="append")
    s.add_argument("--min-games", type=int, default=1500); s.add_argument("--procs", type=int, default=8)
    s.add_argument("--title", default="")
    l = sub.add_parser("loss")
    l.add_argument("--traindir", required=True); l.add_argument("--out", required=True)
    l.add_argument("--ignore-first", type=float, default=1e6); l.add_argument("--smooth", type=int, default=8)
    l.add_argument("--event", action="append", help='"SAMPLES=LABEL"'); l.add_argument("--title", default="")
    a = ap.parse_args()
    {"extract": cmd_extract, "openings": cmd_openings, "score": cmd_score, "selfplay": cmd_selfplay,
     "loss": cmd_loss}[a.cmd](a)


if __name__ == "__main__":
    main()
