#!/usr/bin/env python3
"""Deep-search hints for self-play (docs/BlindSpots.md): positions where a deep search finds a better move than the
self-play search does.

    python -m quoridor_arena.hintminer mine --dirs SELFPLAY_DIR... --model MODEL --out-dir HINT_DIR

mine:   takes positions from the newest self-play's standard games (the opening-tree nodes and the positions before a
        value swing), searches each with the self-play settings (1000 visits, self-play root params, no noise) and
        with a deep search (10000+ visits). When the deep best move differs from the fresh one and is better by a margin
        (winrate 0.05, or a lead margin in decided positions) it writes a hint pose: the moves from the start,
        `hintLoc` = the deep move, a weight. Output: HINT_DIR/mine_<time>_<net>.hintposes.txt, one JSON line per hint
        (the format of KataGo's hintPosesDir: Sgf::PositionSample::toJsonLine; the board field is not used, the
        position is its move list from the empty board). Old hints expire: the hint files of the last --keep-runs
        runs are kept, and before mining a hint that the current net's fresh search already plays is dropped.
check:  re-searches the hints in a directory with a net and reports which ones the fresh search finds.
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

from quoridor_arena import blindspot as bs

HINT_SUFFIX = ".hintposes.txt"


def hint_line(moves, hint, weight, metadata):
    """One hint-pose JSON line for Sgf::PositionSample::ofJsonLine. `moves` are 9x9 moves from the empty board."""
    return json.dumps({
        "xSize": 17, "ySize": 17, "board": "", "nextPla": "B" if len(moves) % 2 == 0 else "W",
        "moveLocs": list(moves), "movePlas": ["B" if i % 2 == 0 else "W" for i in range(len(moves))],
        "initialTurnNumber": 0, "hintLoc": hint, "weight": weight, "metadata": json.dumps(metadata, sort_keys=True),
    }, sort_keys=True)


def parse_hint_line(line):
    d = json.loads(line)
    return {"moves": d["moveLocs"], "hint": d["hintLoc"], "weight": d.get("weight", 1.0),
            "metadata": json.loads(d["metadata"]) if d.get("metadata") else {}}


def read_hint_file(path):
    with open(path) as f:
        return [parse_hint_line(l) for l in f if l.strip()]


def write_hint_file(path, hints):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for h in hints:
            f.write(hint_line(h["moves"], h["hint"], h["weight"], h["metadata"]) + "\n")
    os.replace(tmp, path)


def hint_files(out_dir):
    return sorted(glob.glob(os.path.join(out_dir, "mine_*" + HINT_SUFFIX)))


def margin_of(deep_children, deep_best, fresh_best, symmetric, args):
    """How much better the deep best move is than the fresh search's move, by the deep search's own numbers.

    Returns (kind, margin, fresh_stats) with kind 'wr' (winrate margin >= args.margin), 'lead' (the position is decided
    and the lead margin >= args.lead_margin) or None. `deep_children` are all the deep search's children, best first."""
    by_move = {c["move"]: c for c in deep_children}
    fresh = by_move.get(fresh_best) or (by_move.get(bs.mirror(fresh_best)) if symmetric else None)
    if fresh is None:
        return None, 0.0, None
    best = deep_children[0]
    wr_margin = best["wr"] - fresh["wr"]
    if wr_margin >= args.margin:
        return "wr", wr_margin, fresh
    decided = abs(best["wr"] - 0.5) >= args.decided - 0.5 and abs(fresh["wr"] - 0.5) >= args.decided - 0.5
    if decided and best["lead"] - fresh["lead"] >= args.lead_margin:
        return "lead", best["lead"] - fresh["lead"], fresh
    return None, wr_margin, fresh


def make_hint(pos, fresh_children, deep_children, args, net):
    """A hint record for a position, or None if the fresh search is fine."""
    sym = pos["symmetric"]
    fresh_best, deep_best = fresh_children[0]["move"], deep_children[0]["move"]
    if bs.same_move(fresh_best, deep_best, sym):
        return None
    label = bs.make_label(deep_children, args.deep_visits, net, args.min_child_visits)
    if not label["clear"]:                    # the deep search is not sure about its own best move
        return None
    kind, margin, fresh = margin_of(deep_children, deep_best, fresh_best, sym, args)
    if kind is None:
        return None
    best = deep_children[0]
    # weight: 1 at the margin, up to 5 for a margin of five times that (a lead margin counts as the minimum)
    weight = round(min(5.0, max(1.0, margin / args.margin)) if kind == "wr" else 1.0, 2)
    meta = {"net": net, "kind": kind, "margin": round(margin, 3), "fresh_best": fresh_best, "fresh_visits": args.fresh_visits,
            "deep_visits": args.deep_visits, "deep_wr": round(best["wr"], 3), "deep_lead": round(best["lead"], 2),
            "deep_share": label["share"], "fresh_move_deep_wr": round(fresh["wr"], 3), "fresh_move_deep_visits": fresh["visits"],
            "ply": pos["ply"], "group": pos["group"], "mined": time.strftime("%F")}
    return {"moves": pos["moves"].split(), "hint": deep_best, "weight": weight, "metadata": meta, "id": pos["id"]}


def search_phase(positions, visits, params, args, label):
    factory = lambda w: bs.make_engine(args.model, args.katago, args.config, bs.PARAM_SETS[params], args.threads if params == "gtp" else 1,
                                       args.cache_bits, "%s%d" % (label, w), args.log_dir)
    work = lambda eng, pos: {"id": pos["id"], "children": bs.run_search(eng, pos["moves"].split(), visits)}
    out = bs.run_parallel(positions, args.engines, factory, work, None, label)
    return {r["id"]: r["children"] for r in out}


def prune_old(out_dir, args, net):
    """Drop the oldest hint files (keep --keep-runs - 1 for the new run) and the hints the current net already finds."""
    files = hint_files(out_dir)
    for f in files[:max(0, len(files) - (args.keep_runs - 1))]:
        print("expired: %s" % os.path.basename(f), file=sys.stderr)
        os.remove(f)
    files = hint_files(out_dir)
    old = []
    for f in files:
        for h in read_hint_file(f):
            old.append((f, h))
    if not old or args.no_recheck:
        return {tuple(h["moves"]) for _, h in old}
    pos = []
    for i, (f, h) in enumerate(old):
        key, sym = bs.state_key(h["moves"])
        pos.append({"id": str(i), "moves": " ".join(h["moves"]), "symmetric": sym})
    fresh = search_phase(pos, args.fresh_visits, "selfplay-nonoise", args, "recheck")
    kept = collections.defaultdict(list)
    dropped = 0
    for i, (f, h) in enumerate(old):
        ch = fresh.get(str(i))
        solved = ch and bs.same_move(ch[0]["move"], h["hint"], pos[i]["symmetric"])
        if solved:
            dropped += 1
        else:
            kept[f].append(h)
    for f in files:
        if kept[f]:
            write_hint_file(f, kept[f])
        else:
            os.remove(f)
    print("old hints: %d re-checked, %d now found by the fresh search (dropped), %d kept" % (len(old), dropped, len(old) - dropped))
    return {tuple(h["moves"]) for hs in kept.values() for h in hs}


def cmd_mine(args):
    os.makedirs(args.out_dir, exist_ok=True)
    net = args.net or os.path.basename(os.path.dirname(os.path.abspath(args.model)))
    have = prune_old(args.out_dir, args, net)
    bargs = bs.build_args(n_opening=args.n_opening, n_swing=args.n_swing, n_random=0, seed=args.seed, opening_min_games=args.min_games)
    positions, info = bs.build_positions(bs.iter_games(args.dirs), bargs)
    positions = [p for p in positions if tuple(p["moves"].split()) not in have]
    print("%d candidate positions from %d standard games" % (len(positions), info["games"]))
    t0 = time.time()
    fresh = search_phase(positions, args.fresh_visits, "selfplay-nonoise", args, "fresh")
    deep = search_phase(positions, args.deep_visits, "gtp", args, "deep")
    hints, differ = [], 0
    for p in positions:
        f, d = fresh.get(p["id"]), deep.get(p["id"])
        if not f or not d:
            continue
        differ += not bs.same_move(f[0]["move"], d[0]["move"], p["symmetric"])
        h = make_hint(p, f, d, args, net)
        if h:
            hints.append(h)
    path = os.path.join(args.out_dir, "mine_%s_%s%s" % (time.strftime("%Y%m%d-%H%M%S"), net, HINT_SUFFIX))
    write_hint_file(path, hints)
    print("%d positions searched in %.0f s; fresh and deep best differ in %d; %d hints -> %s" % (
        len(positions), time.time() - t0, differ, len(hints), path))
    for h in sorted(hints, key=lambda h: -h["metadata"]["margin"])[:args.show]:
        m = h["metadata"]
        print("  %-3d plies  %-40s fresh %-5s -> hint %-5s %s margin %.3f (deep wr %.3f, share %.2f)  w=%.1f" % (
            len(h["moves"]), " ".join(h["moves"][-8:]) if len(h["moves"]) > 8 else " ".join(h["moves"]), m["fresh_best"], h["hint"],
            m["kind"], m["margin"], m["deep_wr"], m["deep_share"], h["weight"]))


def cmd_check(args):
    hints = [h for f in hint_files(args.hint_dir) for h in read_hint_file(f)]
    pos = []
    for i, h in enumerate(hints):
        pos.append({"id": str(i), "moves": " ".join(h["moves"]), "symmetric": bs.state_key(h["moves"])[1]})
    fresh = search_phase(pos, args.fresh_visits, args.params, args, "check")
    found = sum(bs.same_move(fresh[str(i)][0]["move"], h["hint"], pos[i]["symmetric"]) for i, h in enumerate(hints))
    print("%d hints, %d found by a fresh %d-visit search (%s)" % (len(hints), found, args.fresh_visits, args.params))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        bs.add_engine_args(p)
        p.add_argument("--fresh-visits", type=int, default=1000)
        p.add_argument("--min-child-visits", type=int, default=100)
        p.set_defaults(threads=16)

    m = sub.add_parser("mine", help="mine hint poses with a net")
    m.add_argument("--dirs", nargs="+", required=True, help="the newest self-play model dirs (with sgfs/)")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--deep-visits", type=int, default=10000)
    m.add_argument("--n-opening", type=int, default=250)
    m.add_argument("--n-swing", type=int, default=150)
    m.add_argument("--min-games", type=int, default=10, help="games that must reach an opening-tree node")
    m.add_argument("--margin", type=float, default=0.05, help="deep winrate of the deep move minus that of the fresh move")
    m.add_argument("--lead-margin", type=float, default=2.0, help="the same for the lead, in decided positions")
    m.add_argument("--decided", type=float, default=0.95, help="winrate beyond which (either way) a position counts as decided")
    m.add_argument("--keep-runs", type=int, default=4, help="hint files of this many mining runs stay")
    m.add_argument("--no-recheck", action="store_true", help="do not drop old hints that the fresh search now finds")
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--show", type=int, default=20)
    common(m)
    m.set_defaults(fn=cmd_mine, engines=2)
    c = sub.add_parser("check", help="which hints does a fresh search find")
    c.add_argument("--hint-dir", required=True)
    c.add_argument("--params", choices=list(bs.PARAM_SETS), default="selfplay-nonoise")
    common(c)
    c.set_defaults(fn=cmd_check, engines=2)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
