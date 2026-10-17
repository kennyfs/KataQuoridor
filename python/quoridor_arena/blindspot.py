#!/usr/bin/env python3
"""Blind-spot suite: how well does a net, at self-play search size, find what a deep search finds? (docs/BlindSpots.md)

    python -m quoridor_arena.blindspot build  --dirs SELFPLAY_DIR... --out positions.jsonl
    python -m quoridor_arena.blindspot label  --positions positions.jsonl --model MODEL --out suite.jsonl [--visits 50000]
    python -m quoridor_arena.blindspot eval   --suite suite.jsonl --model MODEL --out-dir OUT [--visits 256,1000]
    python -m quoridor_arena.blindspot report OUT/eval_*.json...

build:  positions from standard self-play games (komi -0.5, 10 / 10 walls, gtype=normal): the opening-tree nodes of the
        common lines (plies 2-24, weighted by how many games reach them), a random sample (plies 4-80) and positions
        where the self-play search value swung by more than 0.3 within a few plies; deduped by position and mirror.
label:  a deep search per position (default 50000 visits) with one net: best move, its visit share, winrate and lead,
        and the winrate / lead / LCB of every child with enough visits. Positions whose best move beats the best
        other child by 0.05 winrate (or 0.5 lead in a decided position) are `clear`; the rest are flagged.
eval:   a fresh search of V visits per clear position; top-move agreement with the label and the regret in label
        winrate, overall and by group and by ply range.

Moves are in 9x9 notation (e8 pawn, e2h / e2v walls), as in GTP. The engine is `gtp` with cpp/configs/gtp_quoridor.cfg:
komi -0.5, 10 / 10 walls, no repetition rule.
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from quoridor_arena.openings_from_selfplay import mirror, to_9x9
from quoridor_arena.qtp import QTPEngine

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SUITE_VERSION = 1

# --- self-play search parameters (selfplay_quoridor_v2.cfg), for the GTP engine as -override-config -----------------
# Only the keys that differ from gtp_quoridor.cfg or are missing there. rootNoiseEnabled and nnRandomize are the
# difference between a "self-play like" search and a GTP search that is repeatable.
SELFPLAY_PARAMS = {
    "numSearchThreads": 1, "rootDesiredPerChildVisitsCoeff": 4, "rootPolicyTemperatureEarly": 1.6,
    "rootPolicyTemperature": 1.5, "chosenMoveTemperatureHalflife": 19, "useLcbForSelection": "true", "lcbStdevs": 5.0,
    "minVisitPropForLCB": 0.15, "staticScoreUtilityFactor": 0.0, "dynamicScoreUtilityFactor": 0.30,
    "rootEndingBonusPoints": 0.5, "rootPruneUselessMoves": "true", "cpuctExploration": 1.1, "cpuctExplorationLog": 0.0,
    "fpuReductionMax": 0.2, "rootFpuReductionMax": 0.0, "valueWeightExponent": 0.5, "useGraphSearch": "true",
    "fpuParentWeightByVisitedPolicy": "true", "fpuParentWeightByVisitedPolicyPow": 2.0, "policyOptimism": 1.0,
    "rootPolicyOptimism": 0.0, "nnRandomize": "true",
}
SELFPLAY_NOISE = {"rootNoiseEnabled": "true", "rootDirichletNoiseTotalConcentration": 10.83, "rootDirichletNoiseWeight": 0.25}
PARAM_SETS = {
    "gtp": {},                                                                     # gtp_quoridor.cfg as it is
    "selfplay": dict(SELFPLAY_PARAMS, **SELFPLAY_NOISE),                           # what a self-play game searches with
    "selfplay-nonoise": dict(SELFPLAY_PARAMS, rootNoiseEnabled="false"),           # ... without the root noise
}

PLY_BINS = [(0, 8), (9, 16), (17, 24), (25, 40), (41, 80), (81, 10 ** 6)]
GROUPS = ["opening", "swing", "random"]       # priority of the primary group of a position in several groups
MIN_FULL_VISITS = 500                         # a self-play comment with fewer visits is a cheap search


# --- notation, positions, mirror ---------------------------------------------------------------------------------
def mirror_all(moves):
    return [mirror(m) for m in moves]


def state_key(moves):
    """Canonical key of the position after `moves` (alternating, Black first): pawns, walls with their owner counts,
    side to move. Transpositions and the left-right mirror give the same key. Returns (key, is_symmetric)."""
    def build(ms):
        pawns = {"b": "e9", "w": "e1"}
        walls = set()
        used = {"b": 0, "w": 0}
        side = "b"
        for m in ms:
            if len(m) == 2:
                pawns[side] = m
            else:
                walls.add(m)
                used[side] += 1
            side = "w" if side == "b" else "b"
        return "%s|%s|%s|%d,%d|%s" % (pawns["b"], pawns["w"], ",".join(sorted(walls)), used["b"], used["w"], side)
    a, b = build(moves), build(mirror_all(moves))
    return min(a, b), a == b


def position_id(key):
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def to_move_of(moves):
    return "b" if len(moves) % 2 == 0 else "w"


def ply_bin(ply):
    for lo, hi in PLY_BINS:
        if lo <= ply <= hi:
            return "%d-%d" % (lo, hi) if hi < 10 ** 6 else "%d+" % lo
    return "?"


# --- reading self-play SGFs --------------------------------------------------------------------------------------
MOVE_RE = re.compile(r";([BW])\[([a-q]{2})\](?:C\[([^\]]*)\])?")


def _prop(key, s):
    m = re.search(r"\b%s\[([^\]]*)\]" % key, s)
    return m.group(1) if m else None


def parse_comment(c):
    """Self-play move comment 'wWin wLoss noResult score v=N weight=W lead=L' (White's view), or None."""
    if not c:
        return None
    t = c.split()
    try:
        d = {"white": float(t[0]) + 0.5 * float(t[2])}
    except (ValueError, IndexError):
        return None
    for x in t[3:]:
        if x.startswith("v="):
            d["v"] = int(x[2:])
    return d


def parse_game(line):
    """One SGF line of a self-play .sgfs file -> dict, or None if it is not a standard game."""
    head = line[:line.find(";B[")] if ";B[" in line else line
    if _prop("KM", head) != "-0.5" or _prop("WB", head) != "10" or _prop("WW", head) != "10":
        return None
    c = _prop("C", head) or ""
    if "gtype=normal" not in c:
        return None
    start = int((re.search(r"startTurnIdx=(\d+)", c) or [0, 0])[1])
    found = MOVE_RE.findall(line)
    return {"start": start, "moves": [to_9x9(m) for _, m, _ in found], "comments": [parse_comment(cm) for _, _, cm in found],
            "rep": int((re.search(r"repetitionDrawCount=(\d+)", _prop("RU", head) or "") or [0, 0])[1])}


def iter_games(dirs):
    """Standard games of self-play model dirs (DIR/sgfs/*.sgfs) or of .sgfs files."""
    for d in dirs:
        files = [d] if d.endswith(".sgfs") else sorted(glob.glob(os.path.join(d, "sgfs", "*.sgfs")))
        for f in files:
            with open(f) as fh:
                for line in fh:
                    g = parse_game(line)
                    if g is not None:
                        yield g


# --- build -------------------------------------------------------------------------------------------------------
def collect(games, args, rng):
    """One pass over the games: opening-tree counts, a reservoir of random positions, the biggest value swings."""
    opening = {}                              # id -> [count, moves, symmetric]
    reservoir, seen = [], 0
    swings, n_swings = [], 0                  # reservoir of (swing, ply, moves)
    n_games = 0
    for g in games:
        n_games += 1
        moves, start = g["moves"], g["start"]
        in_game = set()
        for p in range(max(2, start), min(args.opening_max_ply, len(moves) - 1) + 1):
            key, sym = state_key(moves[:p])
            pid = position_id(key)
            if pid in in_game:
                continue
            in_game.add(pid)
            e = opening.get(pid)
            if e is None:
                opening[pid] = [1, moves[:p], sym]
            else:
                e[0] += 1
        # random positions: a reservoir over every eligible (game, ply)
        lo, hi = max(args.random_min_ply, start), min(args.random_max_ply, len(moves) - 1)
        for p in range(lo, hi + 1):
            seen += 1
            item = (p, moves[:p])
            if len(reservoir) < args.reservoir:
                reservoir.append(item)
            else:
                j = rng.randrange(seen)
                if j < args.reservoir:
                    reservoir[j] = item
        # value swings: White's win probability of full searches
        cm = g["comments"]
        for t in range(max(4, start), len(moves) - 1):
            a = cm[t]
            if a is None or a.get("v", 0) < MIN_FULL_VISITS:
                continue
            best, bestk = 0.0, None
            for k in range(1, args.swing_window + 1):
                if t + k >= len(moves):
                    break
                b = cm[t + k]
                if b is not None and b.get("v", 0) >= MIN_FULL_VISITS and abs(b["white"] - a["white"]) > best:
                    best, bestk = abs(b["white"] - a["white"]), k
            if best > args.swing_threshold:        # a uniform sample of all the swings (a reservoir)
                n_swings += 1
                item = (best, t, moves[:t])
                if len(swings) < args.swing_pool:
                    swings.append(item)
                else:
                    j = rng.randrange(n_swings)
                    if j < args.swing_pool:
                        swings[j] = item
    return opening, reservoir, swings, n_games, n_swings


def build_args(**over):
    """The defaults of the `build` command as a namespace (for other tools)."""
    d = dict(seed=1, n_opening=150, n_random=150, n_swing=120, opening_min_games=10, opening_max_ply=24, random_min_ply=4,
             random_max_ply=80, reservoir=5000, swing_threshold=0.3, swing_window=4, swing_pool=2000)
    d.update(over)
    return argparse.Namespace(**d)


def build_positions(games, args):
    rng = random.Random(args.seed)
    opening, reservoir, swings, n_games, n_swings = collect(games, args, rng)
    chosen = collections.OrderedDict()        # id -> record

    def add(group, moves, **extra):
        key, sym = state_key(moves)
        pid = position_id(key)
        rec = chosen.get(pid)
        if rec is None:
            rec = {"id": pid, "moves": " ".join(moves), "to_move": to_move_of(moves), "ply": len(moves), "symmetric": sym,
                   "groups": [], "komi": -0.5, "walls": [10, 10], "rep": 0}
            chosen[pid] = rec
        if group not in rec["groups"]:
            rec["groups"].append(group)
        rec.update(extra)
        return rec

    # opening-tree nodes: the most common positions, at most `cap` per ply so that deep nodes are not crowded out
    per_ply = collections.Counter()
    cap = max(8, args.n_opening // 8)
    ranked = sorted(opening.items(), key=lambda kv: (-kv[1][0], kv[0]))
    n = 0
    for pid, (count, moves, sym) in ranked:
        if n >= args.n_opening:
            break
        if count < args.opening_min_games or per_ply[len(moves)] >= cap:
            continue
        per_ply[len(moves)] += 1
        add("opening", moves, count=count)
        n += 1
    # swings: a random subset of the positions before a swing
    n = 0
    rng.shuffle(swings)
    for swing, t, moves in swings:
        if n >= args.n_swing:
            break
        key, _ = state_key(moves)
        if position_id(key) in chosen and "swing" in chosen[position_id(key)]["groups"]:
            continue
        add("swing", moves, swing=round(swing, 3))
        n += 1
    # random positions
    rng.shuffle(reservoir)
    n = 0
    for p, moves in reservoir:
        if n >= args.n_random:
            break
        key, _ = state_key(moves)
        if position_id(key) in chosen and "random" in chosen[position_id(key)]["groups"]:
            continue
        add("random", moves)
        n += 1
    for rec in chosen.values():
        rec["group"] = next(g for g in GROUPS if g in rec["groups"])
    return list(chosen.values()), {"games": n_games, "opening_nodes": len(opening), "swings": n_swings}


# --- engine ------------------------------------------------------------------------------------------------------
def default_katago():
    return os.environ.get("KATAGO") or os.path.join(REPO, "cpp", "katago")


def default_config():
    return os.path.join(REPO, "cpp", "configs", "gtp_quoridor.cfg")


def make_engine(model, katago=None, config=None, params=None, threads=None, cache_bits=None, name="e", log_dir=None,
                timeout=3600):
    ov = {"logAllGTPCommunication": "false", "logSearchInfo": "false", "logDir": log_dir or os.path.join(REPO, "gtp_logs")}
    ov.update(params or {})
    if threads is not None:
        ov["numSearchThreads"] = threads
    if cache_bits is not None:
        ov["nnCacheSizePowerOfTwo"] = cache_bits
    argv = [katago or default_katago(), "gtp", "-model", model, "-config", config or default_config(),
            "-override-config", ",".join("%s=%s" % kv for kv in ov.items())]
    eng = QTPEngine(name, argv, os.path.join(log_dir or os.path.join(REPO, "gtp_logs"), "%s.stderr" % name), default_timeout=timeout)
    eng.start()
    eng.must("name", timeout=900)
    return eng


INFO_SPLIT = re.compile(r"(?:^|\s)info move ")


def parse_search(text):
    """kata-search_analyze output -> list of children dicts (move, visits, wr, lead, lcb, prior, order, sym), best first.

    Values are from the side to move. `sym` is the move this one is the mirror image of (identical stats), or None."""
    out = []
    for chunk in INFO_SPLIT.split(text)[1:]:
        toks = chunk.split()
        if not toks:
            continue
        d = {"move": toks[0]}
        i = 1
        while i + 1 < len(toks) and toks[i] != "pv":
            d[toks[i]] = toks[i + 1]
            i += 2
        try:
            out.append({"move": d["move"], "visits": int(d["visits"]), "wr": float(d["winrate"]), "lead": float(d.get("scoreLead", 0)),
                        "lcb": float(d.get("lcb", 0)), "prior": float(d.get("prior", 0)), "order": int(d.get("order", 10 ** 6)),
                        "sym": d.get("isSymmetryOf")})
        except (KeyError, ValueError):
            continue
    out.sort(key=lambda c: (c["order"], -c["visits"]))
    return out


def run_search(eng, moves, visits):
    """Fresh search of the position after `moves` (list of 9x9 moves). Returns the children, best first."""
    eng.must("clear_board")
    side = "b"
    for m in moves:
        eng.must("play %s %s" % (side, m))
        side = "w" if side == "b" else "b"
    eng.must("kata-set-param maxVisits %d" % visits)
    eng.must("clear_cache")
    return parse_search(eng.must("kata-search_analyze %s" % side))


# --- running positions on several engines, with an append-only output ------------------------------------------------
class Progress:
    def __init__(self, total, label):
        self.total, self.label, self.done, self.t0 = total, label, 0, time.time()
        self.lock = threading.Lock()

    def tick(self):
        with self.lock:
            self.done += 1
            if self.done % 10 == 0 or self.done == self.total:
                el = time.time() - self.t0
                eta = el / self.done * (self.total - self.done)
                print("%s %d/%d  %.0fs elapsed, ~%.0fs left" % (self.label, self.done, self.total, el, eta), file=sys.stderr, flush=True)


def run_parallel(items, n_workers, factory, work, out_path=None, label="work"):
    """Calls work(engine, item) -> record for every item, on n_workers engines (one thread each). Records are appended to
    out_path as they come. Returns the records in item order."""
    n_workers = max(1, min(n_workers, len(items) or 1))
    lock = threading.Lock()
    results = {}
    prog = Progress(len(items), label)
    fh = open(out_path, "a") if out_path else None

    def worker(w):
        eng = factory(w)
        try:
            for i in range(w, len(items), n_workers):
                for attempt in range(3):
                    try:
                        rec = work(eng, items[i])
                        break
                    except Exception as e:   # engine crash or timeout: restart it and retry the position
                        print("worker %d: %s: %s" % (w, items[i].get("id"), e), file=sys.stderr, flush=True)
                        eng.ensure_running()
                else:
                    continue
                with lock:
                    results[i] = rec
                    if fh:
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                prog.tick()
        finally:
            eng.close()

    with ThreadPoolExecutor(n_workers) as ex:
        for f in [ex.submit(worker, w) for w in range(n_workers)]:
            f.result()
    if fh:
        fh.close()
    return [results[i] for i in sorted(results)]


def read_jsonl(path):
    recs, meta = [], None
    if not os.path.exists(path):
        return recs, meta
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "meta" in r:
                meta = r["meta"]
            else:
                recs.append(r)
    return recs, meta


# --- labels ------------------------------------------------------------------------------------------------------
def make_label(children, visits, net, min_child_visits, min_gap_wr=0.05, min_gap_lead=0.5, decided_wr=0.95,
               second_min_share=0.01):
    """The label of a deep search: best move (the engine's own choice), its numbers, the children with enough visits,
    and whether the label is clear: the best move beats the best rival by min_gap_wr winrate (or min_gap_lead lead if
    the position is decided), or it is the only move with a real share of the visits."""
    total = sum(c["visits"] for c in children)
    best = children[0]
    kids = {c["move"]: {"visits": c["visits"], "wr": round(c["wr"], 4), "lead": round(c["lead"], 3), "lcb": round(c["lcb"], 4),
                        "prior": round(c["prior"], 4)} for c in children if c["visits"] >= min_child_visits}
    kids.setdefault(best["move"], {"visits": best["visits"], "wr": round(best["wr"], 4), "lead": round(best["lead"], 3),
                                   "lcb": round(best["lcb"], 4), "prior": round(best["prior"], 4)})
    # a child and its mirror image are the same move: the "others" are the children that are not best or its mirror
    same = {best["move"], best["sym"] or best["move"]} | {c["move"] for c in children if c["sym"] == best["move"]}
    # the rivals of the best move: children with a real share of the visits (a 100-visit winrate is too noisy to be
    # taken as "better than the best move" or as a gap)
    others = [c for c in children if c["move"] not in same and c["visits"] >= max(min_child_visits, second_min_share * total)]
    decided = abs(best["wr"] - 0.5) >= decided_wr - 0.5
    lab = {"net": net, "visits": visits, "best": best["move"], "best_visits": best["visits"],
           "share": round(best["visits"] / max(1, total), 4), "wr": round(best["wr"], 4), "lead": round(best["lead"], 3),
           "lcb": round(best["lcb"], 4), "children": kids, "decided": decided}
    if others:
        second = max(others, key=lambda c: c["wr"])
        lab.update({"second": second["move"], "second_wr": round(second["wr"], 4), "second_lead": round(second["lead"], 3),
                    "gap_wr": round(best["wr"] - second["wr"], 4), "gap_lcb": round(best["lcb"] - second["wr"], 4),
                    "gap_lead": round(best["lead"] - max(c["lead"] for c in others), 3)})
        gap_wr, gap_lead = lab["gap_wr"], lab["gap_lead"]
        lab["clear"] = gap_wr >= min_gap_wr or (decided and gap_lead >= min_gap_lead)
        lab["forced"] = False
    else:                                      # only one move got enough visits: nothing to confuse it with
        lab.update({"second": None, "gap_wr": 1.0, "gap_lead": 0.0, "clear": True, "forced": True})
    return lab


def label_position(eng, pos, args):
    children = run_search(eng, pos["moves"].split(), args.visits)
    if not children:
        raise RuntimeError("no search output for %s" % pos["id"])
    rec = dict(pos)
    rec["label"] = make_label(children, args.visits, args.net or os.path.basename(os.path.dirname(args.model)),
                              args.min_child_visits)
    return rec


# --- evaluation --------------------------------------------------------------------------------------------------
def same_move(a, b, symmetric):
    return a == b or (symmetric and mirror(a) == b)


def judge(pos, children):
    """(chosen move, agrees, regret or None) of a fresh search on a labelled position."""
    lab = pos["label"]
    chosen = children[0]["move"]
    sym = pos.get("symmetric", False)
    agrees = same_move(chosen, lab["best"], sym)
    kids = lab["children"]
    if agrees:
        return chosen, True, 0.0
    key = chosen if chosen in kids else (mirror(chosen) if sym and mirror(chosen) in kids else None)
    if key is None:
        return chosen, False, None
    return chosen, False, max(0.0, kids[lab["best"]]["wr"] - kids[key]["wr"])


def eval_position(eng, pos, visits):
    children = run_search(eng, pos["moves"].split(), visits)
    chosen, agrees, regret = judge(pos, children)
    top = children[0]
    return {"id": pos["id"], "group": pos["group"], "ply": pos["ply"], "count": pos.get("count", 1), "chosen": chosen,
            "best": pos["label"]["best"], "agree": agrees, "regret": regret, "wr": round(top["wr"], 4),
            "chosen_share": round(top["visits"] / max(1, sum(c["visits"] for c in children)), 3)}


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    agree = sum(r["agree"] for r in rows)
    known = [r["regret"] for r in rows if r["regret"] is not None]
    p = agree / n
    out = {"n": n, "agree": round(p, 4), "agree_se": round(math.sqrt(p * (1 - p) / n), 4),
           "regret_mean": round(sum(known) / len(known), 4) if known else None, "regret_known": len(known),
           "regret_unknown": n - len(known), "disagree": n - agree}
    w = sum(r["count"] for r in rows)
    out["agree_weighted"] = round(sum(r["count"] for r in rows if r["agree"]) / w, 4) if w else None
    return out


def summarize_all(rows):
    res = {"overall": summarize(rows), "by_group": {}, "by_ply": {}}
    for g in GROUPS:
        res["by_group"][g] = summarize([r for r in rows if r["group"] == g])
    for lo, hi in PLY_BINS:
        sub = [r for r in rows if lo <= r["ply"] <= hi]
        if sub:
            res["by_ply"][ply_bin(lo)] = summarize(sub)
    return res


def fmt_row(name, s):
    if not s.get("n"):
        return "%-14s %5d" % (name, 0)
    rm = "%.3f" % s["regret_mean"] if s["regret_mean"] is not None else "  -  "
    return "%-14s %5d  agree %5.1f%% ± %.1f   regret %s (known %d, unknown %d)" % (
        name, s["n"], 100 * s["agree"], 100 * s["agree_se"], rm, s["regret_known"], s["regret_unknown"])


def print_summary(title, res, out=sys.stdout):
    print("== " + title, file=out)
    print(fmt_row("all", res["overall"]), file=out)
    for g, s in res["by_group"].items():
        if s.get("n"):
            print(fmt_row("group " + g, s), file=out)
    for b, s in res["by_ply"].items():
        print(fmt_row("ply " + b, s), file=out)


# --- commands ----------------------------------------------------------------------------------------------------
def cmd_build(args):
    positions, info = build_positions(iter_games(args.dirs), args)
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {"type": "positions", "version": SUITE_VERSION, "sources": [os.path.basename(d.rstrip("/")) for d in args.dirs],
                                     "games": info["games"], "seed": args.seed, "created": time.strftime("%F %T")}}) + "\n")
        for r in positions:
            f.write(json.dumps(r) + "\n")
    c = collections.Counter(r["group"] for r in positions)
    multi = sum(len(r["groups"]) > 1 for r in positions)
    print("%d positions from %d standard games (%d opening-tree nodes, %d swings seen): %s; %d in more than one group" % (
        len(positions), info["games"], info["opening_nodes"], info["swings"], dict(c), multi))
    print("plies: %s" % dict(sorted(collections.Counter(ply_bin(r["ply"]) for r in positions).items())))


def cmd_label(args):
    positions, _ = read_jsonl(args.positions)
    done, _ = read_jsonl(args.out)
    done_ids = {r["id"] for r in done}
    todo = [p for p in positions if p["id"] not in done_ids]
    args.net = args.net or os.path.basename(os.path.dirname(os.path.abspath(args.model)))
    if not os.path.exists(args.out) or not done:
        with open(args.out, "w") as f:
            f.write(json.dumps({"meta": {"type": "suite", "version": SUITE_VERSION, "label_net": args.net, "label_visits": args.visits,
                                         "label_params": args.params, "min_child_visits": args.min_child_visits,
                                         "created": time.strftime("%F %T"),
                                         "rules": "komi -0.5, 10/10 walls, no repetition rule (gtp_quoridor.cfg)"}}) + "\n")
    print("%d positions to label (%d already done)" % (len(todo), len(done)), file=sys.stderr)
    factory = lambda w: make_engine(args.model, args.katago, args.config, PARAM_SETS[args.params], args.threads, args.cache_bits,
                                    "label%d" % w, args.log_dir)
    t0 = time.time()
    run_parallel(todo, args.engines, factory, lambda eng, pos: label_position(eng, pos, args), args.out, "label")
    recs, _ = read_jsonl(args.out)
    clear = sum(r["label"]["clear"] for r in recs)
    print("labelled %d positions in %.0f s; %d clear, %d unclear" % (len(recs), time.time() - t0, clear, len(recs) - clear))


def load_suite(path, include_unclear=False):
    recs, meta = read_jsonl(path)
    if not include_unclear:
        recs = [r for r in recs if r["label"]["clear"]]
    return recs, meta


def cmd_eval(args):
    suite, meta = load_suite(args.suite, args.include_unclear)
    if args.limit:
        suite = suite[:args.limit]
    net = args.net or os.path.basename(os.path.dirname(os.path.abspath(args.model)))
    os.makedirs(args.out_dir, exist_ok=True)
    for v in [int(x) for x in args.visits.split(",")]:
        tag = "%s_v%d_%s" % (net, v, args.params)
        pos_path = os.path.join(args.out_dir, "eval_%s.jsonl" % tag)
        if os.path.exists(pos_path):
            os.remove(pos_path)
        factory = lambda w: make_engine(args.model, args.katago, args.config, PARAM_SETS[args.params], args.threads, args.cache_bits,
                                        "eval%d" % w, args.log_dir)
        t0 = time.time()
        rows = run_parallel(suite, args.engines, factory, lambda eng, pos: eval_position(eng, pos, v), pos_path, "eval " + tag)
        res = summarize_all(rows)
        res.update({"net": net, "visits": v, "params": args.params, "suite_label_net": (meta or {}).get("label_net"),
                    "suite_label_visits": (meta or {}).get("label_visits"), "seconds": round(time.time() - t0, 1)})
        with open(os.path.join(args.out_dir, "eval_%s.json" % tag), "w") as f:
            json.dump(res, f, indent=1)
        print_summary("%s v%d (%s params), %.0f s" % (net, v, args.params, res["seconds"]), res)


def cmd_report(args):
    rows = []
    for p in args.files:
        with open(p) as f:
            rows.append(json.load(f))
    print("| net | visits | params | n | agree | regret | known | " + " | ".join("agree " + g for g in GROUPS) + " |")
    print("|---|---|---|---|---|---|---|" + "---|" * len(GROUPS))
    for r in rows:
        o = r["overall"]
        gs = " | ".join("%.1f%% (%d)" % (100 * r["by_group"][g]["agree"], r["by_group"][g]["n"]) if r["by_group"].get(g, {}).get("n") else "-"
                        for g in GROUPS)
        rm = "%.3f" % o["regret_mean"] if o["regret_mean"] is not None else "-"
        print("| %s | %d | %s | %d | %.1f%% ± %.1f | %s | %d | %s |" % (
            r["net"], r["visits"], r["params"], o["n"], 100 * o["agree"], 100 * o["agree_se"], rm, o["regret_known"], gs))
    print()
    bins = list(rows[0]["by_ply"]) if rows else []
    print("| net | visits | " + " | ".join("ply " + b for b in bins) + " |")
    print("|---|---|" + "---|" * len(bins))
    for r in rows:
        print("| %s | %d | %s |" % (r["net"], r["visits"], " | ".join(
            "%.1f%% (%d)" % (100 * r["by_ply"][b]["agree"], r["by_ply"][b]["n"]) if b in r["by_ply"] else "-" for b in bins)))


def add_engine_args(ap):
    ap.add_argument("--model", required=True)
    ap.add_argument("--net", help="name stored in the output (default: the model's directory name)")
    ap.add_argument("--katago", help="binary (default $KATAGO or cpp/katago)")
    ap.add_argument("--config", help="GTP config (default cpp/configs/gtp_quoridor.cfg)")
    ap.add_argument("--engines", type=int, default=4, help="engine processes sharing the GPU")
    ap.add_argument("--threads", type=int, help="numSearchThreads of each engine (default: the config's / the param set's)")
    ap.add_argument("--cache-bits", type=int, help="nnCacheSizePowerOfTwo")
    ap.add_argument("--log-dir", help="engine stderr and logDir (default gtp_logs/ in the repo)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="positions from self-play SGFs")
    b.add_argument("--dirs", nargs="+", required=True, help="self-play model dirs (with sgfs/) or .sgfs files")
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--n-opening", type=int, default=150)
    b.add_argument("--n-random", type=int, default=150)
    b.add_argument("--n-swing", type=int, default=120)
    b.add_argument("--opening-min-games", type=int, default=10, help="games that must reach an opening-tree node")
    b.add_argument("--opening-max-ply", type=int, default=24)
    b.add_argument("--random-min-ply", type=int, default=4)
    b.add_argument("--random-max-ply", type=int, default=80)
    b.add_argument("--reservoir", type=int, default=5000)
    b.add_argument("--swing-threshold", type=float, default=0.3)
    b.add_argument("--swing-window", type=int, default=4, help="plies after the position in which the value may swing")
    b.add_argument("--swing-pool", type=int, default=2000)
    b.set_defaults(fn=cmd_build)
    lb = sub.add_parser("label", help="deep-search labels")
    lb.add_argument("--positions", required=True)
    lb.add_argument("--out", required=True, help="the suite (appended to, so that an interrupted run resumes)")
    lb.add_argument("--visits", type=int, default=50000)
    lb.add_argument("--params", choices=list(PARAM_SETS), default="gtp")
    lb.add_argument("--min-child-visits", type=int, default=100)
    add_engine_args(lb)
    lb.set_defaults(fn=cmd_label, engines=1, threads=16)
    ev = sub.add_parser("eval", help="evaluate a net on the suite")
    ev.add_argument("--suite", required=True)
    ev.add_argument("--visits", default="256,1000", help="comma-separated search sizes")
    ev.add_argument("--params", choices=list(PARAM_SETS), default="gtp",
                    help="gtp: gtp_quoridor.cfg; selfplay: the self-play search params with root noise; selfplay-nonoise: without")
    ev.add_argument("--out-dir", required=True)
    ev.add_argument("--include-unclear", action="store_true")
    ev.add_argument("--limit", type=int)
    add_engine_args(ev)
    ev.set_defaults(fn=cmd_eval)
    rp = sub.add_parser("report", help="markdown tables from eval_*.json files")
    rp.add_argument("files", nargs="+")
    rp.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
