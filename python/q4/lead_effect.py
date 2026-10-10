#!/usr/bin/env python3
"""The numbers of the Q4 lead effect check (docs/q4/Q4Lead.md, docs/q4/rounds/RLead.md) from .sgfs games.

Per table and per player (the seat's name in the record), from the search values of each move (the move comments):

- hopeless: a move by a seat whose own win probability is below 2% and that has walls in hand. Share of those moves
  that place a wall ("wall share"), and the share of those walls that lengthen the mover's own walls-only path
  (style.descriptors, the mover's distance change > 0; "self-hurting walls").
- decided: a game with a winner. From the first move whose comment gives the winner more than 0.9 to the end: the
  number of plies and the share of the game's plies ("decided phase").

    python q4/lead_effect.py games.sgfs [more.sgfs ...] [--json out.json]

Games without move comments count for nothing in the search-based numbers; the hopeless numbers need the comments of
the moving seat (every match game has them).
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q4.record import load_records  # noqa: E402
from q4.reference import Pos, eliminate, play, str_to_action  # noqa: E402
from q4.style import descriptors  # noqa: E402

HOPELESS_P = 0.02
DECIDED_P = 0.9


def new_bucket():
    return collections.Counter()


def game_stats(rec):
    """Per-seat counters of one game: {seat name: Counter}, and the game-level decided counters."""
    rules = rec.get("rules", {})
    pos = Pos(max_plies=rules.get("maxPlies", 400), repetition_draw_count=rules.get("repetitionDrawCount", 0),
              initial_walls=rules.get("initialWalls", [7, 7, 7, 7]))
    comments = rec.get("moveComments") or [None] * len(rec["events"])
    winner = int(rec["result"][0]) - 1 if rec["result"].endswith("+") else -1
    per_seat = collections.defaultdict(new_bucket)
    plies = 0
    decided_start = None
    action_comments = []
    for i, ev in enumerate(rec["events"]):
        if "elim" in ev:
            pos = eliminate(pos, ev["elim"] - 1)
            continue
        action = str_to_action(ev["a"])
        before = pos
        after = play(before, action)
        mover = before.to_move
        mc = comments[i]
        name = rec["players"][mover]["name"]
        bucket = per_seat[name]
        bucket["moves"] += 1
        if mc is not None:
            bucket["commented"] += 1
            own = mc["p"][mover]
            if own < HOPELESS_P and before.walls_left[mover] > 0:
                bucket["hopeless"] += 1
                if action[0] != 'p':
                    bucket["hopelessWalls"] += 1
                    if descriptors(before, after, action)[3 + mover] > 0:
                        bucket["selfHurtingWalls"] += 1
            if winner >= 0 and decided_start is None and mc["p"][winner] > DECIDED_P:
                decided_start = plies
            action_comments.append(mc)
        plies += 1
        pos = after
    game = collections.Counter()
    game["games"] = 1
    game["plies"] = plies
    if winner >= 0 and decided_start is not None:
        game["decidedGames"] = 1
        game["decidedPlies"] = plies - decided_start
    return per_seat, game


def summarize(counts):
    out = {}
    h = counts["hopeless"]
    out["hopelessMoves"] = h
    out["wallShare"] = counts["hopelessWalls"] / h if h else None
    hw = counts["hopelessWalls"]
    out["selfHurtingShare"] = counts["selfHurtingWalls"] / hw if hw else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", help="write the numbers to this file")
    args = ap.parse_args(argv)

    by_table = collections.defaultdict(lambda: {"seat": collections.defaultdict(new_bucket), "game": new_bucket()})
    for path in args.files:
        for rec in load_records(path):
            table = (rec.get("match") or {}).get("table", "?")
            per_seat, game = game_stats(rec)
            tab = by_table[table]
            for name, c in per_seat.items():
                tab["seat"][name].update(c)
            tab["game"].update(game)

    report = {}
    for table, tab in sorted(by_table.items()):
        g = tab["game"]
        entry = {
            "games": g["games"],
            "meanPlies": g["plies"] / g["games"] if g["games"] else None,
            "decidedGames": g["decidedGames"],
            "decidedPlyShare": g["decidedPlies"] / g["plies"] if g["plies"] else None,
            "decidedMeanPlies": g["decidedPlies"] / g["decidedGames"] if g["decidedGames"] else None,
            "players": {},
        }
        for name, c in sorted(tab["seat"].items()):
            entry["players"][name] = dict(summarize(c), moves=c["moves"], commented=c["commented"])
        report[table] = entry

    for table, entry in report.items():
        print(f"table {table}: {entry['games']} games, mean {entry['meanPlies']:.1f} plies; decided phase in "
              f"{entry['decidedGames']} games: {entry['decidedMeanPlies'] or 0:.1f} plies, "
              f"{100 * (entry['decidedPlyShare'] or 0):.1f}% of the plies")
        for name, p in entry["players"].items():
            ws = "-" if p["wallShare"] is None else f"{100 * p['wallShare']:.1f}%"
            sh = "-" if p["selfHurtingShare"] is None else f"{100 * p['selfHurtingShare']:.1f}%"
            print(f"  {name}: hopeless moves {p['hopelessMoves']}, wall share {ws}, self-hurting walls {sh}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
