#!/usr/bin/env python3
"""Opening suite for kq_ladder from self-play SGFs: the most common first-N-ply lines of standard games.

    python -m quoridor_arena.openings_from_selfplay OUT.json SELFPLAY_DIR... [--plies 4,6,8] [--count 64] [--min-games 10]

--plies takes one or more line lengths; each length contributes its most common lines (at least --min-games games,
at most --count lines per length), longer lines after shorter ones.
Standard games: komi -0.5, 10 / 10 walls, gtype=normal (no forks), any repetition rule. Lines that are mirror images
(a <-> i) are the same opening and are counted together; the suite keeps the more common of the two spellings. OUT is
kq_ladder's openings file ({"params", "openings"}, plus "games" per opening), used with
"openings": {"file": "OUT.json"} in a ladder config.

Random 4-ply openings (the default ladder) measure play in all sorts of positions but rarely reach the lines strong
play actually takes; this suite measures play in the standard game's openings.
"""
import argparse, collections, glob, json, os, re

MOVE = re.compile(r";([BW])\[([a-q]{2})\]")


def to_9x9(m):
    """SGF 17x17 move -> GTP 9x9 notation (e8, e2h, d6v)."""
    x, y = ord(m[0]) - 97, ord(m[1]) - 97
    s = chr(97 + x // 2) + str(y // 2 + 1)
    if x % 2 == 0 and y % 2 == 0:
        return s
    return s + ("h" if x % 2 and y % 2 else "v")


def mirror(move):
    """Left-right mirror: pawn columns c -> 8 - c, wall anchor columns c -> 7 - c."""
    col = ord(move[0]) - 97
    wall = move[-1] in "hv"
    return chr(97 + ((7 if wall else 8) - col)) + move[1:]


def standard_lines(dirs, plies):
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "sgfs", "*.sgfs"))):
            with open(f) as fh:
                for line in fh:
                    if "KM[-0.5]" not in line or "WB[10]" not in line or "WW[10]" not in line or "gtype=normal" not in line:
                        continue
                    mv = [to_9x9(m) for _, m in MOVE.findall(line)[:plies]]
                    if len(mv) == plies:
                        yield tuple(mv)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("dirs", nargs="+", help="self-play model dirs (with sgfs/)")
    ap.add_argument("--plies", default="4,6,8", help="line lengths, comma-separated")
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--min-games", type=int, default=10)
    args = ap.parse_args()
    plies_list = [int(p) for p in args.plies.split(",")]
    lines = list(standard_lines(args.dirs, max(plies_list)))
    total = len(lines)
    top = []
    for plies in plies_list:
        spelled = collections.Counter(l[:plies] for l in lines)
        merged = collections.Counter()
        best = {}
        for line, n in spelled.items():
            key = min(line, tuple(mirror(m) for m in line))
            merged[key] += n
            if key not in best or n > spelled[best[key]]:
                best[key] = line
        kept = [(best[k], n) for k, n in merged.most_common() if n >= args.min_games][:args.count]
        top += kept
        print("%d plies: %d lines (%d up to mirroring); kept %d covering %.1f%% of the games"
              % (plies, len(spelled), len(merged), len(kept), 100.0 * sum(n for _, n in kept) / max(1, total)))
    params = {"source": "selfplay", "plies": plies_list, "count": len(top), "min_games": args.min_games,
              "dirs": [os.path.basename(os.path.normpath(d)) for d in args.dirs]}
    with open(args.out + ".tmp", "w") as f:
        json.dump({"params": params, "openings": [list(l) for l, _ in top], "games": [n for _, n in top],
                   "standard_games": total}, f, indent=1)
    os.replace(args.out + ".tmp", args.out)
    print("%d standard games (at least %d plies); %d openings" % (total, max(plies_list), len(top)))


if __name__ == "__main__":
    main()
