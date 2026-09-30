#!/usr/bin/env python3
"""A tiny QTP engine for tests. It plays a toy race game, not Quoridor:

- each pawn moves one row towards its goal (Black e9 -> row 1, White e1 -> row 9) or one column sideways;
- "walls" a1h..c1h are always legal and do nothing (they exist to exercise opening weights);
- the first pawn to reach its goal row wins.

It speaks the same QTP subset the arena uses, so it can serve as both arbiter and player.
Flags: --illegal (genmove answers "z9"), --crash-after N (exit after N genmoves),
--hang-after N (stop answering after N genmoves), --no-legal (unknown legal_moves).
"""
import argparse
import sys
import time

COLS = "abcdefghi"


class Game:
    def __init__(self):
        self.pos = {"b": (4, 9), "w": (4, 1)}
        self.to_move = "b"
        self.moves = []

    def goal(self, c):
        return 1 if c == "b" else 9

    def winner(self):
        for c in "bw":
            if self.pos[c][1] == self.goal(c):
                return c.upper()
        return None

    def legal(self, c=None):
        c = c or self.to_move
        if self.winner():
            return []
        x, y = self.pos[c]
        dy = -1 if c == "b" else 1
        out = ["%s%d" % (COLS[x], y + dy)]
        for dx in (-1, 1):
            if 0 <= x + dx < 9:
                out.append("%s%d" % (COLS[x + dx], y))
        return out + ["a1h", "b1h", "c1h"]

    def play(self, c, mv):
        if c != self.to_move or mv not in self.legal(c):
            return False
        if mv[-1] not in "hv":
            self.pos[c] = (COLS.index(mv[0]), int(mv[1:]))
        self.moves.append((c, mv))
        self.to_move = "w" if c == "b" else "b"
        return True

    def dist(self, c):
        return abs(self.pos[c][1] - self.goal(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--illegal", action="store_true")
    ap.add_argument("--crash-after", type=int, default=-1)
    ap.add_argument("--hang-after", type=int, default=-1)
    ap.add_argument("--no-legal", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    g = Game()
    genmoves = 0

    def reply(ok, text=""):
        sys.stdout.write(("= " if ok else "? ") + text + "\n\n")
        sys.stdout.flush()

    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        cmd, rest = parts[0], parts[1:]
        if cmd == "quit":
            reply(True)
            return
        elif cmd == "known_command":
            known = {"play", "genmove", "clear_board", "winner", "dist", "printsgf", "quit", "known_command",
                     "showboard"}
            if not args.no_legal:
                known.add("legal_moves")
            reply(True, "true" if rest and rest[0] in known else "false")
        elif cmd == "clear_board":
            g = Game()
            reply(True)
        elif cmd == "play":
            ok = len(rest) == 2 and g.play(rest[0].lower()[0], rest[1].lower())
            reply(ok, "" if ok else "illegal move")
        elif cmd == "genmove":
            genmoves += 1
            if genmoves == args.crash_after:
                sys.exit(3)
            if genmoves == args.hang_after:
                time.sleep(3600)
            c = rest[0].lower()[0]
            mv = "z9" if args.illegal else g.legal(c)[0]
            if not args.illegal:
                g.play(c, mv)
            reply(True, mv)
        elif cmd == "legal_moves" and not args.no_legal:
            reply(True, " ".join(g.legal()))
        elif cmd == "winner":
            reply(True, g.winner() or "none")
        elif cmd == "dist":
            reply(True, "B: %d W: %d" % (g.dist("b"), g.dist("w")))
        elif cmd == "printsgf":
            body = "".join(";%s[%s]" % (c.upper(), m) for c, m in g.moves)
            reply(True, "(;FF[4]GM[1]SZ[17]PB[]PW[]KM[7.5]RU[Quoridor]%s)" % body)
        elif cmd == "showboard":
            reply(True, "line one\nline two\nline three")
        else:
            reply(False, "unknown command")


if __name__ == "__main__":
    main()
