#!/usr/bin/env python3
"""A tiny QTP engine for tests. It plays a toy race game, not Quoridor:

- each pawn moves one row towards its goal (Black e9 -> row 1, White e1 -> row 9) or one column sideways;
- "walls" a1h..c1h are legal while the player has walls left (10 each) and do nothing (they exist to
  exercise opening weights and wall counting);
- the first pawn to reach its goal row wins.

It speaks the same QTP subset the arena uses, so it can serve as both arbiter and player. It also answers
the commands the play GUI (python/play_gui) uses: name, version, walls, undo, kata-set-param maxVisits,
kata-genmove_analyze, kata-search_analyze and kata-raw-nn. Their output imitates KataQuoridor's: analysis
values are from the side to move, kata-raw-nn values from White's, a 1-visit search prints no "info" line,
and kata-raw-nn on a finished game kills the process (like KataQuoridor 0.1.0's "legalCount > 0" assert).

Flags: --illegal (genmove answers "z9"), --crash-after N (exit after N genmoves),
--hang-after N (stop answering after N genmoves), --no-legal (unknown legal_moves),
--think-delay S (sleep S seconds in every genmove variant).
"""
import argparse
import math
import sys
import time

COLS = "abcdefghi"
WALLS = 10


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

    def walls_left(self, c):
        return WALLS - sum(1 for col, mv in self.moves if col == c and mv[-1] in "hv")

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
        return out + (["a1h", "b1h", "c1h"] if self.walls_left(c) > 0 else [])

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

    def copy(self):
        g = Game()
        for c, mv in self.moves:
            g.play(c, mv)
        return g

    def undo(self):
        if not self.moves:
            return False
        g = Game()
        for c, mv in self.moves[:-1]:
            g.play(c, mv)
        self.__dict__.update(g.__dict__)
        return True

    def value(self, c):
        """(win probability, score lead) for c, as if c were to move: a smooth function of the race."""
        lead = self.dist(opp(c)) - self.dist(c) + 0.5
        return 1.0 / (1.0 + math.exp(-0.8 * lead)), lead


def opp(c):
    return "w" if c == "b" else "b"


def color_arg(s):
    s = s.lower()
    return "b" if s in ("b", "black") else "w" if s in ("w", "white") else None


def analysis(g, c, visits):
    """KataGo-style analysis text for c to move in g (None for a 1-visit search, which has no children)."""
    if visits <= 1:
        return None
    infos = []
    for order, mv in enumerate(g.legal(c)[:3]):
        after = g.copy()
        after.play(c, mv)
        if after.winner():
            wr, lead = 1.0, float(after.dist(opp(c)))
        else:
            wr, lead = after.value(c)
        reply = after.legal(opp(c))[:1]
        v = max(1, visits // (2 ** (order + 1)))
        infos.append("info move %s visits %d edgeVisits %d utility 0 winrate %.6f scoreMean %.6f scoreStdev 1 "
                     "scoreLead %.6f scoreSelfplay %.6f prior 0.5 lcb %.6f utilityLcb 0 weight %d order %d pv %s"
                     % (mv, v, v, wr, lead, lead, lead, wr, v, order, " ".join([mv] + reply)))
    wr, lead = g.value(c)
    root = ("rootInfo visits %d utility 0 winrate %.6f scoreMean %.6f scoreStdev 1 scoreLead %.6f "
            "scoreSelfplay %.6f weight %d" % (visits, wr, lead, lead, lead, visits))
    return " ".join(infos + [root])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--illegal", action="store_true")
    ap.add_argument("--crash-after", type=int, default=-1)
    ap.add_argument("--hang-after", type=int, default=-1)
    ap.add_argument("--no-legal", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--think-delay", type=float, default=0.0)
    args = ap.parse_args()
    g = Game()
    genmoves = 0
    max_visits = 100

    def reply(ok, text=""):
        sys.stdout.write(("= " if ok else "? ") + text + "\n\n")
        sys.stdout.flush()

    def begin_genmove():
        nonlocal genmoves
        genmoves += 1
        if genmoves == args.crash_after:
            sys.exit(3)
        if genmoves == args.hang_after:
            time.sleep(3600)
        if args.think_delay > 0:
            time.sleep(args.think_delay)

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
                     "showboard", "name", "version", "walls", "undo", "kata-set-param", "kata-genmove_analyze",
                     "kata-search_analyze", "kata-raw-nn"}
            if not args.no_legal:
                known.add("legal_moves")
            reply(True, "true" if rest and rest[0] in known else "false")
        elif cmd == "name":
            reply(True, "MockQuoridor")
        elif cmd == "version":
            reply(True, "0.0 (mock)")
        elif cmd == "clear_board":
            g = Game()
            reply(True)
        elif cmd == "play":
            c = color_arg(rest[0]) if rest else None
            if len(rest) != 2 or c is None:
                reply(False, "illegal move")
            elif g.winner():
                reply(False, "game is over")
            else:
                ok = g.play(c, rest[1].lower())
                reply(ok, "" if ok else "illegal move")
        elif cmd == "undo":
            reply(True) if g.undo() else reply(False, "cannot undo")
        elif cmd == "genmove":
            begin_genmove()
            c = color_arg(rest[0])
            mv = "z9" if args.illegal else g.legal(c)[0]
            if not args.illegal:
                g.play(c, mv)
            reply(True, mv)
        elif cmd in ("kata-genmove_analyze", "kata-search_analyze"):
            c = color_arg(rest[0]) if rest and color_arg(rest[0]) else g.to_move
            if g.winner():
                reply(False, "game is over")
                continue
            if c != g.to_move:
                reply(False, "not %s's turn" % ("black" if c == "b" else "white"))
                continue
            begin_genmove()
            text = analysis(g, c, max_visits)
            mv = g.legal(c)[0]
            if cmd == "kata-genmove_analyze":
                g.play(c, mv)
            reply(True, "\n" + (text + "\n" if text else "") + "play " + mv)
        elif cmd == "kata-raw-nn":
            if g.winner():
                sys.stderr.write("FATAL ERROR:\nFailed test assert: legalCount > 0\n")
                sys.exit(1)
            wr, lead = g.value(g.to_move)
            if g.to_move == "b":
                wr, lead = 1.0 - wr, -lead
            reply(True, "symmetry 0\nwhiteWin %.6f\nwhiteLoss %.6f\nnoResult 0.000000\nwhiteLead %.3f\n"
                        "whiteScoreSelfplay %.3f\npolicy\n NAN 0.1\npolicyPass NAN\nwhiteOwnership\n0.0"
                  % (wr, 1.0 - wr, lead, lead))
        elif cmd == "kata-set-param":
            if len(rest) == 2 and rest[0] == "maxVisits" and rest[1].isdigit():
                max_visits = int(rest[1])
                reply(True)
            else:
                reply(False, "unknown parameter")
        elif cmd == "legal_moves" and not args.no_legal:
            reply(True, " ".join(g.legal()))
        elif cmd == "winner":
            reply(True, g.winner() or "none")
        elif cmd == "dist":
            reply(True, "B: %d W: %d" % (g.dist("b"), g.dist("w")))
        elif cmd == "walls":
            reply(True, "B: %d W: %d" % (g.walls_left("b"), g.walls_left("w")))
        elif cmd == "printsgf":
            body = "".join(";%s[%s]" % (c.upper(), m) for c, m in g.moves)
            reply(True, "(;FF[4]GM[1]SZ[17]PB[]PW[]KM[7.5]RU[Quoridor]%s)" % body)
        elif cmd == "showboard":
            reply(True, "line one\nline two\nline three")
        else:
            reply(False, "unknown command")


if __name__ == "__main__":
    main()
