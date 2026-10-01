#!/usr/bin/env python3
"""A tiny QTP engine for tests. It plays a toy race game, not Quoridor:

- each pawn moves one row towards its goal (Black e9 -> row 1, White e1 -> row 9) or one column sideways;
- "walls" a1h..c1h are always legal and do nothing (they exist to exercise opening weights);
- the game ends when a pawn reaches its goal row, and is scored like KataQuoridor (docs/QuoridorIOv2.md): the margin
  is the other pawn's distance, the tempo t = margin if White arrived, 1 - margin if Black did, and White wins iff
  t + komi > 0 (komi -0.5 by default); the result is the lead |t + komi|;
- after --max-plies plies (default 300) without an arrival, the game is a draw;
- with kata-set-rule repetitionDrawCount N > 0, so is the N-th occurrence of a position (both pawns and the side to
  move; the toy walls reset the history, like real ones), and printsgf says DR[repetition] (DR[maxPlies] for the
  ply limit).

It speaks the same QTP subset the arena uses, including komi and kata-set-rule blackInitialWalls / whiteInitialWalls /
repetitionDrawCount (the walls only show in printsgf), so it can serve as both arbiter and player.
Flags: --shuffle (genmove steps sideways and back instead of forward, so positions repeat), --illegal (genmove answers "z9"), --crash-after N (exit after N genmoves),
--hang-after N (stop answering after N genmoves), --no-legal (unknown legal_moves),
--no-rules (unknown komi / kata-set-rule, like SimpleQuoridor), --max-plies N,
--kata (answers kata-genmove_analyze like KataQuoridor, from the side to move: winrate 0.7, scoreLead 2.5,
scoreSelfplay 3 for Black to move; winrate 0.2, scoreLead -1.5, scoreSelfplay -2 for White).
"""
import argparse
import sys
import time

COLS = "abcdefghi"


STANDARD_RULES = {"komi": -0.5, "blackInitialWalls": 10, "whiteInitialWalls": 10, "repetitionDrawCount": 0}


class Game:
    def __init__(self, rules, max_plies):
        self.pos = {"b": (4, 9), "w": (4, 1)}
        self.to_move = "b"
        self.moves = []
        self.rules = rules
        self.max_plies = max_plies
        self.since_wall = [self.key()]

    def key(self):
        return (self.pos["b"], self.pos["w"], self.to_move)

    def repeated(self):
        n = self.rules["repetitionDrawCount"]
        return n > 0 and self.since_wall.count(self.key()) >= n

    def goal(self, c):
        return 1 if c == "b" else 9

    def lead(self):
        """White's lead t + komi once a pawn arrived, else None."""
        for c in "bw":
            if self.pos[c][1] == self.goal(c):
                margin = max(1, self.dist("w" if c == "b" else "b"))
                t = margin if c == "w" else 1 - margin
                return t + self.rules["komi"]
        return None

    def winner(self):
        """'B', 'W', 'Draw' or None."""
        s = self.lead()
        if s is not None:
            return "W" if s > 0 else "B"
        if self.repeated() or len(self.moves) >= self.max_plies:
            return "Draw"
        return None

    def draw_reason(self):
        if self.winner() != "Draw":
            return None
        return "repetition" if self.repeated() else "maxPlies"

    def result(self):
        w = self.winner()
        if w is None:
            return None
        return "0" if w == "Draw" else "%s+%g" % (w, abs(self.lead()))

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
        if mv[-1] in "hv":
            self.since_wall = []
        self.since_wall.append(self.key())
        return True

    def dist(self, c):
        return abs(self.pos[c][1] - self.goal(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--illegal", action="store_true")
    ap.add_argument("--crash-after", type=int, default=-1)
    ap.add_argument("--hang-after", type=int, default=-1)
    ap.add_argument("--no-legal", action="store_true")
    ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kata", action="store_true")
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()
    rules = dict(STANDARD_RULES)
    g = Game(rules, args.max_plies)
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
            if not args.no_rules:
                known.update({"komi", "get_komi", "kata-set-rule"})
            if args.kata:
                known.add("kata-genmove_analyze")
            reply(True, "true" if rest and rest[0] in known else "false")
        elif cmd == "clear_board":
            # Like KataQuoridor, the rules persist across clear_board.
            g = Game(rules, args.max_plies)
            reply(True)
        elif cmd == "komi" and not args.no_rules:
            k = float(rest[0]) if len(rest) == 1 else 0.0
            ok = (k * 2) == int(k * 2) and int(k * 2) % 2 == 1
            if ok:
                rules["komi"] = k
            reply(ok, "" if ok else "bad komi")
        elif cmd == "get_komi" and not args.no_rules:
            reply(True, "%g" % rules["komi"])
        elif cmd == "kata-set-rule" and not args.no_rules:
            ok = len(rest) == 2 and rest[0] in ("blackInitialWalls", "whiteInitialWalls", "repetitionDrawCount") \
                and not g.moves
            if ok:
                rules[rest[0]] = int(rest[1])
                g = Game(rules, args.max_plies)
            reply(ok, "" if ok else "cannot set rule")
        elif cmd == "play":
            ok = len(rest) == 2 and g.play(rest[0].lower()[0], rest[1].lower())
            reply(ok, "" if ok else "illegal move")
        elif cmd in ("genmove", "kata-genmove_analyze") and (cmd == "genmove" or args.kata):
            genmoves += 1
            if genmoves == args.crash_after:
                sys.exit(3)
            if genmoves == args.hang_after:
                time.sleep(3600)
            c = rest[0].lower()[0]
            legal = g.legal(c)
            mv = "z9" if args.illegal else legal[0]
            if args.shuffle and not args.illegal:
                # Sideways from column e and back.
                x = g.pos[c][0]
                mv = "%s%d" % (COLS[3 if x == 4 else 4], g.pos[c][1])
            if not args.illegal:
                g.play(c, mv)
            if cmd == "genmove":
                reply(True, mv)
            else:
                wr, lead, sp = (0.7, 2.5, 3.0) if c == "b" else (0.2, -1.5, -2.0)
                other_mv = "a1h"
                sys.stdout.write(
                    "=\ninfo move %s visits 9 edgeVisits 9 utility 0.1 winrate %g scoreMean %g scoreStdev 2 "
                    "scoreLead %g scoreSelfplay %g noResultValue 0 prior 0.5 lcb 0.5 utilityLcb 0 weight 9 order 0 "
                    "pv %s info move %s visits 1 edgeVisits 1 utility 0 winrate 0.5 scoreMean 0 scoreStdev 2 "
                    "scoreLead 0 scoreSelfplay 0 noResultValue 0.5 prior 0.1 lcb 0 utilityLcb 0 weight 1 order 1 "
                    "pv %s rootInfo visits 10 utility 0.1 winrate %g scoreMean %g scoreStdev 2 scoreLead %g "
                    "scoreSelfplay %g weight 10\nplay %s\n\n" % (
                        mv, wr, lead, lead, sp, mv, other_mv, other_mv, wr, lead, lead, sp, mv))
                sys.stdout.flush()
        elif cmd == "legal_moves" and not args.no_legal:
            reply(True, " ".join(g.legal()))
        elif cmd == "winner":
            reply(True, g.winner() or "none")
        elif cmd == "dist":
            reply(True, "B: %d W: %d" % (g.dist("b"), g.dist("w")))
        elif cmd == "printsgf":
            body = "".join(";%s[%s]" % (c.upper(), m) for c, m in g.moves)
            res = g.result()
            dr = g.draw_reason()
            ru = "Quoridor" if not rules["repetitionDrawCount"] else \
                "Quoridor:repetitionDrawCount=%d" % rules["repetitionDrawCount"]
            reply(True, "(;FF[4]GM[1]SZ[17]PB[]PW[]KM[%g]WB[%d]WW[%d]RU[%s]%s%s%s)" % (
                rules["komi"], rules["blackInitialWalls"], rules["whiteInitialWalls"], ru,
                "" if res is None else "RE[%s]" % res, "" if dr is None else "DR[%s]" % dr, body))
        elif cmd == "showboard":
            reply(True, "line one\nline two\nline three")
        else:
            reply(False, "unknown command")


if __name__ == "__main__":
    main()
