#!/usr/bin/env python3
"""A tiny QTP engine for tests. It plays a toy race game, not Quoridor:

- each pawn moves one row towards its goal (Black e9 -> row 1, White e1 -> row 9) or one column sideways;
- "walls" a1h..c1h are legal while the player has walls left (10 each) and do nothing (they exist to
  exercise opening weights and wall counting);
- the game ends when a pawn reaches its goal row, and is scored like KataQuoridor (docs/QuoridorIOv2.md): the margin
  is the other pawn's distance, the tempo t = margin if White arrived, 1 - margin if Black did, and White wins iff
  t + komi > 0 (komi -0.5 by default); the result is the lead |t + komi|;
- after --max-plies plies (default 300) without an arrival, the game is a draw;
- with kata-set-rule repetitionDrawCount N > 0, so is the N-th occurrence of a position (both pawns and the side to
  move; the toy walls reset the history, like real ones), and printsgf says DR[repetition] (DR[maxPlies] for the
  ply limit).

It speaks the same QTP subset the arena uses, including komi and kata-set-rule blackInitialWalls / whiteInitialWalls /
repetitionDrawCount (the walls only show in printsgf), so it can serve as both arbiter and player. It also answers
the commands the play GUI (python/play_gui) uses: name, version, walls, undo, kata-set-param maxVisits,
kata-genmove_analyze, kata-search_analyze, kata-analyze (streams until the next command) and kata-raw-nn. Their
output imitates KataQuoridor's: analysis values are from the side to move, kata-raw-nn values from White's, a 1-visit
search prints no "info" line, and kata-raw-nn on a finished game kills the process (like KataQuoridor 0.1.0's
"legalCount > 0" assert).

Flags: --shuffle (genmove steps sideways and back instead of forward, so positions repeat), --illegal (genmove
answers "z9"), --crash-after N (exit after N genmoves), --hang-after N (stop answering after N genmoves), --no-legal
(unknown legal_moves), --no-rules (unknown komi / kata-set-rule, like SimpleQuoridor), --max-plies N,
--think-delay S (sleep S seconds in every genmove variant), --kata (kata-genmove_analyze gives fixed values instead:
winrate 0.7, scoreLead 2.5, scoreSelfplay 3 for Black to move; winrate 0.2, scoreLead -1.5, scoreSelfplay -2 for
White). genmove and kata-genmove_analyze play the same move.
"""
import argparse
import math
import sys
import threading
import time

COLS = "abcdefghi"
WALLS = 10


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
        if mv[-1] in "hv":
            self.since_wall = []
        self.since_wall.append(self.key())
        return True

    def dist(self, c):
        return abs(self.pos[c][1] - self.goal(c))

    def copy(self):
        g = Game(self.rules, self.max_plies)
        for c, mv in self.moves:
            g.play(c, mv)
        return g

    def undo(self):
        if not self.moves:
            return False
        g = Game(self.rules, self.max_plies)
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
    ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kata", action="store_true")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--think-delay", type=float, default=0.0)
    args = ap.parse_args()
    rules = dict(STANDARD_RULES)
    g = Game(rules, args.max_plies)
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

    def choose(c):
        """The move genmove and kata-genmove_analyze play for c (see --illegal, --shuffle)."""
        if args.illegal:
            return "z9"
        if args.shuffle:  # sideways from column e and back
            x = g.pos[c][0]
            return "%s%d" % (COLS[3 if x == 4 else 4], g.pos[c][1])
        return g.legal(c)[0]

    streamer = None   # (thread, stop event, is_cancellable) of a running analysis

    def stop_streamer(cancelled=False):
        nonlocal streamer
        if streamer is not None:
            t, stop, is_cancellable = streamer
            stop.set()
            t.join()
            streamer = None
            if is_cancellable:
                sys.stdout.write("play cancelled\n\n" if cancelled else "\n")
            else:
                sys.stdout.write("\n")
            sys.stdout.flush()

    def stream(c, interval, stop, is_cancellable=False):
        nonlocal streamer
        visits = 2
        while not stop.wait(interval):
            sys.stdout.write(analysis(g, c, visits) + "\n")
            sys.stdout.flush()
            if is_cancellable and max_visits > 0 and visits >= max_visits:
                sys.stdout.write("play " + g.legal(c)[0] + "\n\n")
                sys.stdout.flush()
                streamer = None
                break
            visits *= 2

    for line in sys.stdin:
        parts = line.split()
        if not parts:
            stop_streamer(cancelled=True)
            continue
        stop_streamer(cancelled=False)
        cmd, rest = parts[0], parts[1:]
        if cmd == "quit":
            reply(True)
            return
        elif cmd == "known_command":
            known = {"play", "genmove", "clear_board", "winner", "dist", "printsgf", "quit", "known_command",
                     "showboard", "name", "version", "walls", "undo", "kata-set-param", "kata-genmove_analyze",
                     "kata-search_analyze", "kata-search_analyze_cancellable", "kata-raw-nn", "kata-analyze", "stop"}
            if not args.no_legal:
                known.add("legal_moves")
            if not args.no_rules:
                known.update({"komi", "get_komi", "kata-set-rule"})
            reply(True, "true" if rest and rest[0] in known else "false")
        elif cmd == "name":
            reply(True, "MockQuoridor")
        elif cmd == "version":
            reply(True, "0.0 (mock)")
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
            c = color_arg(rest[0]) if rest else None
            if len(rest) != 2 or c is None:
                reply(False, "illegal move")
            elif g.winner():
                reply(False, "game is over")
            else:
                ok = g.play(c, rest[1].lower())
                reply(ok, "" if ok else "illegal move")
        elif cmd == "kata-genmove_analyze" and args.kata:
            begin_genmove()
            c = rest[0].lower()[0]
            mv = choose(c)
            if not args.illegal:
                g.play(c, mv)
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
        elif cmd == "undo":
            reply(True) if g.undo() else reply(False, "cannot undo")
        elif cmd == "genmove":
            begin_genmove()
            c = color_arg(rest[0])
            mv = choose(c)
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
            mv = choose(c) if cmd == "kata-genmove_analyze" else g.legal(c)[0]
            if cmd == "kata-genmove_analyze" and not args.illegal:
                g.play(c, mv)
            reply(True, "\n" + (text + "\n" if text else "") + "play " + mv)
        elif cmd in ("kata-analyze", "kata-search_analyze_cancellable"):
            c = color_arg(rest[0]) if rest and color_arg(rest[0]) else g.to_move
            if g.winner() or c != g.to_move:
                reply(False, "cannot analyze")
                continue
            interval = float(rest[1]) / 100 if len(rest) > 1 and rest[1].isdigit() else 0.1
            sys.stdout.write("=\n")
            sys.stdout.flush()
            stop = threading.Event()
            is_cancellable = (cmd == "kata-search_analyze_cancellable")
            t = threading.Thread(target=stream, args=(c, max(0.01, interval), stop, is_cancellable), daemon=True)
            t.start()
            streamer = (t, stop, is_cancellable)
        elif cmd == "stop":
            reply(True)
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
