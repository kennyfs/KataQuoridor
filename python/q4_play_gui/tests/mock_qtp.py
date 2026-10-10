#!/usr/bin/env python3
"""A mock of KataQuoridor's q4qtp for the play GUI tests: the same command names, answers and framing, with simple
rules (pawns step to free orthogonal neighbours, walls are only counted, no winning path is checked).

The controller's logic (cursor, undo / redo, eliminations, seat settings, the analysis stream, the SGF round trip) is
what these tests exercise; the real rules are the engine's and are tested by test_engine_real.py.

Env: MOCK_CRASH_ON=<command prefix>  exit(3) when such a command arrives (engine death tests);
     MOCK_HANG_ON=<command prefix>   never answer it (timeout tests).
"""
import copy
import json
import os
import queue
import re
import sys
import threading
import time

START = [(5, 0), (0, 5), (5, 10), (10, 5)]          # f1 a6 f11 k6
SEATS = "SWNE"


def cell_name(x, y):
    return "%s%d" % (chr(97 + x), y + 1)


def cell_xy(name):
    return ord(name[0]) - 97, int(name[1:]) - 1


class Board:
    def __init__(self, rules):
        self.rules = rules
        self.pawns = list(START)
        self.alive = [True] * 4
        self.walls = []                 # placed wall strings
        self.left = list(rules["initialWalls"])
        self.to_move = 0
        self.plies = 0
        self.finished = False
        self.winner = -1
        self.draw = False

    def next_alive(self, seat):
        for k in range(1, 5):
            s = (seat + k) % 4
            if self.alive[s]:
                return s
        return seat

    def pawn_moves(self, seat):
        x, y = self.pawns[seat]
        out = []
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < 11 and 0 <= ny < 11 and (nx, ny) not in self.occupied_by_others(seat):
                out.append(cell_name(nx, ny))
        return out

    def occupied_by_others(self, seat):
        return {p for s, p in enumerate(self.pawns) if s != seat and self.alive[s]}

    def legal(self, seat):
        moves = self.pawn_moves(seat)
        if self.left[seat] > 0 and self.alive[seat]:
            moves += ["a1h", "b1v"] if "a1h" not in self.walls and "b1v" not in self.walls else []
        return moves

    def to_json(self):
        return {
            "plies": self.plies, "toMove": self.to_move, "finished": self.finished, "winner": self.winner,
            "draw": self.draw, "alive": list(self.alive),
            "pawns": [cell_name(*p) if self.alive[s] else "" for s, p in enumerate(self.pawns)],
            "wallsLeft": list(self.left),
            "dist": [7 if self.alive[s] else 255 for s in range(4)],
            "walls": list(self.walls),
        }


class Mock:
    def __init__(self):
        self.rules = {"maxPlies": 400, "repetitionDrawCount": 0, "initialWalls": [7, 7, 7, 7]}
        self.board = Board(self.rules)
        self.history = []
        self.events = []                 # (kind, seat, text) applied
        self.params = {"maxVisits": 800}
        self.opp = {"q4OpponentMode": "maxn", "q4OwnSeats": 0}
        self.stdin_lines = queue.Queue()
        self.pending = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in sys.stdin:
            self.stdin_lines.put(line.rstrip("\n"))
        self.stdin_lines.put(None)

    def out(self, text=""):
        sys.stdout.write(text)
        sys.stdout.flush()

    def ok(self, msg=""):
        self.out("= %s\n\n" % msg)

    def err(self, msg):
        self.out("? %s\n\n" % msg)

    def snapshot(self):
        self.history.append((copy.deepcopy(self.board), len(self.events)))

    def apply_event(self, kind, seat, text):
        self.snapshot()
        b = self.board
        if kind == "elim":
            b.alive[seat] = False
            if b.to_move == seat:
                b.to_move = b.next_alive(seat)
            if sum(b.alive) == 1:
                b.finished = True
                b.winner = b.alive.index(True)
        else:
            if text[-1] in "hv":
                b.walls.append(text)
                b.left[seat] -= 1
            else:
                b.pawns[seat] = cell_xy(text)
            b.plies += 1
            b.to_move = b.next_alive(seat)
            if b.plies >= b.rules["maxPlies"]:
                b.finished = True
                b.draw = True
        self.events.append((kind, seat, text))

    def handle(self, cmd, args, line):
        crash = os.environ.get("MOCK_CRASH_ON")
        if crash and line.lower().startswith(crash):
            os._exit(3)
        hang = os.environ.get("MOCK_HANG_ON")
        if hang and line.lower().startswith(hang):
            time.sleep(3600)
        b = self.board
        if cmd == "clear_board":
            self.board = Board(self.rules)
            self.history = []
            self.events = []
            return self.ok()
        if cmd == "set_rules":
            if self.board.plies > 0:
                return self.err("Cannot set rules after game has started")
            parts = dict(item.split("=", 1) for item in re.split(r",(?=[A-Za-z])", args[0].removeprefix("Q4:")))
            self.rules = {"maxPlies": int(parts["maxPlies"]), "repetitionDrawCount": int(parts["repetitionDrawCount"]),
                          "initialWalls": [int(w) for w in parts["initialWalls"].split(",")]}
            self.board = Board(self.rules)
            self.history = []
            self.events = []
            return self.ok()
        if cmd == "q4-set-param":
            if len(args) < 2:
                return self.err("Usage: q4-set-param <key> <value>")
            key, value = args[0], args[1]
            if key in ("maxVisits",):
                self.params["maxVisits"] = int(value)
            elif key == "q4OpponentMode":
                if value not in ("maxn", "style", "expect"):
                    return self.err("bad mode")
                self.opp["q4OpponentMode"] = value
            elif key == "q4OwnSeats":
                self.opp["q4OwnSeats"] = int(value)
            elif key in ("dynamicScoreUtilityFactor", "q4LeadWeightEps", "cpuctExploration", "rootPolicyTemperature"):
                self.params[key] = float(value)
            else:
                return self.err("Unknown key: %s" % key)
            return self.ok()
        if cmd == "play":
            seat = int(args[0]) - 1
            if b.finished:
                return self.err("Game is already finished")
            if seat != b.to_move:
                return self.err("Not seat %d's turn (it is seat %d's turn)" % (seat + 1, b.to_move + 1))
            if args[1] not in b.legal(seat):
                return self.err("Illegal move: %s" % args[1])
            self.apply_event("move", seat, args[1])
            return self.ok()
        if cmd == "legal_moves":
            seat = int(args[0]) - 1 if args else b.to_move
            return self.ok(" ".join(b.legal(seat)))
        if cmd == "eliminate":
            seat = int(args[0]) - 1
            if b.finished:
                return self.err("Cannot eliminate in finished game")
            if not b.alive[seat]:
                return self.err("Seat %d is already eliminated" % (seat + 1))
            self.apply_event("elim", seat, "")
            return self.ok()
        if cmd == "undo":
            if not self.history:
                return self.err("Cannot undo: already at start of game")
            self.board, n = self.history.pop()
            self.events = self.events[:n]
            return self.ok()
        if cmd == "q4-position":
            return self.ok(json.dumps(b.to_json(), separators=(",", ":"), sort_keys=True))
        if cmd == "q4-rawnn":
            return self.ok("Symmetry 0, seat %d (North) to move\nWin probability by absolute seat:\n"
                           "  seat 1 (South):  30.00%%\n  seat 2 (West ):  20.00%%\n  seat 3 (North):  25.00%%\n"
                           "  seat 4 (East ):  25.00%%\n  draw          :   0.00%%\n"
                           "Plies to the end of the game: 100.0\nTop legal moves of the search policy (2 legal):\n"
                           "  f2     50.00%%\n  e5h    10.00%%" % (b.to_move + 1))
        if cmd == "genmove":
            if b.finished:
                return self.err("Game is already finished")
            move = b.legal(b.to_move)[0]
            self.apply_event("move", b.to_move, move)
            return self.ok(move)
        if cmd == "q4-genmove-bot":
            if b.finished:
                return self.err("Game is already finished")
            if args[0] not in ("greedy", "basher", "randomPawn", "random", "grudge"):
                return self.err("Unknown Q4 bot type: %s" % args[0])
            if args[0] == "grudge" and len(args) < 2:
                return self.err("grudge needs a target seat: q4-genmove-bot grudge <seat>")
            move = b.legal(b.to_move)[-1]
            self.apply_event("move", b.to_move, move)
            return self.ok(move)
        if cmd == "printrecord":
            nodes = []
            for kind, seat, text in self.events:
                nodes.append(";EL[%s]" % SEATS[seat] if kind == "elim" else ";%s[%s]" % (SEATS[seat], text))
            rules = self.rules
            head = ("(;FF[4]GM[Q4]SZ[11]PS[P1]PW[P2]PN[P3]PE[P4]WS[%d]WW[%d]WN[%d]WE[%d]"
                    "RU[Q4:repetitionDrawCount=%d,maxPlies=%d]RE[%s]C[gtype=normal,startTurnIdx=0,type0=bot,type1=bot,type2=bot,type3=bot]" % (
                        *rules["initialWalls"], rules["repetitionDrawCount"], rules["maxPlies"],
                        "?" if not b.finished else (SEATS[b.winner] if b.winner >= 0 else "0")))
            return self.ok(head + "".join(nodes) + ")")
        if cmd == "to_move":
            return self.ok(str(b.to_move + 1))
        if cmd == "winner":
            return self.ok("none" if not b.finished else ("draw" if b.draw else SEATS[b.winner]))
        if cmd == "q4-search_analyze_cancellable":
            return self.analyze(args)
        if cmd == "quit":
            self.ok()
            sys.exit(0)
        return self.err("unknown command: %s" % cmd)

    def analyze(self, args):
        """The stream of q4-search_analyze_cancellable: "=", info lines per interval, then the answer."""
        cs = int(args[0]) if args else 25
        cap = int(args[1]) if len(args) > 1 else 1000000
        b = self.board
        self.out("=\n")
        visits = 0
        while True:
            try:
                line = self.stdin_lines.get(timeout=cs / 100.0)
            except queue.Empty:
                line = None
            if line is not None:
                self.pending.insert(0, line)     # the cancelling line is the next command
                return self.out("play cancelled\n\n")
            visits += 100
            self.out("info root visits %d value 0.2500,0.2500,0.2500,0.2500,0.0000 utility 0.0,0.0,0.0,0.0 "
                     "lead 0.0,0.0,0.0,0.0\n" % visits)
            for m in b.legal(b.to_move)[:3]:
                self.out("info move %s visits %d prior 0.1000 utility 0.0100 lcb 0.0000 value 0.2500,0.2500,"
                         "0.2500,0.2500,0.0000 pv %s %s\n" % (m, visits // 3, SEATS[b.to_move], m))
            if visits >= cap:
                return self.out("done\n\n")

    def run(self):
        while True:
            line = self.pending.pop(0) if self.pending else self.stdin_lines.get()
            if line is None:
                return
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cmd = parts[0].lower()
            self.handle(cmd, parts[1:], line)


if __name__ == "__main__":
    Mock().run()
