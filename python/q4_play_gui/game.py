"""Game controller of the Q4 play GUI: one KataQuoridor QTP process (q4qtp), one game, many HTTP threads.

The engine is the only rules authority: legality, turn order, eliminations, the winner and undo come from its
answers (`play`, `legal_moves`, `eliminate`, `undo`, `q4-position`). The controller keeps the event list, a cursor
(the number of events applied in the engine: the engine always stands at position `cursor`) and one snapshot per
position (the engine's `q4-position`, the legal moves and the raw net value of that position, and the search values
the analysis stored for it).

Undo / redo / review are all engine moves: `undo` takes back the last applied event, `redo` replays the next one
(the same move, the same position), so a review of an earlier position is the engine at that position.
Playing a move there drops the events after the cursor.

Locking: `engine_lock` serialises every engine command and is held for a whole search; `lock` protects the
Python-side state and is only held briefly, so `state()` answers while the AI thinks. When both are needed,
`engine_lock` is taken first. Every command goes through `_session()`, which stops a running analysis first and
starts it again afterwards when it is on (the analysis streams until stopped: the same design as Duel's game.py).
"""
import contextlib
import json
import os
import re
import threading

from . import evaluation as ev
from .engine import QTPCrash, QTPError, QTPTimeout, StreamingQTP

BOTS = ("greedy", "basher", "randomPawn", "random", "grudge")
OPPONENT_MODES = ("maxn", "style", "expect")
BIG_VISITS = 1000000000        # "unlimited" visits (empty max visits field)
MAX_AI_MOVES_PER_CALL = 20     # "AI plays until a human seat": the browser repeats the call
DEFAULT_RULES = {"maxPlies": 400, "repetitionDrawCount": 0, "initialWalls": [7, 7, 7, 7]}
RE_RULES = re.compile(r"^Q4:")
SGF_NODE_RE = re.compile(r";(EL|S|W|N|E)\[([^\]]*)\]")
KEEP = object()   # "not given" for set_seats (the "me" argument)


class ActionError(Exception):
    """A request that cannot be carried out in the current state (HTTP 4xx)."""

    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


class EngineDown(Exception):
    """The engine process died, hung or never started (HTTP 503). The state carries the details."""


def default_seat():
    return {"kind": "human", "visits": 0, "bot": "greedy", "target": None}


def seat_mask_text(mask):
    return ",".join(ev.SEAT_LETTERS[s] for s in range(4) if mask >> s & 1) or "none"


class Q4Game:
    def __init__(self, argv, log_dir, move_timeout=600.0, default_visits=400, interval_cs=25):
        self.move_timeout = move_timeout
        self.engine = StreamingQTP("q4qtp", argv, stderr_path=os.path.join(log_dir, "q4qtp.stderr.log"),
                                   default_timeout=move_timeout)
        self.lock = threading.Lock()
        self.engine_lock = threading.Lock()
        self.status = "down"                 # "ready" | "down"
        self.message = "engine not started"
        self.restarts = 0
        self.rules = {"maxPlies": 400, "repetitionDrawCount": 0, "initialWalls": [7, 7, 7, 7]}
        self.seats = [default_seat() for _ in range(4)]
        self.me = None                       # the seat of "me" (style / expect own-seat mask), or None: the humans
        self.settings = {"maxVisits": default_visits, "opponentMode": "maxn", "dynamicScoreUtilityFactor": 0.3,
                         "q4LeadWeightEps": 0.02, "cpuctExploration": 1.1, "rootPolicyTemperature": 1.1}
        self.events = []                     # every event of the game, applied or not (see cursor)
        self.cursor = 0                      # events applied in the engine
        self.snaps = []                      # snaps[k] = the position after k applied events, len(events) + 1 at most
        self.analysis = {"on": False, "cs": interval_cs, "cap": 0, "ply": None, "root": None, "moves": [],
                         "done": False}
        self._stream = None                  # token of the running analysis (None: no analysis running)
        self._stream_cmd = None
        self._engine_visits = None           # maxVisits the engine currently has

    # -- engine plumbing ----------------------------------------------------------------------------------
    def _fail(self, e):
        with self.lock:
            self.status = "down"
            self.message = str(e)
            self._stream = None
        return EngineDown(str(e))

    def _cmd(self, command, timeout=None):
        """One command, its answer text. An "?" answer is an ActionError; a dead or hung engine an EngineDown."""
        try:
            ok, text = self.engine.send(command, timeout=timeout or self.move_timeout)
        except (QTPCrash, QTPTimeout) as e:
            raise self._fail(e)
        except QTPError as e:
            raise self._fail(e)
        if not ok:
            raise ActionError(text)
        return text

    def start_engine(self):
        with self.engine_lock:
            self._start_locked()

    def _start_locked(self):
        """Start the process, configure it and replay the applied events (the restart after a crash)."""
        self._stream = None
        self._engine_visits = None
        try:
            self.engine.kill()
            self.engine.start()
        except (OSError, ValueError) as e:
            raise self._fail(e)
        with self.lock:
            rules_text = self._rules_text()
            mode = self.settings["opponentMode"]
            visits = self.settings["maxVisits"]
            mask = self._own_mask()
            eps = self.settings["q4LeadWeightEps"]
            dsuf = self.settings["dynamicScoreUtilityFactor"]
            cpuct = self.settings["cpuctExploration"]
            rpt = self.settings["rootPolicyTemperature"]
            applied = self.events[:self.cursor]
        self._cmd("clear_board")
        self._cmd("set_rules " + rules_text)
        self._cmd("q4-set-param q4OpponentMode " + mode)
        self._cmd("q4-set-param q4OwnSeats %d" % mask)
        self._cmd("q4-set-param q4LeadWeightEps %s" % eps)
        self._cmd("q4-set-param dynamicScoreUtilityFactor %s" % dsuf)
        self._cmd("q4-set-param cpuctExploration %s" % cpuct)
        self._cmd("q4-set-param rootPolicyTemperature %s" % rpt)
        self._cmd("q4-set-param maxVisits %d" % visits)
        self._engine_visits = visits
        old = self.snaps
        self.snaps = [self._snapshot_cmd()]
        for k, event in enumerate(applied):
            self._apply_cmd(event)
            self.snaps.append(self._snapshot_cmd())
        for k, s in enumerate(old[:len(self.snaps)]):
            if s.get("eval") is not None and self.snaps[k].get("eval") is None:
                self.snaps[k]["eval"] = s["eval"]
        with self.lock:
            self.status = "ready"
            self.message = "ready"

    def restart(self):
        with self.engine_lock:
            self.restarts += 1
            self._start_locked()

    def close(self):
        with self.engine_lock:
            self._stream = None
            self.engine.close()

    def _rules_text(self):
        r = self.rules
        return "Q4:maxPlies=%d,repetitionDrawCount=%d,initialWalls=%s" % (
            r["maxPlies"], r["repetitionDrawCount"], ",".join(str(w) for w in r["initialWalls"]))

    def _own_mask(self):
        if self.me is not None:
            return 1 << self.me
        return sum(1 << s for s in range(4) if self.seats[s]["kind"] == "human")

    # -- the position ---------------------------------------------------------------------------------------
    def _snapshot_cmd(self):
        pos = json.loads(self._cmd("q4-position"))
        snap = {"pos": pos, "legal": [], "raw": None, "eval": None}
        if not pos["finished"]:
            snap["legal"] = self._cmd("legal_moves").split()
            try:
                snap["raw"] = ev.parse_rawnn(self._cmd("q4-rawnn"))
            except ActionError:
                snap["raw"] = None
        return snap

    def _apply_cmd(self, event):
        if event.get("elim"):
            self._cmd("eliminate %d" % (event["seat"] + 1))
        else:
            self._cmd("play %d %s" % (event["seat"] + 1, event["a"]))

    def _to_move(self):
        return self.snaps[self.cursor]["pos"]["toMove"]

    def _position(self):
        return self.snaps[self.cursor]["pos"]

    # -- analysis streaming (the same design as Duel's game.py) ----------------------------------------
    @contextlib.contextmanager
    def _session(self):
        """Hold the engine for one command: stop a running analysis, restart it afterwards when it is on."""
        with self.engine_lock:
            if self.status != "ready":
                raise EngineDown("the engine is not running: %s" % self.message)
            self._stop_analysis_locked()
            try:
                yield
            finally:
                try:
                    self._maybe_start_analysis_locked()
                except EngineDown:
                    pass

    def _stop_analysis_locked(self):
        with self.lock:
            if self._stream is None:
                return
            self._stream = None
            self._stream_cmd = None
        try:
            self.engine.write("")                                   # a line cancels the analysis
            self.engine.read_response("q4-search_analyze_cancellable", self.move_timeout)
        except QTPError as e:
            raise self._fail(e)
        finally:
            self.engine.streaming = False
            self.engine.on_info = None

    def _maybe_start_analysis_locked(self):
        with self.lock:
            a = self.analysis
            if not (a["on"] and self.status == "ready" and self._stream is None):
                return
            if self._position()["finished"]:
                return
            ply = self.cursor
            if a["ply"] != ply:
                a.update(ply=ply, root=None, moves=[], done=False)
            if a["done"]:
                return
            token = object()
            self._stream = token
            self._stream_cmd = "q4-search_analyze_cancellable"
            cap = a["cap"] or BIG_VISITS
            cs = a["cs"]
        self.engine.on_info = lambda line: self._on_info(token, ply, line)
        self.engine.streaming = True
        try:
            self.engine.write("q4-search_analyze_cancellable %d %d" % (cs, cap))
        except QTPError as e:
            raise self._fail(e)

    def _on_info(self, token, ply, line):
        """One analysis report (the engine's reader thread)."""
        info = ev.parse_info(line)
        if info is None:
            return
        with self.lock:
            if self._stream is not token:
                return
            a = self.analysis
            if info["kind"] == "root":
                a["root"] = info
                a["moves"] = []
                if info["value"] is not None and ply < len(self.snaps):
                    self.snaps[ply]["eval"] = {"p": info["value"], "visits": info["visits"], "lead": info["lead"],
                                               "utility": info["utility"]}
                if info["visits"] >= (a["cap"] or BIG_VISITS):
                    a["done"] = True          # the engine stops by itself and answers "done"
            else:
                a["moves"].append(info)

    # -- reading state --------------------------------------------------------------------------------------
    def state(self):
        with self.lock:
            snap = self.snaps[self.cursor] if self.snaps else None
            pos = snap["pos"] if snap else None
            a = self.analysis
            evals = [None if s["eval"] is None else s["eval"]["p"] for s in self.snaps]
            events = []
            for k, e in enumerate(self.events):
                item = dict(e)
                item["applied"] = k < self.cursor
                events.append(item)
            mode = "analysis" if all(s["kind"] == "human" for s in self.seats) else "play"
            return {
                "engine": {"status": self.status, "message": self.message, "restarts": self.restarts},
                "rules": dict(self.rules, initialWalls=list(self.rules["initialWalls"])),
                "seats": [dict(s) for s in self.seats],
                "me": self.me,
                "mode": mode,
                "settings": dict(self.settings),
                "events": events,
                "cursor": self.cursor,
                "length": len(self.events),
                "position": None if pos is None else dict(pos, legal=snap["legal"], raw=snap["raw"],
                                                          eval=snap["eval"]),
                "evals": evals,
                "analysis": {"on": a["on"], "running": self._stream is not None, "ply": a["ply"],
                             "root": a["root"], "moves": list(a["moves"]), "done": a["done"], "cs": a["cs"],
                             "cap": a["cap"]},
            }

    def legal_for(self, seat):
        """The legal actions of any seat in the current position (for a click on a pawn of another seat)."""
        with self._session():
            return self._cmd("legal_moves %d" % (seat + 1)).split()

    def search_params(self):
        """q4-search-params: the effective search values of the engine (the panel of the GUI)."""
        with self._session():
            return self._cmd("q4-search-params")

    # -- moves ---------------------------------------------------------------------------------------------
    def _check_ready(self):
        if self._position()["finished"]:
            raise ActionError("The game is over (undo to continue)")

    def _record(self, event, snap):
        """Append an applied event at the cursor (dropping the events after it) and keep its snapshot."""
        del self.events[self.cursor:]
        del self.snaps[self.cursor + 1:]
        self.events.append(event)
        self.snaps.append(snap)
        self.cursor += 1

    def play(self, text):
        """A pawn move or a wall for the seat to move (any seat, human or not)."""
        text = text.strip().lower()
        if not ev.MOVE_RE.match(text):
            raise ActionError("Not a move: %r (pawn e.g. f2, wall e5h or j10v)" % text, 400)
        with self._session():
            self._check_ready()
            seat = self._to_move()
            event = {"seat": seat, "a": text}
            self._cmd("play %d %s" % (seat + 1, text))
            snap = self._snapshot_cmd()
            with self.lock:
                self._record(event, snap)
        return {"ok": True}

    def eliminate(self, seat, confirmed=False):
        if not 0 <= seat <= 3:
            raise ActionError("No such seat", 400)
        with self._session():
            pos = self._position()
            if pos["finished"]:
                raise ActionError("The game is over (undo to continue)")
            if not pos["alive"][seat]:
                raise ActionError("%s is already eliminated" % ev.SEAT_LETTERS[seat])
            alive_after = sum(1 for s in range(4) if pos["alive"][s] and s != seat)
            if alive_after == 1 and not confirmed:
                raise ActionError("Eliminating %s leaves one seat: the game ends. Confirm first."
                                  % ev.SEAT_LETTERS[seat], 409)
            event = {"seat": seat, "elim": True}
            self._cmd("eliminate %d" % (seat + 1))
            snap = self._snapshot_cmd()
            with self.lock:
                self._record(event, snap)
        return {"ok": True}

    def undo(self):
        with self._session():
            with self.lock:
                if self.cursor == 0:
                    raise ActionError("Nothing to undo")
            self._cmd("undo")
            with self.lock:
                self.cursor -= 1
        return {"ok": True}

    def redo(self):
        with self._session():
            with self.lock:
                if self.cursor >= len(self.events):
                    raise ActionError("Nothing to redo")
                event = self.events[self.cursor]
            self._apply_cmd(event)
            with self.lock:
                self.cursor += 1
        return {"ok": True}

    def goto(self, ply):
        try:
            target = int(ply)
        except (TypeError, ValueError):
            raise ActionError("ply must be a number", 400)
        with self._session():
            with self.lock:
                if not 0 <= target <= len(self.events):
                    raise ActionError("ply out of range (0..%d)" % len(self.events), 400)
            while self.cursor > target:
                self._cmd("undo")
                with self.lock:
                    self.cursor -= 1
            while self.cursor < target:
                with self.lock:
                    event = self.events[self.cursor]
                self._apply_cmd(event)
                with self.lock:
                    self.cursor += 1
        return {"ok": True}

    # -- AI moves -------------------------------------------------------------------------------------------
    def _ai_move_locked(self):
        """One AI move for the seat to move (engine_lock held, the analysis stopped)."""
        with self.lock:
            seat_idx = self._to_move()
            spec = dict(self.seats[seat_idx])
            visits = spec.get("visits") or self.settings["maxVisits"]
        if spec["kind"] == "human":
            raise ActionError("%s is a human seat: play its move yourself" % ev.SEAT_LETTERS[seat_idx])
        if spec["kind"] == "net":
            if visits != self._engine_visits:
                self._cmd("q4-set-param maxVisits %d" % visits)
                self._engine_visits = visits
            text = self._cmd("genmove %d" % (seat_idx + 1))
        else:
            target = spec.get("target")
            tail = " %d" % (target + 1) if target is not None else ""
            text = self._cmd("q4-genmove-bot %s%s" % (spec["bot"], tail))
        event = {"seat": seat_idx, "a": text.strip(), "ai": True}
        snap = self._snapshot_cmd()
        with self.lock:
            self._record(event, snap)

    def ai_move(self, all_seats=False):
        """The AI plays the seat to move; with all_seats, the AI seats up to a human seat (at most 20 moves per call)."""
        with self._session():
            with self.lock:
                self._check_ready()
            moves = 0
            while True:
                with self.lock:
                    if self._position()["finished"]:
                        break
                    kind = self.seats[self._to_move()]["kind"]
                if kind == "human":
                    if moves == 0 and not all_seats:
                        raise ActionError("the seat to move is a human seat: play its move yourself")
                    break
                self._ai_move_locked()
                moves += 1
                if not all_seats or moves >= MAX_AI_MOVES_PER_CALL:
                    break
        return {"ok": True, "moves": moves, "more": bool(all_seats and moves >= MAX_AI_MOVES_PER_CALL)}

    # -- settings -------------------------------------------------------------------------------------------
    def set_seats(self, seats, me=KEEP):
        """Seat kinds, visits, bots and targets. `me` (a seat index or None) is the own seat of style / expect;
        left out, it stays as it was. The engine's own-seat mask follows (when the engine is up)."""
        checked = []
        for i, s in enumerate(seats):
            kind = s.get("kind", "human")
            if kind not in ("human", "net", "bot"):
                raise ActionError("seat %s: kind must be human, net or bot" % ev.SEAT_LETTERS[i], 400)
            try:
                visits = int(s.get("visits") or 0)
            except (TypeError, ValueError):
                raise ActionError("visits must be a number", 400)
            if visits < 0:
                raise ActionError("visits must be >= 0", 400)
            bot = s.get("bot", "greedy")
            target = None
            if kind == "bot":
                if bot not in BOTS:
                    raise ActionError("unknown bot %r (%s)" % (bot, ", ".join(BOTS)), 400)
                if bot == "grudge":
                    if s.get("target") is None or not 0 <= int(s["target"]) <= 3:
                        raise ActionError("grudge needs a target seat (S W N E)", 400)
                    target = int(s["target"])
            checked.append({"kind": kind, "visits": visits, "bot": bot, "target": target})
        if me is not KEEP and me is not None and not 0 <= me <= 3:
            raise ActionError("me must be a seat or none", 400)
        with self.lock:
            self.seats = checked
            if me is not KEEP:
                self.me = me
        if self.status == "ready":
            with self._session():
                self._cmd("q4-set-param q4OwnSeats %d" % self._own_mask())
        return {"ok": True}

    def set_settings(self, values):
        """q4-set-param for each known key; a change clears the tree. Keys: maxVisits, opponentMode,
        dynamicScoreUtilityFactor, q4LeadWeightEps, cpuctExploration, rootPolicyTemperature."""
        names = {"maxVisits": "maxVisits", "opponentMode": "q4OpponentMode",
                 "dynamicScoreUtilityFactor": "dynamicScoreUtilityFactor", "q4LeadWeightEps": "q4LeadWeightEps",
                 "cpuctExploration": "cpuctExploration", "rootPolicyTemperature": "rootPolicyTemperature"}
        clean = {}
        for k, v in values.items():
            if k not in names:
                raise ActionError("unknown setting %r" % k, 400)
            if k == "opponentMode":
                if v not in OPPONENT_MODES:
                    raise ActionError("opponentMode must be maxn, style or expect", 400)
                clean[k] = v
            elif k == "maxVisits":
                clean[k] = int(v)
                if clean[k] < 1:
                    raise ActionError("maxVisits must be >= 1", 400)
            else:
                clean[k] = float(v)
        with self._session():
            for k, v in clean.items():
                if k == "opponentMode":
                    self._cmd("q4-set-param q4OpponentMode %s" % v)
                else:
                    self._cmd("q4-set-param %s %s" % (names[k], v))
                    if k == "maxVisits":
                        self._engine_visits = v
                with self.lock:
                    self.settings[k] = v
        return {"ok": True}

    def set_analysis(self, on=None, cs=None, cap=None):
        with self._session():
            with self.lock:
                a = self.analysis
                if on is not None:
                    a["on"] = bool(on)
                if cs is not None:
                    cs = int(cs)
                    if cs < 1:
                        raise ActionError("interval must be at least 1 centisecond", 400)
                    a["cs"] = cs
                if cap is not None:
                    cap = int(cap or 0)
                    if cap < 0:
                        raise ActionError("max visits must be >= 0", 400)
                    a["cap"] = cap
                    a["done"] = False
                    a["ply"] = None
        return {"ok": True}

    def new_game(self, rules=None, seats=None):
        """An empty game with the given rules (and seat settings), the engine's board cleared."""
        if rules is not None:
            self.rules = self._check_rules(rules)
        with self._session():
            self._cmd("clear_board")
            self._cmd("set_rules " + self._rules_text())
            snap = self._snapshot_cmd()
            with self.lock:
                self.events = []
                self.cursor = 0
                self.snaps = [snap]
                self.analysis.update(ply=None, root=None, moves=[], done=False)
        if seats is not None:
            self.set_seats(seats)
        return {"ok": True}

    @staticmethod
    def _check_rules(rules):
        try:
            max_plies = int(rules.get("maxPlies", 400))
            rep = int(rules.get("repetitionDrawCount", 0))
            walls = [int(w) for w in rules.get("initialWalls", [7, 7, 7, 7])]
        except (TypeError, ValueError, AttributeError):
            raise ActionError("rules: maxPlies, repetitionDrawCount (numbers) and initialWalls (4 numbers)", 400)
        if not 1 <= max_plies <= 1000 or not 0 <= rep <= 50 or len(walls) != 4 or not all(0 <= w <= 10 for w in walls):
            raise ActionError("rules out of range (maxPlies 1..1000, repetitionDrawCount 0..50, walls 0..10)", 400)
        return {"maxPlies": max_plies, "repetitionDrawCount": rep, "initialWalls": walls}

    # -- loading and replaying ------------------------------------------------------------------------------
    def load(self, text, ply=None):
        """A Q4 .sgfs line or a pasted move list. The game is replayed into the engine (every move and
        elimination checked against the rules and the seat to move); the cursor is left at `ply` (default: the end)."""
        text = text.strip()
        if not text:
            raise ActionError("nothing to load", 400)
        if text.startswith("(;"):
            from q4.record import Q4Record   # the Python reader of the .sgfs format (python/q4/record.py)
            try:
                rec = Q4Record.from_sgf_line(text.splitlines()[0])
            except (ValueError, IndexError) as e:
                raise ActionError("not a Q4 game record: %s" % e, 400)
            rules = self._check_rules(rec.rules)
            events = [{"elim": e["elim"] - 1} if "elim" in e else {"a": e["a"]} for e in rec.events]
        else:
            try:
                events = ev.parse_move_list(text)
            except ValueError as e:
                raise ActionError(str(e), 400)
            rules = None
        if not events:
            raise ActionError("the move list has no moves", 400)
        with self._session():
            if rules is not None:
                self.rules = rules
            self._cmd("clear_board")
            self._cmd("set_rules " + self._rules_text())
            snap = self._snapshot_cmd()
            with self.lock:
                self.events = []
                self.cursor = 0
                self.snaps = [snap]
            for k, e in enumerate(events):
                with self.lock:
                    seat = self._to_move()
                    finished = self._position()["finished"]
                if finished:
                    raise ActionError("event %d: the game is over before it" % (k + 1), 400)
                if "elim" in e:
                    event = {"seat": e["elim"], "elim": True}
                    self._cmd("eliminate %d" % (e["elim"] + 1))
                else:
                    if "seat" in e and e["seat"] != seat:
                        raise ActionError("event %d: %s is not the seat to move (%s is)" % (
                            k + 1, ev.SEAT_LETTERS[e["seat"]], ev.SEAT_LETTERS[seat]), 400)
                    event = {"seat": seat, "a": e["a"]}
                    self._cmd("play %d %s" % (seat + 1, e["a"]))
                snap = self._snapshot_cmd()
                with self.lock:
                    self.events.append(event)
                    self.snaps.append(snap)
                    self.cursor = len(self.events)
            target = len(self.events) if ply is None else int(ply)
            if not 0 <= target <= len(self.events):
                raise ActionError("ply out of range", 400)
            while self.cursor > target:
                self._cmd("undo")
                with self.lock:
                    self.cursor -= 1
        return {"ok": True, "events": len(events)}

    # -- record out -----------------------------------------------------------------------------------------
    def sgf(self):
        """The game as the engine writes it (printrecord: the applied events), the evaluation of each
        position as a comment on the move that follows it (Q4Sgf.md §4)."""
        with self._session():
            line = self._cmd("printrecord")
            with self.lock:
                evals = [s["eval"] for s in self.snaps[:self.cursor]]
                applied = self.events[:self.cursor]
            # The root's type<i> of each seat follows the seat settings now: human, search (a net) or the bot's name.
            with self.lock:
                kinds = [dict(s) for s in self.seats]
            for i, sp in enumerate(kinds):
                label = {"human": "human", "net": "search"}.get(sp["kind"], sp.get("bot", "bot"))
                line = re.sub(r"type%d=[^,\]]*" % i, "type%d=%s" % (i, label), line, count=1)
            nodes = list(SGF_NODE_RE.finditer(line))
            if len(nodes) != len(applied):
                return line
            out = []
            last = 0
            for k, m in enumerate(nodes):
                out.append(line[last:m.end()])
                last = m.end()
                e = evals[k] if k < len(evals) else None
                if applied[k].get("elim") or e is None:
                    continue
                p = e["p"]
                lead = ",".join("%.1f" % x for x in (e["lead"] or [0, 0, 0, 0]))
                out.append("C[%.2f %.2f %.2f %.2f %.2f v=%d weight=1.00 lead=%s]" % (
                    p[0], p[1], p[2], p[3], p[4], e["visits"], lead))
            out.append(line[last:])
            return "".join(out)
