"""Game controller for the play GUI: one KataQuoridor process, one game, many HTTP threads.

The engine is the only rules authority. The controller keeps the move list and, per position, what the
frontend needs to draw it (pawn cells and walls, which follow directly from the engine-accepted moves, plus
the engine's `walls` and `dist` answers and an evaluation). Legality, the winner and the side to move always
come from the engine (`legal_moves`, `winner`, `play` answers).

Locking: `engine_lock` serialises every engine command and is held for a whole search; `lock` protects the
Python-side state and is only held briefly, so `/api/state` answers while the AI thinks. When both are
needed, `engine_lock` is taken first.
"""
import os
import random
import re
import threading
import time

from quoridor_arena.qtp import QTPEngine, QTPError

from . import evaluation as ev

MOVE_RE = re.compile(r"^(?:[a-i][1-9]|[a-h][1-8][hv])$")
START_CELLS = {"b": "e9", "w": "e1"}
WALLS_PER_PLAYER = 10
REQUIRED_COMMANDS = ("play", "undo", "clear_board", "legal_moves", "winner", "dist", "walls",
                     "kata-genmove_analyze", "kata-search_analyze", "kata-raw-nn", "kata-set-param", "printsgf")


def opp(c):
    return "w" if c == "b" else "b"


class ActionError(Exception):
    """A request that cannot be carried out in the current state (HTTP 4xx)."""

    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


class EngineDown(Exception):
    """The engine process died, hung or never started (HTTP 503). The state carries the details."""


def parse_pair(text):
    """Parse "B: 7 W: 9" into {"b": 7, "w": 9}."""
    m = re.search(r"B:\s*(-?\d+)\s+W:\s*(-?\d+)", text)
    if not m:
        raise ValueError("cannot parse %r" % text)
    return {"b": int(m.group(1)), "w": int(m.group(2))}


def is_wall(move):
    return move[-1] in "hv"


def decorate_sgf(sgf, positions, black_name, white_name):
    """Add player names and per-move evaluation comments to `printsgf` output.

    The comments use the format of self-play SGFs, which python/sgfs_viewer understands: the comment on
    move i holds "whiteWin whiteLoss noResult whiteScore [v=visits]" for the position *before* move i,
    from White's perspective. Existing comments (e.g. "result=W+2" on the last move) are kept.
    """
    sgf = re.sub(r"PB\[[^\]]*\]", lambda _: "PB[%s]" % black_name, sgf, count=1)
    sgf = re.sub(r"PW\[[^\]]*\]", lambda _: "PW[%s]" % white_name, sgf, count=1)
    counter = [0]

    def repl(m):
        i = counter[0]
        counter[0] += 1
        old = m.group(3)[2:-1] if m.group(3) else ""
        e = positions[i].get("eval") if i < len(positions) else None
        parts = []
        if e is not None:
            parts.append("%.2f %.2f %.2f %.1f" % (e["white_win"], e["black_win"],
                                                  max(0.0, 1.0 - e["white_win"] - e["black_win"]), -e["black_lead"]))
            if e.get("visits"):
                parts.append("v=%d" % e["visits"])
        if old:
            parts.append(old)
        comment = "C[%s]" % " ".join(parts) if parts else ""
        return ";%s[%s]%s" % (m.group(1), m.group(2), comment)

    return re.sub(r";([BW])\[([^\]]*)\]((?:C\[[^\]]*\])?)", repl, sgf)


class GameController:
    def __init__(self, argv, log_dir, model_name="", move_timeout=600.0, start_timeout=600.0, hint_visits=400,
                 demo_delay=1.0, rng=None):
        self.argv = list(argv)
        self.log_dir = log_dir
        self.stderr_path = os.path.join(log_dir, "engine_stderr.log")
        self.model_name = model_name
        self.move_timeout = move_timeout
        self.start_timeout = start_timeout
        self.hint_visits = hint_visits
        self.demo_delay = demo_delay
        self.rng = rng or random.Random()
        self.lock = threading.Lock()
        self.engine_lock = threading.Lock()
        self.engine = QTPEngine("katago", self.argv, stderr_path=self.stderr_path, default_timeout=60.0)
        self.engine_status = "stopped"   # stopped | starting | ready | error
        self.engine_error = None
        self.engine_name = ""
        self.gen = 0                     # bumped by every new game / restart; stale AI threads check it
        self.started = False
        self.human = "b"                 # "b", "w", or None for AI vs AI
        self.human_choice = "b"
        self.visits = 16
        self.thinking = False
        self.busy = None
        self.ai_error = None
        self.hint = None
        self._reset_position_state()

    def _reset_position_state(self):
        self.moves = []
        self.positions = [self._snapshot([], {"b": WALLS_PER_PLAYER, "w": WALLS_PER_PLAYER}, None, None)]
        self.legal = []
        self.winner = None
        self.margin = None

    @staticmethod
    def _snapshot(moves, walls_left, dist, evaluation):
        pawns = dict(START_CELLS)
        walls = []
        for i, (c, mv) in enumerate(moves):
            if is_wall(mv):
                walls.append({"move": mv, "color": c, "ply": i + 1})
            else:
                pawns[c] = mv  # a pawn move is written as its destination cell, jumps included
        return {"pawns": pawns, "walls": walls, "walls_left": walls_left, "dist": dist, "eval": evaluation}

    # -- engine plumbing (call with engine_lock held) -------------------------------------------------
    def _stderr_tail(self, n=6):
        try:
            with open(self.stderr_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 4000))
                lines = f.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return ""
        lines = [l for l in lines if l.strip() and not l.startswith("====")]
        return "\n".join(lines[-n:])

    def _fail(self, exc):
        tail = self._stderr_tail()
        msg = str(exc) + ("\n" + tail if tail else "")
        with self.lock:
            self.engine_status = "error"
            self.engine_error = msg
            self.thinking = False
            self.busy = None
        self.engine.kill()
        return EngineDown(msg)

    def _send(self, cmd, timeout=None):
        if self.engine_status != "ready" and self.engine_status != "starting":
            raise EngineDown(self.engine_error or "engine is not running")
        try:
            return self.engine.send(cmd, timeout=timeout)
        except QTPError as e:
            raise self._fail(e)

    def _must(self, cmd, timeout=None):
        ok, text = self._send(cmd, timeout)
        if not ok:
            raise self._fail(QTPError("unexpected engine answer to %r: ? %s" % (cmd, text)))
        return text

    def _query_position(self):
        """winner, dist, walls and legal moves of the engine's current position."""
        w = self._must("winner").strip().lower()
        winner = w if w in ("b", "w") else None
        dist = parse_pair(self._must("dist"))
        walls_left = parse_pair(self._must("walls"))
        legal = [] if winner else self._must("legal_moves").split()
        return winner, dist, walls_left, legal

    def _raw_eval(self):
        """Net-only evaluation of the current (unfinished) position. Never call it on a finished game:
        KataQuoridor 0.1.0's kata-raw-nn aborts the process there."""
        ok, text = self._send("kata-raw-nn 0", timeout=self.move_timeout)
        if not ok:
            return None
        try:
            return ev.from_raw_nn(ev.parse_raw_nn(text))
        except ValueError:
            return None

    def _position_eval(self, winner, dist):
        if winner:
            return ev.final_eval(winner, self._margin(winner, dist))
        return self._raw_eval()

    @staticmethod
    def _margin(winner, dist):
        return max(1, dist[opp(winner)])

    def _apply(self, color, move, winner, dist, walls_left, legal, evaluation):
        """Record an engine-accepted move (call with self.lock held)."""
        self.moves.append((color, move))
        self.positions.append(self._snapshot(self.moves, walls_left, dist, evaluation))
        self.legal = legal
        self.winner = winner
        self.margin = self._margin(winner, dist) if winner else None
        self.hint = None

    def _to_move(self):
        if self.winner:
            return None
        return "b" if len(self.moves) % 2 == 0 else "w"

    # -- engine lifecycle -------------------------------------------------------------------------------
    def start_engine(self):
        """Start (or restart) the engine and replay the current game into it. Blocks while the net loads."""
        with self.engine_lock:
            with self.lock:
                self.engine_status = "starting"
                self.engine_error = None
                self.gen += 1
                self.thinking = False
                self.ai_error = None
            try:
                self.engine.kill()
                self.engine.start()
                self.engine.send("name", timeout=self.start_timeout)  # waits until the net is loaded
            except (QTPError, OSError) as e:
                raise self._fail(e)
            ok, version = self._send("version")
            try:
                missing = [c for c in REQUIRED_COMMANDS if not self.engine.known_command(c)]
            except QTPError as e:
                raise self._fail(e)
            if missing:
                raise self._fail(QTPError("the engine does not support: %s (is it KataQuoridor >= 0.1.0?)"
                                          % ", ".join(missing)))
            # Replay the current game (after a crash) or just reset the board.
            self._must("clear_board")
            self._must("kata-set-param maxVisits %d" % self.visits)
            for color, move in self.moves:
                self._must("play %s %s" % (color, move))
            winner, dist, walls_left, legal = self._query_position()
            with self.lock:
                self.engine_name = version.strip() if ok else ""
                self.engine_status = "ready"
                self.legal = legal
                self.winner = winner
                need_ai = self.started and self._ai_should_move()
                if need_ai:
                    self.thinking = True
                gen = self.gen
        if need_ai:
            self._spawn_ai(gen)

    def close(self):
        with self.lock:
            self.gen += 1
        self.engine.close()

    def _ai_should_move(self):
        tm = self._to_move()
        return tm is not None and tm != self.human

    def _spawn_ai(self, gen, delay=0.0):
        t = threading.Thread(target=self._ai_turn, args=(gen, delay), daemon=True)
        t.start()

    # -- AI move ------------------------------------------------------------------------------------------
    def _ai_turn(self, gen, delay=0.0):
        if delay > 0:
            time.sleep(delay)
        again = False
        try:
            with self.engine_lock:
                with self.lock:
                    if gen != self.gen or not self.thinking:
                        return
                    color = self._to_move()
                ok, text = self._send("kata-genmove_analyze %s 1000 rootInfo true" % color,
                                      timeout=self.move_timeout)
                if not ok:
                    with self.lock:
                        self.ai_error = "engine refused to move: %s" % text
                        self.thinking = False
                    return
                a = ev.parse_analysis(text)
                move = a["move"]
                if not move or not MOVE_RE.match(move):
                    with self.lock:
                        self.ai_error = "engine returned %r" % move
                        self.thinking = False
                    return
                root_eval = None
                if a["root"] and a["infos"]:
                    r = a["root"]
                    root_eval = ev.from_side_to_move(r["winrate"], r["scoreLead"], color, "search",
                                                     r.get("visits"), a["infos"][0].get("pv"))
                winner, dist, walls_left, legal = self._query_position()
                info = ev.info_for(a, move)
                if winner:
                    move_eval = ev.final_eval(winner, self._margin(winner, dist))
                elif info is not None:
                    visits = a["root"].get("visits") if a["root"] else info.get("visits")
                    move_eval = ev.from_side_to_move(info["winrate"], info["scoreLead"], color, "search",
                                                     visits, info.get("pv", [])[1:])
                else:  # 1-visit search: no analysis output
                    move_eval = self._raw_eval()
                with self.lock:
                    if gen != self.gen:
                        return
                    if root_eval is not None:
                        self.positions[-1]["eval"] = root_eval
                    self._apply(color, move, winner, dist, walls_left, legal, move_eval)
                    again = self.human is None and not winner
                    self.thinking = again
        except EngineDown:
            return
        finally:
            with self.lock:
                if gen == self.gen and not again:
                    self.thinking = False
        if again:
            self._spawn_ai(gen, self.demo_delay)

    # -- actions ----------------------------------------------------------------------------------------
    def _check_ready(self):
        if self.engine_status == "starting":
            raise ActionError("the engine is still starting", 503)
        if self.engine_status != "ready":
            raise ActionError("the engine is not running; restart it", 503)

    def new_game(self, human="b", visits=16):
        if human not in ("b", "w", "random", "none"):
            raise ActionError("human must be b, w, random or none", 400)
        try:
            visits = int(visits)
        except (TypeError, ValueError):
            raise ActionError("visits must be an integer", 400)
        if not 1 <= visits <= 100000:
            raise ActionError("visits must be between 1 and 100000", 400)
        with self.lock:
            self._check_ready()
            if self.busy:
                raise ActionError("busy: %s" % self.busy)
            self.busy = "new"
            self.gen += 1          # an AI thread that is still searching will discard its result
        try:
            with self.engine_lock:
                self._must("clear_board")
                self._must("kata-set-param maxVisits %d" % visits)
                winner, dist, walls_left, legal = self._query_position()
                evaluation = self._position_eval(winner, dist)
                with self.lock:
                    self.gen += 1
                    gen = self.gen
                    self.human_choice = human
                    self.human = None if human == "none" else self.rng.choice("bw") if human == "random" else human
                    self.visits = visits
                    self._reset_position_state()
                    self.positions[0].update(walls_left=walls_left, dist=dist, eval=evaluation)
                    self.legal = legal
                    self.started = True
                    self.ai_error = None
                    self.hint = None
                    need_ai = self._ai_should_move()
                    self.thinking = need_ai
        finally:
            with self.lock:
                self.busy = None
        if need_ai:
            self._spawn_ai(gen)
        return self.state()

    def play(self, move):
        move = (move or "").strip().lower()
        if not MOVE_RE.match(move):
            raise ActionError("not a move in QTP notation (e.g. e8 or e2h): %r" % move[:20], 400)

        def check():
            self._check_ready()
            if not self.started:
                raise ActionError("no game in progress")
            if self.winner:
                raise ActionError("the game is over")
            if self.thinking:
                raise ActionError("the AI is thinking")
            if self.busy:
                raise ActionError("busy: %s" % self.busy)
            if self._to_move() != self.human:
                raise ActionError("it is not your turn")

        with self.lock:
            check()
            self.busy = "move"
        try:
            with self.engine_lock:
                with self.lock:
                    color = self.human
                ok, text = self._send("play %s %s" % (color, move))
                if not ok:
                    raise ActionError("illegal move %s: %s" % (move, text), 400)
                winner, dist, walls_left, legal = self._query_position()
                evaluation = self._position_eval(winner, dist)
                with self.lock:
                    self._apply(color, move, winner, dist, walls_left, legal, evaluation)
                    need_ai = self._ai_should_move()
                    self.thinking = need_ai
                    self.ai_error = None
                    gen = self.gen
        finally:
            with self.lock:
                self.busy = None
        if need_ai:
            self._spawn_ai(gen)
        return self.state()

    def undo(self):
        with self.lock:
            self._check_ready()
            if self.thinking:
                raise ActionError("cannot undo while the AI is thinking")
            if self.busy:
                raise ActionError("busy: %s" % self.busy)
            if self.human is None:
                raise ActionError("nothing to undo in AI vs AI mode")
            human_plies = [i for i, (c, _) in enumerate(self.moves) if c == self.human]
            if not human_plies:
                raise ActionError("nothing to undo")
            count = len(self.moves) - human_plies[-1]
            self.busy = "undo"
        try:
            with self.engine_lock:
                for _ in range(count):
                    self._must("undo")
                winner, dist, walls_left, legal = self._query_position()
                with self.lock:
                    del self.moves[len(self.moves) - count:]
                    del self.positions[len(self.moves) + 1:]
                    self.legal = legal
                    self.winner = winner
                    self.margin = None
                    self.hint = None
                    self.ai_error = None
        finally:
            with self.lock:
                self.busy = None
        return self.state()

    def request_hint(self):
        with self.lock:
            self._check_ready()
            if not self.started or self.winner:
                raise ActionError("no game in progress")
            if self.thinking or self.busy:
                raise ActionError("wait for the AI move")
            if self._to_move() != self.human:
                raise ActionError("it is not your turn")
            color = self.human
            ply = len(self.moves)
            self.busy = "hint"
            visits = max(self.hint_visits, self.visits)
        try:
            with self.engine_lock:
                self._must("kata-set-param maxVisits %d" % visits)
                try:
                    ok, text = self._send("kata-search_analyze %s 1000 rootInfo true" % color,
                                          timeout=self.move_timeout)
                finally:
                    self._must("kata-set-param maxVisits %d" % self.visits)
            if not ok:
                raise ActionError("engine refused the hint search: %s" % text)
            a = ev.parse_analysis(text)
            # The engine's own choice ("play X") comes first; the others follow in the engine's order.
            infos = sorted(a["infos"], key=lambda d: d.get("move") != a["move"])
            moves = []
            for d in infos[:3]:
                e = ev.from_side_to_move(d["winrate"], d["scoreLead"], color, "search", d.get("visits"), d.get("pv"))
                e["move"] = d["move"]
                moves.append(e)
            if not moves and a["move"]:
                moves.append(dict(ev.make_eval(0.5, 0.0, "search"), move=a["move"]))
            hint = {"ply": ply, "color": color, "moves": moves,
                    "visits": a["root"].get("visits") if a["root"] else visits}
            with self.lock:
                if len(self.moves) == ply:
                    self.hint = hint
        finally:
            with self.lock:
                self.busy = None
        return hint

    def sgf(self):
        with self.lock:
            self._check_ready()
            if self.thinking or self.busy:
                raise ActionError("wait for the AI move")
            human = self.human
            positions = [dict(p) for p in self.positions]
            ai_name = "KataQuoridor%s (%d visit%s)" % (" " + self.model_name if self.model_name else "", self.visits,
                                                      "" if self.visits == 1 else "s")
        with self.engine_lock:
            text = self._must("printsgf")
        names = {c: ("Human" if c == human else ai_name) for c in "bw"}
        return decorate_sgf(text.strip(), positions, names["b"], names["w"])

    # -- state ----------------------------------------------------------------------------------------------
    def state(self):
        with self.lock:
            cur = self.positions[-1]
            return {
                "engine": {"status": self.engine_status, "error": self.engine_error, "name": self.engine_name,
                           "model": self.model_name},
                "started": self.started,
                "game_id": self.gen,
                "settings": {"human": self.human, "human_choice": self.human_choice, "visits": self.visits,
                             "hint_visits": max(self.hint_visits, self.visits)},
                "moves": [{"color": c, "move": m} for c, m in self.moves],
                "to_move": self._to_move(),
                "pawns": cur["pawns"],
                "walls": cur["walls"],
                "walls_left": cur["walls_left"],
                "dist": cur["dist"],
                "eval": cur["eval"],
                "legal_moves": list(self.legal),
                "winner": self.winner,
                "margin": self.margin,
                "thinking": self.thinking,
                "busy": self.busy,
                "ai_error": self.ai_error,
                "hint": self.hint,
                "positions": [dict(p) for p in self.positions],
            }
