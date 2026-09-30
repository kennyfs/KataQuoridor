"""Plays one game between two QTP engines, mirrored into an arbiter process.

The arbiter (a KataQuoridor process without a neural net) validates every move and decides the
result. The referee never reimplements the rules; it only enforces the ply limit.
"""
import re
import time

from .qtp import QTPCrash, QTPEngine, QTPError, QTPTimeout


class VerifyMismatch(Exception):
    """--verify found different legal-move sets in the arbiter and an engine. Aborts the run."""


class RefereeError(Exception):
    """The arbiter itself failed; the game is not recorded (it is retried on the next run)."""


def other(color):
    return "w" if color == "b" else "b"


class Arbiter:
    def __init__(self, engine):
        self.engine = engine

    def start(self):
        self.engine.ensure_running()

    def close(self):
        self.engine.close()

    def _must(self, cmd):
        try:
            ok, text = self.engine.send(cmd)
        except QTPError as e:
            self.engine.kill()
            raise RefereeError("arbiter failed on %r: %s" % (cmd, e))
        if not ok:
            raise RefereeError("arbiter rejected %r: %s" % (cmd, text))
        return text

    def clear(self):
        self.engine.ensure_running()
        self._must("clear_board")

    def play(self, color, move):
        try:
            ok, _ = self.engine.send("play %s %s" % (color, move))
        except QTPError as e:
            self.engine.kill()
            raise RefereeError("arbiter failed on play %s %s: %s" % (color, move, e))
        return ok

    def legal_moves(self):
        return self._must("legal_moves").split()

    def winner(self):
        """'B', 'W', 'Draw' or None."""
        w = self._must("winner").strip()
        return None if w.lower() == "none" else w

    def dist(self):
        m = re.match(r"B:\s*(\d+)\s+W:\s*(\d+)", self._must("dist"))
        if not m:
            raise RefereeError("cannot parse dist")
        return int(m.group(1)), int(m.group(2))

    def printsgf(self):
        return self._must("printsgf")


class Player:
    """A roster engine owned by one worker. Reused across games via clear_board."""

    def __init__(self, spec, stderr_path, cwd=None, command_timeout=120.0, genmove_timeout=600.0):
        self.spec = spec
        self.name = spec["name"]
        self.engine = QTPEngine(self.name, spec["argv"], stderr_path=stderr_path, cwd=cwd,
                                default_timeout=command_timeout)
        self.genmove_timeout = genmove_timeout
        self._has_legal = None

    def new_game(self):
        """Reset for a new game; restarts the process once if it is dead or clear_board fails."""
        for attempt in range(2):
            try:
                self.engine.ensure_running()
                ok, text = self.engine.send("clear_board")
                if ok:
                    return
                err = "clear_board failed: " + text
            except QTPError as e:
                err = str(e)
            self.engine.kill()
        raise QTPCrash("%s: cannot start a new game: %s" % (self.name, err))

    def supports_legal_moves(self):
        if self._has_legal is None:
            try:
                self._has_legal = self.engine.known_command("legal_moves")
            except QTPError:
                self._has_legal = False
        return self._has_legal

    def close(self):
        self.engine.close()


def fix_sgf(sgf, black, white, result, opening_plies, extra_comment=""):
    """Turn the arbiter's printsgf output into a self-play style one-line SGF record."""
    sgf = " ".join(sgf.split("\n")).strip()
    semi = sgf.find(";", 2)
    root = sgf if semi < 0 else sgf[:semi]
    body = "" if semi < 0 else sgf[semi:]
    for prop in ("PB", "PW", "RE", "KM", "C"):
        root = re.sub(r"%s\[[^\]]*\]" % prop, "", root)
    comment = "startTurnIdx=%d,initTurnNum=0,gtype=arena" % opening_plies
    if extra_comment:
        comment += "," + extra_comment
    root += "PB[%s]PW[%s]KM[0]RE[%s]C[%s]" % (black, white, result, comment)
    return root + body


def play_game(arbiter, black, white, opening, max_plies=300, verify=False, log=print):
    """Play one game. black/white are Player objects. Returns a dict with the result fields plus
    'sgf'. Raises RefereeError (arbiter failure) or VerifyMismatch."""
    t0 = time.time()
    players = {"b": black, "w": white}
    arbiter.clear()
    moves = []
    result = None  # (winner_color or None, reason, detail)

    def forfeit(loser_color, reason, detail):
        log("!!! %s (%s) loses by %s: %s" % (players[loser_color].name, loser_color.upper(), reason, detail))
        return (other(loser_color), reason, detail)

    for color in ("b", "w"):
        try:
            players[color].new_game()
        except QTPError as e:
            result = forfeit(color, "crash", str(e))
            break

    color = "b"
    if result is None:
        for mv in opening:
            if not arbiter.play(color, mv):
                raise RefereeError("arbiter rejected opening move %s %s" % (color, mv))
            for c in ("b", "w"):
                try:
                    ok, text = players[c].engine.send("play %s %s" % (color, mv))
                except QTPTimeout as e:
                    result = forfeit(c, "timeout", str(e))
                    break
                except QTPError as e:
                    result = forfeit(c, "crash", str(e))
                    break
                if not ok:
                    result = forfeit(c, "illegal", "rejected opening move %s %s: %s" % (color, mv, text))
                    break
            if result is not None:
                break
            moves.append(mv)
            color = other(color)

    while result is None:
        w = arbiter.winner()
        if w in ("B", "W"):
            result = (w.lower(), "goal", "")
            break
        if w is not None:
            result = (None, "draw", "arbiter says %s" % w)
            break
        if len(moves) >= max_plies:
            result = (None, "draw%d" % max_plies, "")
            break
        mover = players[color]
        if verify and mover.supports_legal_moves():
            try:
                ok, text = mover.engine.send("legal_moves")
            except QTPError as e:
                result = forfeit(color, "crash", str(e))
                break
            ref = set(arbiter.legal_moves())
            got = set(text.split()) if ok else None
            if got != ref:
                raise VerifyMismatch(
                    "%s legal_moves differ from the arbiter after %s: only engine %s, only arbiter %s"
                    % (mover.name, " ".join(moves) or "(start)",
                       sorted(got - ref) if got is not None else "(error: %s)" % text,
                       sorted(ref - got) if got is not None else ""))
        try:
            ok, text = mover.engine.send("genmove %s" % color, timeout=mover.genmove_timeout)
        except QTPTimeout as e:
            result = forfeit(color, "timeout", str(e))
            break
        except QTPError as e:
            result = forfeit(color, "crash", str(e))
            break
        mv = text.strip().split()[0].lower() if (ok and text.strip()) else ""
        if not ok or mv in ("", "resign", "pass"):
            result = forfeit(color, "illegal", "genmove returned %r (ok=%s)" % (text, ok))
            break
        if not arbiter.play(color, mv):
            result = forfeit(color, "illegal", "illegal move %s after %s" % (mv, " ".join(moves)))
            break
        opp = players[other(color)]
        try:
            ok, text = opp.engine.send("play %s %s" % (color, mv))
        except QTPTimeout as e:
            moves.append(mv)
            result = forfeit(other(color), "timeout", str(e))
            break
        except QTPError as e:
            moves.append(mv)
            result = forfeit(other(color), "crash", str(e))
            break
        moves.append(mv)
        if not ok:
            result = forfeit(other(color), "illegal", "rejected legal move %s %s: %s" % (color, mv, text))
            break
        color = other(color)

    winner, reason, detail = result
    bd, wd = arbiter.dist()
    if winner is None:
        margin = 0
        re_str = "0"
    else:
        loser_dist = wd if winner == "b" else bd
        margin = max(1, loser_dist)
        re_str = "%s+%d" % (winner.upper(), margin)
    sgf = fix_sgf(arbiter.printsgf(), black.name, white.name, re_str, len(opening),
                  "reason=%s" % reason)
    return {
        "black": black.name,
        "white": white.name,
        "winner": winner,  # 'b', 'w' or None
        "winner_name": None if winner is None else players[winner].name,
        "margin": margin,
        "plies": len(moves),
        "reason": reason,
        "detail": detail,
        "moves": moves,
        "seconds": round(time.time() - t0, 2),
        "sgf": sgf,
    }
