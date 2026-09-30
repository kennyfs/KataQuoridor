"""Plays one game between two QTP engines, mirrored into an arbiter process.

The arbiter (a KataQuoridor process without a neural net) validates every move and decides the
result. The referee never reimplements the rules; it only enforces the ply limit.
"""
import re
import time

from .qtp import QTPCrash, QTPEngine, QTPError, QTPTimeout


class VerifyMismatch(Exception):
    """--verify found different legal-move sets in the arbiter and an engine. Aborts the run."""


class RulesError(Exception):
    """An engine or the arbiter rejected a komi / initial-walls command. Aborts the run."""


# -- game rules (docs/QuoridorIOv2.md) ------------------------------------------

# The rules the arena can set per game, by their QTP names. Engines start a process with these.
STANDARD_RULES = {"komi": -0.5, "blackInitialWalls": 10, "whiteInitialWalls": 10}
MAX_KOMI = 20.5
MAX_WALLS = 10


def make_rules(*layers, where="rules"):
    """The standard rules updated by each layer (a dict with some of the keys of STANDARD_RULES), validated."""
    rules = dict(STANDARD_RULES)
    for layer in layers:
        for k, v in (layer or {}).items():
            if k not in STANDARD_RULES:
                raise ValueError("%s: unknown key %r (known: %s)" % (where, k, ", ".join(STANDARD_RULES)))
            rules[k] = v
    komi = float(rules["komi"])
    if abs(komi) > MAX_KOMI or komi * 2 != int(komi * 2) or int(komi * 2) % 2 != 1:
        raise ValueError("%s: komi must be a half-integer with |komi| <= %g, got %r" % (where, MAX_KOMI, rules["komi"]))
    rules["komi"] = komi
    for k in ("blackInitialWalls", "whiteInitialWalls"):
        if not isinstance(rules[k], int) or not 0 <= rules[k] <= MAX_WALLS:
            raise ValueError("%s: %s must be an integer 0..%d, got %r" % (where, k, MAX_WALLS, rules[k]))
    return rules


def is_standard(rules):
    return rules is None or rules == STANDARD_RULES


def rules_tag(rules):
    """'' for the standard game, else e.g. '_k+1.5_w9-10' (komi, Black's and White's initial walls)."""
    if is_standard(rules):
        return ""
    return "_k%+g_w%d-%d" % (rules["komi"], rules["blackInitialWalls"], rules["whiteInitialWalls"])


class RulesState:
    """What rules an engine process has. QTP rules persist across clear_board, so they are sent only when they
    change; a (re)started process has the standard rules."""

    def __init__(self):
        self.rules = None
        self.starts = None

    def sync(self, engine, rules, owner):
        """Sends the commands that change the engine's rules to `rules` (after clear_board, since the initial
        walls can only change before the first move). Raises RulesError if the engine rejects one, QTPError if
        it fails."""
        current = self.rules if self.starts == engine.starts else STANDARD_RULES
        if current == rules:
            return
        cmds = []
        if rules["komi"] != current["komi"]:
            cmds.append("komi %g" % rules["komi"])
        for k in ("blackInitialWalls", "whiteInitialWalls"):
            if rules[k] != current[k]:
                cmds.append("kata-set-rule %s %d" % (k, rules[k]))
        self.rules = None
        for cmd in cmds:
            ok, text = engine.send(cmd)
            if not ok:
                raise RulesError("%s rejected %r: %s (engines without komi / walls support can only play the "
                                 "standard game)" % (owner, cmd, text))
        self.rules = dict(rules)
        self.starts = engine.starts


class RefereeError(Exception):
    """The arbiter itself failed; the game is not recorded (it is retried on the next run)."""


def other(color):
    return "w" if color == "b" else "b"


class Arbiter:
    def __init__(self, engine):
        self.engine = engine
        self.rules_state = RulesState()

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

    def clear(self, rules=None):
        self.engine.ensure_running()
        self._must("clear_board")
        try:
            self.rules_state.sync(self.engine, rules or STANDARD_RULES, "arbiter")
        except RulesError:
            raise
        except QTPError as e:
            self.engine.kill()
            raise RefereeError("arbiter failed setting rules: %s" % e)

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


def sgf_root_prop(sgf, prop):
    """The value of a root property of an SGF text, or None."""
    semi = sgf.find(";", sgf.find(";") + 1)
    root = sgf if semi < 0 else sgf[:semi]
    m = re.search(r"(?<![A-Z])%s\[([^\]]*)\]" % prop, root)
    return m.group(1) if m else None


class Player:
    """A roster engine owned by one worker. Reused across games via clear_board."""

    def __init__(self, spec, stderr_path, cwd=None, command_timeout=120.0, genmove_timeout=600.0):
        self.spec = spec
        self.name = spec["name"]
        self.engine = QTPEngine(self.name, spec["argv"], stderr_path=stderr_path, cwd=cwd,
                                default_timeout=command_timeout)
        self.genmove_timeout = genmove_timeout
        self._has_legal = None
        self.rules_state = RulesState()

    def new_game(self, rules=None):
        """Reset for a new game with the given rules (default standard); restarts the process once if it is dead
        or clear_board fails. Raises RulesError if the engine rejects the rules."""
        rules = rules or STANDARD_RULES
        for attempt in range(2):
            try:
                self.engine.ensure_running()
                ok, text = self.engine.send("clear_board")
                if ok:
                    self.rules_state.sync(self.engine, rules, self.name)
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
    """Turn the arbiter's printsgf output into a self-play style one-line SGF record. KM, WB, WW and RU come from
    the arbiter; RE is `result`."""
    sgf = " ".join(sgf.split("\n")).strip()
    semi = sgf.find(";", 2)
    root = sgf if semi < 0 else sgf[:semi]
    body = "" if semi < 0 else sgf[semi:]
    for prop in ("PB", "PW", "RE", "C"):
        root = re.sub(r"(?<![A-Z])%s\[[^\]]*\]" % prop, "", root)
    comment = "startTurnIdx=%d,initTurnNum=0,gtype=arena" % opening_plies
    if extra_comment:
        comment += "," + extra_comment
    root += "PB[%s]PW[%s]RE[%s]C[%s]" % (black, white, result, comment)
    return root + body


def play_game(arbiter, black, white, opening, max_plies=300, verify=False, log=print, rules=None):
    """Play one game under `rules` (see make_rules; default standard), set in both engines and the arbiter.
    black/white are Player objects. Returns a dict with the result fields plus 'sgf'. Raises RefereeError
    (arbiter failure), VerifyMismatch or RulesError."""
    t0 = time.time()
    rules = rules or STANDARD_RULES
    players = {"b": black, "w": white}
    arbiter.clear(rules)
    moves = []
    result = None  # (winner_color or None, reason, detail)

    def forfeit(loser_color, reason, detail):
        log("!!! %s (%s) loses by %s: %s" % (players[loser_color].name, loser_color.upper(), reason, detail))
        return (other(loser_color), reason, detail)

    for color in ("b", "w"):
        try:
            players[color].new_game(rules)
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
            # KataQuoridor >= 0.2 ends a game itself at its maxPlies (default 300, a game rule) and says "Draw";
            # label it like the referee's own cutoff below.
            result = (None, "draw%d" % len(moves), "arbiter says %s" % w)
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
    sgf = arbiter.printsgf()
    # The arbiter's result (RE, the lead s = tempo + komi, e.g. W+0.5, or 0 for its maxPlies draw) when it ended the
    # game itself; the referee's own for forfeits (B+F / W+F, as SGF has it) and its --max-plies cutoff (0).
    arbiter_re = sgf_root_prop(sgf, "RE")
    if reason == "goal" or (winner is None and arbiter_re):
        re_str = arbiter_re or "?"
        lead = 0.0 if winner is None else abs(float(re_str[2:]))
        # The margin: the distance of the pawn that did not arrive (with komi, it can be the winner's).
        margin = 0 if winner is None else max(1, bd, wd)
    elif winner is None:
        re_str, lead, margin = "0", 0.0, 0
    else:
        re_str, lead = "%s+F" % winner.upper(), None
        margin = max(1, wd if winner == "b" else bd)
    sgf = fix_sgf(sgf, black.name, white.name, re_str, len(opening), "reason=%s" % reason)
    return {
        "black": black.name,
        "white": white.name,
        "winner": winner,  # 'b', 'w' or None
        "winner_name": None if winner is None else players[winner].name,
        "margin": margin,
        "lead": lead,  # |s| from the SGF result (0 for a draw), None for a forfeit
        "result": re_str,
        "komi": rules["komi"],
        "black_walls": rules["blackInitialWalls"],
        "white_walls": rules["whiteInitialWalls"],
        "plies": len(moves),
        "reason": reason,
        "detail": detail,
        "moves": moves,
        "seconds": round(time.time() - t0, 2),
        "sgf": sgf,
    }
