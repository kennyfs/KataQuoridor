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
        self._kata = spec.get("kata")  # None: auto-detect
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

    def is_kata(self):
        """Whether to ask this engine for its search info with KATA_GENMOVE (roster "kata": true / false, else
        auto-detected with known_command)."""
        if self._kata is None:
            try:
                self._kata = self.engine.known_command(KATA_GENMOVE)
            except QTPError:
                self._kata = False
        return self._kata

    def genmove(self, color):
        """Ask for a move. Returns (ok, move or error text, eval dict or None). For a KataQuoridor engine the
        move comes from kata-genmove_analyze (the same search as genmove, the analysis is printed once at the
        end) and the eval is its root info, see parse_kata_genmove."""
        if not self.is_kata():
            ok, text = self.engine.send("genmove %s" % color, timeout=self.genmove_timeout)
            return ok, (text.strip().split()[0].lower() if (ok and text.strip()) else text), None
        ok, text = self.engine.send(KATA_GENMOVE_CMD % color, timeout=self.genmove_timeout)
        if not ok:
            return ok, text, None
        mv, ev = parse_kata_genmove(text, color, self.spec.get("kata_perspective", "sidetomove"))
        return True, mv if mv is not None else text, ev

    def close(self):
        self.engine.close()


# -- search info (KataQuoridor's kata-genmove_analyze) ---------------------------

KATA_GENMOVE = "kata-genmove_analyze"
# No interval: the analysis is printed once, for the final search, then "play <move>".
KATA_GENMOVE_CMD = KATA_GENMOVE + " %s rootInfo true noResultValue true"


def _kv_after(tokens, start):
    out = {}
    for i in range(start, len(tokens) - 1, 2):
        out[tokens[i]] = tokens[i + 1]
    return out


def parse_kata_genmove(text, color, perspective="sidetomove"):
    """Parse a kata-genmove_analyze response: "info move .. info move .. rootInfo k v ..." lines, then
    "play <move>". Returns (move or None, eval or None); the eval is White's view of the search at the position
    before the move: {"win", "loss", "noResult", "score" (scoreSelfplay: the utility score u, with the time bonus),
    "lead" (scoreLead: the predicted lead s), "visits" (root)}.

    `perspective` is the engine's reportAnalysisWinratesAs (KataQuoridor's default: the side to move). The root
    info has no noResult value; it is taken from the played move's info (noResultValue), 0 if missing.
    """
    move = None
    info_lines = []
    for line in text.splitlines():
        t = line.split()
        if len(t) >= 2 and t[0] == "play":
            move = t[1].lower()
        elif t and t[0] == "info":
            info_lines.append(line)
    if move is None:
        return None, None
    # Several "info move" blocks may share one line; rootInfo is last.
    blob = " ".join(info_lines)
    root = None
    no_result = 0.0
    if " rootInfo " in blob + " ":
        head, _, tail = blob.partition(" rootInfo ")
        root = _kv_after(tail.split(), 0)
    else:
        head = blob
    for chunk in head.split("info ")[1:]:
        kv = _kv_after(chunk.split(), 0)
        if kv.get("move", "").lower() == move and "noResultValue" in kv:
            no_result = float(kv["noResultValue"])
    if root is None or "winrate" not in root:
        return move, None
    winrate, score, lead = float(root["winrate"]), float(root["scoreSelfplay"]), float(root["scoreLead"])
    persp = perspective.lower()
    white_view = persp in ("w", "white") or (persp not in ("b", "black") and color == "w")
    if not white_view:
        winrate, score, lead = 1.0 - winrate, -score, -lead
    win = min(1.0, max(0.0, winrate - no_result / 2))
    loss = min(1.0, max(0.0, 1.0 - no_result - win))
    return move, {"win": win, "loss": loss, "noResult": no_result, "score": score, "lead": lead,
                  "visits": int(float(root.get("visits", 0)))}


def eval_comment(ev):
    """The self-play move comment (cpp/dataio/sgf.cpp: "win loss noResult score v=..", White's view) plus the
    predicted lead as a named token, e.g. "0.48 0.52 0.00 -0.2 v=256 lead=-0.17"."""
    return "%.2f %.2f %.2f %.1f v=%d lead=%.2f" % (ev["win"], ev["loss"], ev["noResult"], ev["score"],
                                                   ev["visits"], ev["lead"])


def add_move_comments(sgf, comments):
    """Put comments[i] (or nothing for None) on the i-th move node of a one-line SGF."""
    idx = [0]

    def sub(m):
        i = idx[0]
        idx[0] += 1
        c = comments[i] if i < len(comments) else None
        return m.group(0) if not c else m.group(0) + "C[%s]" % c
    return re.sub(r";[BW]\[[^\]]*\]", sub, sgf)


def fix_sgf(sgf, black, white, result, opening_plies, extra_comment="", move_comments=None):
    """Turn the arbiter's printsgf output into a self-play style one-line SGF record. KM, WB, WW and RU come from
    the arbiter; RE is `result`. move_comments[i] (if not None) becomes the comment of the i-th move."""
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
    if move_comments:
        body = add_move_comments(body, move_comments)
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
    evals = []  # per move: the mover's search info (White's view) or None
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
            evals.append(None)
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
            ok, mv, ev = mover.genmove(color)
        except QTPTimeout as e:
            result = forfeit(color, "timeout", str(e))
            break
        except QTPError as e:
            result = forfeit(color, "crash", str(e))
            break
        text = mv
        if not ok or mv in ("", "resign", "pass"):
            result = forfeit(color, "illegal", "genmove returned %r (ok=%s)" % (text, ok))
            break
        if not arbiter.play(color, mv):
            result = forfeit(color, "illegal", "illegal move %s after %s" % (mv, " ".join(moves)))
            break
        opp = players[other(color)]
        evals.append(ev)
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
    sgf = fix_sgf(sgf, black.name, white.name, re_str, len(opening), "reason=%s" % reason,
                  move_comments=[eval_comment(e) if e else None for e in evals])
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
        # per move [White win, score (utility, with the time bonus), lead, visits] of the mover's search, or None
        "evals": [None if e is None else [round(e["win"], 3), round(e["score"], 2), round(e["lead"], 2), e["visits"]]
                  for e in evals] if any(evals) else None,
        "seconds": round(time.time() - t0, 2),
        "sgf": sgf,
    }
