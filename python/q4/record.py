"""Q4 game record reader and writer: one game per line as a Duel-style SGF (docs/q4/Q4Sgf.md), the same semantics as
cpp/q4/q4record.cpp.

`load_records(path)` yields one dict per game with the keys of the former JSON record:

    rules       {"maxPlies", "repetitionDrawCount", "initialWalls": [4]}
    players     [{"name", "type", "net"?, "visits"?} x4]   (seat 0..3 = S, W, N, E)
    result      "1+" .. "4+" (the winning seat, 1-based), "Draw" or "none" (unfinished)
    events      [{"a": "f2"} | {"elim": seat1to4}]
    comments    [str] one per event (only if some move has a comment)
    gameHash    32 hex digits (only if the root comment has one)
    match       {"table", "opening", "rotation", "openingPlies", "drawReason"} (only for match / gatekeeper games)

plus the keys "gtype", "startTurnIdx", "drawReason" ("" if none) and, when comments exist, "moveComments": one entry
per event, None or {"p": [pS, pW, pN, pE, pDraw], "visits", "weight"}.
"""

import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from q4.reference import Pos, action_to_str, eliminate, play, str_to_action

SEAT_LETTERS = "SWNE"
PLAYER_IDS = ("PS", "PW", "PN", "PE")
WALL_IDS = ("WS", "WW", "WN", "WE")

_PROP_RE = re.compile(r"([A-Z]+)\[((?:[^\\\]]|\\.)*)\]", re.S)
_UNESCAPE_RE = re.compile(r"\\(.)", re.S)


def escape_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("]", "\\]").replace("\n", " ").replace("\r", " ")


def unescape_text(s: str) -> str:
    return _UNESCAPE_RE.sub(r"\1", s)


def parse_nodes(line: str) -> List[List[Tuple[str, str]]]:
    """Splits '(;ID[v]ID[v];ID[v])' into nodes, each a list of (id, value)."""
    s = line.strip()
    if not s.startswith("(") or not s.endswith(")"):
        raise ValueError("not an SGF line")
    nodes: List[List[Tuple[str, str]]] = []
    i, n = 1, len(s) - 1
    while i < n:
        c = s[i]
        if c == ";":
            nodes.append([])
            i += 1
        elif c.isspace():
            i += 1
        else:
            m = _PROP_RE.match(s, i)
            if m is None or not nodes:
                raise ValueError(f"malformed SGF near position {i}")
            nodes[-1].append((m.group(1), unescape_text(m.group(2))))
            i = m.end()
    if not nodes:
        raise ValueError("SGF line has no root node")
    return nodes


def parse_kv(text: str) -> Dict[str, str]:
    kv = {}
    for part in text.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def parse_move_comment(text: str) -> Optional[Dict[str, Any]]:
    """'0.31 0.22 0.27 0.15 0.05 v=600 weight=1.00' -> {"p": [5 floats], "visits", "weight"}; None if not a move
    comment."""
    parts = text.split()
    if len(parts) < 6:
        return None
    try:
        p = [float(x) for x in parts[:5]]
    except ValueError:
        return None
    kv = {}
    for part in parts[5:]:
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k] = v
    if "v" not in kv:
        return None
    try:
        return {"p": p, "visits": int(kv["v"]), "weight": float(kv.get("weight", 0.0))}
    except ValueError:
        return None


def format_move_comment(c: Dict[str, Any]) -> str:
    p = c["p"]
    return "%.2f %.2f %.2f %.2f %.2f v=%d weight=%.2f" % (p[0], p[1], p[2], p[3], p[4], c["visits"], c["weight"])


def _result_to_letter(result: str) -> str:
    if result == "Draw":
        return "0"
    if result == "none":
        return "?"
    if len(result) == 2 and result[1] == "+" and result[0] in "1234":
        return SEAT_LETTERS[int(result[0]) - 1]
    raise ValueError(f"unknown result {result!r}")


def _letter_to_result(letter: str) -> str:
    if letter == "0":
        return "Draw"
    if letter == "?":
        return "none"
    if len(letter) == 1 and letter in SEAT_LETTERS:
        return f"{SEAT_LETTERS.index(letter) + 1}+"
    raise ValueError(f"bad result {letter!r}")


class Q4Record:
    def __init__(self,
                 rules: Optional[Dict[str, Any]] = None,
                 players: Optional[List[Dict[str, Any]]] = None,
                 result: str = "none",
                 events: Optional[List[Dict[str, Any]]] = None,
                 comments: Optional[List[str]] = None,
                 game_hash: Optional[str] = None,
                 match: Optional[Dict[str, Any]] = None,
                 gtype: str = "normal",
                 start_turn_idx: int = 0,
                 draw_reason: str = ""):
        self.rules = rules or {
            "maxPlies": 400,
            "repetitionDrawCount": 0,
            "initialWalls": [7, 7, 7, 7],
        }
        self.players = players or [
            {"name": f"P{i+1}", "type": "bot"} for i in range(4)
        ]
        self.result = result
        self.events = events or []
        self.comments = comments or []
        self.game_hash = game_hash
        self.match = match
        self.gtype = gtype
        self.start_turn_idx = start_turn_idx
        self.draw_reason = draw_reason

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "rules": self.rules,
            "players": self.players,
            "result": self.result,
            "events": self.events,
            "gtype": self.gtype,
            "startTurnIdx": self.start_turn_idx,
            "drawReason": self.draw_reason,
        }
        if self.comments:
            d["comments"] = self.comments
            d["moveComments"] = [parse_move_comment(c) if c else None for c in self.comments]
        if self.game_hash:
            d["gameHash"] = self.game_hash
        if self.match is not None:
            d["match"] = self.match
        return d

    def to_sgf_line(self) -> str:
        rules = self.rules
        walls = rules.get("initialWalls", [7, 7, 7, 7])
        out = ["(;FF[4]GM[Q4]SZ[11]"]
        for i in range(4):
            out.append(f"{PLAYER_IDS[i]}[{escape_text(self.players[i]['name'])}]")
        for i in range(4):
            out.append(f"{WALL_IDS[i]}[{walls[i]}]")
        out.append(f"RU[Q4:repetitionDrawCount={rules.get('repetitionDrawCount', 0)},"
                   f"maxPlies={rules.get('maxPlies', 400)}]")
        out.append(f"RE[{_result_to_letter(self.result)}]")
        dr = self.draw_reason
        plies = sum(1 for ev in self.events if "a" in ev)
        if self.result == "none":
            dr = "unfinished"
        elif self.result == "Draw":
            dr = dr or ("maxPlies" if plies >= rules.get("maxPlies", 400) else "repetition")
        else:
            dr = ""
        if dr:
            out.append(f"DR[{dr}]")
        c = [f"gtype={self.gtype}"]
        if self.game_hash:
            c.append(f"gameHash={self.game_hash}")
        c.append(f"startTurnIdx={self.start_turn_idx}")
        for i, p in enumerate(self.players):
            c.append(f"type{i}={p.get('type', 'bot')}")
            if p.get("net"):
                c.append(f"net{i}={p['net']}")
            if p.get("visits", 0) > 0:
                c.append(f"v{i}={p['visits']}")
        if self.match is not None:
            c += [f"table={self.match['table']}", f"opening={self.match['opening']}",
                  f"rotation={self.match.get('rotation', 0)}"]
        out.append(f"C[{escape_text(','.join(c))}]")

        pos = Pos(max_plies=rules.get("maxPlies", 400), repetition_draw_count=rules.get("repetitionDrawCount", 0),
                  initial_walls=walls)
        for i, ev in enumerate(self.events):
            if "elim" in ev:
                seat = ev["elim"] - 1
                node = f"EL[{SEAT_LETTERS[seat]}]"
                pos = eliminate(pos, seat)
            else:
                action = str_to_action(ev["a"])
                node = f"{SEAT_LETTERS[pos.to_move]}[{ev['a']}]"
                pos = play(pos, action)
            if i < len(self.comments) and self.comments[i]:
                node += f"C[{escape_text(self.comments[i])}]"
            out.append(";" + node)
        out.append(")")
        return "".join(out)

    @classmethod
    def from_sgf_line(cls, line: str) -> 'Q4Record':
        nodes = parse_nodes(line)
        root = dict(nodes[0])
        if root.get("GM") != "Q4":
            raise ValueError(f"not a Q4 game (GM[{root.get('GM', '')}])")
        rules = {"maxPlies": 400, "repetitionDrawCount": 0, "initialWalls": [7, 7, 7, 7]}
        for k, v in parse_kv(root.get("RU", "").removeprefix("Q4:")).items():
            if k in ("maxPlies", "repetitionDrawCount"):
                rules[k] = int(v)
        players = []
        for i in range(4):
            players.append({"name": root.get(PLAYER_IDS[i], f"P{i+1}"), "type": "bot"})
            if WALL_IDS[i] in root:
                rules["initialWalls"][i] = int(root[WALL_IDS[i]])
        rec = cls(rules=rules, players=players, result=_letter_to_result(root.get("RE", "?")))
        rec.draw_reason = root.get("DR", "")
        kv = parse_kv(root.get("C", ""))
        rec.gtype = kv.get("gtype", "normal")
        rec.start_turn_idx = int(kv.get("startTurnIdx", 0))
        rec.game_hash = kv.get("gameHash")
        for i in range(4):
            if f"type{i}" in kv:
                players[i]["type"] = kv[f"type{i}"]
            if f"net{i}" in kv:
                players[i]["net"] = kv[f"net{i}"]
            if f"v{i}" in kv:
                players[i]["visits"] = int(kv[f"v{i}"])
        if "table" in kv:
            rec.match = {"table": kv["table"], "opening": int(kv.get("opening", 0)),
                         "rotation": int(kv.get("rotation", 0)), "openingPlies": rec.start_turn_idx,
                         "drawReason": rec.draw_reason}
        comments = []
        for node in nodes[1:]:
            props = dict(node)
            if "EL" in props:
                rec.events.append({"elim": SEAT_LETTERS.index(props["EL"]) + 1})
            else:
                moves = [(k, v) for k, v in node if k in ("S", "W", "N", "E")]
                if len(moves) != 1:
                    raise ValueError("node without a move")
                rec.events.append({"a": moves[0][1]})
            comments.append(props.get("C", ""))
        if any(comments):
            rec.comments = comments
        return rec

    def replay(self) -> Pos:
        max_plies = self.rules.get("maxPlies", 400)
        rep = self.rules.get("repetitionDrawCount", 0)
        walls = self.rules.get("initialWalls", [7, 7, 7, 7])
        pos = Pos(max_plies=max_plies, repetition_draw_count=rep, initial_walls=walls)

        for ev in self.events:
            if "elim" in ev:
                seat_1based = ev["elim"]
                pos = eliminate(pos, seat_1based - 1)
            elif "a" in ev:
                action = str_to_action(ev["a"])
                pos = play(pos, action)
            else:
                raise ValueError(f"Unknown event format: {ev}")

        return pos


def record_dict_to_sgf_line(d: Dict[str, Any]) -> str:
    """The SGF line of a record dict as load_records yields it (or as tests build it: rules / players / result /
    events, optionally comments, gameHash, match, gtype, startTurnIdx, drawReason)."""
    rec = Q4Record(
        rules=d.get("rules"), players=d.get("players"), result=d.get("result", "none"), events=d.get("events", []),
        comments=d.get("comments", []), game_hash=d.get("gameHash"), match=d.get("match"),
        gtype=d.get("gtype", "normal"), start_turn_idx=d.get("startTurnIdx", 0), draw_reason=d.get("drawReason", ""))
    rec.rules = {"maxPlies": 400, "repetitionDrawCount": 0, "initialWalls": [7, 7, 7, 7], **rec.rules}
    return rec.to_sgf_line()


def load_records(path: str) -> Iterator[Dict[str, Any]]:
    """The games of a .sgfs file, one dict per line (see the module docstring)."""
    with open(path) as f:
        for line in f:
            if line.strip():
                yield Q4Record.from_sgf_line(line).to_dict()
