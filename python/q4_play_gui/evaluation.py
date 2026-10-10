"""Pure helpers of the Q4 play GUI: the engine's analysis lines, the q4-rawnn text, the move list text.

No I/O here, so the tests can call these without an engine. Seat letters are S W N E (seat 0..3), the
engine's seat numbers are 1..4 (QTP `play 1 f2`).
"""
import re

SEAT_LETTERS = "SWNE"
SEAT_NAMES = ("South", "West", "North", "East")
MOVE_RE = re.compile(r"^(?:[a-k](?:1[01]|[1-9])|[a-j](?:10|[1-9])[hv])$")


def seat_letter(seat):
    return SEAT_LETTERS[seat]


def seat_from_letter(text):
    """'S' / 's' / '1' .. '4' -> 0..3; ValueError otherwise."""
    t = text.strip()
    if len(t) == 1 and t.upper() in SEAT_LETTERS:
        return SEAT_LETTERS.index(t.upper())
    if t.isdigit() and 1 <= int(t) <= 4:
        return int(t) - 1
    raise ValueError("not a seat: %r (use S W N E or 1..4)" % text)


def _floats(text, n):
    parts = text.split(",")
    if len(parts) != n:
        raise ValueError("expected %d values, got %r" % (n, text))
    return [float(p) for p in parts]


def parse_info(line):
    """One engine "info" line of q4-search_analyze_cancellable -> dict, or None for other lines.

    {"kind": "root", "visits", "value" [5], "utility" [4], "lead" [4]}
    {"kind": "move", "move", "visits", "prior", "utility", "lcb", "value" [5], "pv": [(seat letter, move)]}
    """
    tok = line.split()
    if len(tok) < 2 or tok[0] != "info":
        return None
    if tok[1] == "root":
        out = {"kind": "root", "visits": 0, "value": None, "utility": None, "lead": None}
        i = 2
        while i + 1 < len(tok):
            key, val = tok[i], tok[i + 1]
            if key == "visits":
                out["visits"] = int(val)
            elif key == "value":
                out["value"] = _floats(val, 5)
            elif key == "utility":
                out["utility"] = _floats(val, 4)
            elif key == "lead":
                out["lead"] = _floats(val, 4)
            i += 2
        return out
    if tok[1] == "move" and len(tok) >= 3:
        out = {"kind": "move", "move": tok[2], "visits": 0, "prior": 0.0, "utility": 0.0, "lcb": 0.0,
               "value": None, "pv": []}
        i = 3
        while i < len(tok):
            key = tok[i]
            if key == "pv":
                rest = tok[i + 1:]
                out["pv"] = [(rest[j], rest[j + 1]) for j in range(0, len(rest) - 1, 2)]
                break
            val = tok[i + 1]
            if key == "visits":
                out["visits"] = int(val)
            elif key == "prior":
                out["prior"] = float(val)
            elif key == "utility":
                out["utility"] = float(val)
            elif key == "lcb":
                out["lcb"] = float(val)
            elif key == "value":
                out["value"] = _floats(val, 5)
            i += 2
        return out
    return None


RAWNN_SEAT_RE = re.compile(r"^\s*seat (\d) \([A-Za-z ]+\)\s*:\s*([\d.]+)%")
RAWNN_DRAW_RE = re.compile(r"^\s*draw\s*:\s*([\d.]+)%")
RAWNN_PLIES_RE = re.compile(r"Plies to the end of the game:\s*([\d.]+)")
RAWNN_TOP_RE = re.compile(r"^\s*([a-k](?:1[01]|[1-9])|[a-j](?:10|[1-9])[hv])\s+([\d.]+)%")


def parse_rawnn(text):
    """The text of q4-rawnn -> {"value" [S W N E D] in [0,1], "plies": float or None, "policy": [(move, pct)]}.

    The net's own opinion of the position, with no search. The policy is the top of the net's list of moves."""
    value = [0.0] * 5
    plies = None
    policy = []
    for line in text.splitlines():
        m = RAWNN_SEAT_RE.match(line)
        if m:
            value[int(m.group(1)) - 1] = float(m.group(2)) / 100.0
            continue
        m = RAWNN_DRAW_RE.match(line)
        if m:
            value[4] = float(m.group(1)) / 100.0
            continue
        m = RAWNN_PLIES_RE.search(line)
        if m:
            plies = float(m.group(1))
            continue
        m = RAWNN_TOP_RE.match(line)
        if m:
            policy.append((m.group(1), float(m.group(2))))
    return {"value": value, "plies": plies, "policy": policy}


def parse_move_list(text):
    """A pasted move list -> [{"seat": letter, "a": move}] / [{"elim": letter}] in order.

    Accepts "S f2, W b6, EL N, E j10v", optional move numbers ("1." / "12)") and line breaks.
    Raises ValueError with the offending token."""
    events = []
    body = re.sub(r"\d+\s*[.)]", " ", text)           # move numbers
    body = body.replace(",", " ").replace(";", " ")
    tok = body.split()
    i = 0
    while i < len(tok):
        t = tok[i]
        if t.upper() == "EL":
            if i + 1 >= len(tok):
                raise ValueError("EL needs a seat")
            events.append({"elim": seat_from_letter(tok[i + 1])})
            i += 2
            continue
        if t.upper() in ("S", "W", "N", "E") and i + 1 < len(tok):
            move = tok[i + 1]
            if not MOVE_RE.match(move):
                raise ValueError("not a move: %r" % move)
            events.append({"seat": SEAT_LETTERS.index(t.upper()), "a": move})
            i += 2
            continue
        if MOVE_RE.match(t):
            events.append({"a": t})   # the seat is the one to move (checked against the engine)
            i += 1
            continue
        raise ValueError("cannot read %r (write moves as 'S f2', eliminations as 'EL W')" % t)
    return events
