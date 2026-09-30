"""Parse KataQuoridor's evaluation output and convert it to an explicit Black/White perspective.

KataQuoridor reports values from different perspectives:

- `kata-genmove_analyze` / `kata-search_analyze` ("info move ... winrate ... scoreLead ..." and
  "rootInfo ..."): from the side to move at the searched position, because the play GUI starts the engine
  with `reportAnalysisWinratesAs=SIDETOMOVE`. A move's `winrate`/`scoreLead` is the value for the player
  who makes that move, i.e. an estimate of the position after it.
- `kata-raw-nn`: always from White's perspective (`whiteWin`, `whiteLoss`, `whiteLead`), whoever is to move.

Scores are Quoridor margins in moves (the loser's remaining shortest-path distance at the end), so a
positive `black_lead` reads as "Black leads by n moves".

Every evaluation handed to the frontend is a dict:
    {"black_win": float, "white_win": float, "black_lead": float, "source": "search"|"net"|"final",
     "visits": int|None, "pv": [moves]}
"""


def make_eval(black_win, black_lead, source, visits=None, pv=None, draw=0.0):
    black_win = min(1.0, max(0.0, float(black_win)))
    white_win = min(1.0, max(0.0, 1.0 - black_win - float(draw)))
    return {"black_win": black_win, "white_win": white_win, "black_lead": float(black_lead),
            "source": source, "visits": visits, "pv": list(pv or [])}


def from_side_to_move(winrate, score_lead, player, source="search", visits=None, pv=None):
    """Convert a (winrate, scoreLead) pair given from `player`'s point of view ("b" or "w")."""
    if player == "b":
        return make_eval(winrate, score_lead, source, visits, pv)
    if player == "w":
        return make_eval(1.0 - float(winrate), -float(score_lead), source, visits, pv)
    raise ValueError("player must be 'b' or 'w', got %r" % (player,))


def from_raw_nn(values):
    """Convert parsed `kata-raw-nn` values (White's perspective) to the Black/White form."""
    white_win = float(values["whiteWin"])
    white_loss = float(values["whiteLoss"])
    no_result = float(values.get("noResult", 0.0))
    total = white_win + white_loss + no_result
    if total > 0:  # the three are softmax outputs and should already sum to 1
        white_win, white_loss, no_result = white_win / total, white_loss / total, no_result / total
    return make_eval(white_loss, -float(values["whiteLead"]), "net", draw=no_result)


def final_eval(winner, margin):
    """The exact value of a finished game: winner "b"/"w", margin in moves (>= 1)."""
    if winner == "b":
        return make_eval(1.0, margin, "final")
    if winner == "w":
        return make_eval(0.0, -margin, "final")
    raise ValueError("winner must be 'b' or 'w', got %r" % (winner,))


# -- parsing ------------------------------------------------------------------------------------------
_RAW_NN_SCALARS = ("whiteWin", "whiteLoss", "noResult", "whiteLead", "whiteScoreSelfplay", "varTimeLeft")


def parse_raw_nn(text):
    """Pick the scalar keys out of `kata-raw-nn <symmetry>` output (the policy/ownership arrays are skipped)."""
    toks = text.split()
    out = {}
    for i, t in enumerate(toks[:-1]):
        if t in _RAW_NN_SCALARS and t not in out:
            try:
                out[t] = float(toks[i + 1])
            except ValueError:
                pass
    for k in ("whiteWin", "whiteLoss", "whiteLead"):
        if k not in out:
            raise ValueError("kata-raw-nn output has no %s: %r" % (k, text[:200]))
    return out


_FLOAT_KEYS = {"winrate", "scoreLead", "scoreMean", "scoreStdev", "scoreSelfplay", "utility", "lcb", "prior",
               "utilityLcb", "weight"}
_INT_KEYS = {"visits", "edgeVisits", "order"}


def _parse_fields(toks):
    d = {}
    i = 0
    while i < len(toks):
        k = toks[i]
        if k == "pv":
            j = i + 1
            while j < len(toks) and toks[j] not in ("pvVisits", "pvEdgeVisits", "ownership", "movesOwnership"):
                j += 1
            d["pv"] = toks[i + 1:j]
            i = j
            continue
        if i + 1 < len(toks):
            v = toks[i + 1]
            try:
                d[k] = int(v) if k in _INT_KEYS else float(v) if k in _FLOAT_KEYS else v
            except ValueError:
                d[k] = v
        i += 2
    return d


def parse_analysis(text):
    """Parse the final response of `kata-genmove_analyze` / `kata-search_analyze`.

    Returns {"move": str|None, "infos": [dict sorted by order], "root": dict|None}. Only the last
    "info" line counts (the engine may print intermediate ones every `interval`). A 1-visit search prints
    no info line at all, so `infos` can be empty.
    """
    move = None
    last_info = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("play "):
            move = line.split()[1]
        elif line.startswith("info "):
            last_info = line
    infos, root = [], None
    if last_info:
        head, _, root_part = last_info.partition(" rootInfo ")
        for chunk in head.split("info ")[1:]:
            d = _parse_fields(chunk.split())
            if "move" in d:
                infos.append(d)
        if root_part:
            root = _parse_fields(root_part.split())
        infos.sort(key=lambda d: d.get("order", 0))
    return {"move": move, "infos": infos, "root": root}


def info_for(analysis, move):
    for d in analysis["infos"]:
        if d.get("move") == move:
            return d
    return None
