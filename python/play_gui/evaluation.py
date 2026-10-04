"""Parse KataQuoridor's evaluation output and convert it to an explicit Black/White perspective.

KataQuoridor reports values from different perspectives:

- `kata-genmove_analyze` / `kata-search_analyze` ("info move ... winrate ... scoreLead ..." and
  "rootInfo ..."): from the side to move at the searched position, because the play GUI starts the engine
  with `reportAnalysisWinratesAs=SIDETOMOVE`. A move's `winrate`/`scoreLead` is the value for the player
  who makes that move, i.e. an estimate of the position after it.
- `kata-raw-nn`: always from White's perspective (`whiteWin`, `whiteLoss`, `whiteLead`), whoever is to move.

Scores are KataQuoridor tempo leads (`scoreLead`; the result of a finished game, e.g. 0.5 for B+0.5, see
docs/QuoridorIOv2.md), so a positive `black_lead` reads as "Black leads by n".

Every evaluation handed to the frontend is a dict:
    {"black_win": float, "white_win": float, "black_lead": float, "source": "search"|"net"|"final",
     "visits": int|None, "pv": [moves]}
"""


EVAL_KEYS = ("black_win", "white_win", "black_lead", "source", "visits", "pv")


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
    """The exact value of a finished game: winner "b"/"w", margin = the winner's tempo lead (> 0)."""
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
               "utilityLcb", "weight", "rawStWrError", "rawStScoreError", "rawVarTimeLeft"}
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


def candidates(analysis, player):
    """The searched children of `parse_analysis` output for `player` to move, in the engine's order.

    Each is an evaluation dict (Black/White perspective, for the position after the move) plus the raw
    side-to-move numbers: winrate, lead, lcb, utility, score_stdev, prior (the net's policy for the move),
    visits / edge_visits and order. Children listed only because of `minmoves` have 0 visits; their values
    are meaningless, only `prior` counts.
    """
    out = []
    for d in analysis["infos"]:
        visits = d.get("visits", 0)
        e = from_side_to_move(d.get("winrate", 0.5), d.get("scoreLead", 0.0), player, "search", visits, d.get("pv"))
        e.update(move=d["move"], order=d.get("order", len(out)), prior=d.get("prior"), winrate=d.get("winrate"),
                 lead=d.get("scoreLead"), lcb=d.get("lcb"), utility=d.get("utility"),
                 score_stdev=d.get("scoreStdev"), edge_visits=d.get("edgeVisits", visits))
        out.append(e)
    return out


def root_summary(analysis, player):
    """The rootInfo of `parse_analysis` output as an evaluation dict plus the raw numbers, or None."""
    r = analysis["root"]
    if not r or "winrate" not in r:
        return None
    pv = analysis["infos"][0].get("pv") if analysis["infos"] else None
    e = from_side_to_move(r["winrate"], r.get("scoreLead", 0.0), player, "search", r.get("visits"), pv)
    e.update(winrate=r["winrate"], lead=r.get("scoreLead"), utility=r.get("utility"),
             score_stdev=r.get("scoreStdev"), raw_wr_error=r.get("rawStWrError"),
             raw_score_error=r.get("rawStScoreError"), raw_time_left=r.get("rawVarTimeLeft"))
    return e
