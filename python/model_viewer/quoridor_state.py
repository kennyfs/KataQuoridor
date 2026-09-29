"""Minimal Quoridor state + a Python replica of QuoridorNN::fillRow (I/O v1).

Coordinates follow cpp/game/board.h: pawn cell (c, r) with c, r in 0..8 (SGF / 17x17 grid point
(2c, 2r)); a vertical wall anchor (c, r) has its upper arm at (2c+1, 2r) and blocks the east edges of
(c, r) and (c, r+1); a horizontal wall anchor (c, r) is centered at (2c+1, 2r+1) and blocks the south
edges of (c, r) and (c+1, r). Black starts on row 8 and heads to row 0; White the reverse.
Move strings match Location::toString / GTP: "e8" (pawn), "e2h" / "e2v" (walls), row = r + 1.

The replay does not check legality: it only applies moves taken from real games.
"""
import numpy as np

N = 9
NUM_SPATIAL = 17
NUM_GLOBAL = 15
POLICY_LEN = 3 * N * N

SPATIAL_NAMES = [
    "On-board mask",
    "My pawn",
    "Opponent pawn",
    "Blocked north",
    "Blocked south",
    "Blocked east",
    "Blocked west",
    "Goal row",
    "My goal distance /32",
    "Opp goal distance /32",
    "Distance from my pawn /32",
    "Distance from opp pawn /32",
    "My shortest-path cells",
    "Opp shortest-path cells",
    "Vertical wall anchors",
    "Horizontal wall anchors",
    "Wall-anchor domain",
]
SPATIAL_DESCS = [
    "1 on every cell (the model also uses it as the attention/pooling mask).",
    "1 at the side-to-move's pawn.",
    "1 at the opponent's pawn.",
    "1 if the cell cannot step toward the side-to-move's goal (row above in canonical view): wall or board edge.",
    "1 if the cell cannot step away from the side-to-move's goal (row below in canonical view).",
    "1 if the cell cannot step east (wall or board edge).",
    "1 if the cell cannot step west (wall or board edge).",
    "1 on the side-to-move's goal row (always the top row in canonical view).",
    "BFS steps from the cell to my goal row, /32 (1 if unreachable).",
    "BFS steps from the cell to the opponent's goal row, /32 (1 if unreachable).",
    "BFS steps from my pawn to the cell, /32 (1 if unreachable).",
    "BFS steps from the opponent's pawn to the cell, /32 (1 if unreachable).",
    "1 on cells lying on any of my shortest paths to goal (pawns ignored).",
    "1 on cells lying on any of the opponent's shortest paths to goal.",
    "1 at vertical-wall anchors (8x8 domain).",
    "1 at horizontal-wall anchors (8x8 domain).",
    "1 on the 8x8 wall-anchor domain.",
]
GLOBAL_NAMES = [
    "Side to move is White",
    "My fences /10",
    "Opp fences /10",
    "My fences exp(-(n-1)/1)",
    "My fences exp(-(n-1)/2)",
    "My fences exp(-(n-1)/4)",
    "My fences exp(-(n-1)/8)",
    "Opp has any fence",
    "Opp fences exp(-(n-1)/1)",
    "Opp fences exp(-(n-1)/2)",
    "Opp fences exp(-(n-1)/4)",
    "Opp fences exp(-(n-1)/8)",
    "Jump-tempo parity (+1/-1)",
    "My shortest distance /32",
    "Opp shortest distance /32",
]
GLOBAL_DESCS = [
    "1 if White is to move (the board itself is always shown from the mover's side).",
    "Remaining fences of the side to move, /10.",
    "Remaining fences of the opponent, /10.",
    "exp(-(n-1)/1) of my remaining fences n (0 if n = 0): sharp near the last fence.",
    "exp(-(n-1)/2) of my remaining fences.",
    "exp(-(n-1)/4) of my remaining fences.",
    "exp(-(n-1)/8) of my remaining fences.",
    "1 if the opponent still has at least one fence.",
    "exp(-(n-1)/1) of the opponent's remaining fences.",
    "exp(-(n-1)/2) of the opponent's remaining fences.",
    "exp(-(n-1)/4) of the opponent's remaining fences.",
    "exp(-(n-1)/8) of the opponent's remaining fences.",
    "+1 if the pawns' Manhattan distance is odd (side to move gets the jump), else -1.",
    "My BFS shortest distance to goal, /32.",
    "Opponent's BFS shortest distance to goal, /32.",
]
# Spatial channels whose C++ value is continuous. The training data stores spatial inputs as packed
# bits (TrainingWriteBuffers::addRow -> packBits casts each float to uint8), so these train as (x >= 1).
CONTINUOUS_SPATIAL = [8, 9, 10, 11]


def move_str(kind, c, r):
    s = chr(ord("a") + c) + str(r + 1)
    return s if kind == "pawn" else s + ("h" if kind == "hwall" else "v")


def parse_move_str(s):
    s = s.strip().lower()
    c = ord(s[0]) - ord("a")
    if s[-1] in "hv":
        return ("hwall" if s[-1] == "h" else "vwall", c, int(s[1:-1]) - 1)
    return ("pawn", c, int(s[1:]) - 1)


def sgf_point_to_move(pt):
    x, y = ord(pt[0]) - 97, ord(pt[1]) - 97
    c, r = x >> 1, y >> 1
    if x % 2 == 0 and y % 2 == 0:
        return ("pawn", c, r)
    if x % 2 == 1 and y % 2 == 1:
        return ("hwall", c, r)
    if x % 2 == 1 and y % 2 == 0:
        return ("vwall", c, r)
    raise ValueError("bad point " + pt)


class QState:
    def __init__(self):
        self.pawn = {"B": (4, 8), "W": (4, 0)}
        self.vwalls = np.zeros((8, 8), dtype=bool)  # [c][r], like Board::vWalls
        self.hwalls = np.zeros((8, 8), dtype=bool)
        self.fences = {"B": 10, "W": 10}
        self.to_move = "B"
        self.history = []  # move strings in GTP notation, with player

    def copy(self):
        s = QState()
        s.pawn = dict(self.pawn)
        s.vwalls = self.vwalls.copy()
        s.hwalls = self.hwalls.copy()
        s.fences = dict(self.fences)
        s.to_move = self.to_move
        s.history = list(self.history)
        return s

    def play(self, pla, mv):
        kind, c, r = mv
        if kind == "pawn":
            self.pawn[pla] = (c, r)
        elif kind == "vwall":
            self.vwalls[c, r] = True
            self.fences[pla] -= 1
        else:
            self.hwalls[c, r] = True
            self.fences[pla] -= 1
        self.history.append((pla, move_str(kind, c, r)))
        self.to_move = "W" if pla == "B" else "B"

    def num_walls(self):
        return int(self.vwalls.sum() + self.hwalls.sum())

    # blocked[d][c][r]: step from (c, r) in direction d (0 N=r-1, 1 S=r+1, 2 E=c+1, 3 W=c-1) is impossible
    def blocked(self):
        b = np.zeros((4, N, N), dtype=bool)
        b[0, :, 0] = True
        b[1, :, N - 1] = True
        b[2, N - 1, :] = True
        b[3, 0, :] = True
        for c in range(8):
            for r in range(8):
                if self.vwalls[c, r]:
                    for rr in (r, r + 1):
                        b[2, c, rr] = True
                        b[3, c + 1, rr] = True
                if self.hwalls[c, r]:
                    for cc in (c, c + 1):
                        b[1, cc, r] = True
                        b[0, cc, r + 1] = True
        return b


DIRS = ((0, -1), (0, 1), (1, 0), (-1, 0))  # N S E W as (dc, dr)


def bfs(blocked, sources):
    """blocked: (4, N, N) indexed [dir][c][r]. Returns dist[c][r], -1 if unreachable."""
    dist = np.full((N, N), -1, dtype=np.int32)
    q = []
    for (c, r) in sources:
        dist[c, r] = 0
        q.append((c, r))
    h = 0
    while h < len(q):
        c, r = q[h]
        h += 1
        for d, (dc, dr) in enumerate(DIRS):
            nc, nr = c + dc, r + dr
            if 0 <= nc < N and 0 <= nr < N and dist[nc, nr] < 0 and not blocked[d, c, r]:
                dist[nc, nr] = dist[c, r] + 1
                q.append((nc, nr))
    return dist


def _dist_feature(d):
    return np.where(d < 0, 1.0, np.minimum(1.0, d / 32.0)).astype(np.float32)


def fill_row(st):
    """Returns (spatial[17, 9, 9] in canonical NCHW = [ch][rCanon][c], global[15]), as the C++ backend sees it."""
    pla = st.to_move
    opp = "W" if pla == "B" else "B"
    blk = st.blocked()
    cur_goal = 0 if pla == "B" else 8
    opp_goal = 8 if pla == "B" else 0
    (cc, cr), (oc, orr) = st.pawn[pla], st.pawn[opp]

    d_goal_cur = bfs(blk, [(c, cur_goal) for c in range(N)])
    d_goal_opp = bfs(blk, [(c, opp_goal) for c in range(N)])
    d_pawn_cur = bfs(blk, [(cc, cr)])
    d_pawn_opp = bfs(blk, [(oc, orr)])
    sd_cur = d_goal_cur[cc, cr]
    sd_opp = d_goal_opp[oc, orr]
    on_cur = (sd_cur >= 0) & (d_pawn_cur >= 0) & (d_goal_cur >= 0) & (d_pawn_cur + d_goal_cur == sd_cur)
    on_opp = (sd_opp >= 0) & (d_pawn_opp >= 0) & (d_goal_opp >= 0) & (d_pawn_opp + d_goal_opp == sd_opp)

    # board-space [c][r] planes; converted to canonical [rCanon][c] at the end
    planes = np.zeros((NUM_SPATIAL, N, N), dtype=np.float32)  # [ch][c][rBoard]
    planes[0] = 1.0
    planes[1, cc, cr] = 1.0
    planes[2, oc, orr] = 1.0
    if pla == "B":
        planes[3], planes[4] = blk[0], blk[1]
    else:
        planes[3], planes[4] = blk[1], blk[0]
    planes[5], planes[6] = blk[2], blk[3]
    planes[7, :, cur_goal] = 1.0
    planes[8] = _dist_feature(d_goal_cur)
    planes[9] = _dist_feature(d_goal_opp)
    planes[10] = _dist_feature(d_pawn_cur)
    planes[11] = _dist_feature(d_pawn_opp)
    planes[12] = on_cur
    planes[13] = on_opp
    spatial = np.zeros((NUM_SPATIAL, N, N), dtype=np.float32)
    for ch in range(14):
        p = planes[ch].T  # [rBoard][c]
        spatial[ch] = p if pla == "B" else p[::-1]
    # wall anchors: canonical row rCanon <-> board row r (black) or 7 - r (white)
    v, h = st.vwalls.T.astype(np.float32), st.hwalls.T.astype(np.float32)  # [r][c]
    if pla == "W":
        v, h = v[::-1], h[::-1]
    spatial[14, :8, :8] = v
    spatial[15, :8, :8] = h
    spatial[16, :8, :8] = 1.0

    g = np.zeros(NUM_GLOBAL, dtype=np.float32)
    g[0] = 1.0 if pla == "W" else 0.0
    mf, of = st.fences[pla], st.fences[opp]
    g[1] = mf / 10.0
    g[2] = of / 10.0
    if mf > 0:
        d = mf - 1.0
        g[3:7] = [np.exp(-d / 1.0), np.exp(-d / 2.0), np.exp(-d / 4.0), np.exp(-d / 8.0)]
    g[7] = 1.0 if of >= 1 else 0.0
    if of > 0:
        d = of - 1.0
        g[8:12] = [np.exp(-d / 1.0), np.exp(-d / 2.0), np.exp(-d / 4.0), np.exp(-d / 8.0)]
    (bc, br), (wc, wr) = st.pawn["B"], st.pawn["W"]
    g[12] = 1.0 if (abs(bc - wc) + abs(br - wr)) % 2 != 0 else -1.0
    g[13] = 1.0 if sd_cur < 0 else min(1.0, sd_cur / 32.0)
    g[14] = 1.0 if sd_opp < 0 else min(1.0, sd_opp / 32.0)
    return spatial, g


def binarize_like_training(spatial):
    """What the training data holds: packBits casts each float to uint8, so a value in [0, 1] survives only if it is 1."""
    return (spatial >= 1.0).astype(np.float32)


def continuous_from_training(spatial_bin):
    """Rebuilds the inference-time continuous channels 8..11 of rows read from training data (batched, (B,17,9,9)).

    Works in canonical space: ch3/ch4 block steps toward row rCanon-1 / rCanon+1, ch5/ch6 east / west,
    my goal is canonical row 0, the opponent's is row 8. Returns a copy with ch8..11 replaced.
    """
    out = spatial_bin.copy()
    for i in range(spatial_bin.shape[0]):
        x = spatial_bin[i]
        # blocked as [dir][c][r] in canonical coords
        blk = np.stack([x[3].T, x[4].T, x[5].T, x[6].T]).astype(bool)
        me = np.argwhere(x[1] > 0.5)
        op = np.argwhere(x[2] > 0.5)
        d_goal_cur = bfs(blk, [(c, 0) for c in range(N)])
        d_goal_opp = bfs(blk, [(c, 8) for c in range(N)])
        d_pawn_cur = bfs(blk, [(int(me[0][1]), int(me[0][0]))]) if len(me) else np.full((N, N), -1)
        d_pawn_opp = bfs(blk, [(int(op[0][1]), int(op[0][0]))]) if len(op) else np.full((N, N), -1)
        out[i, 8] = _dist_feature(d_goal_cur).T
        out[i, 9] = _dist_feature(d_goal_opp).T
        out[i, 10] = _dist_feature(d_pawn_cur).T
        out[i, 11] = _dist_feature(d_pawn_opp).T
    return out


VALID_POLICY_MASK = np.zeros(POLICY_LEN, dtype=bool)  # same as Metrics.valid_action_mask
VALID_POLICY_MASK[0:81] = True
for _r in range(8):
    for _c in range(8):
        VALID_POLICY_MASK[81 + _r * 9 + _c] = True
        VALID_POLICY_MASK[162 + _r * 9 + _c] = True


def policy_index_to_move(idx, pla):
    """Canonical policy index (plane*81 + rCanon*9 + c) -> (kind, c, rBoard) for the side to move."""
    plane, rem = divmod(idx, 81)
    rc, c = divmod(rem, 9)
    if plane == 0:
        return ("pawn", c, rc if pla == "B" else 8 - rc)
    return ("vwall" if plane == 1 else "hwall", c, rc if pla == "B" else 7 - rc)


def move_to_policy_index(mv, pla):
    kind, c, r = mv
    if kind == "pawn":
        return 0 * 81 + (r if pla == "B" else 8 - r) * 9 + c
    plane = 1 if kind == "vwall" else 2
    return plane * 81 + (r if pla == "B" else 7 - r) * 9 + c


def parse_sgfs_line(line):
    """One game per line (.sgfs). Returns dict(moves=[(pla, mv, comment)], result, props)."""
    import re
    semi = line.find(";", 2)
    root = line if semi < 0 else line[:semi]
    props = {}
    for m in re.finditer(r"([A-Z]+)((?:\[[^\]]*\])+)", root):
        props[m.group(1)] = re.findall(r"\[([^\]]*)\]", m.group(2))
    moves = []
    body = "" if semi < 0 else line[semi:]
    for m in re.finditer(r";([BW])\[([a-z]*)\](?:C\[([^\]]*)\])?", body):
        if len(m.group(2)) != 2:
            continue
        moves.append((m.group(1), sgf_point_to_move(m.group(2)), m.group(3)))
    kv = {}
    for part in (props.get("C", [""])[0]).split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip()] = v.strip()
    start = {"B": (4, 8), "W": (4, 0)}
    if "AB" in props:
        _, c, r = sgf_point_to_move(props["AB"][0]); start["B"] = (c, r)
    if "AW" in props:
        _, c, r = sgf_point_to_move(props["AW"][0]); start["W"] = (c, r)
    return {"moves": moves, "result": (props.get("RE", ["?"])[0]), "kv": kv, "start": start, "props": props}


def parse_eval_comment(c):
    """SGF move comment: 'wWin wLoss noRes wScore v=.. weight=..' = search result BEFORE the move (White's view)."""
    if not c:
        return None
    t = c.split()
    if len(t) < 4:
        return None
    try:
        ev = {"wWin": float(t[0]), "wLoss": float(t[1]), "wScore": float(t[3])}
    except ValueError:
        return None
    for tok in t[4:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            ev[k] = v
    return ev
