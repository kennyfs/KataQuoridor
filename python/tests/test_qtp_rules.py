"""QTP rule enforcement in `katago gtp` (regression tests for the 0.1.0 QTP fixes).

1. Game over: once a pawn has reached its goal, play/move/wall and every genmove/search/analyze variant fail with
   "? game is over", legal_moves is empty, and winner keeps the result. undo, clear_board, loadsgf and set_position
   still work.
2. Turn order: Quoridor strictly alternates with no pass, so play/move/wall/genmove with an explicit colour that is
   not the side to move fail (and do not change the side to move).
3. Quoridor I/O v2 rules (docs/QuoridorIOv2.md): komi / get_komi, kata-get-rules / kata-set-rule(s) (maxPlies,
   timeBonusPerPly, initial walls), the draw at maxPlies (winner -> Draw), komi deciding the winner, the fence
   handicap, and SGF KM / WB / WW / RU / RE.

The engine runs without a net (`-model /dev/null`, debugSkipNeuralNet=true), like the arena's arbiter. The binary is
taken from $KATAGO_BIN, else the most recently built cpp/build*/katago.
"""
import glob
import json
import os
import subprocess
import tempfile

import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GTP_CONFIG = os.path.join(repo_root, "cpp", "configs", "gtp_quoridor.cfg")

# Black starts at e9 and walks down column e, White starts at e1 and walks up column d; Black reaches row 1 first.
BLACK_WINS = ["b e8", "w d1", "b e7", "w d2", "b e6", "w d3", "b e5", "w d4",
              "b e4", "w d5", "b e3", "w d6", "b e2", "w d7", "b e1"]


def _katago_bin():
    candidates = [os.environ["KATAGO_BIN"]] if os.environ.get("KATAGO_BIN") else (
        glob.glob(os.path.join(repo_root, "build", "katago")) + glob.glob(os.path.join(repo_root, "cpp", "build*", "katago")))
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        pytest.skip("KataGo binary not found (set KATAGO_BIN or build under build/ or cpp/build*/).")
    return os.path.abspath(max(candidates, key=os.path.getmtime))


class Engine:
    def __init__(self, logdir):
        overrides = ",".join(["debugSkipNeuralNet=true", "logAllGTPCommunication=false", "logSearchInfo=false",
                              "logToStderr=false", "ponderingEnabled=false", "logDir=" + logdir])
        self.p = subprocess.Popen(
            [_katago_bin(), "gtp", "-model", "/dev/null", "-config", GTP_CONFIG, "-override-config", overrides],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=repo_root)

    def send(self, cmd):
        """Returns (ok, text) for one command."""
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        lines = []
        while True:
            line = self.p.stdout.readline()
            assert line != "", "katago exited during %r" % cmd
            line = line.rstrip("\n")
            if line == "" and lines:
                break
            if line != "" or lines:
                lines.append(line)
        head = lines[0]
        assert head[0] in "=?", "unexpected response to %r: %r" % (cmd, lines)
        return head[0] == "=", "\n".join([head[1:].strip()] + lines[1:]).strip()

    def ok(self, cmd):
        ok, text = self.send(cmd)
        assert ok, "%r failed: %s" % (cmd, text)
        return text

    def err(self, cmd):
        ok, text = self.send(cmd)
        assert not ok, "%r should have failed but returned %r" % (cmd, text)
        return text

    def close(self):
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


@pytest.fixture
def eng():
    with tempfile.TemporaryDirectory() as d:
        e = Engine(d)
        yield e
        e.close()


def _play_to_black_win(eng):
    for i, m in enumerate(BLACK_WINS):
        assert eng.ok("winner") == "none", "game ended early before %r" % m
        eng.ok("play " + m)
    assert eng.ok("winner") == "B"


def test_name_and_version(eng):
    assert eng.ok("name") == "KataQuoridor"
    assert eng.ok("version").startswith("KataQuoridor 0.1.0 (based on KataGo 1.18.2, git ")


GAME_OVER_COMMANDS = [
    "play w d8", "play b d1", "move d8", "move w d8", "wall a1h", "wall w a1h",
    "genmove w", "genmove b", "genmove_debug w", "kata-search w", "kata-search_debug w",
    "kata-genmove_analyze w", "lz-genmove_analyze w", "kata-search_analyze w", "kata-analyze w", "lz-analyze w",
]


def test_game_over(eng):
    _play_to_black_win(eng)
    for cmd in GAME_OVER_COMMANDS:
        assert eng.err(cmd) == "game is over", cmd
        assert eng.ok("winner") == "B", "winner changed after %r" % cmd
    assert eng.ok("legal_moves") == ""
    assert eng.ok("legal_moves w") == ""
    assert eng.ok("legal_moves b") == ""
    # Board state is unchanged: Black on the goal row, White where it was.
    board = eng.ok("showboard")
    assert "Black (B): e1" in board and "White (W): d7" in board, board


def test_winner_cannot_step_off_goal(eng):
    # Before the fix, a winner stepping back off its goal row reset `winner` to none.
    _play_to_black_win(eng)
    eng.err("play w d8")
    eng.err("play b e2")
    assert eng.ok("winner") == "B"


def test_undo_and_clear_board_after_game_over(eng):
    _play_to_black_win(eng)
    eng.ok("undo")
    assert eng.ok("winner") == "none"
    assert "e1" in eng.ok("legal_moves").split()
    eng.ok("play b e1")
    assert eng.ok("winner") == "B"
    eng.ok("clear_board")
    assert eng.ok("winner") == "none"
    eng.ok("play b e8")
    assert eng.ok("genmove w") != ""


def test_loadsgf_after_game_over(eng):
    _play_to_black_win(eng)
    sgf = eng.ok("printsgf")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "game.sgf")
        with open(path, "w") as f:
            f.write(sgf)
        eng.ok("clear_board")
        # The whole finished game: the result is restored and nothing can be played.
        eng.ok("loadsgf " + path)
        assert eng.ok("winner") == "B"
        assert eng.err("play w d8") == "game is over"
        assert eng.ok("legal_moves") == ""
        # Up to (not including) move 15, Black's winning move: the game is open again.
        eng.ok("loadsgf %s 15" % path)
        assert eng.ok("winner") == "none"
        eng.err("play w d8")
        eng.ok("play b e1")
        assert eng.ok("winner") == "B"


def test_set_position_after_game_over(eng):
    _play_to_black_win(eng)
    eng.ok("set_position")
    assert eng.ok("winner") == "none"
    eng.ok("play b e8")


def test_white_win_is_final(eng):
    # White walks up column e, Black detours through column d; White reaches row 9 first.
    moves = ["b d9", "w e2", "b d8", "w e3", "b d7", "w e4", "b d6", "w e5",
             "b d5", "w e6", "b d4", "w e7", "b d3", "w e8", "b d2", "w e9"]
    for m in moves:
        eng.ok("play " + m)
    assert eng.ok("winner") == "W"
    assert eng.err("play b d1") == "game is over"
    assert eng.err("genmove b") == "game is over"
    assert eng.ok("winner") == "W"


TURN_COMMANDS = ["play w e2", "move w e2", "wall w a1h", "genmove w", "kata-search w", "kata-genmove_analyze w",
                 "kata-analyze w"]


def test_turn_order_at_start(eng):
    for cmd in TURN_COMMANDS:
        text = eng.err(cmd)
        assert "not white's turn" in text, (cmd, text)
    # The failed commands did not hand White the move.
    eng.ok("play b e8")
    for cmd in ["play b e7", "move b e7", "wall b a1h", "genmove b"]:
        assert "not black's turn" in eng.err(cmd), cmd
    eng.ok("play w e2")
    # Colour-less move/wall default to the side to move.
    eng.ok("move e7")
    eng.ok("wall a1h")
    assert eng.ok("walls") == "B: 10 W: 9"
    eng.ok("play b e6")


def test_play_error_is_illegal_move(eng):
    # Controllers matching on "illegal move" still recognise an out-of-turn play.
    assert eng.err("play w e2").startswith("illegal move")
    assert eng.err("play b e7") == "illegal move"


def test_undo_restores_turn(eng):
    eng.ok("play b e8")
    eng.ok("play w e2")
    eng.ok("undo")
    eng.err("play b e7")
    eng.ok("play w e2")


def test_genmove_alternates(eng):
    color = "b"
    for _ in range(20):
        if eng.ok("winner") != "none":
            break
        other = "w" if color == "b" else "b"
        assert "not" in eng.err("genmove " + other)
        mv = eng.ok("genmove " + color)
        assert mv not in ("pass", "resign", ""), mv
        color = other


# --- Quoridor I/O v2 rules (docs/QuoridorIOv2.md): komi, maxPlies draw, time bonus, fence handicap ----------------

# Black and White shuffle sideways on their start rows: nobody ever gets closer to the goal.
def _shuffle(n):
    moves = []
    for i in range(n):
        color = "b" if i % 2 == 0 else "w"
        row = "9" if color == "b" else "1"
        col = "d" if (i // 2) % 2 == 0 else "e"
        moves.append("%s %s%s" % (color, col, row))
    return moves


# An equal race: Black steps to d9 and walks down column d, White steps to f1 and walks up column f. Black, moving
# first, arrives on ply 17 with White 1 step away: B+1, tempo 0.
EQUAL_RACE = ["b d9", "w f1"] + [m for r in range(8) for m in ("b d%d" % (8 - r), "w f%d" % (2 + r))][:15]


def _json(text):
    return json.loads(text)


def test_komi_commands(eng):
    assert eng.ok("get_komi") == "-0.5"
    for bad in ("komi 0", "komi 7", "komi 0.25", "komi 21.5", "komi -21.5", "komi abc", "komi"):
        assert "komi" in eng.err(bad), bad
    assert eng.ok("get_komi") == "-0.5"
    eng.ok("komi 1.5")
    assert eng.ok("get_komi") == "1.5"
    assert "Komi: 1.5" in eng.ok("showboard")
    eng.ok("komi -0.5")
    assert "Komi" not in eng.ok("showboard")


def test_rules_commands(eng):
    assert _json(eng.ok("kata-get-rules")) == {
        "maxPlies": 300, "timeBonusPerPly": 0.0, "blackInitialWalls": 10, "whiteInitialWalls": 10}
    eng.ok("kata-set-rule maxPlies 40")
    eng.ok("kata-set-rule timeBonusPerPly 0.05")
    eng.ok("kata-set-rule whiteInitialWalls 7")
    assert _json(eng.ok("kata-get-rules")) == {
        "maxPlies": 40, "timeBonusPerPly": 0.05, "blackInitialWalls": 10, "whiteInitialWalls": 7}
    assert eng.ok("walls") == "B: 10 W: 7"
    for bad in ("kata-set-rule komi 1.5", "kata-set-rule maxPlies 0", "kata-set-rule blackInitialWalls 11",
                "kata-set-rule timeBonusPerPly -1", "kata-set-rule ko SIMPLE", "kata-set-rule maxPlies"):
        eng.err(bad)
    # maxPlies and the initial walls define the game from its start: only before the first move.
    eng.ok("play b e8")
    assert "before the first move" in eng.err("kata-set-rule maxPlies 50")
    assert "before the first move" in eng.err("kata-set-rule blackInitialWalls 5")
    eng.ok("kata-set-rule timeBonusPerPly 0.1")
    eng.ok("clear_board")
    eng.ok("kata-set-rules quoridor")
    assert _json(eng.ok("kata-get-rules"))["maxPlies"] == 300
    assert eng.ok("walls") == "B: 10 W: 10"


def test_showboard_ply(eng):
    assert "Ply: 0 (draw at 300, 300 left)" in eng.ok("showboard")
    eng.ok("play b e8")
    eng.ok("play w e2")
    assert "Ply: 2 (draw at 300, 298 left)" in eng.ok("showboard")
    # set_position starts counting at 0 again.
    eng.ok("set_position")
    assert "Ply: 0 (draw at 300, 300 left)" in eng.ok("showboard")


def test_draw_at_300_plies(eng):
    moves = _shuffle(300)
    for m in moves[:-1]:
        eng.ok("play " + m)
    assert eng.ok("winner") == "none"
    eng.ok("play " + moves[-1])
    assert eng.ok("winner") == "Draw"
    board = eng.ok("showboard")
    assert "Game finished: Draw (300-ply limit)" in board and "Ply: 300 (draw at 300, 0 left)" in board
    assert eng.err("play b e9") == "game is over"
    assert eng.err("genmove b") == "game is over"
    assert eng.ok("legal_moves") == ""
    assert "RE[0]" in eng.ok("printsgf")
    # Undo across the draw reopens the game with one ply to go.
    eng.ok("undo")
    assert eng.ok("winner") == "none"
    assert "Ply: 299 (draw at 300, 1 left)" in eng.ok("showboard")
    eng.ok("play " + moves[-1])
    assert eng.ok("winner") == "Draw"


def test_win_on_last_ply_is_a_win(eng):
    # The equal race ends on ply 17. With maxPlies 17 Black's arrival is still a win; with 16 the game is a draw.
    eng.ok("kata-set-rule maxPlies 17")
    for m in EQUAL_RACE:
        eng.ok("play " + m)
    assert eng.ok("winner") == "B"
    eng.ok("clear_board")
    eng.ok("kata-set-rule maxPlies 16")
    for m in EQUAL_RACE[:-1]:
        eng.ok("play " + m)
    assert eng.ok("winner") == "Draw"


def test_komi_decides_winner(eng):
    for m in EQUAL_RACE:
        eng.ok("play " + m)
    assert eng.ok("winner") == "B"
    assert "RE[B+0.5]" in eng.ok("printsgf")
    # Komi +0.5: the equal race goes to White, although Black's pawn reached its goal.
    eng.ok("komi 0.5")
    assert eng.ok("winner") == "W"
    assert "Game finished: winner = White (W+0.5)" in eng.ok("showboard")
    sgf = eng.ok("printsgf")
    assert "KM[0.5]" in sgf and "RE[W+0.5]" in sgf
    assert eng.err("play w f9") == "game is over"
    eng.ok("komi -0.5")
    assert eng.ok("winner") == "B"


def test_fence_handicap(eng):
    eng.ok("kata-set-rule blackInitialWalls 0")
    assert eng.ok("walls") == "B: 0 W: 10"
    assert eng.err("wall b a1h").startswith("illegal move")
    eng.ok("play b e8")
    eng.ok("wall w a1h")
    assert eng.ok("walls") == "B: 0 W: 9"
    assert "WB[0]WW[10]" in eng.ok("printsgf")
    # clear_board keeps the rules.
    eng.ok("clear_board")
    assert eng.ok("walls") == "B: 0 W: 10"


def test_sgf_round_trip(eng):
    eng.ok("kata-set-rule maxPlies 100")
    eng.ok("kata-set-rule timeBonusPerPly 0.05")
    eng.ok("kata-set-rule whiteInitialWalls 6")
    eng.ok("komi 1.5")
    for m in EQUAL_RACE:
        eng.ok("play " + m)
    assert eng.ok("winner") == "W"
    sgf = eng.ok("printsgf")
    for prop in ("KM[1.5]", "WB[10]WW[6]", "RU[Quoridor:maxPlies=100,timeBonusPerPly=0.05,whiteInitialWalls=6]",
                 "RE[W+1.5]"):
        assert prop in sgf, (prop, sgf)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "game.sgf")
        with open(path, "w") as f:
            f.write(sgf)
        eng.ok("komi -0.5")
        eng.ok("clear_board")
        eng.ok("kata-set-rules quoridor")
        eng.ok("loadsgf " + path)
        assert eng.ok("get_komi") == "1.5"
        assert _json(eng.ok("kata-get-rules")) == {
            "maxPlies": 100, "timeBonusPerPly": 0.05, "blackInitialWalls": 10, "whiteInitialWalls": 6}
        assert eng.ok("winner") == "W"
        # Up to move 10: plies count from the start of the SGF.
        eng.ok("loadsgf %s 11" % path)
        assert "Ply: 10 (draw at 100, 90 left)" in eng.ok("showboard")
        assert eng.ok("walls") == "B: 10 W: 6"


def test_load_old_sgf(eng):
    # KataQuoridor 0.1.0 wrote KM[0] and no WB/WW for the standard game; without KM the komi is standard too.
    old = "(;FF[4]GM[1]SZ[17]PB[a]PW[b]HA[0]KM[0]RU[Quoridor]RE[B+R]AB[iq]AW[ia];B[io];W[ic])"
    no_km = "(;FF[4]GM[1]SZ[17]AB[iq]AW[ia];B[io];W[ic])"
    eng.ok("komi 2.5")
    for text in (old, no_km):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old.sgf")
            with open(path, "w") as f:
                f.write(text)
            eng.ok("loadsgf " + path)
            assert eng.ok("get_komi") == "-0.5"
            assert eng.ok("walls") == "B: 10 W: 10"
            assert "Ply: 2 (draw at 300, 298 left)" in eng.ok("showboard")
