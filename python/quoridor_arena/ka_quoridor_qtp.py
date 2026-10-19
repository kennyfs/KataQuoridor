"""QTP adapter for Ka_Quoridor (github.com/sugiyama2718/Quoridor), so the arena can play it against KataQuoridor.

Ka_Quoridor is an AlphaZero/KataGo-style Python engine (TensorFlow 1 graph, Python MCTS). It runs in its own
environment (Python 3.11 for its compiled State module, TensorFlow 2 with tf.compat.v1), so this script is started
with that environment's interpreter, e.g. in a roster entry:

    {"name": "ka-n500", "command": "/home/kenny/arena/ka_venv/bin/python",
     "args": ["{repo}/python/quoridor_arena/ka_quoridor_qtp.py", "--ka-dir", "/path/to/Ka_Quoridor",
              "--data-dir", "/path/to/Ka_Quoridor/from_google_drive/application_data", "--nodes", "500",
              "--seed", "{seed}"], "seed": 11}

Moves: Ka_Quoridor's internal (Glendenning) notation is KataQuoridor's QTP notation: the first player (Black)
starts on e9 and heads to row 1, walls are the anchor cell plus h / v (checked against KataQuoridor's legal moves
by ka_quoridor_check.py). Only the standard game (10 walls each, komi -0.5, no repetition
rule) is supported; Ka_Quoridor draws at ply 300 like the arena's default maxPlies.

Commands: protocol_version, name, version, known_command, list_commands, clear_board, komi, kata-set-rule
(standard values only), play, genmove, legal_moves (all legal moves, for tests), quit.
"""
import argparse
import os
import sys
import time

COMMANDS = ["protocol_version", "name", "version", "known_command", "list_commands", "clear_board", "komi",
            "kata-set-rule", "play", "genmove", "legal_moves", "quit"]
STANDARD_RULES = {"blackInitialWalls": "10", "whiteInitialWalls": "10", "repetitionDrawCount": "0"}


class KaEngine:
    def __init__(self, args):
        self.args = args
        self.data_dir = os.path.abspath(args.data_dir)
        sys.path.insert(0, os.path.abspath(args.ka_dir))
        # Ka_Quoridor loads ./State_util.so and application_data/config.json relative to the working directory.
        os.chdir(os.path.abspath(args.ka_dir))
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        global State_mod, Agent_mod, BasicAI_mod
        import State as State_mod
        import Agent as Agent_mod
        import BasicAI as BasicAI_mod
        self.AIs = None
        self.state = None
        self.use_prev_trees = [True, True]
        self.bestmoves = [None, None]
        self.games = 0

    def _prepare(self):
        from engine_util import prepare_AI
        param = os.path.join(self.data_dir, "parameter")
        seed = self.args.seed * 1000003 + self.games
        self.AIs = [prepare_AI(param, c, search_nodes=self.args.nodes, tau=self.args.tau, level=self.args.level,
                               seed=seed + c) for c in (0, 1)]

    def clear_board(self):
        if self.AIs is None and not self.args.no_net:
            self._prepare()
        for ai in self.AIs or []:
            ai.prev_action = None
        self.games += 1
        self.state = State_mod.State()
        State_mod.State_init(self.state)
        self.use_prev_trees = [False, False]
        self.bestmoves = [None, None]

    def color_to_move(self):
        return "b" if self.state.turn % 2 == 0 else "w"

    def legal_moves(self):
        """All legal moves (not the search's pruned action set): every action string that the state accepts."""
        out = set()
        for i in range(128 + 9):
            mv = Agent_mod.actionid2str(self.state, i)
            if mv in out or not ("a" <= mv[0] <= "i" and mv[1:2].isdigit() and 1 <= int(mv[1]) <= 9):
                continue
            s = BasicAI_mod.state_copy(self.state)
            if State_mod.accept_action_str(s, mv):
                out.add(mv)
        return sorted(out)

    def play(self, color, mv):
        if color != self.color_to_move():
            raise ValueError("not %s's turn" % color)
        c = self.state.turn % 2
        backup = BasicAI_mod.state_copy(self.state)
        if not State_mod.accept_action_str(self.state, mv):
            raise ValueError("illegal move " + mv)
        if self.AIs:
            self.AIs[1 - c].prev_action = Agent_mod.str2actionid(backup, mv)
        self.use_prev_trees[c] = self.bestmoves[c] == mv
        self.bestmoves[c] = None

    def genmove(self, color):
        if color != self.color_to_move():
            raise ValueError("not %s's turn" % color)
        c = self.state.turn % 2
        t0 = time.time()
        action_id, _, _, v_post, _ = self.AIs[c].act_and_get_pi(self.state, use_prev_tree=self.use_prev_trees[c])
        mv = Agent_mod.actionid2str(self.state, action_id)
        sys.stderr.write("genmove %s %s v_post %s %.2fs\n" % (color, mv, v_post, time.time() - t0))
        self.play(color, mv)
        self.bestmoves[c] = mv
        self.use_prev_trees[c] = True
        return mv


def parse_color(s):
    s = s.lower()
    if s in ("b", "black"):
        return "b"
    if s in ("w", "white"):
        return "w"
    raise ValueError("bad color " + s)


def serve(engine, qtp_out):
    for line in sys.stdin:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        toks = line.split()
        cid = ""
        if toks[0].isdigit():
            cid, toks = toks[0], toks[1:]
        cmd, a = toks[0], toks[1:]
        ok, out = True, ""
        try:
            if cmd == "protocol_version":
                out = "2"
            elif cmd == "name":
                out = "Ka_Quoridor"
            elif cmd == "version":
                out = "n%d" % engine.args.nodes
            elif cmd == "known_command":
                out = "true" if a and a[0] in COMMANDS else "false"
            elif cmd == "list_commands":
                out = "\n".join(COMMANDS)
            elif cmd == "clear_board":
                engine.clear_board()
            elif cmd == "komi":
                if float(a[0]) != -0.5:
                    raise ValueError("only komi -0.5 is supported")
            elif cmd == "kata-set-rule":
                if STANDARD_RULES.get(a[0]) != a[1]:
                    raise ValueError("only the standard rules are supported")
            elif cmd == "play":
                engine.play(parse_color(a[0]), a[1].lower())
            elif cmd == "genmove":
                out = engine.genmove(parse_color(a[0]))
            elif cmd == "legal_moves":
                out = " ".join(engine.legal_moves())
            elif cmd == "quit":
                qtp_out.write("=%s\n\n" % cid)
                qtp_out.flush()
                return
            else:
                ok, out = False, "unknown command"
        except Exception as e:  # report and keep serving
            ok, out = False, "%s: %s" % (type(e).__name__, e)
        qtp_out.write("%s%s %s\n\n" % ("=" if ok else "?", cid, out))
        qtp_out.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ka-dir", required=True)
    ap.add_argument("--data-dir", required=True, help="Ka_Quoridor's application_data (with parameter/)")
    ap.add_argument("--nodes", type=int, default=500, help="search nodes per move")
    ap.add_argument("--tau", type=float, default=0.32, help="move temperature (Ka_Quoridor's evaluation value)")
    ap.add_argument("--level", type=int, default=-1, help="-1: the newest epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-net", action="store_true", help="load no net (play / legal_moves only, for tests)")
    args = ap.parse_args()
    # Keep the QTP stream clean: Ka_Quoridor's C code, Python code and TensorFlow print to stdout, so QTP gets a
    # duplicate of fd 1 and fd 1 itself (and sys.stdout) go to stderr.
    qtp_out = os.fdopen(os.dup(1), "w")
    sys.stdout.flush()
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    serve(KaEngine(args), qtp_out)


if __name__ == "__main__":
    main()
