"""NN tools around the Q4 net: the NN cache (T18) and `q4qtp` with a model (`bot = nnpolicy`, `q4-rawnn`)."""

import json
import os
import subprocess
import sys

import pytest

from q4_testutil import find_katago, run, run_evalnn
from q4.make_random_model import export_random_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def eigen():
    path = find_katago("eigen")
    if path is None:
        pytest.skip("no Eigen katago build")
    return path


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return export_random_model("b1c32_q4", str(tmp_path_factory.mktemp("nntools")), "tools", seed=7, scale_heads=True)


@pytest.mark.parametrize("kind", ["eigen", "cuda"])
def test_nn_cache_hits_and_misses_t18(model, kind):
    katago = find_katago(kind)
    if katago is None:
        pytest.skip(f"no {kind} katago build")
    out = run([katago, "q4tool", "nncache", "-model", model, "-seed", "5"]).stdout
    rows = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    names = [r["scenario"] for r in rows]
    for r in rows:
        assert r["hit"] == r["expectedHit"], f"{r['scenario']}: hit={r['hit']}, expected {r['expectedHit']}"
    print(f"T18 ({kind}): {len(rows)} scenarios")
    # the scenarios of T18 are all there
    for needed in ["same position, same symmetry hits", "same position, another symmetry hits",
                   "same board, ply count 8 instead of 0, misses", "other maxPlies misses",
                   "other repetitionDrawCount misses", "repetition rule off misses",
                   "transposition: other order hits", "same board and ply count, other repetition state, misses"]:
        assert needed in names, f"missing scenario {needed!r}"


class Qtp:
    """A `katago q4qtp` session."""

    def __init__(self, katago, args):
        self.p = subprocess.Popen([katago, "q4qtp"] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, cwd=REPO_ROOT)

    def send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        lines = []
        while True:
            line = self.p.stdout.readline()
            assert line != "", f"engine exited during {cmd!r}: {self.p.stderr.read()[-500:]}"
            line = line.rstrip("\n")
            if line == "" and lines:
                break
            if line != "" or lines:
                lines.append(line)
        return lines[0].startswith("="), "\n".join([lines[0][2:]] + lines[1:]).strip()

    def close(self):
        try:
            self.send("quit")
        except Exception:
            pass
        self.p.terminate()
        self.p.wait()


def argmax_of_policy(katago, model_path, actions):
    """The legal action with the highest search-policy probability of the network, from `q4tool evalnn` (symmetry 0)."""
    events = [{"action": a} for a in actions]
    res = run_evalnn(katago, model_path, [{"events": events, "sym": 0}])[0]["decoded"]
    probs = res["searchPolicyProbs"]
    return max(res["legalActions"], key=lambda a: probs[a])


def test_qtp_nnpolicy_plays_the_argmax(eigen, model):
    from q4.reference import action_to_str, index_to_action
    qtp = Qtp(eigen, ["-bot", "nnpolicy", "-model", model])
    try:
        played = []
        for _ in range(8):
            ok, move = qtp.send("genmove")
            assert ok, move
            expected = action_to_str(index_to_action(argmax_of_policy(eigen, model, played)))
            assert move == expected, f"after {played}: genmove played {move}, the argmax of the policy is {expected}"
            played.append(move)
    finally:
        qtp.close()


def test_qtp_nnpolicy_samples_with_temperature(eigen, model):
    seen = set()
    for seed in range(6):
        qtp = Qtp(eigen, ["-bot", "nnpolicy", "-model", model, "-temp", "2.0", "-seed", str(seed)])
        try:
            ok, move = qtp.send("genmove")
            assert ok
            seen.add(move)
        finally:
            qtp.close()
    assert len(seen) >= 2, f"temperature 2 sampled only {seen}"
    # the same seed gives the same move
    moves = []
    for _ in range(2):
        qtp = Qtp(eigen, ["-bot", "nnpolicy", "-model", model, "-temp", "2.0", "-seed", "3"])
        try:
            moves.append(qtp.send("genmove")[1])
        finally:
            qtp.close()
    assert moves[0] == moves[1]


def test_qtp_rawnn(eigen, model):
    qtp = Qtp(eigen, ["-bot", "nnpolicy", "-model", model])
    try:
        for sym in range(8):
            ok, text = qtp.send(f"q4-rawnn {sym}")
            assert ok, text
            assert f"Symmetry {sym}, seat 1 (South) to move" in text
            probs = [float(l.split(":")[1].strip().rstrip("%")) for l in text.splitlines()
                     if l.strip().startswith(("seat ", "draw")) and ":" in l and "%" in l]
            assert len(probs) == 5 and abs(sum(probs) - 100.0) < 0.05, text
            assert "Top legal moves of the search policy (203 legal)" in text
        # a played move changes the report: seat 2 to move
        ok, mv = qtp.send("genmove 1")
        assert ok
        assert "seat 2 (West) to move" in qtp.send("q4-rawnn")[1]
    finally:
        qtp.close()


def test_qtp_nn_error_cases(eigen, model, tmp_path):
    # bot = nnpolicy without a model: refused at start
    res = subprocess.run([eigen, "q4qtp", "-bot", "nnpolicy"], input="quit\n", capture_output=True, text=True)
    assert res.returncode != 0 and "needs a model" in res.stderr
    # q4-rawnn without a model: a GTP error, the session goes on
    qtp = Qtp(eigen, ["-bot", "greedy"])
    try:
        ok, text = qtp.send("q4-rawnn")
        assert not ok and "No NN model loaded" in text
        assert qtp.send("genmove")[0]
    finally:
        qtp.close()
    # a Duel net: a clear error at genmove
    duel = str(tmp_path / "duel")
    run([sys.executable, os.path.join(REPO_ROOT, "python", "export_model_pytorch.py"),
         "-export-random-initialized-model", "b2c64_quoridor_v3", "-export-dir", duel, "-model-name", "duel",
         "-filename-prefix", "model"])
    qtp = Qtp(eigen, ["-bot", "nnpolicy", "-model", os.path.join(duel, "model.bin.gz")])
    try:
        ok, text = qtp.send("genmove")
        assert not ok and "not a Q4" in text
    finally:
        qtp.close()
