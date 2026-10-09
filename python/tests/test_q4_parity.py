"""T17: NN parity of the Q4 nets, PyTorch vs the C++ engine (Eigen FP32, CUDA FP32, CUDA FP16).

For b1c32_q4 (convnet), b2c64_q4 and tf2_b4c192_q4 (transformer), random weights (output heads scaled up so that
the outputs are far from uniform; saved as a checkpoint and exported from it, so that PyTorch and C++ run the same
weights), on 200 positions x 8 symmetries (1600 evaluations per net and backend):

1. raw outputs, layer by layer: for the transformed input rows of python/q4/features.py (symmetry applied), the 6
   policy channels (variant * 3 + plane), the 5 value logits, the 6 misc values and the trajectory logits of the
   C++ evaluation (`q4tool evalnn`, field "raw") equal PyTorch's;
2. decoded outputs: the search-policy probabilities over the legal actions, the value probabilities per absolute
   seat and the trajectory probabilities in game space, decoded by C++ (symmetry undone, seats rotated) equal the
   same decoding done in python on the PyTorch outputs;
3. the average over the 8 symmetries is invariant under symmetries of the position (`q4tool symavg`).

CUDA FP32 runs with NVIDIA_TF32_OVERRIDE=0: otherwise NHWC convolutions run in TF32 on Ampere GPUs (measured error
1.2e-3 on policy logits for b1c32_q4), which is not what the FP32 comparison is for.

Tolerances: FP32 1e-4 (absolute on probabilities, 1e-4 * (1 + |reference|) on logits and misc values), as the Duel
parity test (test_nn_parity.py). FP16 (CUDA with half-precision weights and activations): 5e-3 on probabilities, 2e-2
* (1 + |reference|) on logits; the worst measured FP16 errors (printed by the test) are 1e-3 and 4e-3.

Binaries: $Q4_EIGEN_BIN / $Q4_CUDA_BIN, else cpp/build-eigen (or build-eigen-release) and cpp/build-cuda; a backend
without a binary is skipped. $Q4_PARITY_OUTDIR keeps the exported models.

Trained nets (round 5): with $Q4_PARITY_CHECKPOINT = a training checkpoint (model.ckpt, its SWA weights if it has
them, as export_model_pytorch.py -use-swa exports) and $Q4_PARITY_MODEL = the model.bin.gz exported from it (e.g. by
the self-play loop), the same checks run on that net instead of the random ones.
"""

import json
import os
import subprocess
import tempfile

import numpy as np
import pytest

from q4_testutil import (dump_positions, export_model, find_katago, make_random_model, run_evalnn,
                         style_metadata_of_dump, torch_raw_outputs)
from q4.features import (decode_shortterm_value_error, decode_trajectory, extract_features,
                         map_policy_to_game, unapply_raw_policy, unapply_raw_trajectory)
from q4.reference import action_to_index, legal_moves

# The "-meta" nets are Q4 I/O v2 (S2, Round 7): the metadata input is the style features of the position.
CONFIGS = ["b1c32_q4", "b2c64_q4", "tf2_b4c192_q4", "b1c32_q4-meta", "b2c64_q4-meta", "tf2_b4c192_q4-meta"]
PARITY_CHECKPOINT = os.environ.get("Q4_PARITY_CHECKPOINT")
PARITY_MODEL = os.environ.get("Q4_PARITY_MODEL")
if PARITY_CHECKPOINT:
    assert PARITY_MODEL, "Q4_PARITY_CHECKPOINT needs Q4_PARITY_MODEL (the model.bin.gz exported from it)"
    CONFIGS = ["checkpoint"]
NUM_POSITIONS = 200
NUM_SYMS = 8
FP32_TOL = 1e-4
FP16_PROB_TOL = 5e-3
FP16_LOGIT_TOL = 2e-2

# CUDA "FP32" convolutions run in TF32 (10 mantissa bits, errors around 1e-3) on Ampere GPUs unless TF32 is switched
# off; the FP32 comparison is meant to test the layout and the weights, so it runs with NVIDIA_TF32_OVERRIDE=0.
NO_TF32 = {"NVIDIA_TF32_OVERRIDE": "0"}

# backend name -> (binary kind, config lines, tolerance on probabilities, tolerance on logits, environment)
BACKENDS = {
    "eigen-fp32": ("eigen", [], FP32_TOL, FP32_TOL, {}),
    "cuda-fp32": ("cuda", ["cudaUseFP16 = false", "cudaUseNHWC = true"], FP32_TOL, FP32_TOL, NO_TF32),
    "cuda-fp16": ("cuda", ["cudaUseFP16 = true", "cudaUseNHWC = true"], FP16_PROB_TOL, FP16_LOGIT_TOL, {}),
}


@pytest.fixture(scope="module")
def workdir():
    outdir = os.environ.get("Q4_PARITY_OUTDIR")
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        yield outdir
    else:
        with tempfile.TemporaryDirectory() as d:
            yield d


@pytest.fixture(scope="module")
def positions():
    katago = find_katago("eigen")
    if katago is None:
        pytest.skip("no Eigen katago build (cpp/build-eigen)")
    dumped = dump_positions(katago, NUM_POSITIONS, 77777)
    assert len(dumped) == NUM_POSITIONS
    return dumped


@pytest.fixture(scope="module", params=CONFIGS)
def net(request, workdir):
    name = request.param
    if name == "checkpoint":
        from katago.train.load_model import load_checkpoint, load_model
        has_swa = "swa_model" in load_checkpoint(PARITY_CHECKPOINT)
        model, swa_model, _ = load_model(PARITY_CHECKPOINT, use_swa=has_swa, device="cpu")
        model = swa_model if swa_model is not None else model
        model.eval()
        return model.config.get("name", os.path.basename(os.path.dirname(PARITY_CHECKPOINT))), model, PARITY_MODEL
    model = make_random_model(name, seed=12345, scale_heads=True)
    return name, model, export_model(model, name, workdir, name + "-parity")


@pytest.fixture(scope="module")
def reference(net, positions):
    """PyTorch raw outputs and their python decoding for every (position, symmetry)."""
    name, model, _ = net
    spatial, glob_, meta, rows = [], [], [], []
    has_meta = model.get_has_metadata_encoder()
    for i, (data, pos) in enumerate(positions):
        row_meta = style_metadata_of_dump(data) if has_meta else None
        for sym in range(NUM_SYMS):
            sp, gl = extract_features(pos, sym)
            spatial.append(sp)
            glob_.append(gl)
            meta.append(row_meta)
            rows.append((i, sym))
    if has_meta:
        # non-zero style inputs (the positions are random games with walls): the test would be trivial otherwise
        assert np.abs(np.stack(meta)[:, :76]).max() > 0.1 and (np.abs(np.stack(meta)[:, :76]) > 0).any(axis=1).mean() > 0.9
    policy, value, misc, traj = torch_raw_outputs(
        model, np.stack(spatial), np.stack(glob_), np.stack(meta) if has_meta else None)
    return dict(rows=rows, policy=policy, value=value, misc=misc, traj=traj)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def decode_reference(pos, sym, policy6, value, misc, traj):
    """Python decoding: search-policy probabilities over the legal actions (dict action -> p), value probabilities by
    absolute seat + draw, trajectory probabilities in game space, and shorttermWinlossError."""
    logits = map_policy_to_game(policy6, sym)[0]
    legal = sorted(action_to_index(m) for m in legal_moves(pos))
    probs = dict(zip(legal, softmax(logits[legal])))
    rel = softmax(value)
    absolute = np.array([rel[(s - pos.to_move) % 4] for s in range(4)] + [rel[4]])
    st_err = decode_shortterm_value_error(misc[5], multiplier=0.25)
    return probs, absolute, decode_trajectory(traj, sym), st_err


def check(name, ref, cpp, tol, worst, rel=False):
    ref, cpp = np.asarray(ref, dtype=np.float64), np.asarray(cpp, dtype=np.float64)
    diff = np.abs(ref - cpp) / (1.0 + np.abs(ref)) if rel else np.abs(ref - cpp)
    worst[name] = max(worst.get(name, 0.0), float(diff.max()))
    return diff.max() <= tol


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_nn_parity_t17(net, positions, reference, backend):
    name, model, model_path = net
    kind, config_lines, prob_tol, logit_tol, env = BACKENDS[backend]
    katago = find_katago(kind)
    if katago is None:
        pytest.skip(f"no {kind} katago build")
    queries = []
    for i, sym in reference["rows"]:
        data = positions[i][0]
        queries.append({"events": data["events"], "rules": data["rules"], "sym": sym})
    results = run_evalnn(katago, model_path, queries, config_lines, env)

    worst = {}
    failures = []
    top_disagree = 0
    for row, ((i, sym), res) in enumerate(zip(reference["rows"], results)):
        pos = positions[i][1]
        raw = res["raw"]
        ref_policy = unapply_raw_policy(reference["policy"][row], sym)
        ref_traj_raw = unapply_raw_trajectory(reference["traj"][row], sym)
        ok = True
        ok &= check("policy logits", ref_policy, np.array(raw["policy"]).reshape(6, 11, 11), logit_tol, worst, rel=True)
        ok &= check("value logits", reference["value"][row], raw["value"], logit_tol, worst, rel=True)
        ok &= check("misc", reference["misc"][row], raw["misc"], logit_tol, worst, rel=True)
        ok &= check("trajectory logits", ref_traj_raw, np.array(raw["trajectory"]).reshape(11, 11), logit_tol, worst, rel=True)

        ref_probs, ref_abs, ref_traj, ref_st_err = decode_reference(pos, 0, ref_policy, reference["value"][row], reference["misc"][row], ref_traj_raw)
        dec = res["decoded"]
        legal = dec["legalActions"]
        assert sorted(legal) == sorted(ref_probs), f"legal actions differ (position {i})"
        cpp_probs = np.array(dec["searchPolicyProbs"])
        ok &= check("policy probabilities", [ref_probs[a] for a in legal], [cpp_probs[a] for a in legal], prob_tol, worst)
        ok &= check("value probabilities (absolute seats)", ref_abs, dec["valueAbs"], prob_tol, worst)
        ok &= check("short-term value error", ref_st_err, dec["shorttermWinlossError"], prob_tol, worst)
        ok &= check("trajectory probabilities", ref_traj, np.array(dec["trajectory"]).reshape(11, 11), prob_tol, worst)
        top_disagree += max(ref_probs, key=ref_probs.get) != max(legal, key=lambda a: cpp_probs[a])
        if not ok:
            failures.append((i, sym))
    print(f"\nT17 {name} {backend}: {len(results)} evaluations, worst differences {worst}, "
          f"top move differs in {top_disagree}")
    assert not failures, f"{name} {backend}: {len(failures)} evaluations over tolerance, e.g. {failures[:5]}; worst {worst}"
    if prob_tol == FP32_TOL:
        assert top_disagree == 0
    else:
        assert top_disagree <= 0.05 * len(results)


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_symmetry_average_invariance_t17(net, positions, backend):
    name, model, model_path = net
    kind, config_lines, prob_tol, _, env = BACKENDS[backend]
    katago = find_katago(kind)
    if katago is None:
        pytest.skip(f"no {kind} katago build")
    lines = [json.dumps({"events": d["events"], "rules": d["rules"]}) for d, _ in positions[:40]]
    cmd = [katago, "q4tool", "symavg", "-model", model_path, "-stdin"]
    cfg_path = model_path + ".symavg.cfg"
    if config_lines:
        with open(cfg_path, "w") as f:
            f.write("\n".join(config_lines) + "\n")
        cmd += ["-config", cfg_path]
    proc = subprocess.run(cmd, input="\n".join(lines) + "\n", capture_output=True, text=True, env=dict(os.environ, **env))
    assert proc.returncode == 0, proc.stderr[-2000:]
    rows = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith("{")]
    assert len(rows) == len(lines)
    worst_policy = max(r["maxPolicyDiff"] for r in rows)
    worst_value = max(r["maxValueDiff"] for r in rows)
    worst_traj = max(r["maxTrajectoryDiff"] for r in rows)
    spread = max(r["singleSymPolicySpread"] for r in rows)
    print(f"\nsymmetry average {name} {backend}: worst policy {worst_policy:.2e}, value {worst_value:.2e}, "
          f"trajectory {worst_traj:.2e}; single-symmetry policy spread {spread:.2e}")
    assert spread > 10 * prob_tol, "the net is (nearly) symmetric already: the invariance check would be trivial"
    assert max(worst_policy, worst_value, worst_traj) <= prob_tol
