# KataQuoridor: Code Review and Recommended Direction

*Review of `main` at `63aa91c` against upstream KataGo `v1.18.2` (`fd0723f`), written 2026-09-28.*

This document answers the questions in the project brief: 17x17 search board vs. 9x9 network, model/input
versioning, whether to delete KataGo's Go input code, CUDA/Eigen vs. ONNX, and which of the three proposed directions
to take. It then lays out a concrete, staged plan sized for Claude Code sessions.

---

## 0. TL;DR

1. **Choose direction 3, done surgically.** Keep the game layer you already have: `Board`, `BoardHistory`, `Rules`,
   lazy BFS, QTP, the V1 features, training-data writing, and the Python heads/losses. **Revert only the
   neural-net integration layer** (`desc.cpp`, the backends, most of `nneval.cpp`, `modelversion.*`,
   `export_model_pytorch.py`) to `fd0723f`, then re-apply a small, principled set of changes. Don't restart from
   scratch (direction 2): that throws away KataGo's search improvements, which are the main reason to fork KataGo.
2. **Keep the 17x17 board for search. Keep the 9x9 network with 3 policy planes.** Both decisions are right. What
   is wrong is that the code conflates them. The conversion between "search space" (17x17 `Loc`, 290 policy slots)
   and "tensor space" (9x9, 3 planes) must live in **exactly one module**, not in `inputsVersion == 1 ? 9 : nnXLen`
   ternaries spread over a dozen files.
3. **Don't use `modelVersion = 1`, and don't use 100 either.** In KataGo, `modelVersion` means *network
   architecture/serialization format*, and every backend branches on it (`>= 15`, `>= 17`, …). Reusing `1` makes
   every one of those branches believe your model is a pre-2019 Go net. Instead:
   - Export every Quoridor model as **KataGo architecture version 17**, unchanged.
   - Add a separate **Quoridor I/O version** (start at 1) in one of the **spare header slots** that v17 reserved for
     exactly this purpose ("model option D" in `ModelDesc`).
4. **Yes, delete KataGo's Go featurization** (V3–V7 inputs; this is mostly done already) and the Go-only tests.
   **Don't touch** KataGo's architecture/version gating in the backends. That code is not Go-specific, and keeping
   it pristine is what lets you merge future upstream backend improvements.
5. **Make `.bin.gz` the canonical model format, and support CUDA (GPU) and Eigen (CPU) first.** With the model
   contract fixed, the required backend change is small: support *N policy planes* instead of one. ONNX becomes
   optional (the upstream ONNX backend can build its graph from `.bin.gz`). Disable OpenCL/TensorRT/Metal with a
   clear error instead of maintaining them.
6. **The backends are closer to working than they look.** I built the Eigen backend and exported random
   `b2c64_quoridor` and `tf3_b4c192_quoridor` models. In a `Release` build (asserts off), the Eigen output matches
   ONNX Runtime to every printed digit, at both symmetries. In the default `RelWithDebInfo` build (asserts on), it
   stops at `assert(output->nnXLen == nnXLen)`: 17 (search) vs. 9 (net). With that bypassed, it next stops in the
   Go anti-mirror code in `searchmirror.cpp`. That is the whole story of "blocked by assertions" (§2.3).
7. **Use the aux heads you already train.** Your net predicts a distance margin, a variance time, and so on, but
   inference throws them away and reports score = 0. Feeding the margin into KataGo's score utility is the natural
   fix for "winning side shuffles its pawn until move 300", which currently wastes self-play games (they get
   discarded).
8. **Version the release as KataQuoridor's own version** (e.g. `0.1.0`, or `1.0.0` once gatekeeping works), and
   print "based on KataGo 1.18.2" in `version` output. Don't publish it as "1.18.2".

---

## 1. What I did

- Read the full diff `fd0723f..HEAD` (67 files, +6.5k/−14.8k lines), focusing on `cpp/game`, `cpp/neuralnet`,
  `cpp/dataio`, `cpp/program`, and `python/katago/train`. Note that **`cpp/search/` is untouched**, which is good.
- Built the Eigen backend twice: `CMAKE_BUILD_TYPE=Release` (`-DNDEBUG`) and the project default `RelWithDebInfo`
  (asserts on).
- Exported random-initialized `b2c64_quoridor`, `tf3_b4c192_quoridor`, and an ad-hoc `b6c96`-shaped Quoridor conv
  net with `export_model_pytorch.py`. Each export writes both `.bin.gz` and `.onnx`.
- Hand-built the V1 input tensor for the initial position in Python. I ran the exported `.onnx` in ONNX Runtime
  and compared it to `kata-raw-nn` from the Eigen build at symmetries 0 and 1.
- Ran `genmove` loops (60 plies, 4 search threads) in the asserts-on build.
- Ran the Python tests: `test_quoridor_model.py` passes (5 tests). `test_end_to_end_training.py` fails only because
  it expects a binary at `build/katago`.

I have no GPU here, so the CUDA findings below come from reading the code. The CUDA code has the same
`output->nnXLen == nnXLen` assert (`cudaandrocmbackend.inc:5639`) and the same `modelVersion <= 1` branches, so I
expect identical behavior.

---

## 2. Assessment of the current code

### 2.1 What is good and worth keeping

| Area | Verdict |
|---|---|
| `Board` on 17x17 with pawn cells at even/even and fence arms/centers at odd coordinates | Good. It is the standard trick (KataGomo does the same), and it lets all of KataGo's `Loc`-indexed search code work unchanged. |
| Lazy BFS (`CompactPath` cached shortest paths, re-BFS only when a new wall crosses the cached path) | Good and thread-safe. The wall-conflict logic via shared arm cells matches `GameRules.md` (same orientation conflicts iff offset ≤ 1; crossing iff same center). |
| `fillRowV1` features (pawns, 4 blocked-edge planes, goal row, 4 BFS distance fields, 2 on-path masks, 2 wall-anchor planes, wall-domain mask; 15 globals including wall counts and jump-tempo parity) | A sensible, cheap, rule-derived feature set in the KataGo spirit. Keep it as Quoridor I/O v1. |
| Canonical perspective (flip rows for White) + x-mirror as the only symmetry | Correct for Duel. |
| Python `QuoridorValueHead` + losses (value, TD values, variance time, game margin, trajectory, final-wall targets; 6 policy targets × 3 planes) | Good. These are the right auxiliary targets for this game. |
| Self-play: 300-move cutoff discards data; match/gatekeeping scores it as a draw | Reasonable. See §6.1 for how to make it rarely trigger. |
| `cpp/search/` untouched | Very valuable. Keep it that way as much as possible. |

### 2.2 The root cause: three concepts are conflated

Almost every ugly `inputsVersion == 1` / `modelVersion <= 1` branch comes from one of these three conflations.

**(A) Architecture version vs. game I/O version.**
KataGo's `modelVersion` controls the *serialization and architecture* of the net: whether the header has
post-processing multipliers (≥13), a metadata encoder and trunk norm kind (≥15), spare option slots (≥17), whether the
policy head has a separate pass MLP (≥15), how many policy channels there are, and so on. `inputsVersion` is derived from
it. Setting `modelVersion = 1` to mean "Quoridor" forces hacks such as:

```cpp
// desc.cpp
if(modelVersion >= 11 || modelVersion <= 1) {   // "<= 1" means Quoridor here
```

It also means every header field added since v13 is silently skipped when a Quoridor `.bin` is parsed. Today this
works only because `tf3_b4c192_quoridor` uses `norm_kind: fixup`. An upstream-style transformer config with an
RMSNorm trunk tip cannot be expressed at version 1 at all: `trunkNormKind` is parsed only for version ≥ 15, and
the exporter itself auto-upgrades such configs to 17.

**(B) Tensor size (9x9) vs. search board size (17x17).**
`NNEvaluator::nnXLen` is 17 because search indexes policy by 17x17 `Loc`. The backend's `nnXLen` is 9 because the
net is 9x9. Several files paper over this with `int modelXLen = (inputsVersion == 1 ? NN_X_LEN : nnXLen)`. Every
backend has `assert(output->nnXLen == nnXLen)` in `getOutput` (Eigen, CUDA, ONNX, OpenCL, and TRT all have it),
and that assert is exactly what fires.

**(C) Generic grid symmetry vs. Quoridor symmetry.**
Mirroring a 9x9 grid maps column `c` to `8 - c`. Wall anchors live on the 8x8 sub-grid and must map `c` to `7 - c`.
The current code handles this correctly but in scattered places:
- `SymmetryHelpers::copyInputsWithSymmetry` was globally replaced by a Quoridor routine with hard-coded channel
  numbers (inside the backends).
- The backend policy symmetry is bypassed for `modelVersion <= 1`.
- `applyPolicyMap` in `nneval` un-mirrors the policy.
- The Python loader has a third copy of the same logic.

It is correct (I verified symmetry 1 numerically) but fragile. Three independent implementations must stay in
sync, and the input half runs inside the backends while the output half runs in `nneval`.

### 2.3 Concrete findings (with evidence)

1. **Eigen `Release` build: numerically correct.** For the initial position with the random tf3 model:

   | Quantity | ONNX Runtime | Eigen `kata-raw-nn` |
   |---|---|---|
   | P(win) for the side to move (Black) | 0.625453 | whiteLoss 0.625453 |
   | Pawn e8 / d9 / f9 | 0.008837 / 0.008414 / 0.008469 | same |
   | V-wall (0,0) / H-wall (0,0) | 0.008233 / 0.007414 | same |
   | Symmetry 1: V-wall (0,0) / H-wall (0,0) | 0.008396 / 0.006920 | same |

   So `.bin.gz` export → `desc.cpp` parse → Eigen forward pass → `applyPolicyMap` is already correct for this model.
   The ONNX exporter and the `.bin` exporter agree.

2. **Eigen `RelWithDebInfo` (the CMake default, asserts on)** aborts on the first evaluation:
   ```
   eigenbackend.cpp:2546: Assertion `output->nnXLen == nnXLen' failed.
   ```
   This is conflation (B).

3. **After bypassing that assert**, `genmove` aborts in Go-specific search code:
   ```
   board.h:61: Color getOpp(Color): Assertion `c == C_BLACK || c == C_WHITE' failed.
     at Search::updateMirroring (searchmirror.cpp:71)
   ```
   GTP enables `antiMirror` for `genmove` by default (`gtp.cpp:2069`). The Go mirror-go detector then runs on a
   Quoridor board. Go-only search heuristics need to be explicitly disabled for this game (§5, Phase 1).

4. **With both bypassed** (`antiMirror=false`), 60-ply games with 4 search threads run cleanly under asserts, for
   both the conv and transformer models. Once walls run out, the random nets shuffle pawns back and forth
   (`e1 e8 e2 e9 e1 e8 …`). That is expected for a random net, but it is a preview of the "winning side dithers
   until move 300" problem (§6.1).

5. **The exporter requires `onnx` even to write the `.bin`.** It runs the ONNX export first and raises if the `onnx`
   module is missing.

6. **`modelconfigs.is_quoridor()` returns true for *any* config with `version == 1`.** Key it on an explicit
   `"game": "quoridor"` only.

7. **`nneval.cpp` silently accepts `legalCount == 0`** by filling the policy with −1. I believe a position with no
   legal move is unreachable in Duel under the no-full-block rule. For the side to move to have no move, it and the
   adjacent opponent must be sealed into a two-cell region. That region contains at most one goal row, so the wall
   that sealed it would have been illegal for the other player. This should be a `testAssert` plus a fuzz test,
   not a quiet fallback.

8. **`whiteOwnerMap`** is sized 17x17 by `nneval` and never filled by the backends for Quoridor models. Nothing
   reads it in self-play, but analysis output would print uninitialized or zero data.

9. **Self-play configs are still Go configs** (`fancyKomiVarying`, handicap and komi options, `sekiForkHack`, etc.).
   Most are inert, but each one is a trap for future readers.

10. **CPU throughput, rough data point.** Release Eigen, 4 vCPUs, random nets, tiny batches (~2 rows/batch):

    | Net | Params | FLOPs/eval (fwd, incl. training-only heads) | NN rows/s |
    |---|---|---|---|
    | `b2c64_quoridor` (nbt) | 0.10 M | 15 M | ~3,200 |
    | `b6c96`-shaped conv | 0.99 M | 156 M | ~600 |
    | `tf3_b4c192_quoridor` | 1.39 M | 223 M | ~260 |

    The absolute numbers mean little on this machine. The ratio does mean something: **on Eigen, tf3-b4c192 has
    1.4x the FLOPs of b6c96 but runs 2.3x slower**. It is also not "about the size of b6c96"; it is closer to
    1.5x in compute. Benchmark on the actual Xeon 8352V before committing to CPU self-play with a transformer
    (§6.4).

---

## 3. Answers to the specific questions

### 3.1 Keep 17x17 in search?

**Yes.** Search, the NN cache, transpositions, `NNOutput::policyProbs`, the QTP move parser, SGF-ish I/O, and the
training policy targets all speak `Loc`. The 290-slot policy array is 39% waste, but that waste is a few hundred
bytes per node, which is negligible. Switching to a dense 209-action index would touch `cpp/search/`, which you
have kept pristine so far. The only rule: **nothing outside the game layer and the one NN-layout module should
know that 17 = 2·9 − 1.**

### 3.2 Keep the 9x9 network with 3 policy planes?

**Yes.** The alternative (feed the 17x17 grid to the net, as KataGomo's Quoridor branch does) would make the net
"just a Go net", with one policy plane plus pass and no backend changes. But it costs:

- **3.6x** more compute per conv/FFN layer (289 vs. 81 positions),
- **12.7x** larger attention score matrices (289² vs. 81²),
- half the effective receptive field per 3x3 conv, measured in pawn cells,
- and a board where 72% of positions are never a pawn cell.

For CPU self-play in particular, 9x9 is clearly better. The price is the plane mapping plus the `7 − c` wall
symmetry, and both are cheap if they live in one place (§4.3). The dead row 8 / column 8 of the two wall planes is
harmless: they are masked by legality in C++ and by `valid_action_mask` in training.

### 3.3 Which version numbers?

Split what is currently one number into independent ones:

| Concept | Where it lives | Value | Changes when… |
|---|---|---|---|
| **Architecture / serialization version** (`modelVersion`) | `.bin` header line 2 | **17** (KataGo's) | …you merge an upstream KataGo that adds v18. You then inherit their backend code for free. |
| **Quoridor I/O version** (features + head semantics + policy layout) | `.bin` header spare slot "model option D"; ONNX metadata key | **1** | …you change `fillRow`, the plane layout, or head semantics. |
| **Policy planes** | policy-head spare slot "option A" | **3** | Never, for Duel. |
| **Training-data format** | `TrainingWriteBuffers` shapes, keyed by I/O version | = I/O version | Together with the I/O version. |
| **Engine version** | `version` / `--version` | `KataQuoridor 0.1.0 (based on KataGo 1.18.2)` | Your releases. |

Why not 100? A value of 100 passes every `modelVersion >= N` architecture gate, so it looks like "newer than
anything". That works until upstream ships v18 features. Then `100 >= 18` claims you have them, and `== 16`-style
checks become meaningless. The spare slots are there precisely so that "option D" can be claimed. The comment in
`desc.cpp` even lists what else to update when you claim one: the exporter, the ONNX metadata key, and the docs.

A Go model loaded into KataQuoridor then fails with a clear message ("I/O version 0: this is a Go network"). A
Quoridor model loaded into upstream KataGo fails at "unknown/unsupported model option D". Both are the right
behavior.

### 3.4 Delete KataGo's Go input code?

- **Featurization (V3–V7), Go-only `NNInputs` helpers, Go rules tests:** yes, delete. Most of it is already gone.
  Also delete (don't stub) Go-only tests such as `testrules.cpp` and `testnninputs.cpp` scenarios, replacing them
  with Quoridor tests. Stubbed Go tests just give false confidence.
- **Architecture/version gating in `desc.cpp` and the backends:** do **not** delete. It is not Go code. It is how
  KataGo supports trunk and head variants, and it is where upstream improvements (FlashMMA, new block kinds) land.
  Every line you change there is a future merge conflict.
- **Go stubs in `Board`/`BoardHistory`/`Rules`** (`getNumLiberties`, `koRule`, `encorePhase`, …): keep for now,
  since `cpp/search/` still calls some of them. Remove them later in a dedicated cleanup phase (§5, Phase 4), one at
  a time, letting the compiler find the callers.

### 3.5 CUDA + Eigen, or ONNX?

**Canonical format: `.bin.gz`. First-class backends: CUDA (RTX 3070) and Eigen (Xeon).**

- KataGo's CUDA backend is far more tuned than ONNX Runtime's CUDA EP (fused kernels, FP16 paths, and in 1.18
  FlashMMA attention). That matters a lot for transformers.
- Eigen is the simplest CPU path and is already numerically correct (§2.3).
- The upstream ONNX backend in 1.18.2 can **build** its graph from a `ModelDesc` (`onnxmodelbuilder.cpp`
  `build()`), so once the `.bin.gz` contract is right, ONNX needs no separate Python exporter. Keep it as a
  *later* option if benchmarks show ONNX Runtime's CPU EP (oneDNN/MLAS with AVX-512 on the 8352V) beating Eigen
  by a wide margin. This is plausible for the transformer, and worth one afternoon of measurement.
- Remove the custom Quoridor ONNX exporter (`QuoridorOnnxExportWrapper`) or make it optional. It is a second model
  contract (no `InputMask`, no pass output, 2-channel value) that every change must keep in sync.
- OpenCL, TensorRT, Metal: make the model loader throw `"KataQuoridor supports CUDA, Eigen (and ONNX)"` in those
  builds instead of carrying half-patched code.

### 3.6 Which of the three directions?

**Direction 3, but "revert the NN integration layer, keep the game layer"**, which matches your instinct.

- **Direction 1** (keep patching in place) keeps the conflated design and makes every future upstream merge
  painful. You would be fixing symptoms (asserts) rather than the contract.
- **Direction 2** (rewrite from scratch, SimpleQuoridor-style) loses the things that make KataGo strong per unit of
  compute, and that you would not realistically re-implement well:
  - playout cap randomization,
  - forced playouts and policy target pruning,
  - uncertainty-weighted playouts,
  - graph search with transpositions,
  - subtree value bias,
  - dynamic score utility,
  - policy-surprise data weighting,
  - side positions,
  - the tuned backends,
  - and the whole shuffle/train/export/gatekeep loop.

  It only makes sense if the goal is learning rather than strength.
- **Direction 3** (as below) keeps ~80% of your work. That 80% (rules, BFS, features, heads, data writing, QTP) is
  exactly the game-specific part, which is the part upstream can never give you. It replaces the ~20% that is
  fighting upstream.

---

## 4. Target architecture

### 4.1 Layering

```
+--------------------------------------------------------------------------------+
| cpp/search/*                (upstream, untouched; Go-only heuristics disabled)  |
|   speaks Loc on the 17x17 board; policy index = NNPos::locToPos(loc,17,17,17)   |
+-------------------------------------+------------------------------------------+
                                      |
+-------------------------------------v------------------------------------------+
| cpp/game/*   Board(17x17), BoardHistory, Rules, lazy BFS      (Quoridor-owned) |
+-------------------------------------+------------------------------------------+
                                      |
+-------------------------------------v------------------------------------------+
| cpp/neuralnet/quoridornn.{h,cpp}    THE ONLY PLACE THAT KNOWS BOTH SPACES      |
|   constants: MODEL_LEN = 9, NUM_POLICY_PLANES = 3, NUM_SPATIAL/GLOBAL(ioVer)   |
|   fillRow(board, hist, pla, ioVersion, float* spatial, float* global)          |
|   applyInputSymmetry(spatial, symmetry)          // pawn c->8-c, walls c->7-c   |
|   mapPolicyToSearch(rawPlanes, pla, symmetry, float* policy290)                |
|   (later) mapTrajectoryToOwnership(...)                                        |
+-------------------------------------+------------------------------------------+
                                      |
+-------------------------------------v------------------------------------------+
| cpp/neuralnet/nneval.cpp  batching, cache, postprocess  (upstream + ~50 lines)  |
|   nnXLen/nnYLen = 17 (search space) ; modelXLen/modelYLen = 9 (tensor space)    |
|   symmetry chosen here, applied here via quoridornn; backends see identity      |
+-------------------------------------+------------------------------------------+
                                      |
+-------------------------------------v------------------------------------------+
| desc.cpp + backends (CUDA, Eigen)   generic KataGo v17 evaluators               |
|   only Quoridor-motivated change: policy head may have numPolicyPlanes > 1      |
+--------------------------------------------------------------------------------+
```

### 4.2 Model file contract (Quoridor I/O v1 on KataGo architecture v17)

**Header.** Upstream v17 layout, unchanged, except:
- `numInputChannels = 17`, `numInputGlobalChannels = 15`.
- `metaEncoderVersion = 0`, `preferPassAlive… = 0`, `preferExcludeTerritory… = 0`.
- **option D = Quoridor I/O version (1)**. Options E–H stay 0.
- The post-processing multipliers (v13+) are written as usual, with Quoridor meanings (below).

**Trunk.** Exactly upstream: any conv, nbt, or transformer block kind the exporter supports.

**Policy head.** Upstream v17 structure, with **policy option A = `numPolicyPlanes` = 3**.
- `p2Conv.outChannels = policyOutChannels × numPolicyPlanes = 2 × 3 = 6`.
- Channel layout is variant-major: `channel = variant * 3 + plane`, with variant ∈ {policy, optimistic} and
  plane ∈ {pawn, vertical wall, horizontal wall}. This matches the PyTorch head's `(target, plane)` layout, so the
  exporter picks channels `[0,1,2]` (target 0) and `[15,16,17]` (target 5, short-term optimistic), exactly as
  upstream picks channels 0 and 5.
- The pass MLP (`gpoolToPassMul`, `…Bias`, `…Mul2`) is written with zero weights and ignored. It costs nothing, and
  keeping it means no structural change to the parser or backends.

**Value head.** Upstream v17 structure. Map the PyTorch `QuoridorValueHead` outputs onto the v17 slots at export
time:

| v17 output | Channels | Quoridor meaning (export mapping) |
|---|---|---|
| value | 3 | [win, loss] from `linear_value`; **noResult = zero weights, bias −30**. The engine then never predicts the 300-move draw, which is correct since self-play never trains on it. |
| scoreValue[0] (`scoreMean`) | 1 | **game margin** (terminal distance margin, "moves ahead") |
| scoreValue[1] (`scoreStdev`, pre-softplus) | 1 | a new small head, or a constant bias. Train it (cheap), because dynamic score utility uses it. |
| scoreValue[2] (`lead`) | 1 | game margin (same row as `[0]`), or a separate head later |
| scoreValue[3] (`varTimeLeft`) | 1 | `linear_variance_time` |
| scoreValue[4] (`shorttermWinlossError`) | 1 | **add this head in Python**. Upstream trains it from TD targets; it drives uncertainty-weighted playouts. |
| scoreValue[5] (`shorttermScoreError`) | 1 | same, for the margin |
| ownership | 1×9×9 | zeros for now. Later: `trajectory[me] − trajectory[opp]` makes a nice "where will pawns go" overlay for analysis. |

The ownership output is 9x9. Search never requests ownership during self-play. For analysis, `quoridornn` can map
9x9 to the 17x17 pawn cells. Until that exists, make `nneval` refuse `includeOwnership` requests.

**Inputs.** Channel 0 is the all-ones on-board mask, so upstream's mask logic automatically yields "all
positions valid". Set `requireExactNNLen = true` (always 9x9), and CUDA skips masking work. Revert the
Python-side `mask = ones_like(...)` special case, because `input_spatial[:,0:1]` already equals it.

### 4.3 Symmetry, owned by `nneval` + `quoridornn`

- `NNEvaluator::serve` chooses the symmetry (0 or 1) as today. It then calls
  `QuoridorNN::applyInputSymmetry(rowSpatial, sym)` on the already-filled row and hands the backend **symmetry 0**.
- `NNEvaluator::evaluate` calls `QuoridorNN::mapPolicyToSearch(raw, pla, sym, policy)`. That is today's
  `applyPolicyMap`, moved.
- Restore `SymmetryHelpers::copyInputsWithSymmetry/copyOutputsWithSymmetry` to upstream's generic versions.
  Backends then do identity copies, and upstream's own symmetry unit tests keep passing.
- The Python loader keeps its mirror of this logic. It is the second, unavoidable copy, so pin it with a parity test
  (§5, Phase 0).

### 4.4 The only backend change: N policy planes

In each supported backend (CUDA, Eigen, and later the ONNX builder):
- Size the policy result buffer as `numPolicyChannels × numPolicyPlanes × H × W`.
- In `getOutput`, loop over planes when mixing policy/optimistic, and write plane `p` to
  `policyProbs[p*H*W + pos]` (identity symmetry, per §4.3). The pass logit is ignored.
- Replace `assert(output->nnXLen == nnXLen)` with
  `assert(numPolicyPlanes * nnXLen * nnYLen <= NNPos::MAX_NN_POLICY_SIZE)`.
  `NNOutput::policyProbs` is used as a raw staging area, and `nneval` remaps it into search space in place. Document
  that in a comment.

This should be roughly 30–60 lines per backend, all of them additive. Compare that with today's diff, which rewrites
value-head, ownership, and score paths in each backend.

---

## 5. Step-by-step plan

Each phase is sized for one or two focused Claude Code sessions. Each ends with an objective acceptance check.
**Do not start a phase until the previous phase's check passes.**

### Phase 0: Safety net (do this first; about 1 session)

1. **Branch:** `git switch -c nn-contract main`. Keep `main` runnable in Release builds as a fallback.
2. **Build matrix:** document and script two builds, `RelWithDebInfo` (asserts on) and `Release`, for Eigen.
   Always test with asserts **on**, since that build shows the real state of the code.
3. **Game-rule tests** (C++ `runtests`):
   - **Perft-style move counts** from a handful of fixed positions. Pin them against an independent
     implementation (SimpleQuoridor or a 50-line Python reference).
   - **Lazy-BFS fuzz test:** play random legal games; after every move, assert
     `isLegalWallPlacement` (lazy) == a naive full-BFS legality check, for all 128 walls.
   - **No-legal-move invariant:** assert ≥ 1 legal move in every non-terminal position of the fuzz games.
   - **Undo round-trip:** `playMoveRecorded` + `undo` restores the board exactly, including the hash.
4. **NN parity test**, the single most valuable test in this project:
   - Add a C++ command (e.g. `katago dumpnninputs -n 200 -seed …`) that plays random games and writes
     `(spatial, global, legal-move list)` rows to an `.npz`.
   - A Python test loads the `.npz` and runs the PyTorch model. It checks that the C++ features equal a Python
     reimplementation (or at least have the expected shapes and invariants), and saves reference outputs.
   - A second C++ command evaluates the same rows with `-model x.bin.gz` on any backend, at symmetries 0 and 1, and
     compares policy (after legal masking) and value against the reference with a tolerance (1e-4 FP32, 2e-2 FP16).
   - Run it for a conv net and a transformer net.

   **Acceptance:** fuzz and perft tests pass under asserts. The parity harness exists; it may still fail on
   today's code, which is fine.

### Phase 1: Fix the NN contract (the core; 2–3 sessions)

1. **Revert the NN integration files to upstream:**
   ```bash
   git checkout fd0723f -- \
     cpp/neuralnet/desc.cpp cpp/neuralnet/desc.h \
     cpp/neuralnet/cudaandrocmbackend.inc cpp/neuralnet/eigenbackend.cpp \
     cpp/neuralnet/onnxbackend.cpp cpp/neuralnet/onnxmodelbuilder.cpp \
     cpp/neuralnet/openclbackend.cpp cpp/neuralnet/trtbackend.cpp cpp/neuralnet/metalbackend.cpp \
     cpp/neuralnet/modelversion.cpp cpp/neuralnet/modelversion.h
   ```
   Keep `nninputs.cpp` and `nneval.cpp` from `main`, but plan to rewrite their Quoridor parts per §4.
2. **Create `cpp/neuralnet/quoridornn.{h,cpp}`.** Move `fillRowV1`, the Quoridor input symmetry, and
   `applyPolicyMap` into it, along with the constants (`MODEL_LEN`, `NUM_POLICY_PLANES`, feature counts per I/O
   version). Delete `NN_X_LEN/NN_Y_LEN/NN_POLICY_SIZE` from `nninputs.h`.
3. **Update `desc.cpp`:**
   - Claim option D as `ModelDesc::quoridorIOVersion`. Reject 0 ("Go network") and anything newer than supported.
   - Claim policy option A as `PolicyHeadDesc::numPolicyPlanes`. 0 means 1.
   - Replace `NNModelVersion::getInputsVersion(modelVersion)` uses with `desc.quoridorIOVersion`.
4. **Update `nneval.cpp`:**
   - Add explicit `modelXLen/modelYLen` (= `QuoridorNN::MODEL_LEN`) for `createComputeContext`,
     `createInputBuffers`, and `fillRowBufs`. `nnXLen/nnYLen` stay 17 for search.
   - Move symmetry handling in here (§4.3).
   - Delete every `inputsVersion == 1` / `modelVersion <= 1` branch. After Phase 1, `grep -rn "Version <= 1\|Version == 1" cpp/neuralnet` should return nothing.
   - Restore `testAssert(legalCount > 0)`.
5. **Backends:** add N-policy-plane support to **Eigen first**. Get the parity test green. Then do CUDA. Make
   OpenCL, TRT, and Metal throw a clear "unsupported" error at model load.
6. **Python exporter:**
   - Revert to upstream's `export_model_pytorch.py`.
   - Add one `if is_quoridor(config):` branch that writes `version = 17`, option D = 1, and policy option A = 3,
     and maps heads per §4.2.
   - Make ONNX export optional, and make `import onnx` lazy.
   - `is_quoridor()` checks `config.get("game") == "quoridor"` only. Quoridor configs get `"version": 17`, meaning
     *architecture*.
7. **Python model:** revert the mask override. Add the two short-term error outputs and a margin-stdev output to
   `QuoridorValueHead` with upstream-style losses; the targets already exist or are cheap to write. Keep
   `num_policy_outputs = 18`.
8. **Disable Go-only search heuristics for this game:** `antiMirror` (confirmed crash), `avoidMYTDaggerHack`,
   `conservativePass`, passing hacks, `fillDameBeforePass`, friendly-pass logic. Restrict `rootSymmetryPruning` to
   {identity, x-mirror}. Put this in one place, e.g. a `SearchParams::sanitizeForQuoridor()` called from `setup.cpp`,
   rather than editing each call site.

   **Acceptance:**
   - The parity test passes on Eigen and CUDA, FP32 and FP16, at symmetries 0 and 1, for a conv and a transformer
     net, under asserts.
   - `selfplay` → `shuffle` → `train` → `export` → `gatekeeper` completes two full generations on the 3070.

### Phase 2: Make the search use what the net knows (1–2 sessions)

1. Set post-processing multipliers so `whiteScoreMean`/`whiteLead` are "moves ahead" (the margin).
2. In self-play configs, turn on score utility again:
   - `staticScoreUtilityFactor ≈ 0.05–0.1`,
   - `dynamicScoreUtilityFactor ≈ 0.3`,
   - and use a smaller scale than Go's: `Board::sqrtBoardArea()` returns 17 today, which is probably too large for
     margins of roughly ±20. Make it 9, or make it a Quoridor constant, and tune it.
3. Turn on uncertainty-weighted playouts (`useUncertainty`) once the short-term error heads are trained.
4. Export the optimistic policy channels. Enable `policyOptimism`/`rootPolicyOptimism` as upstream does.
5. Rewrite `selfplay1.cfg`/`gatekeeper1.cfg` into Quoridor-only configs: delete the komi, handicap, rules-variation,
   and seki keys. Keep:
   - playout cap randomization,
   - forked/side positions (with Quoridor-sensible probabilities),
   - early-move temperature for opening diversity,
   - resignation settings.

   **Acceptance:**
   - The rate of self-play games hitting the 300-move cutoff drops below ~1%. **Log this rate per generation.**
   - A new net beats the Phase 1 net at equal visits in gatekeeping.

### Phase 3: Performance and hardware decision (1 session plus measurement)

1. **Add a Quoridor benchmark mode.** Upstream `benchmark` is Go-oriented. Measure NN rows/s and visits/s at several
   `numSearchThreads`/`nnMaxBatchSize` settings for b6c96-conv, b10c128-conv/nbt, and tf3-b4c192:
   - Eigen on the 8352V (try `numNNServerThreadsPerModel` from 1 to 4 × Eigen threads),
   - CUDA FP16 on the 3070,
   - optionally ONNX Runtime CPU on the 8352V.
2. **Decide by visits/s per machine** (§6.4).

### Phase 4: Cleanup (ongoing, low priority)

- Remove Go stubs from `Board`/`BoardHistory`/`Rules` one by one: delete the method, fix callers, run tests.
  `chain_head`/`next_in_chain` alone add ~1.4 KB to every `Board` copy.
- Delete dead Go commands (`evalsgf` Go paths, `startposes` Go logic, …) and dead configs.
- Replace `README.md` with KataQuoridor docs (build, QTP, training loop, model format). Keep upstream's in `docs/`.
- Add CI (GitHub Actions): Eigen `RelWithDebInfo` build + `runtests` + `pytest` + the parity test on a tiny net.
  It takes about 10 minutes per run and catches most regressions a coding agent can introduce.

### Phase 5: Release

- Tag `v0.1.0` once Phase 1 acceptance holds and a trained net clearly beats random and a simple
  shortest-path-greedy bot.
- Tag `v1.0.0` once gatekeeping has promoted a few generations and you have Elo anchors (§6.5).
- `version` output: `KataQuoridor 0.1.0 (KataGo 1.18.2 base, git <sha>)`.

---

## 6. Training, search, and hardware recommendations

### 6.1 The 300-move cutoff and "dithering"

With score utility at 0, a side with a clearly won position sees every non-blundering move at ~100% win rate.
MCTS then has no reason to prefer progress, so it can shuffle, especially after walls run out. Those games hit
300 plies and are discarded. That wastes the compute, and it also removes exactly the "convert the win" positions
from training. The fix is the margin-as-score mapping in Phase 2.

Also consider dropping discarded games more gently: keep the first N plies of a cutoff game as training data
with value weight 0 (policy targets are still good). KataGo's writer already supports per-row target weights.

### 6.2 Features (I/O v2 candidates, only after v1 is stable)

The V1 set is good. Candidates to A/B test later, each behind a new I/O version:

- **Legal-wall planes** (2×8x8, from the lazy BFS; cheap).
- **Wall-impact planes:** for each legal wall, how much it would increase the opponent's (and my) shortest path.
  This is expensive (up to 128 extra BFS runs per eval, most skipped by the lazy check), but it is the Quoridor
  analog of KataGo's ladder features: rule-derived information that a small net struggles to compute. Measure
  both the NN-eval speedup it buys and the featurization cost.
- **Ply count / moves until cutoff** as a global feature, **only** if you ever train on cutoff games.

### 6.3 Network size

- For the bootstrap (first ~20 generations), **prefer a small conv/nbt net**, something like b6c96 or b10c128nbt.
  The reasons:
  - Fewest moving parts while the new model contract is proven.
  - Best CPU/Eigen efficiency (§2.3 #10).
  - KataGo itself bootstrapped this way.
- Switch to tf3 once the loop is stable. Training data is architecture-independent, so you can train the
  transformer on the accumulated conv self-play data (as you planned) and let it take over gatekeeping.
- `tf3_b4c192` has 1.4x the FLOPs of b6c96, not the same. If the transformer is the goal on CPU, consider a
  narrower variant (c128, 3–4 blocks) and compare at equal wall-clock time, not equal visits.

### 6.4 Hardware plan

- **Training:** on the RTX 3070. Training a ~1M-parameter net on CPU is roughly an order of magnitude slower, and
  the trainer is idle most of the time anyway.
- **Self-play:** the 3070 with CUDA is likely worth more than several CPU boxes for small nets with big batches. But
  the school Xeons are free, and KataGo's loop is designed for many heterogeneous self-play clients writing to a
  shared directory. **Use both.** Run self-play on the Xeons (Eigen, or ONNX Runtime if it wins the Phase 3
  benchmark) and on the 3070, sync `selfplay/` data to the trainer (rsync or a shared filesystem), and sync models
  back. The only hard requirement is that every client runs the same I/O version, which the model header now
  enforces.
- **Gatekeeping:** on a Xeon, so it doesn't compete with training for the GPU.

### 6.5 Evaluation

- Keep a fixed ladder of opponents:
  - uniform random,
  - greedy shortest-path,
  - SimpleQuoridor's engine (if runnable),
  - and frozen snapshots of your own nets.
- Run a small fixed-visit round-robin every N generations and track Elo. Gatekeeping alone only says "better than
  the previous net".
- Watch these per-generation metrics:
  - 300-move cutoff rate,
  - average game length,
  - first-player win rate (it should settle to a stable value; large swings between generations indicate
    problems),
  - average walls used per game.

---

## 7. Other observations

- **S8–S11 fix and data upgrade.** Spatial channels 8–11 are continuous BFS distances (`d/32`, 1 if unreachable).
  Training data used to store them through `packBits`, which truncated them to binary, so nets trained on
  near-zero planes while inference fed the real values. Measured: 5.6 pp average win-rate shift, top move changed
  in 17% of positions. Fixed within Quoridor I/O v1, with no v2 and no parallel code path:
  - the writer adds `spatialDistNCHW` (raw uint8 distances, 255 = unreachable), and the bit planes 8–11 are 0;
  - the loader rebuilds ch8–11 before symmetry;
  - one BFS (`QuoridorNN::fillDistances`) serves both `fillRow` and the writer.

  Existing data is upgraded exactly with `python/quoridor_add_dist_planes.py`, and checkpoints are migrated with
  `python/quoridor_migrate_dist_planes.py`, which zeroes the first-layer weights for ch8–11. The operator runbook
  is in `docs/DistPlanesUpgrade.md`. `python/tests/test_dist_planes.py` guards train↔inference input parity.

- **KataGo upstream merges:** add `upstream` as a remote and record the exact upstream base in a `UPSTREAM.md`.
  With the design above, the files you touch in `cpp/neuralnet/` shrink to `desc.cpp` (two spare slots), the policy
  extraction in two backends, and `nneval.cpp`. A future `git merge upstream/stable` should then conflict in only a
  handful of hunks.
- **Coding-agent hygiene** (this matters with Claude Code too). Add a `CLAUDE.md` at the repo root that states the
  invariants:
  - "no `inputsVersion == N` branches outside `quoridornn`",
  - "`cpp/search/` changes need justification",
  - "every NN change must pass the parity test under `RelWithDebInfo`",
  - "don't edit generated/stub Go tests; delete them",
  - "`modelVersion` means architecture; Quoridor semantics use `quoridorIOVersion`".

  Agents follow written invariants far better than inferred ones. Several of today's hacks (e.g. `<= 1`
  conditions) are exactly what an agent produces when no invariant says otherwise.
- **`GameRules.md`** documents Race and Four-at-a-Table as well. The code is Duel-only (2 pawns, fixed goals). That
  is fine, but say so explicitly in the rules doc so that no one "helpfully" generalizes `Board`.
- **Resignation:** KataGo's resignation thresholds were tuned for Go's value calibration. Before enabling
  resignation in gatekeeping (or anywhere else), play a sample of would-be-resigned games to the end and measure
  how often the resigning side would actually have won.
- **Hash/cache:** `NNInputs::getHash` still folds in Go-only state (pass-ends-phase, friendly pass, pass-alive
  modes). It is harmless but wasted work. Simplify it once the stubs are gone.

---

## 8. Suggested Claude Code prompts (one per session)

1. *"Read `docs/KataQuoridor_Review_and_Roadmap.md` §5 Phase 0. Implement the Quoridor perft, lazy-BFS fuzz,
   no-legal-move, and undo round-trip tests in `cpp/tests/`, wired into `runtests`. Build with
   `CMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_BACKEND=EIGEN` and make them pass. Do not modify `cpp/search/`."*
2. *"Implement the NN parity harness from §5 Phase 0 step 4 (C++ dump command + pytest + C++ eval command). It is
   fine if it fails on the current model contract; report the diffs."*
3. *"Do §5 Phase 1 steps 1–4 (revert NN files to `fd0723f`, create `quoridornn`, claim option D and policy
   option A, rework `nneval`). Eigen only. Acceptance: parity test green on Eigen at symmetry 0 and 1 under
   asserts."*
4. *"Phase 1 steps 5–7: exporter and PyTorch head changes, then CUDA backend policy planes."*
5. *"Phase 1 step 8 + Phase 2."*

Each prompt should end with: *"Keep the diff against upstream `fd0723f` minimal in `cpp/neuralnet/` and
`cpp/search/`; explain every hunk you add there."*
