# KataQuoridor Four-at-a-Table (Q4) Neural Network I/O Specification (Q4 I/O v1)

Canonical specification of Q4 I/O v1 (Plan §6–§7). The code that implements it:

| What | Where |
|---|---|
| shared constants (counts, version numbers) | `cpp/q4/nn/q4nnconstants.h` |
| C++ inputs, symmetry, output decoding, cache key | `cpp/q4/nn/q4nn.{h,cpp}` (`Q4NN`) |
| independent Python feature reference | `python/q4/features.py` |
| heads | `Q4PolicyHead`, `Q4ValueHead` in `python/katago/train/model_pytorch.py` |
| export | `python/export_model_pytorch.py` |

---

## 1. Overview and model header

Q4 nets run on the full 11 × 11 board in true board space (no canonical rotation), from the point of view of the
seat to move, and are evaluated and trained under all 8 symmetries of the square.

- **Model option D** (the header slot Duel uses for `quoridorIOVersion`) is `100 + q4_io_version`, i.e. **101** for
  Q4 I/O v1. Values `>= 100` mean "four-player family". `isQ4IOVersion(v)` (`q4nnconstants.h`) is the one test the
  shared C++ code uses.
- The Duel loader rejects option D `> 3` (including every Q4 net) and the Q4 loader rejects option D `< 100`
  (Duel and Go nets), both with a message naming the expected and the found value.
- Python: `"game": "quoridor4"`, `"q4_io_version": 1`.
- Model version is 17 (KataGo architecture version), like Duel.

Tensor shapes (batch `N`):

| Tensor | Shape |
|---|---|
| spatial input | `[N, 27, 11, 11]` (`[N, 11, 11, 27]` when the backend wants NHWC) |
| global input | `[N, 28]` |
| policy | `[N, 2, 3, 11, 11]` |
| value | `[N, 5]` logits |
| misc ("score value") | `[N, 6]` |
| ownership slot (my trajectory) | `[N, 1, 11, 11]` logits |
| training only, not exported | all-seat trajectories `[N, 4, 11, 11]`, wall placements `[N, 8, 11, 11]` (§5.3) |

Cell `c = y * 11 + x` (`x` = column `a..k`, `y` = row `1..11`, y up); tensor index `[y][x]`. Wall anchor
`(ax, ay)` with `ax, ay` in `0..9` is stored at tensor position `[ay][ax]`; the 11-th row and column of every
anchor plane are zero padding.

## 2. Relative seats and value rotation

| Relative slot | Label | Absolute seat |
|:---:|:---:|:---:|
| 0 | `me` | `toMove` |
| 1 | `next` | `(toMove + 1) % 4` |
| 2 | `across` | `(toMove + 2) % 4` |
| 3 | `previous` | `(toMove + 3) % 4` |

Slots are fixed even if seats are eliminated; an eliminated seat has its alive flag 0, its pawn planes zero, its
distance planes all 1.0 and its walls 0.

The relative value probabilities `[v0, v1, v2, v3, vdraw]` (softmax of the 5 logits) are rotated to absolute seats:
`Vabs[s] = v[(s - toMove) mod 4]`, `Vabs[draw] = vdraw`.

## 3. Spatial input channels (27, 11 × 11)

"Binary" channels hold exactly 0.0 or 1.0. "Raw distance" channels are derived from integer distances `d`
(walls only: BFS steps ignoring pawns, `255` = unreachable) as

> `dist01(d) = 1.0` if `d == 255`, else `min(d, 64) / 64`.

| Ch | Type | Feature | Under symmetry S |
|---:|:---:|:---|:---|
| 0 | bin | on-board (all ones) | invariant |
| 1–4 | bin | pawn of me / next / across / previous (all zero if eliminated) | cell map |
| 5 | bin | the goal cell `f6` (index 60) | invariant (fixed point) |
| 6–9 | bin | blocked to the N / E / S / W (wall or board edge) | cell map; the four channels are permuted by `applyDirection` |
| 10 | raw | distance of the cell to the center (walls only) | cell map |
| 11–14 | raw | distance from the pawn of me / next / across / previous to the cell (all 1.0 if eliminated) | cell map |
| 15–18 | bin | cell on **some** shortest path to the center of me / next / across / previous: `distFromPawn[c] + distToCenter[c] == distToCenter[pawn]` (all zero if eliminated or the center is unreachable) | cell map |
| 19, 20 | bin | placed vertical / horizontal walls (anchor grid) | anchor map; channels swap when S exchanges the axes |
| 21 | bin | anchor domain (ones on `[0..9] × [0..9]`) | invariant |
| 22, 23 | bin | geometrically legal vertical / horizontal walls: free of overlap, crossing and the "no seat is walled in" rule, **ignoring the wall supply** | anchor map; swap as 19/20 |
| 24 | bin | legal pawn destinations of the seat to move (including all jumps) | cell map |
| 25 | bin | pawn destinations that repeat an earlier position (since the last wall or elimination); 0 everywhere if the repetition rule is off | cell map |
| 26 | bin | pawn destinations that end the game by repetition; 0 everywhere if the rule is off | cell map |

Symmetry conventions (`cpp/q4/q4symmetry.h`): the 8 symmetries are the identity, the rotations by 90°, 180° and
270°, and the 4 reflections (the group `D4` acting on the 11 × 11 cells). Symmetries 1, 3, 6 and 7 exchange the two
axes: they swap vertical and horizontal walls (channels 19 ↔ 20, 22 ↔ 23, policy planes 1 ↔ 2). Cells map by
`applyCell`; a wall anchor `(ax, ay)` maps through the wall's center (`applyAnchor` returns the new anchor and
orientation); direction channel `d` moves to `applyDirection(d, S)`.

"On some shortest path" (instead of one chosen path) makes channels 15–18 equivariant under all symmetries.

Seat identities do not change under a symmetry: the value rotated back to absolute seats, and the relative slots,
are the same. A reflection reverses the apparent turn order on the board; the net sees both chiralities in
training, because every row gets a random symmetry.

### 3.1 Storage in training rows (round 4)

Binary channels (0–9, 15–26, 22 channels) are bit-packed into `binaryInputNCHWPacked`. The five raw-distance
channels 10–14 are stored as `uint8` in `spatialDistNCHW`: `min(d, 254)`, and `255` for unreachable (always, also
for the eliminated seat's pawn plane). The loader decodes them with `dist01` above. The reason is the one Duel
documents in `docs/DistPlanesUpgrade.md`: bit-packing truncates fractional values.

## 4. Global inputs (28 floats)

Seats in slots 0–3 are relative (me, next, across, previous). `n` = number of alive seats.

| Index | Feature | Formula |
|---:|:---|:---|
| 0–3 | walls left | `wallsLeft / 7` (0 for an eliminated seat) |
| 4–7 | **has at least one wall left** | `1.0` if `wallsLeft > 0`, else `0.0` (0 for an eliminated seat) |
| 8–11 | alive flags | `1.0` / `0.0` |
| 12–15 | distance of the pawn to the center | `dist01(d)`; `0.0` if eliminated |
| 16–19 | arrival estimate in plies from now if nobody walls and pawns do not interact | `min(1.0, ((d − 1) · n + order + 1) / 64)` where `order` is the seat's position in the upcoming alive turn order (me = 0); `1.0` if unreachable; `0.0` if eliminated. Clamped to 1.0. |
| 20–23 | race leader one-hot | `1.0` for the alive, reachable seat with the smallest arrival estimate (the estimates of two seats never tie), else all zero |
| 24 | alive seats | `n / 4` |
| 25 | plies until the draw | `max(0, maxPlies − plies) / 400` (absolute scale, as Duel) |
| 26 | repetition rule on | `1.0` if `repetitionDrawCount >= 2` |
| 27 | repetition progress | with the rule on: `1.0` if `repetitionDrawCount == 2`, else `min(1, (count − 1) / (repetitionDrawCount − 2))`, where `count >= 1` is the number of occurrences of the current position since the last wall or elimination (including this one); `0.0` if the rule is off. As Duel I/O v3. |

Global features are invariant under symmetry.

## 5. Outputs, heads and training targets

### 5.1 Policy head (`Q4PolicyHead`), exported

Shape `[N, 2, 3, 11, 11]`, packed variant-major: channel `variant * 3 + plane` (also in the exported `conv2p`).

- Variant 0, the **search policy**: target = the root visit distribution of a full search (self-play rows only).
- Variant 1, the **style policy**: target = the action the side to move actually played. Unused until round 6 but
  in the format from the start. It takes the place of Duel's "optimistic policy" channel; the Q4 path returns both
  variants separately (no optimism blending).
- Planes: 0 = pawn destination cell, 1 = vertical wall anchor, 2 = horizontal wall anchor (positions in the
  `[ay][ax]` anchor grid of §1). There is no pass.

Action numbering (`Q4Board`): `0..120` pawn move to cell `a`; `121..220` vertical wall at anchor `a − 121`;
`221..320` horizontal wall at anchor `a − 221` (321 actions). The softmax runs over the legal actions only.

Under a symmetry `S` the net is evaluated on the transformed inputs; its raw output at plane/position of
`applyAction(a, S)` is the logit of game action `a`.

Loss: cross-entropy per variant (weights in round 4). Illegal slots are masked in the loss, not in the net.

### 5.2 Value head (`Q4ValueHead`), exported

- **Value**: 5 logits = winner as relative seat (me, next, across, previous) or draw. Cross-entropy.
- **Misc** (the 6 exported "score value" slots; no post-processing multipliers are applied to Q4 misc values):

| Slot | Meaning | Target |
|---:|---|---|
| 0 | plies from now to the end of the game / 100 | final ply count − current ply |
| 1–4 | relative seat's walls-only distance to the center **at the end** of the game / 32 (0 for the winner) | weight 0 for seats eliminated before the end |
| 5 | short-term value error | as Duel |

  Loss: Huber. The final-distance targets (slots 1–4) get a loss weight comparable to Duel's lead loss (Duel:
  coefficient 0.054, Huber δ = 3.0 on the quantity in moves; Q4 does the same on the distance in cells, i.e.
  on `32 ×` the slot value). Slot 0 and 5 get the weights of Duel's remaining-plies and short-term-error terms.
  Exact values are fixed in round 4.
- **Trajectory (ownership slot)**: 1 × 11 × 11 logits, BCE, small weight. Target: 1 for every cell the pawn of the
  seat to move occupies at this row's position and at every later position until the end of the game, else 0.

### 5.3 Training-only spatial heads (PyTorch only, **not exported**, like Duel's `conv_wall_graph`)

Both live in `Q4ValueHead` on the same `conv1` features as the trajectory and are returned by the forward pass
after the exported outputs. All planes are in relative seat order and follow the input symmetry rule of §3
(cell map / anchor map with the H ↔ V swap). They are filled in round 4; the definitions are fixed here.

| Output | Channels | Definition (row at ply `t`, mover = seat to move) | Loss |
|---|---|---|---|
| `paths` | 4 (me, next, across, previous) | channel `k`: 1 on every cell that the pawn of relative seat `k` occupies at the position of this row and after each later action, until the end of the game. Channel 0 equals the exported trajectory target. All zero, weight 0, if the seat is eliminated at this row. | BCE per cell, small weight |
| `walls` | 8 = 4 seats × 2 orientations; channel `2k` = vertical, `2k + 1` = horizontal walls of relative seat `k` | 1 at anchor `[ay][ax]` if relative seat `k` places a wall of that orientation there with an action at ply `t` or later (this row's own action included), until the end of the game. Weight 0 outside the 10 × 10 anchor domain and for a seat eliminated at this row. | BCE per anchor cell, small weight |

The exported "ownership" slot is the trajectory of **me** (1 plane): the exported layout does not depend on these two
heads.

## 6. Raw evaluation entry point and output block (C++)

`NNEvaluator::evaluateQ4Raw` is used only when the loaded model's option D is a Q4 version. The caller (`Q4NN`)
fills the spatial and global rows with the symmetry already applied, passing the symmetry and the 128-bit cache key;
batching, the cache and the server threads are the existing ones, and the server hands the backend symmetry 0. A Q4 row
result carries a `Q4RawNNOutput` (`NNOutput::q4Raw`): both policy variants as raw logits (`2 × 3 × 121`), the 5
value logits, the 6 misc values and the 121 trajectory logits.

The GPU backends may run a net with all activations scaled down by 1/8 (`ModelDesc::applyScale8ToReduceActivations`, which
raises `postProcessParams.outputScaleMultiplier` to 8). The evaluator's server thread multiplies every raw Q4 output
(policy, value, misc, trajectory) by that multiplier, and un-symmetrizes the raw output into symmetry-0 orientation
using `Q4RawSymmetry` before the result is cached. Therefore, cached results and returned results are always in
symmetry-0 orientation. Decoding in `Q4NN` always uses symmetry 0.

Tools (`katago q4tool ...`, `cpp/q4/command/q4nntool.cpp`): `dumpinputs` (inputs of sampled positions, all 8
symmetries), `evalnn -model` (raw and decoded outputs), `symavg -model` (invariance of the 8-symmetry average),
`nncache -model` (cache hits and misses), `nnbench` (input-fill time, NN evaluations per second).
`katago q4qtp -bot nnpolicy -model <file>` plays the argmax (or, with `-temp`, a sample) of the search-policy
probabilities over the legal actions; `q4-rawnn [symmetry]` evaluates under `symmetry` with `skipCache = true` and
prints the decoded outputs.

## 7. NN cache key

The key covers everything the inputs read:

> `key = board.hash` (pawns, walls, wall supply, alive flags, seat to move) `⊕ H(maxPlies, repetitionDrawCount)`
> `⊕ H(max(0, maxPlies − plies))` `⊕ H(repetition state)`.

- *Repetition state* (only if `repetitionDrawCount >= 2`): the occurrence count of the current position and, for
  every legal pawn destination that repeats an earlier position, `(destination, occurrences)`.
- The symmetry is **not** in the key because the evaluator un-symmetrizes the raw output before caching it, so
  any symmetry produces a symmetry-0 result that serves subsequent queries under any symmetry.
- `initialWalls` is not in the key: the inputs only read the current wall supply, which is in `board.hash`.
- From round 6 the style features join the key.

## 8. Training rows (Round 4)

One row per recorded turn, from the perspective of the seat to move. "Relative seat k" means seat
`(toMove + k) mod 4`.
- npz keys are the same as Duel's, so `shuffle.py`'s key check passes: `binaryInputNCHWPacked`, `globalInputNC`,
  `policyTargetsNCMove`, `globalTargetsNC`, `scoreDistrN`, `valueTargetsNCHW`, `spatialDistNCHW`.
- No `metadataInputNC` and no `qValueTargetsNCMove`.
- Every spatial tensor is in **symmetry-0 orientation**. The loader (round 5) applies a random symmetry to inputs and
  targets together.

| Key | dtype, shape | Content |
|---|---|---|
| `binaryInputNCHWPacked` | uint8 `[N, 27, 16]` | the 27 spatial input channels bit-packed per channel (121 bits zero-padded to 128, big-endian bit order as KataGo), with the 5 raw-distance channels 10–14 written as 0 |
| `spatialDistNCHW` | uint8 `[N, 5, 11, 11]` | raw distances of channels 10–14 (`Q4NN::RawDistances`, 255 = unreachable) |
| `globalInputNC` | float32 `[N, 28]` | the 28 global inputs |
| `policyTargetsNCMove` | int16 `[N, 3, 363]` | in the net's policy layout (3 planes × 11 × 11: pawn cells, vertical-wall anchors `[ay][ax]`, horizontal-wall anchors). **C0**: the search's policy target (visit counts after KataGo's play-selection pruning, `extractPolicyTarget`). **C1**: the action actually played this turn, one-hot with value 1 (style-policy target). **C2**: the next seat's search policy target (the next row's C0; KataGo's "policy target next turn"), uniform with weight 0 if unavailable |
| `globalTargetsNC` | float32 `[N, 64]` | the table below |
| `scoreDistrN` | int8 `[N, 1]` | zeros (kept only so the key exists) |
| `valueTargetsNCHW` | int8 `[N, 12, 11, 11]` | **C0–3**: cells visited by the pawn of relative seat 0..3 from this row to the end of the game (Q4IO §5.3 "paths"; C0 is also the exported trajectory target). **C4–11**: walls placed from this row on by relative seat k: channel `4 + 2k` vertical, `5 + 2k` horizontal, at `[ay][ax]`. All 0 when the outcome weight C27 is 0; per-seat channels 0 for seats eliminated at this row |

`globalTargetsNC` (all value vectors are in relative-seat order `[me, next, across, previous, draw]`):

| Column | Content |
|---|---|
| C0–4 | final result: one-hot of the winner's relative seat, or the draw (KataGo C0–3 with `nowFactor = 0`) |
| C5–9 | TD value target, `nowFactor = 1/(1 + 121·0.176)` |
| C10–14 | TD value target, `nowFactor = 1/(1 + 121·0.056)` |
| C15–19 | TD value target, `nowFactor = 1/(1 + 121·0.016)` (the "short-term" target) |
| C20–24 | this turn's search value (`nowFactor = 1`) |
| C25 | row weight (KataGo C25) |
| C26 | weight of the search policy target C0 (KataGo C26) |
| C27 | outcome weight (KataGo C27 / Duel C27): value weight on main rows of a finished game, 0 on side positions |
| C28 | weight of the next-seat policy target (KataGo C28) |
| C29 | weight of the style-policy target (1 on every written row) |
| C30, C31, C32 | policy surprise, policy entropy, search entropy (KataGo C30–32) |
| C33 | 1 − weight of the TD value targets (KataGo C24) |
| C34 | 1 − weight of the value targets (KataGo C35) |
| C35 | plies from this row to the end of the game (weighted by C27) |
| C36–39 | walls-only distance to the center of relative seat 0..3 at the end of the game, in cells (0 for the winner) |
| C40–43 | weight of C36–39: C27, or 0 if that seat was eliminated before the end |
| C44–49 | 128-bit game hash in six 22/22/20/22/22/20-bit chunks (KataGo C41–46) |
| C50 | ply of this row; C51: first ply of the game that is training data (KataGo C53); C52: game mode (0 normal, 2 fork, as KataGo C55) |
| C53 | visits of the search, before reduction (KataGo C60) |
| C54 | raw NN utility of the seat to move (Plan §8.3 formula) |
| C55 | number of alive seats at this row |
| C56 | `repetitionDrawCount` of the game (0 = off); C57: `maxPlies` |
| C58 | 1 if the game hit `maxMovesPerGame` without a result (KataGo C52) |
| C59 | 1 if the game finished and this is not a side position (KataGo C62) |
| C60 | Q4 data format version, 1 |
| C61–63 | 0 |

- The TD value targets are KataGo's `fillValueTDTargets` with the 5-vector in place of win/loss/noResult. Read
  each turn's absolute value vector and rotate it to the relative seats of **this** row.
- The per-turn value vectors are the root `valueAvg[5]` of each turn's search, and the final entry is the game
  result, exactly as `whiteValueTargetsByTurn` in KataGo.
- Rows are written only for turns whose target weight is nonzero after KataGo's surprise weighting and
  integerization (cheap searches have weight `cheapSearchTargetWeight` = 0 in the config).

