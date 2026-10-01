# Quoridor I/O v3: repetition inputs, and continuing a v2 run

*Design document, 2026-10-02. I/O v3 = [I/O v2](QuoridorIOv2.md) + inputs that make the optional repetition draw
rule ([QuoridorIOv2.md §2.10, §10](QuoridorIOv2.md#210-draw-by-repetition-optional)) Markov for the net. Heads,
targets and training-row layout (apart from the input channel counts) are unchanged from v2.*

## 1. Why

With `repetitionDrawCount = N > 0` the game is drawn at the N-th occurrence of a position. Whether a move draws
depends on the path, not on the board: a v2 net can't see it, so its value of a shuffling position is the average of
"this repeats" and "this doesn't", and search corrects that only at terminal nodes. v3 adds what the rule reads, so
the net's evaluation is a function of its input again.

## 2. Inputs

v3 = the 19 spatial / 17 global v2 features, then these, appended (`QuoridorNN` in `cpp/neuralnet/quoridornn.h` is
the one module that knows the layout):

| Index | Kind | Feature | Symmetry (x-mirror) |
|---:|---|---|---|
| spatial **19** | pawn-cell plane, binary | 1 on each legal pawn destination of the side to move whose resulting position already occurred since the last wall placement (the move **repeats** a position: occurrence ≥ 2) | c → 8 − c, like the pawns |
| spatial **20** | pawn-cell plane, binary | 1 on each legal pawn destination whose move **draws by repetition** (occurrence ≥ N) | c → 8 − c |
| global **17** | 0 / 1 | the repetition rule is on (`rules.repetitionDrawCount > 0`) | |
| global **18** | float | **repetition progress** of the current position, `(count − 1) / (N − 2)` (1.0 for N = 2): 0 the first time, 1.0 when one more occurrence of this position draws | |

- All four are 0 when the rule is off, so a v3 net in a standard game (GTP, arena vs other engines) gets exactly the
  v2 inputs plus zeros.
- Rows are canonical (the side to move heads for row 0) like every other plane; the mirror symmetry maps both planes
  like the pawn planes (`QuoridorNN::applyInputSymmetry`, Python `apply_symmetry_quoridor`: a plain flip).
- Wall placements never repeat a position (walls are never removed), so only pawn moves are marked.
- **Encoding choice.** The brief suggested one spatial plane scaled like global 18 (1.0 = the move draws). Training
  rows store spatial planes as **bits** (`TrainingWriteBuffers::fillQuoridorInputRow` asserts binary values; only the
  distance planes 8–11 are stored raw), so a 0.5 in such a plane would be truncated to 0 in training while inference
  saw 0.5, the same class of bug as the old S8–S11 truncation. Two binary planes ("repeats", "draws") carry the
  same information for N = 3 (occurrence 2 vs 3) and the two facts that matter for any N, without a new raw array in
  the training data. The globals are stored as floats, so global 18 keeps the scaled form.
- How they're computed: `QuoridorNN::repeatingPawnMoves` hashes each legal pawn destination's resulting position
  incrementally (`Board::getSitHashAfterPawnMove`, no board copy) and counts it in
  `BoardHistory::positionsSinceLastWall` (`numOccurrencesSinceLastWall`). With N = 3 that is ≤ 5 destinations × the
  plies since the last wall; skipped entirely when the rule is off or no pawn move has happened since the last wall.

## 3. NN cache hash

`NNInputs::getHash` covers board, side to move, rules (N included) and the ply, not the path. For **v3 nets**
`NNEvaluator` folds in `QuoridorNN::repetitionInputsHash`: the current position's count and every repeating pawn
move with its count, i.e. exactly what inputs 18–20 read (17 and N come from the rules). Equal positions on the same
ply with different repetition state get different entries; equal repetition state shares an entry. It is
`Hash128()` when the rule is off, and v1/v2 nets don't add it, so their caching is unchanged. Tested in
`runtests quoridorv3` (hash level and through `NNEvaluator` cache hits).

## 4. Training data, heads, presets

- `QuoridorNN::TRAINING_IO_VERSION = 3`, `MAX_SUPPORTED_IO_VERSION = 3`; Python `QUORIDOR_TRAINING_IO_VERSION = 3`,
  channel counts `{1: 17/15, 2: 19/17, 3: 21/19}`. Self-play writes v3 rows **whatever the net's I/O version** (the
  writer computes the inputs from the history), so a v2 net in `models/` produces v3 data.
- Targets and heads: unchanged (v2). The loss asserts a v3 model; the loader rejects rows with other channel counts.
- Presets `b2c64_quoridor_v3`, `tf2_b4c192_quoridor_v3` (`quoridor_io_version: 3`). `*_v2` stay for inference,
  export and as the upgrade source.
- `python/model_viewer` remains I/O v1 only (it refuses v2 and v3 nets).

## 5. Tools

| Tool | v3 |
|---|---|
| `dumpnninputs` (`-io-version 1..3`) | Every other game from game 2 has the rule (N = 2..4) and shuffling pawn-only players; writes `repInfo` (N, count) and `repMoves` (earlier occurrences per pawn destination) for the Python re-derivation |
| `evalnnparity` | unchanged (picks the inputs by the net's version) |
| `writesampletrainquoridor` | 25% of games have the rule and step back and forth (repetition draws); prints how many |
| `quoridor_upgrade_v2_to_v3.py` | v2 training checkpoint → v3 (§6) |
| `quoridor_convert_tdata_v2_to_v3.py` | v2 training rows → v3, zero repetition inputs, only for data played with the rule off (`-rule-was-off`) |

## 6. Continuing a v2 run as v3

`python/quoridor_upgrade_v2_to_v3.py` zero-pads the input columns of `conv_spatial.weight` (19 → 21) and
`linear_global.weight` (17 → 19) in the model, the SWA model and the optimizer state (every state tensor of those
two shapes, e.g. SGD momentum, Adam moments), sets `config["quoridor_io_version"] = 3` and records
`train_state["quoridor_upgraded_v2_to_v3"] = global_step_samples`. The new inputs are multiplied by 0, so the net is
the v2 net until training moves the new columns. train.py takes the model config from the checkpoint (`MODELKIND`
only matters for a fresh run), and the padded optimizer state loads with `optimizer.load_state_dict` as usual.

**Equality on run3** (copy of `~/q1_run/run3/train/run3/checkpoint.ckpt`, s7865856, `tf2_b4c192`, 400
`dumpnninputs -io-version 3` positions of which 75 have non-zero repetition inputs):

| Check | max \|v3 − v2\| |
|---|---|
| PyTorch, model, all postprocessed outputs (v2 fed the first 19 / 17 channels) | **0.0** |
| PyTorch, SWA model | **0.0** |
| exported `.bin.gz` (`-use-swa`) through `evalnnparity`, Eigen FP32, symmetry 0: policy / value / misc | **0.0 / 0.0 / 0.0** |
| same, symmetry 1 | **0.0 / 0.0 / 0.0** |

(`tests/test_quoridor_upgrade_v3.py` checks the same on small random nets with a 1e-5 tolerance: CPU convolutions
over 19 and 21 input channels can sum in a different order. It also loads the padded SGD state into a v3 model's
optimizer and steps.)

### 6.1 Switch-over procedure for run3 (`synchronous_loop.sh`)

`BASEDIR=~/q1_run/run3`, `TRAININGNAME=run3`. Do it between cycles (after an export, before the next gatekeeper).

1. **Stop** the loop after a cycle's export. `torchmodels_toexport/` should be empty (or let those v2 exports
   finish first; v2 nets still load everywhere).
2. **Binary and code.** Build `cpp/katago` from a checkout containing I/O v3 (the loop uses
   `$GITROOTDIR/cpp/katago` unless `KATAGO_BIN` is set, and copies `python/` into `scripts/dated/<date>` at start, so
   the new Python code is picked up on restart).
3. **Back up** `train/run3/` (e.g. `cp -a train/run3 train/run3.v2`).
4. **Upgrade the checkpoint:**
   `python python/quoridor_upgrade_v2_to_v3.py -checkpoint $BASEDIR/train/run3/checkpoint.ckpt -in-place`
   (keeps `checkpoint.ckpt.pre_v3`). Move `checkpoint_prev*.ckpt` into the backup: they are v2, and train.py falls
   back to them only if `checkpoint.ckpt` is missing; the next save rotates fresh v3 copies in. Leave
   `longterm_checkpoints/` (v2 history).
5. **Training data in the shuffle window.** v2 rows (19 / 17 channels) can't be mixed with v3 rows: the shuffler
   concatenates arrays and the loader asserts the counts. **Recommendation: convert.** Every run3 row so far was
   played by a binary without the repetition rule, so its v3 repetition inputs are exactly 0, and zero-padding is not
   an approximation:
   `python python/quoridor_convert_tdata_v2_to_v3.py -rule-was-off $BASEDIR/selfplay` (rewrites each v2 `.npz`
   atomically; v3 files are skipped; `-dry-run` first to see the counts). The total row count is unchanged, so the
   train bucket and the window carry on (a drop-and-refill would throw away a ~1.2M-row window and make train.py
   reset its train bucket, "Data was deleted or this network was transplanted"). Old `shuffleddata/` directories are
   regenerated each cycle and can be deleted.
6. **Restart** with the v3 model kind: `./synchronous_loop.sh <NAMEPREFIX> $BASEDIR run3 tf2_b4c192_quoridor_v3
   <USEGATING>`. The default configs are now `selfplay_quoridor_v2.cfg` / `gatekeeper_quoridor_v2.cfg` with
   `repetitionDrawCount = 3`. The first cycle self-plays with the newest model in `models/`, a v2 net (it writes v3
   rows; for a v2 net the rule is still enforced by search, it just doesn't see the counts), then trains the upgraded
   checkpoint and exports the first v3 net. With gating, the gatekeeper plays the v3 candidate against the v2
   incumbent as usual.

Optional: to have the first cycle's self-play already use a v3 net (same outputs, plus the v3 NN-cache keys), export
the upgraded checkpoint into `models/` under a new name before restarting. It isn't needed.

**Smoke test of this procedure** (scratch copy: run3's latest model, one model's worth of v2 data, 172 files
converted, upgraded checkpoint; two `synchronous_loop.sh` cycles with small sizes, `KATAGO_BIN` = the v3 build):
self-play with the v2 net wrote v3 rows, shuffle mixed converted and new rows, train.py resumed the upgraded
checkpoint and trained 38,784 samples with no jump in the losses (run3's last v2 metrics: `p0loss` 2.123, `vloss`
0.7527; first v3 print: 2.113, 0.7529; after 38k samples: 2.053, 0.754), and
`export_model_for_selfplay.sh` exported `run3-s7904640-d186886`, a v3 net. That net then played GTP (with the rule on, in
the shuffle position one move before the third occurrence it chose a wall instead of the drawing pawn move) and
300 self-play games (5222 rows, all 21 / 19 channels, rule on; 110 rows with a repeating move, 1 with a drawing
move; 1 repetition draw). The smoke dir held only one model's data, so train.py first reset its bucket (fewer rows
than the checkpoint had seen); I set `train_bucket_level_at_row` to the smoke dir's row count to stand in for the
full directory. With the full `selfplay/` directory converted in place this does not happen.

## 7. The random phase of a from-scratch run

Measured with the v2 self-play config and a random net (400 games each):

| Rule | Draws | Avg plies | Rows |
|---|---|---|---|
| `repetitionDrawCount = 0` | 14.7% (all maxPlies) | 162 | 17,498 |
| `repetitionDrawCount = 3` | 86% (all repetition) | 53 | 5,486 |

A random policy's pawn walks eventually reach a goal within 300 plies, so without the rule most random games are
decisive, and those wins are the first value signal. With the rule, random walks repeat positions long before that,
and 86% of the games become draws (value 0.5 / 0.5, no lead target). The value head would mostly learn "draw", and
the decisive-game signal per GPU-hour drops (fewer, shorter games, mostly draws).

**Recommendation:** start a from-scratch run with the rule **off** (`repetitionDrawCount = 0` in the self-play
config) and turn it on (3) once the nets play purposeful races, e.g. when the decisive rate with the rule on would be
above ~90% (try a few hundred self-play games with `-override-config repetitionDrawCount=3`) or after the first few
accepted models. v3 nets see the rule's state (global 17), so rows from both phases train one net consistently, and
the gatekeeper should use the same setting as self-play at each phase. Not implemented as a config schedule: switching
the key between cycles is enough. A per-game probability (e.g. 50% of games with the rule during a transition) would
be a small `GameInitializer` addition if wanted later. run3 is far past this phase: with its nets the rule draws
0.3–0.6% of games.

## 8. Where things live

| What | Where |
|---|---|
| Layout constants, features, symmetry, `repeatingPawnMoves`, `repetitionProgress`, `repetitionInputsHash` | `cpp/neuralnet/quoridornn.{h,cpp}` |
| Child-position hash, occurrence lookup | `Board::getSitHashAfterPawnMove` (`cpp/game/board.{h,cpp}`), `BoardHistory::numOccurrencesSinceLastWall` |
| NN cache hash for v3 nets | `NNEvaluator::evaluate` (`cpp/neuralnet/nneval.cpp`) |
| Model loading | `cpp/neuralnet/desc.cpp` (option D 1..3, channel counts by version; unchanged code) |
| Tools | `cpp/command/nnparity.cpp` (`dumpnninputs`, `evalnnparity`), `writesampletrainquoridor` (`cpp/command/misc.cpp`) |
| C++ tests | `cpp/tests/testquoridoriov3.cpp` (`runtests quoridorv3`): brute-force recount over the whole game on 80 random games (3349 rows, walls, N = 0, 2..5) incl. 366 undo-by-replay checks; mirror; cache hash; training-row encoding of a repetition-draw game |
| Python | `modelconfigs.py` (presets, counts), `data_processing_pytorch.py` (count check, symmetry note), `metrics_pytorch.py` (v3-only training) |
| Upgrade / conversion | `python/quoridor_upgrade_v2_to_v3.py`, `python/quoridor_convert_tdata_v2_to_v3.py` |
| Python tests | `tests/test_nn_parity.py` (v1/v2/v3 features and parity, conv + transformer, symmetries 0 and 1), `tests/test_quoridor_model.py`, `tests/test_quoridor_upgrade_v3.py`, `tests/test_end_to_end_training.py` |
