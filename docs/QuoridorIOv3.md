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
   <USEGATING>`. The default configs are now `selfplay_quoridor_v2.cfg` / `gatekeeper_quoridor_v2.cfg`: the
   gatekeeper always plays with `repetitionDrawCount = 3`; self-play draws the rule per game (§6.2). The first cycle self-plays with the newest model in `models/`, a v2 net (it writes v3
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

### 6.2 The rule per game in self-play (mix)

Nets must play well with the rule (self-play, KataQuoridor-only matches) and without it (the arena against other
engines and the standard GTP config, where only the 300-ply draw applies). v3 nets see whether it is on (global
17), so self-play trains on a mix instead of only N = 3:

| Config key (`GameInitializer`) | Default | `selfplay_quoridor_v2.cfg` |
|---|---|---|
| `quoridorRepetitionDrawProb` | not set: every game has the config's `repetitionDrawCount` | 0.75 (rule on in 75% of games) |
| `quoridorRepetitionDrawCounts` | 3 | 3, 4 |
| `quoridorRepetitionDrawCountWeights` | equal | 0.9, 0.1 |

Drawn per game from the empty board, independently of the komi and fence randomization. Fork games keep their
game's rule (they start from an `InitialPosition` with the game's history, or replay with its `startHist.rules`);
side positions and reanalysis use the game's histories. SGF-start positions keep the config's `repetitionDrawCount`.
Setting the key makes `mayCreateNonStandardGames` true, so the gatekeeper refuses it (its config has a fixed N = 3).
`runtests quoridorselfplay` (40,000 games, fixed seed): rule on 0.7497 (expected 0.75), of which N = 3 0.8974
(0.9), random komi among them 0.3014 (independent: 0.30), fence handicap 0.1014 (0.10); fork games keep N = 4 or
off. Self-play stats split by rule:
`by repetition rule: on 455 games draw rate 0.002 avg plies 73.5, off 145 games draw rate 0.000 avg plies 72.5`
(600 games, run3-s7865856; SGF `RU`: 412 games with `repetitionDrawCount=3`, 43 with 4, 145 without).

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

With the 75% mix (§6.2) and a random net (600 games): draws 63.8% overall; rule on: 80.8% draws, 56 plies; rule
off: 16.5% draws, 172 plies. The 25% rule-off games supply most of the decisive games (132 of 217) and, being long,
most of the rows.

**Recommendation:** don't turn the rule off entirely while the net is random; use a **lower on-probability early**,
e.g. `quoridorRepetitionDrawProb = 0.25` for the first few models, then the 0.75 of the v2 config once the nets play
purposeful races (decisive rate with the rule on above ~90%: check with a few hundred games and
`-override-config quoridorRepetitionDrawProb=1`). With 0.25 roughly 3/4 of games are rule-off random walks that
mostly end at a goal, which is where the early value signal comes from, while the net still sees rule-on rows from
the start, so raising the probability later is a shift in the game mix, not new inputs. Fully off would also work but
gains little over 0.25. Change the key between cycles; the gatekeeper keeps N = 3 (decisive games between real nets
are unaffected). run3 is far past this phase: with its nets the rule draws 0.2–0.6% of games.

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
| Upgrade / conversion | `python/quoridor_upgrade_v2_to_v3.py`, `python/quoridor_convert_tdata_v2_to_v3.py`, `python/quoridor_convert_tdata_lambda0.py` (§9) |
| Python tests | `tests/test_nn_parity.py` (v1/v2/v3 features and parity, conv + transformer, symmetries 0 and 1), `tests/test_quoridor_model.py`, `tests/test_quoridor_upgrade_v3.py`, `tests/test_end_to_end_training.py` |

## 9. λ = 0 (2026-10-02)

The time bonus is dropped: `timeBonusPerPly = 0` in `selfplay_quoridor_v2.cfg` and `gatekeeper_quoridor_v2.cfg`
(it was 0.05, the placeholder of [QuoridorIOv2.md §5.3](QuoridorIOv2.md#53-the-v2-configs)). Why: the repetition
draw (§6.2) now ends the cycling games early, and the 300-ply draw still makes a winner progress; with λ = 0 the
utility score `u` (`scoreMean`) is the lead `s`, comparable across nets, and the short-term score error is no longer
inflated by the bonus. run3 nets up to `run3-s21886976-d3911658` were trained with λ = 0.05. The gatekeeper also
plays 256 visits now (was 150), and self-play writes files of 10,000 rows (we train without validation; use 1000
with validation, the split is per file).

### 9.1 Columns derived from `u`, and the converter

`python/quoridor_convert_tdata_lambda0.py` rewrites λ > 0 rows (`-lambda-old`, default 0.05) in place, atomically,
keeping each file's compression and **mtime** (shuffle.py picks the window by mtime). `-dry-run` counts; files
already converted (C70 = 1) are skipped. It refuses (writes nothing for that file) anything but v3 inputs with 80
global target columns and C63 = 3, rows with a lead weight but no outcome weight (search lead estimates), non-zero
C20 without a lead weight, and decisive rows that don't satisfy `u − s = sign(s)·λ·k` with an integer `k` in
[0, maxPlies].

| Column (`globalTargetsNC`) | Content | Read by the loss | Converter |
|---|---|---|---|
| C20 | final `u`, weight C27 | utility score + its stdev | decisive rows (C29 > 0, exactly the main rows of a game with a winner; verified on run3's data) := C21 (`u = s`); draws keep 0 |
| C3 | final value targets' score; = C20 on main rows, the side search's `u` on side rows | no | main rows := new C20; side rows unchanged |
| C15 | short-term TD score: a mix of the following searches' `u` and the final `u` | short-term score error loss, short-term optimistic policy weight | **can't be recomputed** (a search's bonus depends on its expected game length): new weight column **C70** := 1 |
| C7, C11, C19 | the other TD score targets (C19 = this row's search `u`) | no | unchanged |
| C58 | the net's raw score (metadata) | no | unchanged |
| policy targets | searches with λ = 0.05 | yes | unchanged (off-policy, like any older data) |

**C70** is new: 1 minus the weight of the short-term score target C15 (the C++ writer writes 0, i.e. full weight;
it zero-filled C70 before, so all new data is unaffected). `metrics_pytorch.py` multiplies the short-term score
error loss by `1 − C70` and drops the score term of the short-term optimistic policy weight on such rows. Zeroing
C24 instead would also have dropped the TD value targets C4–C14, which don't depend on λ; leaving C15 to age out of
the window would have trained the short-term score error head (used by search, `useUncertainty`) on inflated
errors for ~1.5M rows.

Tests: `tests/test_quoridor_convert_lambda0.py` (synthetic round trip with decisive, draw and side rows, stored and
deflated files, mtime, idempotence, refusals; and on real run3 files that only C3, C20, C70 change).

**run3's data** (backup: `~/q1_run/run3/selfplay.lambda005.bak`): 1776 files, 3,911,658 rows before and after
(3,341,719 decisive, 150,606 draw rows, 419,333 side rows); all decisive rows had an integer `k`. Old
`shuffleddata/` deleted.

### 9.2 Restarting run3 with λ = 0

No rebuild is needed (λ is a config value; C70 was already zero-filled). The new configs and the C70 loss change
must be in the checkout the loop runs from (`synchronous_loop.sh` copies `python/` and the configs into
`scripts/dated/<date>` at start):

1. Merge the branch into the checkout the loop runs from.
2. Convert run3's `selfplay/` (done, §9.1) and delete `shuffleddata/`.
3. Restart: `./synchronous_loop.sh <NAMEPREFIX> ~/q1_run/run3 run3 tf2_b4c192_quoridor_v3 <USEGATING>` (with the
   env as before, e.g. `VALIDATE=0`). The row count is unchanged, so the train bucket carries on. The first cycle
   self-plays with `run3-s21886976-d3911658` (a λ = 0.05 net) under λ = 0 rules: its rows have exact λ = 0 targets
   (they come from the game), only its search used a score head with the bonus. With gating, the gatekeeper compares
   the first λ = 0 net with it at λ = 0.

### 9.3 Optimizer and capacity: Aurora, `tf3_b5c256_quoridor_v3`

- **Aurora.** A from-scratch b4c192 bootstrap on run3's newest 2.5M rows with Aurora (`-use-aurora -wd-floor-frac
  0.5`, no warmup, lr scale 8 → 4 at 3.6M → 2 at 5.1M samples) beat run3's SGD net trained on the same data by about
  +84 Elo (kq_ladder3) and became run3a's first net, `run3a-s6097408-d4605550` (metrics and logs:
  `~/q1_run/run3/train/run3a/bootstrap_from_scratch/`). A 300k-sample lr sweep found Aurora stable up to x8 (no
  warmup needed for stability, but its first ~100k samples are slower than SGD) and clearly better than SGD at x2-x8.
  run3a then trains with constant lr scale 2. Note the lr scale is relative to train.py's per-sample lr, which for
  SGD is KataGo's late-training value (KataGo's own schedules start at 8-12x).
- **Capacity.** With Aurora, run3a's Elo per data row fell well below run3's late SGD rate while val losses stayed
  flat, so the next run (run4) uses a bigger net, KataGo's mainline design (b11c768h12nbt3tflrs-fson-silu) scaled
  down: `tf3_b5c256_quoridor_v3`, 5 nested-bottleneck blocks of 3 transformer layers (15 layers), 256 trunk, `mid`
  128, FFN 384, 4 heads of 32, heads 48/48/96/96, fson normalization, SiLU; **3,672,817** parameters (2.8x b4c192).
