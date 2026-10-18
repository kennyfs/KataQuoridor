# Quoridor I/O v4: wall-effect inputs and the draw as its own outcome

*Design and A/B document, 2026-10-18. I/O v4 = [I/O v3](QuoridorIOv3.md) + two independent parts: (1) inputs that
say how much each wall placement would lengthen each player's shortest path, (2) the draw as the noResult value
outcome. The two parts can be used separately (the A/B tests each alone).*

## 1. Why

- run4's weak spots are wall moves with low prior that a deep search finds ([BlindSpots.md](BlindSpots.md)). The net sees
  walls and BFS distances; "how much would this wall cost the opponent" is a non-local computation it has to learn
  through 15 attention layers. v4 hands it over.
- Draws were trained as win 0.5 / loss 0.5 (noResult forced off), so the value head could not tell a certain draw from a
  coin flip. 16% of run4's current window rows belong to drawn games. As its own outcome, a draw gives the search and the
  GUI a draw probability and lets self-play reduce visits in settled draws (`reduceVisitsDrawThreshold`).
- Caution: KataCDC's extra features sped up only the first 1M samples there. Hence the A/B before any switch.

## 2. Inputs

`QuoridorNN` (`cpp/neuralnet/quoridornn.{h,cpp}`) is still the one module that knows the layout.

| Index | Kind | Feature | Symmetry |
|---:|---|---|---|
| spatial **21** | anchor plane | vertical wall: increase of the **side to move's** shortest distance if placed now | c → 7 − c |
| spatial **22** | anchor plane | horizontal wall, side to move | c → 7 − c |
| spatial **23** | anchor plane | vertical wall, the **opponent's** increase | c → 7 − c |
| spatial **24** | anchor plane | horizontal wall, opponent | c → 7 − c |
| global **19** | float | largest increase for the opponent over all legal placements, `min(·, 8) / 8` | |
| global **20** | float | number of legal placements costing the opponent ≥ 2 steps, `/ 16`, clipped to 1 | |

- Distances are the same as channels 8-13: walls only, pawns ignored. Planes sit on the 8×8 anchor grid inside the 9×9
  plane (row and column 8 are 0), canonical orientation (side to move heads for row 0), like channels 14-18.
- Value: legal placement → `min(Δ, 8) / 8`; illegal (overlap, crossing, or it would fully block a player) → **−1**
  (the sentinel; a legal placement with Δ = 0 reads 0). Legality is exactly `Board::isGeometricallyLegalWallPlacement`
  (tested). The globals: why these two: they are what a player would look at first ("is there a damaging wall for me to
  place", "how many are there"), a conv net can only get them by pooling through the trunk; both are cheap given the planes.
- **Stored raw** (packed bit planes can't hold them; the S8-S11 lesson): npz key `wallEffectNCHW`, `uint8 [N,4,9,9]`, the
  increase capped at 254, 255 = illegal, 0 outside the anchors. The packed planes 21-24 are written as 0, the globals are
  floats in `globalInputNC`.
- **Computation** (`fillCanonicalWallEffectU8`): per player, BFS fields from the pawn and from the goal row and the number
  of shortest paths through each edge. A wall closes two edges; if the two together carry fewer shortest paths than exist,
  Δ = 0 and the wall is legal for that player. Otherwise one bit-parallel early-exit BFS (an 81-bit board, 4 shifts per
  step) with the two edges closed gives the new distance, or "unreachable" = illegal. About 19 BFS per row on average.
- **Cost** (RelWithDebInfo, one core, 579 positions of random wall-heavy games): `fillCanonicalWallEffectU8` **8 µs**
  per row; `fillRow` v3 29 µs → v4 39 µs (+10 µs: the BFS plus writing 4 planes). The first version with per-cell BFS was
  19 µs. For scale, a b5c256 forward pass on the RTX 3070 is ~100 µs per row at batch 128 (the NN server also batches).
- NN cache hash: unchanged. The planes depend only on the board (pawns and walls), which `NNInputs::getHash` covers; the
  v3 repetition hash is folded in for I/O ≥ 3 nets as before.

## 3. The draw outcome

- Targets: a drawn game's final value target (C0-C2) is `win 0, loss 0, noResult 1` (`ValueTargets::setOutcomeOfWinner`).
  The TD value targets (C4-C19) are search values and carry the search's own noResult share. Score heads stay "u = 0, lead
  weight 0" for draws, and the loss now trains the u head on decisive games only (the net's score outputs are conditional
  on a result, `nneval.cpp` multiplies them by 1 − P(noResult)).
- Net: `num_value_outputs = 3` (win, loss, noResult) in the value head and each TD horizon. The exporter writes the real
  third logit for v4 nets (before: zero weights, bias −30).
- Search / eval, every place that assumed noResult = 0 for Quoridor:

| Place | Now |
|---|---|
| `nneval.cpp` (`noResultLogits -= 100000` for non-territory rules) | skipped for I/O v4 nets (`predictsNoResult`); the random test net and v1-v3 nets stay forced off |
| terminal draw (`search.cpp` `playoutDescend`, race-solver leaf) | `winLoss 0, noResult 1` instead of `winLoss = 2·drawEquivalentWinsForWhite − 1, noResult 0`; `drawEquivalentWinsForWhite` no longer enters a terminal draw (it still feeds the komi input) |
| utility of a draw | `noResultUtilityForWhite` (0 in every config) = the old win 0.5 / loss 0.5. Tested: a search where every move draws reports noResult 0.99, utility 0.0003; with `noResultUtilityForWhite = 0.4` the utility follows |
| LCB, `getHackedLCBForWinrate`, resign, `findEvenKomi` | all read `winLossValue` (win − loss), which a draw leaves at 0 as before; unchanged |
| self-play value surprise | KL of (win, loss, noResult) against the raw net; for a net that cannot predict a draw (noResult < 1e-9: v1-v3) the draw share is folded into win / loss halves, else every draw would be maximal surprise |
| SGF move comments `win loss noResult score` | win and loss **with a draw worth ½ each** (`win + noResult/2`), noResult = the draw probability. sgfs_viewer plots the first as White's win; `blindspot.parse_comment`, the arena and the play GUI follow this |
| `kata-analyze` / GUI / analysis JSON | winrate = (1 + win − loss)/2 as before (draw = ½); `rootInfo` now has `noResultValue` (the search's draw probability) and `rawNoResultProb`; the GUI shows win / draw / loss |
| gatekeeper resign (`resignThreshold = −0.99` on win − loss) | unchanged |
| `lead` estimate skip (`noResult < 0.3`) | now also skips drawish positions (`estimateLeadProb` is 0 in the Quoridor configs) |

- `reduceVisitsDrawThreshold` (KataCDC e50ca8a, `play.cpp`): off by default (1.0). When the search's noResult value has
  been above it for `reduceVisitsThresholdLookback` turns, visits and weight shrink like for a won game. Documented as a
  commented line in `selfplay_quoridor_v2.cfg`.

## 4. Plumbing

- `TRAINING_IO_VERSION = 4`, `MAX_SUPPORTED_IO_VERSION = 4`; Python `QUORIDOR_TRAINING_IO_VERSION = 4`, channels
  `{…, 4: 25 / 21}`; presets `tf2_b4c192_quoridor_v4`, `tf3_b5c256_quoridor_v4`, `b2c64_quoridor_v4`. Self-play writes v4
  rows whatever the net's I/O version.
- `train.py -quoridor-draw-target {noresult|half}` (default noresult; `half` maps draws back to ½ before the value loss),
  `-zero-wall-effect` (zero planes 21-24 and globals 19, 20 in the loader: the net sees the v3 inputs). New metrics:
  `vhalfloss` (the value loss with draws mapped back to ½ on both sides: comparable between all v4 nets), `pdraw` / `tdraw`.
- `shuffle.py` carries `wallEffectNCHW` (optional: v3 data passes through without it).
- `katago quoridorwalleffects`: the writer's own function on rows of canonical planes (used by the converter).
- Tools: `dumpnninputs -io-version 4`, `evalnnparity` (v4 nets), `writesampletrainquoridor` (v4 rows).

## 5. Old data and old nets

- `python/quoridor_convert_tdata_v3_to_v4.py [-out-dir OUT] DIRS`: **exact** for the wall effects (computed from pawns and
  walls in the stored planes with the writer's C++ function; bit-identical to a writer row in the test) and the 2 globals.
  Draw rows (C0 = C1 = 0.5, C2 = 0, outcome weight on, C27 = 1) become C0 = C1 = 0, C2 = 1. **Not converted:** the TD
  value targets C4-C19 of draw games (a mix of later search values and the final outcome that can't be separated; they
  keep their win / loss form; weight 0.05 of the loss).
- `python/quoridor_upgrade_v3_to_v4.py`: zero columns for the new inputs, zero weights and bias −8 for the new
  noResult rows (value head and the four TD horizons), in the model, the SWA model and the optimizer state. The net
  is the v3 net up to P(noResult) ≈ 3e-4 on v3-equivalent inputs. The −8 is so that the logit can be learned; the initial
  noResult share is a few ‰, not exactly 0.

## 6. Tests

| Test | What |
|---|---|
| `runtests quoridorv4` | wall effects vs a brute force (place every wall on a copy, recompute both distances; 10,615 rows, 171k non-zero effects, max 16, 917k illegal cells), the row, globals and mirror (applyInputSymmetry of a game = the mirrored game's row), training-row encoding, draw search (noResult 0.99, utility 0.0003), cost per row |
| `python/tests/test_nn_parity.py` | v4: an independent Python brute force of planes 21-24 and globals 19, 20 vs the C++ rows (200 positions); v4 nets (conv and transformer) through the exported net and `NNEvaluator`, symmetries 0 and 1: policy / value (incl. noResult) within 5e-7, score outputs with the (1 − noResult) factor |
| `python/tests/test_quoridor_v4.py` | converter == writer rows bit for bit (all arrays, draws included); converter on real run4 rows vs a Python brute force; mirror (loader symmetry vs the brute force of the mirrored position); `-zero-wall-effect`; v3 → v4 migration (same outputs, optimizer state loads and steps, refused twice); shuffler |
| `python/tests/test_quoridor_model.py` | draw-target round trip (a converted draw row trains in `half` mode exactly like the unconverted one), draw rows have no u / lead loss in `noresult` mode, symmetries, ONNX |


## 7. A/B design (home RTX 3070, 2026-10-18)

`scripts/run_ab_io_v4.sh` (steps `prep A B C [D] losses blindspot ladder`; every train.py instance is ≤ 10 epochs of 100k
samples, ~40 min, so no process runs more than an hour).

- **Start:** run4's checkpoint `run4-s28602368` (the newest net in `models/`), migrated to v4 (`quoridor_upgrade_v3_to_v4.py`).
- **Data:** one fresh shuffle of the loop's window with its settings, run on `~/q1_run/run4/selfplay` (so the 1% md5
  validation split is the loop's) with every window row kept: 1,908,872 train rows (window 7.93M-9.86M data rows),
  21,000 validation rows; converted to v4 (16% of the rows are draw rows). Fixed unseen sets of `eval_sets.py` (`old` 8.4k,
  `new` 3.8k, `late` 2.6k rows) converted too.
- **Arms**, Aurora, lr ×2, batch 256, SWA exported, all from the same start checkpoint:
  **A** `-zero-wall-effect -quoridor-draw-target half` (= the v3 net), **B** wall-effect inputs on, **C** draw target on.
  Compared at equal samples: A 30,930,176, B 30,932,736, C 30,931,968 (2.33M samples each; the epochs of an instance are
  file-sized, so B and C were topped up with extra short instances).
- Evaluation: `quoridor_eval_sets_v4.py` (SWA weights, eval mode, rows of the val set and the fixed sets), the
  blind-spot suite v2 (labelled by the start net's lineage `run4-s28602368`; v256 / v1000, gtp params, 212 clear positions),
  and a kq_ladder like `kq_std2` (standard openings `std_run4_s187-s204_p2468.json`, v256, 230 games per pair, 1380 games).
- **Not done:** a noise estimate (a second control arm with another seed). Differences below are single runs.

## 8. A/B results

Held-out losses (policy KL = p0loss − target entropy; `vhalf` = value loss with draws mapped back to ½; the net's own
`vloss` is only comparable between nets trained with the same draw target, S/A/B cannot predict a draw at all):

| arm | set | rows | KL | pacc1 | vhalf | vloss | lead |
|---|---|---:|---:|---:|---:|---:|---:|
| S (start) | val | 20992 | 0.4740 | 0.7054 | 0.5671 | 2.793 | 0.0478 |
| A | val | 20992 | **0.4640** | 0.7073 | **0.5655** | 6.220 | 0.0472 |
| B | val | 20992 | 0.4651 | 0.7098 | 0.5681 | 6.127 | 0.0474 |
| C | val | 20992 | 0.4659 | 0.7103 | 0.5656 | **0.5155** | 0.0475 |
| S | old | 8448 | 0.6868 | 0.6084 | 0.5875 | 0.886 | 0.0663 |
| A | old | 8448 | 0.7072 | 0.6016 | 0.5976 | 1.304 | 0.0669 |
| B | old | 8448 | 0.7141 | 0.5954 | 0.6059 | 1.312 | 0.0670 |
| C | old | 8448 | 0.7104 | 0.6005 | 0.6069 | 0.727 | 0.0673 |
| S | new | 3840 | 0.7228 | 0.5977 | 0.6790 | 1.770 | 0.0639 |
| A | new | 3840 | 0.7604 | 0.5878 | 0.6898 | 3.210 | 0.0642 |
| B | new | 3840 | 0.7548 | 0.5867 | 0.6935 | 3.134 | 0.0649 |
| C | new | 3840 | 0.7454 | 0.5990 | 0.6829 | 0.875 | 0.0647 |
| S | late | 2560 | 0.5824 | 0.6652 | 0.6000 | 2.002 | 0.0466 |
| A | late | 2560 | 0.6175 | 0.6590 | 0.6026 | 3.323 | 0.0470 |
| B | late | 2560 | 0.6195 | 0.6562 | 0.6085 | 3.291 | 0.0464 |
| C | late | 2560 | 0.6124 | 0.6633 | 0.6144 | 0.801 | 0.0469 |

- All arms improve on the in-distribution validation set (KL 0.474 → 0.464-0.466) and get worse on the older fixed sets
  (it is the window's distribution, the same drift the loop shows), by about the same.
- **B (wall-effect inputs): no gain.** Policy KL and `vhalf` are equal to A within ~0.001-0.007 (B marginally worse on the
  old sets). Only the end point at 2.3M samples was measured (intermediate exports exist for part ends only), so an early-samples speedup, as KataCDC saw, is not excluded.
- **C (draw target):** `vhalf` equal to A, policy KL equal; the draw-aware value loss falls from 6.2 (A: 18% draw rows
  predicted as ~0%) to 0.52. So the net learns the draw probability without hurting anything else.

Blind-spot suite v2 (212 clear positions, gtp params; agreement ±2.2 pt at n = 212):

| net | v1000 agree | v1000 regret | v256 agree | v256 regret | swing group v1000 |
|---|---:|---:|---:|---:|---:|
| start | 87.7% | 0.034 | 84.0% | 0.055 | 76.1% (67) |
| A | 88.7% | 0.038 | 84.0% | 0.057 | 73.1% |
| B | 89.6% | 0.033 | 84.9% | 0.043 | 80.6% |
| C | 88.2% | 0.037 | 84.9% | 0.050 | 76.1% |

B is nominally best (swing positions 80.6% vs 73.1%, regret at v256 0.043 vs 0.057), but every difference is within one
standard error.

kq_ladder, standard openings, v256, 1380 games (4.6 min on the GPU), Elo against the start net (95% bootstrap CI):

| net | Elo | CI |
|---|---:|---|
| start (28.6M) | 0 | |
| A (30.93M) | +65 | [43, 89] |
| B | +66 | [43, 89] |
| C | +52 | [30, 74] |

Head to head (first listed bot's score, 230 games): A-B 50%, A-C 53%, B-C 51%. All three gain ~+60 over the start net in 2.3M samples
(that includes the 60k-row drift of the window); B and C are not separable from A.

## 9. Recommendation

- **Wall-effect inputs (B): do not switch for strength.** After 2.3M samples from a strong checkpoint they give nothing
  measurable: not in the held-out losses, the blind-spot agreement (within noise, nominally +1 pt at v1000), or the ladder
  (+66 vs +65). This matches KataCDC's experience (extra features only helped the first 1M samples). It costs +10 µs per NN input row.
  The code stays (it is exact, tested, 8 µs), and `-zero-wall-effect` trains without it. It might still help a net trained from scratch or
  the very early samples of a new run; this test cannot say. A second control arm (seed noise) would be needed to resolve differences below ~±20 Elo.
- **Draw outcome (C): switch.** Neutral to strength within the noise (+52 [30, 74] vs +65 [43, 89]; A-C head to head 53%, which is
  within 1 σ), and it gives what v3 cannot: a draw probability for the search, GUI and kata-analyze (`noResultValue`),
  and `reduceVisitsDrawThreshold` for self-play, where 16% of the rows are draw games. Its cost is nothing at inference.
  Turn `reduceVisitsDrawThreshold` on (0.9) only after the first v4 net exists and look at the draw stats.
- Practically: v4 is one I/O version, so "C" means switching the loop to v4 and training with `-zero-wall-effect`, or leaving the wall-effect
  inputs on (equal in this test; B and A are indistinguishable) at a +10 µs/row self-play cost. I would **switch to v4 with the
  draw target and keep the wall-effect inputs off in training** (`-zero-wall-effect`), unless an extra arm shows B > A; a switch to
  turn the computation off in self-play and inference would then save the 10 µs (not implemented).

## 10. Switch-over procedure for the running loop (`synchronous_loop.sh`, run4)

Do it between cycles. `BASEDIR=~/q1_run/run4`, `TRAININGNAME=run4`; `PY=~/ml_venv/bin/python`.

1. **Stop** the loop after a cycle's export (`torchmodels_toexport/` empty).
2. **Binary and code:** merge this branch into the checkout the loop runs from, build `cpp/katago` (the loop uses
   `$GITROOTDIR/cpp/katago` or `KATAGO_BIN`, and copies `python/` and the configs at start).
3. **Back up** `train/run4/` (`cp -a train/run4 train/run4.v3`).
4. **Upgrade the checkpoint:**
   `$PY python/quoridor_upgrade_v3_to_v4.py -checkpoint $BASEDIR/train/run4/checkpoint.ckpt -in-place`
   (keeps `checkpoint.ckpt.pre_v4`). Move `checkpoint_prev*.ckpt` into the backup: they are v3.
5. **Convert the self-play data** in place (the row count is unchanged, so the train bucket carries on; -num-processes defaults to half the cores):
   `$PY python/quoridor_convert_tdata_v3_to_v4.py -katago cpp/katago $BASEDIR/selfplay` (`-dry-run` first).
   Rewrites each v3 `.npz` atomically, keeping mtimes; ~4 min per 2M rows. Delete `shuffleddata/` and `shufflescratch/`
   (regenerated). Remote (Colab) rows that come in as v3 need the same conversion before the shuffle, and Colab workers need the new binary.
6. **Restart** with the v4 model kind and the zero flag, the draw target is the default:
   `TRAIN_EXTRA_ARGS="... -zero-wall-effect" ./synchronous_loop.sh <NAMEPREFIX> $BASEDIR run4 tf3_b5c256_quoridor_v4 <USEGATING>`
   (drop `-zero-wall-effect` to train with the wall-effect inputs). The first cycle self-plays with the newest model in
   `models/`, a v3 net: it writes v4 rows (draws as noResult). The gatekeeper (v3 incumbent vs v4 candidate) uses the new
   binary for both.
7. Optional, later: `reduceVisitsDrawThreshold = 0.9` in the self-play config.

Equality and smoke checks done: the migrated checkpoint gives the v3 net's outputs up to P(noResult) = 3e-4
(`tests/test_quoridor_v4.py`), an exported v4 net plays self-play (6 games, 66 rows, 25 / 21 channels, `reduceVisitsDrawThreshold = 0.5`
accepted), and the A/B arms started from the migrated checkpoint trained with the loop's `train.sh`.

## 11. Where things live

| What | Where |
|---|---|
| layout, wall effects, symmetry | `cpp/neuralnet/quoridornn.{h,cpp}` |
| training write | `cpp/dataio/trainingwrite.{h,cpp}` (`wallEffectNCHW`, `ValueTargets::setOutcomeOfWinner`) |
| draw search values | `cpp/search/search.cpp`, `cpp/neuralnet/nneval.{h,cpp}`, `cpp/program/play.cpp` |
| `rootInfo noResultValue` | `cpp/command/gtp.cpp`, `cpp/search/searchresults.cpp` |
| converter helper command | `cpp/command/walleffects.cpp` (`katago quoridorwalleffects`) |
| Python | `modelconfigs.py`, `model_pytorch.py` (3 value logits), `metrics_pytorch.py`, `data_processing_pytorch.py`, `shuffle.py`, `train.py`, `export_model_pytorch.py` |
| tools | `python/quoridor_convert_tdata_v3_to_v4.py`, `python/quoridor_upgrade_v3_to_v4.py`, `python/quoridor_eval_sets_v4.py`, `scripts/run_ab_io_v4.sh` |
| tests | `cpp/tests/testquoridoriov4.cpp` (`runtests quoridorv4`), `python/tests/test_quoridor_v4.py`, `test_nn_parity.py`, `test_quoridor_model.py` |
