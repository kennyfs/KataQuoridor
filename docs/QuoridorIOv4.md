# Quoridor I/O v4: the draw as its own outcome

*Design and A/B document, 2026-10-18; updated the same day (wall-effect inputs removed). I/O v4 = the [I/O v3](QuoridorIOv3.md)
inputs (21 spatial / 19 global channels, unchanged) + the draw as the noResult value outcome. v4 was first built with a
second part, wall-effect inputs; the A/B (§7-§9) showed no gain from them, so they were removed before any v4 data or net
existed (§2.2). The A/B tables below are the first A/B, with the wall-effect inputs as one of its arms.*

## 1. Why

- Draws were trained as win 0.5 / loss 0.5 (noResult forced off), so the value head could not tell a certain draw from a
  coin flip. 16% of run4's current window rows belong to drawn games. As its own outcome, a draw gives the search and the
  GUI a draw probability and lets self-play reduce visits in settled draws (`reduceVisitsDrawThreshold`).
- (Removed, see §2.2) run4's weak spots are wall moves with low prior that a deep search finds ([BlindSpots.md](BlindSpots.md)).
  The net sees walls and BFS distances; "how much would this wall cost the opponent" is a non-local computation it has to
  learn through 15 attention layers, so v4 first handed it over as inputs. KataCDC's extra features had sped up only the first
  1M samples there, hence the A/B before any switch.

## 2. Inputs

### 2.1 v4 inputs = v3 inputs

`QuoridorNN` (`cpp/neuralnet/quoridornn.{h,cpp}`) is still the one module that knows the layout. v4 has **21 spatial / 19
global** input channels, exactly the v3 channels and values (`fillRow` fills the same row for I/O 3 and 4; tested row for
row, with and without the mirror). So:

- a v3 net and a v4 net are told apart by `quoridor_io_version` in the model config (Python) / the model file's I/O version
  (C++, model option D), **not** by channel counts. The loader checks the version of the data against the version of the net;
- v3 and v4 *training rows* also have the same arrays; they differ only in the draw targets. §4.1 says how the two kinds of
  data are told apart.

### 2.2 The wall-effect inputs (removed)

The first v4 added, after the v3 channels, spatial planes 21-24 (for each of vertical / horizontal walls and for the side
to move / the opponent, the increase of that player's shortest distance if the wall were placed now; 255-sentinel / −1 for
an illegal or fully blocking placement) and globals 19, 20 (largest opponent increase, number of opponent walls costing
≥ 2), stored raw as `wallEffectNCHW` (`uint8 [N,4,9,9]`). They were computed by `QuoridorNN::fillCanonicalWallEffectU8`
(shortest-path counting per edge, then a bit-parallel early-exit BFS), cost 8 µs per row (`fillRow` 29 → 39 µs), and were
verified bit for bit against two independent brute forces. In the A/B they gave **no measurable gain** (§8, §9), so they
were taken out again before the loop switched to v4: no spatial planes 21-24, no globals 19-20, no `wallEffectNCHW` array,
no `-zero-wall-effect`, no `katago quoridorwalleffects`, no wall-effect parts in the converter and migration.

The code is in git history: the last commit that has it is `cb0b5ed7` (`afecf570` C++, `38fe6c49` Python, `c1369f89` /
`cb0b5ed7` A/B script and docs; e.g. `git show cb0b5ed7:cpp/neuralnet/quoridornn.cpp`, `git show
cb0b5ed7:docs/QuoridorIOv4.md` for the full wall-effect description, and reverting the removal commits brings it back).
It could be tried again for a net trained from scratch (§9).

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
  `{…, 3: 21 / 19, 4: 21 / 19}`; presets `tf2_b4c192_quoridor_v4`, `tf3_b5c256_quoridor_v4`, `b2c64_quoridor_v4` (3 value
  logits). Self-play writes v4 rows whatever the net's I/O version.
- `train.py -quoridor-draw-target {noresult|half}` (default noresult; `half` maps draws back to ½ before the value loss; the
  A/B control). New metrics: `vhalfloss` (the value loss with draws mapped back to ½ on both sides: comparable between all
  v4 nets), `pdraw` / `tdraw`.
- `train.py -stop-at-samples N` (§4.2).
- Tools: `dumpnninputs -io-version 4`, `evalnnparity` (v4 nets), `writesampletrainquoridor` (v4 rows).

### 4.1 Telling v3 data from v4 data: the `quoridorIOVersion` stamp

v3 and v4 rows have the same arrays, so shape can't tell them apart, and training a v4 net on unconverted v3 rows would teach
it that a draw is a coin flip. Every v4 npz therefore carries one extra key, **`quoridorIOVersion`** (`int32 [1,1]`, value 4):

| Who | What |
|---|---|
| C++ writer (`TrainingWriteBuffers::writeToZipFile`) | writes the stamp into every training npz |
| `shuffle.py` | refuses any input npz without the stamp (or with another value) with a message pointing to the converter; its output npz files (shards and final) carry the stamp. `-allow-unstamped-v3` accepts v3 data and writes unstamped output (used only to prepare a v3 shuffle for conversion, the A/B script) |
| `train.py` / `read_npz_training_data` | for a Quoridor model, the data's I/O version (the stamp, or 3 if absent) must equal the model's `quoridor_io_version`, else a `ValueError` naming the file |
| `quoridor_convert_tdata_v3_to_v4.py` | converts the draw targets and adds the stamp; stamped files are skipped, so it is idempotent and can be re-run on a directory that keeps receiving old-binary data |

Files the old binary writes (a Colab worker that was not updated) are unstamped, so the shuffle fails loudly on the first one
instead of silently mixing in coin-flip draws; run the converter on the self-play directory again and restart the shuffle.
(Not covered: someone stamping a v3 file by hand. A content check, e.g. a row with C0 = C1 = 0.5 and outcome weight on, would
catch that, but a v4 row legitimately has such cells in no case, so the stamp is the single source of truth.)

### 4.2 `train.py -stop-at-samples N`

`-max-epochs-this-instance` and `-max-training-samples` act at epoch boundaries, so arms with the same epochs ended at
different samples (the A/B needed top-ups). `-stop-at-samples N` (K/M/B suffixes ok) stops the run when its total trained
samples (`train_state["global_step_samples"]`, over all instances) reach N, even in the middle of an epoch: it ends the
epoch at the first batch boundary at or after N (exactly N when N is a multiple of batch size × world size), then does what
an epoch end does (validation, SWA, checkpoint) and **forces an export** (even with `-export-only-at-end`'s end export or an
`epochs_per_export` > 1). A later instance resumes from that checkpoint; an instance that starts at or past N exits at once
without touching the data. Without the option nothing changes. Tested on the CPU (`tests/test_train_stop_at_samples.py`:
300, then resume to 450, then a no-op).

## 5. Old data and old nets

- `python/quoridor_convert_tdata_v3_to_v4.py [-out-dir OUT] DIRS`: needs no katago binary. For every unstamped npz with 21 / 19
  channels: draw rows (C0 = C1 = 0.5, C2 = 0, outcome weight on, C27 = 1) become C0 = C1 = 0, C2 = 1; the stamp is added;
  everything else is bit-identical (tested against what the C++ writer writes for the same games, and on real run4 rows).
  **Not converted:** the TD value targets C4-C19 of draw games (a mix of later search values and the final outcome that can't
  be separated; they keep their win / loss form; weight 0.05 of the loss).
- `python/quoridor_upgrade_v3_to_v4.py`: only the value head changes: zero weights and bias −8 for the new noResult rows (value
  head and the four TD horizons), in the model, the SWA model and the optimizer state. The inputs are the v3 inputs, so the
  net is the v3 net up to P(noResult) ≈ 3e-4. The −8 is so that the logit can be learned; the initial noResult share is a
  few ‰, not exactly 0. `config["quoridor_io_version"]` becomes 4, so the net is a v4 net from then on.

## 6. Tests

| Test | What |
|---|---|
| `runtests quoridorv4` | v4 rows equal v3 rows (feature counts, values, mirror, 1666 rows of wall-heavy games with the repetition rule on), draw targets, a search of a certain draw (noResult 0.99, utility 0.0003; follows `noResultUtilityForWhite`) |
| `runtests` (others) | the writer's draw target and stamp (`testquoridorscore`, `testquoridoriov3`: training row = v3 row) |
| `python/tests/test_nn_parity.py` | v3 and v4 nets (conv and transformer) through the exported net and `NNEvaluator`, symmetries 0 and 1: policy / value (incl. noResult) within tolerance, score outputs with the (1 − noResult) factor. Use the Eigen build (`KATAGO_BIN=cpp/build-eigen/katago`); the CUDA build's TF32 and fp32 kernels exceed the 1e-4 tolerance on v3 nets too |
| `python/tests/test_quoridor_v4.py` | the writer stamps and writes 21 / 19 channels; converter == writer rows bit for bit (all arrays, draws included; idempotent; wrong channel counts refused); converter on real run4 rows (only the draw targets change); the loader refuses unstamped data for a v4 net (and v4 data for a v3 net); the shuffler stamps, refuses unstamped input, and with `-allow-unstamped-v3` stays unstamped; v3 → v4 migration (same outputs, optimizer state loads and steps, refused twice) |
| `python/tests/test_quoridor_model.py` | draw-target round trip (a converted draw row trains in `half` mode exactly like the unconverted one), draw rows have no u / lead loss in `noresult` mode, symmetries, ONNX |
| `python/tests/test_train_stop_at_samples.py` | `-stop-at-samples` mid-epoch, resume, no-op |
| smoke | a v4 net (run4's checkpoint migrated and exported) plays self-play on the Eigen build (6 games, 47 rows, stamped, 21 / 19 channels, `reduceVisitsDrawThreshold = 0.5` accepted) |


## 7. A/B design (home RTX 3070, 2026-10-18)

Run with `scripts/run_ab_io_v4.sh` as of commit `cb0b5ed7` (steps `prep A B C [D] losses blindspot ladder`; every train.py
instance was ≤ 10 epochs of 100k samples, ~40 min, so no process ran more than an hour). The script in the tree now is the
draw-target-only version (two arms, `-stop-at-samples`, §7.1); the wall-effect arms and the `-zero-wall-effect` flag are gone.

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

### 7.1 The script today

`scripts/run_ab_io_v4.sh [prep A B eval | ladder]`: arm **A** `-quoridor-draw-target half` (the v3 net), arm **B**
`-quoridor-draw-target noresult`, both from the migrated start checkpoint on the converted shuffle of the loop's window.
Each arm trains `SAMPLES` (default 2,000,000) samples as parts of `PART` (default 1,000,000, ~40 min) with
`-stop-at-samples k × PART`, so both arms end at exactly the same sample count and no top-ups are needed. `prep` shuffles
the old data with `-allow-unstamped-v3`, then converts it (draw targets + stamp) with the converter. The first A/B's arms
A and C are this script's A and B. It was run after the change only as far as `bash -n` and the CPU `-stop-at-samples` test
(the GPU was not used for this change); the first A/B's results do not depend on it.

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
  **Decision: removed** (§2.2; the code is in git history). It might still help a net trained from scratch or the very early
  samples of a new run; this test cannot say. A second control arm (seed noise) would be needed to resolve differences below ~±20 Elo.
- **Draw outcome (C): switch.** Neutral to strength within the noise (+52 [30, 74] vs +65 [43, 89]; A-C head to head 53%, which is
  within 1 σ), and it gives what v3 cannot: a draw probability for the search, GUI and kata-analyze (`noResultValue`),
  and `reduceVisitsDrawThreshold` for self-play, where 16% of the rows are draw games. Its cost is nothing at inference.
  Turn `reduceVisitsDrawThreshold` on (0.9) only after the first v4 net exists and look at the draw stats.
- Practically: **v4 = the v3 inputs + the draw outcome**. The loop switches to it with the draw target on and no wall-effect
  inputs (they no longer exist in the tree).

## 10. Switch-over procedure for the running loop (`synchronous_loop.sh`, run4)

Do it between cycles. `BASEDIR=~/q1_run/run4`, `TRAININGNAME=run4`; `PY=~/ml_venv/bin/python`. Same as the v3 switch
([QuoridorIOv3.md](QuoridorIOv3.md) §6), but the inputs do not change: **no new input channels, no wall-effect flag, and the data
conversion rewrites targets only (no katago binary needed).**

1. **Stop** the loop after a cycle's export (`torchmodels_toexport/` empty).
2. **Binary and code:** merge this branch into the checkout the loop runs from, build `cpp/katago` (the loop uses
   `$GITROOTDIR/cpp/katago` or `KATAGO_BIN`, and copies `python/` and the configs at start). Colab workers need the new
   binary too (old workers write unstamped v3 rows, step 5).
3. **Back up** `train/run4/` (`cp -a train/run4 train/run4.v3`).
4. **Upgrade the checkpoint:**
   `$PY python/quoridor_upgrade_v3_to_v4.py -checkpoint $BASEDIR/train/run4/checkpoint.ckpt -in-place`
   (keeps `checkpoint.ckpt.pre_v4`; the value head gets its noResult rows, the config says I/O v4). Move `checkpoint_prev*.ckpt`
   into the backup: they are v3.
5. **Convert the self-play data** in place (row count unchanged, so the train bucket carries on; `-num-processes` defaults to
   half the cores): `$PY python/quoridor_convert_tdata_v3_to_v4.py $BASEDIR/selfplay` (`-dry-run` first to see "N v3 files to
   convert"). It rewrites each unstamped `.npz` atomically (draw targets C0 = C1 = 0.5 → 0 / 0 / 1, plus the stamp), keeping
   mtimes; stamped files are skipped, so it can be run again. Delete `shuffleddata/` and `shufflescratch/` (regenerated).
   **Remote (Colab) rows** that come from a worker with the old binary are unstamped: the shuffle fails on the first one
   ("npz has no quoridorIOVersion stamp ... convert it"); run the converter on that directory again, then restart the shuffle.
6. **Restart** with the v4 model kind (the draw target is the default; no extra flag):
   `./synchronous_loop.sh <NAMEPREFIX> $BASEDIR run4 tf3_b5c256_quoridor_v4 <USEGATING>`.
   The first cycle self-plays with the newest model in `models/`, a v3 net: it writes v4 rows (draws as noResult). The
   gatekeeper (v3 incumbent vs v4 candidate) uses the new binary for both.
7. Optional, later: `reduceVisitsDrawThreshold = 0.9` in the self-play config, after the first v4 net exists and the draw
   statistics look right.

Checks done: the migrated checkpoint gives the v3 net's outputs up to P(noResult) = 3e-4 (`tests/test_quoridor_v4.py`); a v4
net (run4's checkpoint migrated and exported) plays self-play on the Eigen build and writes stamped 21 / 19-channel rows; the
shuffler accepts those and refuses unstamped ones; the loader refuses unstamped data for a v4 net; `train.py` trained a b2c64 v4
net on CPU on shuffled writer rows with `-stop-at-samples`.

## 11. Where things live

| What | Where |
|---|---|
| layout, symmetry | `cpp/neuralnet/quoridornn.{h,cpp}` |
| training write | `cpp/dataio/trainingwrite.{h,cpp}` (`ValueTargets::setOutcomeOfWinner`, the `quoridorIOVersion` stamp) |
| draw search values | `cpp/search/search.cpp`, `cpp/neuralnet/nneval.{h,cpp}`, `cpp/program/play.cpp` |
| `rootInfo noResultValue` | `cpp/command/gtp.cpp`, `cpp/search/searchresults.cpp` |
| Python | `modelconfigs.py`, `model_pytorch.py` (3 value logits), `metrics_pytorch.py`, `data_processing_pytorch.py`, `shuffle.py` (stamp check), `train.py` (`-stop-at-samples`), `export_model_pytorch.py` |
| tools | `python/quoridor_convert_tdata_v3_to_v4.py`, `python/quoridor_upgrade_v3_to_v4.py`, `python/quoridor_eval_sets_v4.py`, `scripts/run_ab_io_v4.sh` |
| tests | `cpp/tests/testquoridoriov4.cpp` (`runtests quoridorv4`), `python/tests/test_quoridor_v4.py`, `test_nn_parity.py`, `test_quoridor_model.py`, `test_train_stop_at_samples.py` |
