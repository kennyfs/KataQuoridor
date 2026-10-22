# KataQuoridor training log

Goal: a state-of-the-art Quoridor agent (Duel mode, 9×9, 10 walls each) built by modifying KataGo, learning by
self-play without Quoridor-specific shortcuts (no analytic endgame solving, no move pruning): problems are fixed
through the rules, training targets, inputs and training setup.

Hardware: one RTX 3070 (8 GB) for self-play and training; occasionally a CPU machine (ws4) for random-net self-play.
Runs live under `~/q0_run`, `~/q1_run`; Elo ladders under `~/arena`. Design details: [QuoridorIOv2.md](QuoridorIOv2.md),
[QuoridorIOv3.md](QuoridorIOv3.md); evaluation tools: [Evaluation.md](Evaluation.md).

Elo below is from the KataQuoridor-only `kq_ladder` (one `katago match` process, v256, paired random openings,
standard rules with the repetition rule off), relative to the 0.1.0 release net `q0 run1-s16141056` = 0.

## 1. q0: the first run (I/O v1), release 0.1.0

- `tf2_b4c192` (1.29M params; the model was meant to be "tf3" but was implemented with 2 transformer layers per
  nested-bottleneck block), SGD, to 16.1M samples / 2.85M data rows. Released as 0.1.0 (`kataq1-tf2-b4c192-s16141056`).
- Late in the run, 300-ply cutoff games rose (0.02% → 0.4%). At first this looked like a bug (a "corridor blockade");
  most long games turned out to be **positions where whoever breaks the deadlock loses** (e.g. a pawn sealed in a
  region can only escape by jumping the opponent and then gets walled off), i.e. real equilibria that are draws. v1
  dropped cutoff games from training and had no notion of time or draws.

## 2. Quoridor I/O v2 and the λ experiment

- I/O v2: the 300-ply draw became a rule (value 0.5/0.5), inputs for legal walls, plies-until-draw and komi; tempo
  komi (standard −0.5) with randomization and fence handicap; separate heads for the utility score `u` and the
  tempo lead `s`; a time bonus λ (`u = s + sign(s)·λ·(maxPlies − T)`) to make winners progress.
- λ experiment (`~/q1_run` run0–run2, b2c64): λ = 0.15 was clearly worse (−253 Elo); λ = 0.05 vs 0 was within noise
  (−19 [−53, 14]). Chose λ = 0.05 for the main run.

## 3. run3 (`~/q1_run/run3`): tf2_b4c192, SGD

- Started from 13,000 random-net games (~560k rows, generated on ws4), then the synchronous loop.
- **Repetition draw rule + I/O v3** (mid-run, via a zero-padded checkpoint upgrade and data conversion): threefold
  repetition (`repetitionDrawCount`) ends equilibrium cycles early; self-play turns it on per game (75%, later 65%) so
  nets learn both with and without it (arena vs other engines has it off). v3 inputs: rule on, repetition count,
  pawn moves that repeat / draw.
- **λ = 0 from run3-s21886976** (data converted: final utility := lead; short-term score target weight off): the
  repetition rule already ends cycles, the 300-ply draw still pushes winners, and `scoreMean` becomes the lead. Elo per
  generation did not drop (5 gens after the switch +161 vs +131 before).
- run3-s26606336 (26.6M samples / 4.61M rows) ≈ the 0.1.0 net (−7). Growth was slowing; KL kept falling but vloss
  flattened.

## 4. Optimizer: SGD → Aurora; run3a

- Discovered that train.py's per-sample lr with `lr_scale` 1 is KataGo's *late*-training value (its own schedules
  start at 8–12×), so q0/run3 SGD trained with a low lr all along.
- Merged upstream KataGo (floored weight decay, Aurora/Muon fixes, cyclic LR, …). Aurora's lr is a per-step size,
  so the defaults were far too small from scratch (first attempt: no learning at all inside the 1/20 warmup).
- Fair lr sweep (b4c192, 300k samples, no warmup): Aurora stable up to ×8 and clearly better per sample than SGD at
  ×2–×8 (val vloss 0.713 vs 0.862 at ×8); Aurora starts slower than SGD in the first ~100k samples.
- Bootstrap b4c192 from scratch with Aurora on run3's newest 2.5M rows (×8 → ×4 at 3.6M → ×2 at 5.1M, 6M samples):
  **+84 Elo over run3's last SGD net on the same data**, won the gatekeeper 100–79, and became run3a's first net
  (`run3a-s6097408-d4605550`; logs in `~/q1_run/run3/train/run3a/bootstrap_from_scratch/`).
- run3a = the run3 loop continued with Aurora, constant lr ×2, `MAX_TRAIN_PER_DATA` 6 (8 showed mild overfit),
  validation on (5% of files, 10k samples), `policyInitAreaProp` 0.08 and the rule off in 35% of games (standard
  rule-off games had collapsed onto two openings that stalled into 300-ply draws).
- An incident: the first run3a launcher lacked the ml_venv PATH, so shuffle/train failed for ~6 h while self-play
  kept running with one net (~2.4M rows); most of those rows were moved to `~/q1_run/run3/selfplay_held/`.
- kq_ladder3: run3a-s13371136 (5.90M rows) = **+173** (72% vs q0 0.1.0, 74% vs run3-s26606336). But Elo per data
  row after the switch (~+75 per 1M rows) is well below run3's late SGD rate (~+220 per 1M rows), val losses are flat
  and lead overfits: b4c192 looks capacity-limited once trained properly.

## 5. run4 (`~/q1_run/run4`): tf3_b5c256

- `tf3_b5c256_quoridor_v3`: KataGo's mainline design (b11c768h12nbt3tflrs-fson-silu) scaled down: 5 nested-bottleneck
  blocks of 3 transformer layers (15 layers), 256 trunk, fson normalization, SiLU; 3.67M params (2.8× b4c192).
- Bootstrap from scratch with Aurora on run3's newest 2.5M rows (its share of `run3a-s6097408` data swapped for 300
  random held files nobody had trained on), 128k-sample ramp 0.5× → 8×, then ×8; ×8 plateaued by ~3M, so it drops to
  ×4 at 3.1M and ×2 at 4.3M, 6M total. At 3M samples it matched the b4c192 bootstrap on policy / value losses and was
  clearly better on lead loss.
- An lr test from the 6.48M checkpoint (×1 vs ×2, same reshuffled data) kept ×2; the loop continued from the ×2
  branch with Aurora ×2, `MAX_TRAIN_PER_DATA` 6, 1% validation files (md5 of the path).
- Self-play visits 600/100 → **1000/250** (full / cheap) from 2026-10-04: v1000 is +52 Elo over v600 for
  run4-s10469632 (`~/arena/visit_scaling_r4`; v256 → v1000 +110).
- Strength: equal time on the RTX 3070 (run4 v1024 vs run3a v1750, run4 is ~1.7× slower per visit): 56.2% (+43);
  kq_ladder9 (v256): run4-s20387072 +72 over run3a's latest, about +283 over 0.1.0 in the joint fit of ladders 1-9.
- **Draw convergence and root exploration** (2026-10-05): standard self-play drifted into drawn lines (standard
  rule-on draws 25% at s17.11M → 72% at s17.67M). A 1000-visit search rarely visited low-prior winning moves that a
  10000-visit search found (`e8 e2 e7 e3`: e6; then `e6 e4`: a6h / h6h, prior 1.6%), and cheap searches (75% of the
  moves) never played them. Forced playouts 2 → 4 and root policy temperature 1.25 / 1.1 → 1.6 / 1.5 (full searches
  only) raised the policy-target share of those moves from 5% to 20% at no cost (`~/arena/explore`). Within two
  cycles standard draws fell to 6% (rule on) / 0% (off) and Black's score at the empty board rose to ~0.8; the net
  learned a6h (raw White win after it 0.58 → 0.39). kq_ladder Elo stayed flat (s17.11M → s20.39M: +7 ± 30): its random
  4-ply openings rarely reach these lines. Openings collapsed onto one line (95.5% of empty-board games
  `e8 e2 e7 e3 e6 e4`), and deeper Black blind spots remain (9 of 30 common positions differ between 1k and 10k).
- The loop stopped after run4-s20387072's self-play (113K rows, not yet trained), 2026-10-05.

### 5.1 Restart (2026-10-18)

- The exploration settings had been left out of the committed config; restored (fa4343da).
- Self-play: 2 NN server threads, +9% NN evals/s (16.4K → 17.9K; 512 game threads was slower).
- A cycle is now `ROWS_PER_CYCLE` = 120K new rows instead of 10000 games: all the self-play rows on disk that the last
  shuffle did not see (`rows_since_shuffle.py`), so remote rows that arrive while the loop shuffles, trains or is
  stopped count towards the next cycle; local self-play makes up the rest (`selfplay -extra-rows-file`).
- Remote self-play on Colab (`python/colab`, ported from KataCDC; own key and share `ws4:tmp2/kq_remote`).
- Measured on run4-s20387072 self-play: both players are out of walls in 35% of the standard games (the last ~14
  plies, 7.8% of their plies). The rule "A has no walls, B to move, dist(B) <= dist(A) → B wins" fires in 55% of
  them but is right only 95.8% of the time (11% wrong in equal races with both at 0 walls: pawns block each other and
  jumps change the race), so the race needs an exact solver, not a distance rule (prompt W).
- kq_ladder can take a fixed opening suite (`openings: {"file": ...}`); `openings_from_selfplay.py` makes one from
  the most common standard-game lines (115 lines of 2-8 plies from run4 s18.7M-s20.4M, mirrors merged).

### 5.2 I/O v4 A/B (2026-10-18)

[QuoridorIOv4.md](QuoridorIOv4.md). From run4-s28602368 (migrated to v4), one fresh shuffle of the window (1.91M rows, 16% draw
rows), 2.33M samples per arm: A control (v3 inputs, draws as ½) +65 Elo [43, 89] over the start net (kq_std2-style ladder, v256);
B wall-effect inputs +66 [43, 89]; C draw outcome +52 [30, 74]. Held-out policy KL and `vhalf` equal across arms, blind-spot
agreement within noise (B nominally best). Wall effects do not help at this stage; the draw outcome costs nothing and gives a
draw probability. Not run: a seed-noise arm. Decision: v4 = the v3 inputs + the draw outcome; the wall-effect inputs were
removed before the loop switched (code in git history, `cb0b5ed7`). The loop's switch to v4 converts old data by stamp
(`quoridorIOVersion`) and draw targets only.

### 5.3 Overnight run, lr ×4, switch to v4 (2026-10-18)

- One night (02:36-09:52): 11 nets, run4-s21083904 → s28602368, +1.28M rows, 1.05M of them from Colab workers (2 L4, some
  T4); window `EXPAND_WINDOW_PER_ROW` 0.6 → 0.3 (3.0M → 1.9M rows), hints on (3% of the games).
- Standard-opening ladder `~/arena/kq_std2` (v256, race solver off, 4298 games, s20387072 = 0): s21.8M +20, s23.4M +81,
  s24.7M +106, s26.2M +139, s27.9M +163, s28.6M +157 (CI ±25). The random-opening ladder would not have shown this
  (§5, draw convergence).
- Blind-spot suites (`~/arena/blindspot2`): each suite favours its labelling net (v1 labelled by s20.4M, v2 by s28.6M),
  so they find weak spots rather than rank nets: value-swing and contested positions ~60-79% agreement in both, late
  and random positions ≥ 92%. Hints re-mined with s28.6M: 27 of the 55 old hints expired, 34 new (62 in use).
- lr test (`~/q1_run/run4_lr4`): from run4-s27906816's export checkpoint (no optimizer state) on the shuffle that
  trained run4-s28602368 (×2, 695552 samples), at ×4: 488K samples (equal time) and 766K (≈ equal samples). Held-out
  (the shuffle's md5 validation files, SWA weights): KL 0.4816 (×2) vs 0.4789 / 0.4766 (×4), value loss 0.5668 vs
  0.5714 / 0.5678; standard ladder vs s28.6M: −7 / +16 [−3, 35]. → the loop runs **×4** (6 samples per new row as
  before) and continues from the ×4 766K checkpoint (`run4lr4-s28673280`, its train bucket set to the loop's).
- Switched to **I/O v4** (draws as noResult) per [QuoridorIOv4.md](QuoridorIOv4.md) §10: checkpoint migrated (v3 backups in
  `train/run4.v3`, `checkpoint.ckpt.s28602368_v3/_v4`), all 9.86M self-play rows converted in place (hard-link backup
  `selfplay.v3`; 6.6% draw rows), `reduceVisitsDrawThreshold = 0.9`, Colab binary rebuilt.

### 5.4 First v4 cycles: s28.67M → s33.97M (2026-10-18 evening)

- **Ladders** (v256, `~/arena/kq_std3` standard openings, `~/arena/kq_rand3` random 4-ply openings), vs s27.91M:
  s33.97M is +31 [10, 51] on standard openings and +39 [21, 56] on random openings. No loss of generality, but the
  standard-opening gain is slower than before v4 (s24.70M → s27.91M was +66 over 3.2M samples; now about +25 over 5.3M).
- **Saturation.** At the standard komi the nets' first self-play eval gives White 10% (komi +0.5: White 90%): the
  standard game is read as a one-tempo Black win, and every komi / handicap setting is near decided. 92% of searched
  positions (s33.1M) are beyond 90 / 10%, up from 62% at s17.1M; so `reduceVisits` (threshold 0.9) acts almost all
  game, and 1000-visit searches fell from 12.3% to 7.7% of positions (mean row weight 0.169 → 0.140).
- **Shuffle:** `shuffle.py -rows-cache` reuses `rows_cache.json` (row counts were read from ~15k npz headers each
  shuffle: 1 s warm, up to 131 s on a cold page cache on the HDD array).
- **Training in bf16** (`-use-bf16`, from the next cycle): 25,600 samples in 16–21 s vs 49–54 s in fp32/TF32 on the
  3070 (~2.7×), same losses (`~/q1_run/bench_bf16`).

- **Self-play changes** (from the next cycle; smoke run of 43k rows with s33.97M, kept in run4's data):
  `reduceVisitsThreshold` 0.9 → 0.97; random komi 30% → 12% (weights 70/25/5); `compensateAfterPolicyInitProb` 0.3
  (the fair komi after the policy init moves); fence handicap 10% → 12% with 80% compensation; hints re-mined with
  s33.97M (26 of the 62 old hints now found by the fresh search and dropped, 26 new) and `hintPosesProb` 0.03 → 0.05.
  Smoke vs s33.1M's self-play: 1000-visit searches 7.7% → 14.2% of positions, mean row weight 0.140 → 0.174,
  positions beyond 90 / 10% 92% → 85%, komi +0.5 games 90/9 → 68/29 (W/B); NN rows per data row 518 → 462.
- **Search-parameter A/B** (`~/arena/kq_search1`, s33.97M vs itself, v512, 300 games each): cpuct 0.8 +25 [−3, 53],
  cpuct 1.4 −23, fpuReductionMax 0.4 −18 / 0.1 +5, dynamicScoreUtilityFactor 0.5 −12, policyOptimism 0.5 +13,
  subtreeValueBiasFactor 0 +5. Nothing significant; nothing changed (cpuct 0.8 is worth a longer test).
- **Race solver CPU:** match used all cores because one-sided proofs are on by default there: every wall the player
  with walls places in the tree is a new configuration and a ~1 ms solve (1024-entry cache). Exact races (both out of
  walls) are cheap and are the solver's main use (leaves after the last walls placed in the tree), so only one-sided
  proofs are now limited to the root's walls (`quoridorRaceSolverOneSidedRootWallsOnly`, default on). CPU for 460
  games at v256: one-sided everywhere 827 s, exact everywhere + one-sided at the root 510 s, exact only 457 s (the
  self-play setting), solver off 482 s.

### 5.5 s33.97M → s53.96M and an outside opponent (2026-10-19)

- The loop ran to run4-s53960960 (gatekeeper at `GATEKEEPER_WIN_PROP` 0.45 accepted every net). Standard self-play:
  Black wins 97-98% of the games from s34M on; the first eval gives White 10% (s32.4M) → 2.3% (s52.9M).
- **Empty-board ladder** (`~/arena/kq_empty4`, v256, no opening suite; match's early temperature gives 55 distinct
  8-ply lines, the top one 37%): vs s33.97M, s38.8M +58, s44.1M +92, s45.8M +115, s51.0M +136, s52.9M +139,
  s53.96M +120 (CI ±20); s27.9M −124. The old standard-opening suite (s18.7M-s20.4M lines) was dropped as stale.
- **Random 4-ply openings** (`~/arena/kq_rand4`): flat from s36.4M (+26) to s53.96M (+31); s27.9M +4. Direct:
  s53.96M scores 54% vs s27.9M, 50% vs s40.9M. The gain is on the self-play main line only.
- **Ka_Quoridor** (an outside AlphaZero/KataGo-style engine, net epoch4100, `ka_quoridor_qtp.py`, Evaluation.md),
  1000 nodes on the CPU, 4-ply random openings, 80 games: s27.9M v256 74%, s40.9M v256 76%, s53.96M v256 68%,
  s53.96M v1000 76% (v64 60%, v16 57%, 40 games). No progress since s28M against an engine the nets never saw.

### 5.6 Openings over run4, policy-init temperature (2026-10-22)

- Opening frequencies in run4's standard self-play games without policy-init moves (`~/arena/joseki`, mirror images
  merged): White's first reply was e2 (both race) in ~100% of games until s27M; within two cycles of the I/O v4 / lr ×4
  switch (s28.6M) it became the wall d7v (85% at s32M), and e2 came back to ~65% from s48M. Inside the e2 line the main
  continuation changed twice (c7h lines until s19M, d6h c3h s20-28M, d6h e3h from s29M). Training changes visibly move
  the nets' openings.
- `policyInitAreaTemperature` (the policy-init moves sample policy^(1/T); 1.0 so far), s53.96M, ~1000 games each, no komi
  randomization / handicap / compensation (`~/arena/policyinit_t`): T 1.0 / 1.3 / 1.6 gave effective 6-ply lines
  (exp entropy) 26 / 85 / 161, positions still open (White 10-90%) at the first search 2% / 5% / 11%, Black score 96% /
  95% / 86%. A higher T does not leave more decided positions: the empty board itself is a Black win for these nets,
  and T 1.0 keeps the init moves on that line.

## 6. q2 (`~/q2_run`): a clean run (from 2026-10-22)

q1's run0-run4 are one continuous line with many mid-run changes (λ 0.05 → 0, I/O v2 → v3 → v4, SGD → Aurora, lr,
exploration, b4 → b5). q2 restarts from a random net with what q1 ended with, data on the home NVMe (`~/q2_run`; reading
many npz files on /thi's HDD array was slow):

- 500K random-net rows from ws4 (native Eigen build `ws4:tmp2/kq_remote/build_eigen`, `gen_random_ws4.sh`), first
  cycle by hand (`first_cycle.sh`: shuffle all, 500K samples), then `run.sh` (`synchronous_loop.sh`, no gatekeeper,
  `ROWS_PER_CYCLE` 50K at first, window min 250K / slope 0.4, 6 samples per row).
- `tf2_b4c192_quoridor_v4` (I/O v4 from the start), Aurora, bf16, lr schedule in `lr_schedule.txt` (ramp 0.5× → 8× over
  128K samples, then 8×; compare a ×4 branch around 2M samples). b5 (`tf3_b5c256`) later, bootstrapped on the newest
  ~2.5M rows when the b4 ladders flatten, as in q1.
- `selfplay_quoridor_q2.cfg` (its header lists the differences): policy-init T 1.5; `compensateAfterPolicyInitProb`
  0.8, randomized-komi games (15%) never compensated; no fence handicap; 600 / 150 visits while b4; no hints; the
  exploration (forced playouts 4, root policy T 1.6 / 1.5), `reduceVisitsThreshold` 0.97, exact race solver and λ = 0
  from the start.
- SGF root comments now say where the komi came from (`komiRandom=1`, `komiComp=policyinit|fair|fancy|emptyboard`),
  and the self-play stats count compensated games separately (smoke test with s53.96M: compensated games Black
  0.505 over 203).

## Lessons

- Long games were mostly game-theoretic draws, not bugs; give the net the rules and inputs to see them.
- Check the lr scale against the optimizer; KataGo's defaults assume a mature net (SGD) or need tuning (Aurora).
- Change one thing at a time when comparing (optimizer, lr, size, data) and compare at equal data where possible.
- Launch scripts must use the venv; check that only one loop runs.
- Measure the standard game with standard openings: a random-opening ladder can hide large progress.
- On the same data and start, compare held-out (SWA) losses first; ladders are noisy and slow.
