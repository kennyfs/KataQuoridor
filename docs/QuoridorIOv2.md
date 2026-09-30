# Quoridor I/O v2: rules, scoring, inputs and self-play

*Design document, started 2026-09-30. It is the source of truth for the three steps of the I/O v2 retrain:*

1. *rules and scoring (game rules, terminal scores, komi, fence handicap, QTP/SGF). **Done**, see
   [§6](#6-step-1-where-things-live);*
2. *the I/O v2 neural net (inputs, heads, training data, exporter, loader);*
3. *self-play randomization and experiments.*

*Steps 2 and 3 are sketched here and will be refined when they start. Where this document and the older
[roadmap](KataQuoridor_Review_and_Roadmap.md) (§5 Phase 6, "margin komi `k`") disagree, this document wins.*

All quantities are from **White's perspective** unless noted: positive favours White.

---

## 1. Motivation

KataQuoridor 0.1.0 has three problems that come from the game's objective rather than from the net:

- **The winner doesn't finish.** In the 0.1.0 Elo ladder, 0.7% of games hit the 300-ply cutoff, and the cases we
  examined were won positions that the winner never closed out. Once the win rate saturates at 100%, only the score
  utility ranks moves. The score is the margin (the loser's remaining distance), so "place more walls, win later
  by more" looks at least as good as "win now". Against an opponent that just blocks a corridor, a low-visit search
  then shuffles until the cutoff. Self-play discards such games, so the net never learns from them either.
- **First-player advantage.** Between strong players Black (the first player) wins about 59–65% of games. There
  is no komi to balance it, so value targets near the start are lopsided and we have no way to measure "how much"
  a position is worth in tempo.
- **The cutoff lives outside the game.** Self-play, the gatekeeper and the arena each cut games at 300 plies, but
  `BoardHistory` and search don't know about it. Search can't see that a shuffle ends in a draw.

We prefer to fix these through **rules and objectives** that the search and the net can learn from, not through
hand-coded endgame logic:

- no analytic endgame solver ("if the race is won, play the shortest path");
- no move pruning ("never step back", "always take an immediate win").

Such rules are hard to get right with walls and jumps, they would only fix the symptoms we have seen, and they
would make the engine play by our heuristics instead of by search. Instead:

1. the **300-ply draw** becomes a game rule, so search and training see it;
2. a **tempo score** with a half-integer **komi** gives a meaningful lead and a way to balance the first-player
   advantage (and to set handicaps);
3. a small **time bonus** in the utility score rewards the winner for finishing earlier and the loser for lasting
   longer, so "win now" beats "win later";
4. a **fence handicap** (fewer initial walls for one side) gives a second, coarser way to handicap or balance games.

## 2. Definitions

### 2.1 Margin (unchanged)

When a pawn reaches its goal row, the **margin** is

    whiteMargin = +max(1, dB)   if White's pawn reached its goal,
                  -max(1, dW)   if Black's pawn reached its goal,

where `dB`, `dW` are the loser's shortest-path distances to its goal row (walls only, pawns ignored). This is the
0.1.0 margin, `Board::whiteMarginWhenWonBy`. In SGF notation W+d / B+d.

### 2.2 Tempo `t` (integer)

    t = whiteMargin       if White's pawn reached its goal   (W+d  ->  t = d)
    t = whiteMargin + 1   if Black's pawn reached its goal   (B+d  ->  t = 1 - d)

So W+1 → 1, B+1 → 0, B+2 → −1, B+3 → −2. One step of `t` is exactly one tempo (one move of the race):

- in an equal race (both need `D` moves), Black, moving first, arrives when White still needs 1: B+1, `t = 0`;
- W+1 means White, moving second, was one move ahead: `t = 1`.

### 2.3 Komi and the winner

`Rules::komi` is a **half-integer** (`n + 0.5`), `|komi| <= 20.5`. **White wins iff `t + komi > 0`**, otherwise
Black wins.

- The **standard game has `komi = -0.5`**: B+1 is a Black win and W+1 a White win, exactly as before.
- Positive komi favours White. With komi, the side whose pawn reached the goal can lose: the game still ends the
  moment a pawn arrives, and `t + komi` decides the winner.
- Komi never causes a draw (it is a half-integer). Draws come only from `maxPlies`.

### 2.4 Lead score `s`

    s = t + komi                  (half-integer, never 0 for a finished-by-goal game)

W+1 → +0.5 and B+1 → −0.5 in the standard game. `s` is the quantity KataGo's `lead` and komi machinery work with:
"how much komi would have to change for the result to flip". `s = 0` for a draw.

### 2.5 Draw by the ply limit

`Rules::maxPlies` (default **300**). Plies are counted from the real game start:
`ply = BoardHistory::initialTurnNumber + moveHistory.size()` (`BoardHistory::getCurrentTurnNumber()`).

- If the game reaches `maxPlies` plies without a pawn reaching its goal, it ends as a **draw**:
  `isGameFinished = true`, `winner = C_EMPTY`, `isNoResult = false`, `isScored = true`, `s = u = 0`.
- A goal reached **on** ply `maxPlies` is a win, not a draw.
- `BoardHistory` decides it after every move, so search sees draws at terminal nodes exactly like wins.
- A draw is a normal result, not "no result". Its value is `ScoreValue::whiteWinsOfWinner(C_EMPTY,
  drawEquivalentWinsForWhite)`, i.e. win 0.5 / loss 0.5 with the default `drawEquivalentWinsForWhite = 0.5`.
- `BoardHistory::pliesUntilDraw() = max(0, maxPlies - ply)`. The v2 input feature "moves until draw" uses it.

### 2.6 Time bonus and the utility score `u`

`Rules::timeBonusPerPly` λ (default **0**, config key `timeBonusPerPly`, range 0..1). With `T` the number of plies
at game end (counted as above),

    u = s + sign(s) * λ * (maxPlies - T)         (u = 0 for a draw)

- `u` is KataGo's **score** (`scoreMean`): terminal nodes, score utility and the training score target use it.
- It rewards the winner for finishing earlier and the loser for lasting longer.
- A win is always better than a draw, and a draw better than a loss: `|u| >= |s| >= 0.5 > 0`.
- λ is a **property of the training run**. The net's score head learns `E[u]` for the λ it was trained with, and
  search must use the same λ, or leaf evaluations and terminal values disagree. That's why λ lives in `Rules`
  (like komi), where search and the training-data writer see the same value. Don't change λ for a net that was
  trained with another λ.

### 2.7 Fence handicap

`Rules::blackInitialFences` / `whiteInitialFences` (default 10 / 10, range 0..`Board::MAX_FENCE_NUM` = 10; config,
QTP and JSON keys `blackInitialWalls` / `whiteInitialWalls`). They are applied when a game is set up
(`clear_board`, self-play game init, SGF load). Nothing in search or the net changes: the remaining walls are
already inputs.

### 2.8 Examples

Standard game (`komi = -0.5`, `maxPlies = 300`); `u` with λ = 0.05:

| End of game | margin | `t` | `s` | winner | `u`, λ = 0.05 | SGF `RE` |
|---|---:|---:|---:|---|---:|---|
| White's pawn arrives on ply 60, Black 1 away | W+1 | 1 | +0.5 | White | 0.5 + 0.05·240 = +12.5 | `W+0.5` |
| White's pawn arrives on ply 80, Black 3 away | W+3 | 3 | +2.5 | White | 2.5 + 0.05·220 = +13.5 | `W+2.5` |
| Black's pawn arrives on ply 59, White 1 away | B+1 | 0 | −0.5 | Black | −0.5 − 0.05·241 = −12.55 | `B+0.5` |
| Black's pawn arrives on ply 41, White 3 away | B+3 | −2 | −2.5 | Black | −2.5 − 0.05·259 = −15.45 | `B+2.5` |
| Black's pawn arrives on ply 299, White 3 away | B+3 | −2 | −2.5 | Black | −2.5 − 0.05·1 = −2.55 | `B+2.5` |
| Ply 300, nobody arrived | — | — | 0 | draw | 0 | `0` |

Non-standard komi (the winner can be the side whose pawn did *not* arrive):

| End of game | `t` | komi −1.5 | komi −0.5 (standard) | komi +0.5 | komi +1.5 |
|---|---:|---|---|---|---|
| W+2 | 2 | `s = +0.5`, White | +1.5, White | +2.5, White | +3.5, White |
| W+1 | 1 | `s = −0.5`, **Black** | +0.5, White | +1.5, White | +2.5, White |
| B+1 | 0 | −1.5, Black | −0.5, Black | `s = +0.5`, **White** | +1.5, **White** |
| B+2 | −1 | −2.5, Black | −1.5, Black | −0.5, Black | `s = +0.5`, **White** |
| B+3 | −2 | −3.5, Black | −2.5, Black | −1.5, Black | −0.5, Black |

### 2.9 Defaults keep v1 behaviour

With `maxPlies = 300`, `komi = -0.5`, `timeBonusPerPly = 0` and 10/10 walls, terminal scores become
`s = u = whiteMargin ∓ 0.5` (W+d → d − 0.5, B+d → −(d − 0.5)). Compared with 0.1.0 that is a ±0.5 shift towards 0,
which is harmless for v1 nets: they still predict the margin, and search only compares scores. The other visible
change is the 300-ply draw rule in `BoardHistory` (QTP `winner` reports `Draw`, search sees draws). Existing I/O v1
nets (e.g. the 0.1.0 release net) keep loading and playing with the default settings.

Using non-default komi or λ > 0 **with a v1 net** is possible but not meaningful: v1 nets have no komi input and
their score head doesn't include a time bonus, so only terminal nodes would see the change.

## 3. Display: `scoreMean` vs `scoreLead`

- `scoreLead` (KataGo's `lead`) is the **tempo lead** `s`: "White is ahead by 1.5 tempo".
- `scoreMean` (KataGo's score, internally) is the **utility score** `u`: it includes the time bonus, so with λ > 0
  it can read around λ·maxPlies (e.g. +15 for λ = 0.05) at the start while the lead is about 0. That is expected.
- **GUIs should show `scoreLead`.** At terminal search nodes `lead = s` while `scoreMean = u`.
- Output field names follow upstream KataGo: `kata-analyze` and the analysis engine print the utility score as
  `scoreSelfplay`, and their `scoreMean` field is a copy of `scoreLead` (upstream's compatibility choice);
  `kata-raw-nn` prints `whiteLead` and `whiteScoreSelfplay`.
- In the standard game (komi −0.5) an equal race reads `scoreLead ≈ −0.5`, and a lead of +0.5 means "White wins by
  one tempo".

## 4. Step 2: Quoridor I/O v2 neural net

**Done**, see [§8](#8-step-2-where-things-live) for where each part lives. This section is the reference for the
v2 tensor layout and targets.

### 4.1 I/O versions

The model header's option D is the **Quoridor I/O version**: 1 (the KataQuoridor 0.1.0 nets) or 2.
`QuoridorNN::MAX_SUPPORTED_IO_VERSION = 2`; `QuoridorNN` fills the inputs of either version, chosen by the net.

- **v1 nets stay supported for inference** (GTP, analysis, arena, match): v1 inputs, and their one margin head is
  exported as both `scoreMean` and `lead`. With a v1 net `scoreMean` and `lead` are therefore the same number, and
  they **ignore komi and the time bonus** (v1 nets have no komi input and were trained on the margin).
- **Training data and training are v2 only** (`QuoridorNN::TRAINING_IO_VERSION = 2`,
  `modelconfigs.QUORIDOR_TRAINING_IO_VERSION`). `TrainingWriteBuffers` / `TrainingDataWriter` throw for any other
  version, the loss asserts on a v1 model, and the loader rejects data with the v1 channel counts. Old v1 training
  data cannot be mixed in (other inputs and targets).

### 4.2 Inputs

v2 = the v1 features followed by the new ones; the channel indices of v1 are unchanged. All from the side to move's
canonical view (its goal row on row 0), before the mirror symmetry.

| Spatial | Feature | Symmetry (x-mirror) |
|---:|---|---|
| 0 | on-board mask (all ones) | c → 8 − c |
| 1, 2 | my pawn, opponent's pawn | c → 8 − c |
| 3, 4 | blocked towards my goal (north), away from it (south) | c → 8 − c |
| 5, 6 | blocked east, west | c → 8 − c, and 5 ↔ 6 |
| 7 | my goal row | c → 8 − c |
| 8–11 | BFS distances / 32: to my goal, to the opponent's goal, from my pawn, from the opponent's pawn (continuous; in training data stored raw in `spatialDistNCHW`) | c → 8 − c |
| 12, 13 | my / the opponent's shortest-path cells | c → 8 − c |
| 14, 15 | placed vertical / horizontal walls (anchor grid) | c → 7 − c |
| 16 | wall-anchor domain (8×8) | c → 7 − c |
| **17, 18** | **v2: geometrically legal vertical / horizontal wall placements** (anchor grid) | c → 7 − c |

Channels 14–18 live on the 8×8 anchor grid in the top-left of the 9×9 plane (row and column 8 are 0); anchor rows
flip as `7 − r` when White is to move. "Geometrically legal" is `Board::isGeometricallyLegalWallPlacement`: legal for
a player with at least one fence (no overlap or crossing with a placed wall, no full block), whatever the fence
counts, which are global inputs. It uses the lazy BFS: the overlap checks and the cached shortest paths decide most
anchors, and only a wall that cuts a cached path runs a BFS.

| Global | Feature |
|---:|---|
| 0 | White to move |
| 1, 2 | my / the opponent's fences / 10 |
| 3–6 | my fences, `exp(−(n − 1) / s)` for s = 1, 2, 4, 8 (0 if none) |
| 7 | the opponent has a fence |
| 8–11 | the opponent's fences, as 3–6 |
| 12 | jump parity (±1) |
| 13, 14 | my / the opponent's shortest distance / 32 |
| **15** | **v2: plies until the draw**, `BoardHistory::pliesUntilDraw() / 300` |
| **16** | **v2: komi from the side to move's view**, `BoardHistory::currentSelfKomi(pla, …) / 5` |

- Global 15 is on an **absolute scale** (`/ 300`, not `/ maxPlies`), so the input keeps meaning "plies left" when
  `maxPlies` changes. It is 1.0 at the start of a standard game and 0 at or past the limit.
- Global 16: the standard komi −0.5 reads **+0.1 for Black and −0.1 for White** (negative komi favours Black).
- The NN cache hash already includes the rules and the ply count (§6.2), so no hash change was needed.
- The mirror symmetry is applied by `QuoridorNN::applyInputSymmetry` (C++) and `apply_symmetry_quoridor` (Python
  loader) with the same channel lists; the C++ test compares a mirrored game's rows channel by channel.

### 4.3 Training targets

Global targets (`globalTargetsNC`, `cpp/dataio/trainingwrite.h`), from the side to move's view, as written by
`TrainingWriteBuffers::addRow`:

| Column | Target | Weight |
|---:|---|---|
| C0–C1 | final value (win, loss); a draw is 0.5 / 0.5 | `1 − C35` |
| C4–C19 | TD value targets; C15 is the short-term TD **score**, the searches' utility score `u` | `1 − C24` |
| **C20** | **the game's final utility score `u`** (`finalWhiteMinusBlackScore`); 0 for a draw | C27 |
| **C21** | **the game's final tempo lead `s`** (`finalWhiteLead`); a search's lead estimate for the position if there is one (`estimateLeadProb`, 0 in the Quoridor configs) | C29 |
| C22 | expected arrival time of the win/loss variance (`varTimeLeft`, unchanged) | C27 |
| **C23** | **plies from this row to the end of the game** (`endHist` ply − row ply), draws included (a draw ends at `maxPlies`) | C27 |
| C27 | outcome weight: the value weight on main rows of a game with a result, draws included; 0 on side positions and on reanalyzed rows without outcome targets | |
| C29 | lead weight: as C27, but **0 for a draw** (a draw carries no tempo information); set by a lead estimate | |
| C47 | komi from the side to move's view (unchanged) | |
| C52, C62 | hit the turn limit (0 for a draw), game finished and not a side position (1 for a draw) | |

C23 is unused in upstream KataGo and was unused before. The trajectory and final-wall targets (`valueTargetsNCHW`)
are weighted by C27, so draws train them with normal weight.

### 4.4 Heads and losses

`QuoridorValueHead` (`python/katago/train/model_pytorch.py`); metric keys as in `metrics_train.json`:

| Head | Module | Target | Loss (weight) | Metric |
|---|---|---|---|---|
| value | `linear_value` | C0–C1 | cross-entropy × 1.5 | `vloss` |
| TD value | `linear_td_value` | C4–C19 | cross-entropy | `tdvloss*` |
| utility score (`scoreMean`) | `linear_utility_score` | `u`, C20 | Huber δ = 1, × 0.04 (C27) | `smloss` |
| its stdev | `linear_misc[0]` | squared error of the utility score | Huber δ = 10, × 0.004 (C27) | `sdregloss` |
| lead | `linear_lead` (v2) | `s`, C21 | Huber δ = 1, × 0.04 (C29) | `leadloss` |
| remaining plies | `linear_remaining_turns` (v2, training only) | C23 / 300 | Huber δ = 0.25, × 1.0 (C27) | `rtloss` |
| short-term win/loss error | `linear_misc[1]` | squared error of the short-term TD value | Huber δ = 0.4, × 2 | `evstloss` |
| short-term score error | `linear_misc[2]` | squared error of the utility score vs C15 | Huber δ = 25, × 0.002 | `esstloss` |
| variance time | `linear_variance_time` | C22 | Huber δ = 5, × 0.01 (C27) | `vtimeloss` |
| trajectory, final walls | `conv_trajectory`, `conv_wall_graph` | `valueTargetsNCHW` | BCE × 0.02 (C27) | `trajloss`, `wallloss` |

"Margin" now only means the terminal distance of §2.1. In v1 checkpoints the score head is named
`linear_game_margin`; `load_model.load_model_state_dict` renames it on load. The postprocessed Quoridor outputs are
`(policy, value, td_value, variance_time, utility_score, utility_score_stdev, shortterm_value_error,
shortterm_score_error, trajectory, wall_graph, lead, remaining_turns)`; for a v1 model `lead` is the utility score
and `remaining_turns` is 0.

### 4.5 Export mapping

The exporter writes option D from the config's `quoridor_io_version`. The v17 value head's `scoreValue` channels:

| Channel | v2 | v1 |
|---:|---|---|
| 0 | utility score `u` mean | margin |
| 1 | utility score stdev (pre-softplus) | margin stdev |
| 2 | lead `s` | margin (the same head as 0) |
| 3 | varTimeLeft | varTimeLeft |
| 4 | short-term win/loss error | same |
| 5 | short-term score error | short-term margin error |

Post-process multipliers: `scoreMean` and `lead` 1 (moves), score stdev 2, varTimeLeft 1, short-term value error
0.25, short-term score error 4 (unchanged from v1). The remaining-plies head is not exported. `nneval.cpp` needed
no change: it already reads channel 2 as the lead. The value channels stay (win, loss, noResult) with noResult forced
off; a draw is trained as win 0.5 / loss 0.5, not as a no-result.

### 4.6 Draw games in self-play

`Play::runGame` no longer marks a `maxPlies` draw as `hitTurnLimit` (`Play::DISCARD_MAX_PLIES_DRAWS` is gone). A
draw is a normal finished game: its rows are written (value 0.5 / 0.5, `u` = 0 with weight, no lead, remaining
plies to `maxPlies`, trajectory and final walls), side positions are searched and reanalysis runs as for any game.
Side-position rows of any game have no outcome targets (C27 = C29 = 0 unless a search estimated a lead). Forks work
on draws as on other games (`maybeForkGame` scores a finished candidate draw 0 and replays with the ply count kept).
The self-play log line `Game stats for ...` reports `finished normally N (draws D, draw rate r)`; "hit cutoff" now
counts only games stopped by `maxMovesPerGame` before their end, which can't happen while it equals `maxPlies`.

### 4.7 Model presets

`b2c64_quoridor_v2` and `tf2_b4c192_quoridor_v2` = the v1 presets plus `"quoridor_io_version": 2`
(`modelconfigs.get_quoridor_io_version`, default 1). `is_quoridor()` stays keyed on `"game": "quoridor"`, and the
presets without `_v2` stay v1 (for existing nets). **The retrain must use a `_v2` preset** (`-model-kind`).

## 5. Step 3: self-play

*Implemented except the λ experiments, which need I/O v2 nets (step 2). Code locations in
[§9](#9-step-3-where-things-live).* Every new randomization and compensation is **off by default in code**, so
GTP, `match`, the gatekeeper and the step-1 configs create the same games as before; they are on only in
`cpp/configs/training/selfplay_quoridor_v2.cfg`. (The gatekeeper's scoring of draws changed, §5.4.)

### 5.1 Komi and fence-handicap randomization

For games from the empty board (not forks, side positions or SGF start positions), `GameInitializer` draws,
**independently**:

| Config key | Default | v2 self-play | Meaning |
|---|---|---|---|
| `quoridorKomiRandomProb` | 0 | 0.3 | Probability of a non-standard komi `komiMean ± n` (sign 50/50). |
| `quoridorKomiRandomWeights` | `0.6,0.3,0.1` | same | Relative weights of n = 1, 2, 3, … (up to 20 entries). |
| `quoridorFenceHandicapProb` | 0 | 0.1 | Probability that one side (50/50) starts with `n` fewer walls; the other keeps its walls. |
| `quoridorFenceHandicapWeights` | `0.6,0.3,0.1` | same | Relative weights of n = 1, 2, 3, … (up to 10 entries). |

With `komiMean = -0.5` the v2 self-play komi is −0.5 in 70% of normal games and −1.5 / +0.5 (18%), −2.5 / +1.5
(9%), −3.5 / +2.5 (3%) otherwise; 10% of normal games have 9, 8 or 7 walls (60/30/10) for one side. The komi goes
into `ExtraBlackAndKomi::komiMean` (so the policy-init and compensation steps of `Play::runGame` start from it) and
the walls into the game's `Rules` (so SGF `WB`/`WW` and the training data's rules carry them). Fork games keep the
komi and walls of the game they fork from. Measured over 40000 sampled games (`runtests quoridorselfplay`, fixed
seed): random komi 29.96%, n = 1/2/3 60.8/29.7/9.5%, sign 49.6% positive; fence handicap 10.19%, n = 1/2/3
59.9/30.2/9.9%, Black handicapped 51.0%; both 3.18% (independent: 3%). In a 400-game smoke run with the v2 config,
the SGFs had 69% `KM[-0.5]` (forks included) and 12% non-10/10 walls.

**Why new keys rather than upstream's `komiStdev` / `komiBigStdevProb` / `komiBigStdev`:** those draw a truncated
Gaussian (scaled by board size), randomly rounded to the komi grid. They can't give "exactly standard in 70% of
games, otherwise ±1..3 with given weights": a Gaussian's rounding puts much of its mass back on −0.5, its tail goes
past ±3, and the fraction of standard games and the offset weights can't be set separately. The upstream keys still
work (applied on top, with the Quoridor komi rounding), and are 0 in the Quoridor configs.

### 5.2 Komi compensation

| Config key | Default | v2 self-play | Games |
|---|---|---|---|
| `forkCompensateKomiProb` | = `handicapCompensateKomiProb` (upstream) | 0.8 | Fork games (early forks and forks): a fair komi for the fork position. |
| `handicapCompensateKomiProb` | 0 | 0.5 | Fence-handicap games: a fair komi after the handicap is applied (it replaces a random komi). |
| `compensateKomiVisits` | 20 | 20 | Visits per evaluation of the fair-komi search. |

A fair komi comes from `PlayUtils::adjustKomiToEven`: a few steps along the lead, then a binary search on the
winrate over the Quoridor komi grid (steps of 1), interpolated between the two neighbouring komis and randomly
rounded to one of them (`PlayUtils::findEvenKomi`, `roundKomiRandomly`).

**Compensation window.** If the fair komi is more than 5 tempi from the standard −0.5 (outside [−5.5, 4.5],
`PlayUtils::MAX_COMPENSATED_KOMI_DELTA`), the game keeps its komi (`PlayUtils::compensatedKomiOrKeep`). An
untrained lead head doesn't track komi, and the search then runs to the edge of the komi range: in a smoke run
with a random-init v2 net, every compensated game got komi +20.5, where Black can practically only play for the
`maxPlies` draw, which is the stalling this design is meant to remove.

**Only with nets that see komi.** `Play::runGame` compensates only if both nets have Quoridor I/O version ≥ 2
(`PlayUtils::nnEvalSeesKomi`, from the model header). With an **I/O v1 net it is a no-op**, logged once
("WARNING: skipping komi compensation …"): the game keeps the komi `GameInitializer` gave it. v1 nets see komi only
at terminal nodes, so a fair-komi search with them would drift. The same gate covers `compensateAfterPolicyInitProb`
and `fancyKomiVarying` (both 0 / off in the Quoridor configs). A no-op rather than an error, so the v2 config can
be smoke-tested with the 0.1.0 net. Test gap: there is no I/O v2 test net yet, so `runtests quoridorselfplay` tests
the search on a synthetic evaluator that sees komi (lead = t₀ + komi; it finds the fair komi −t₀ within 0.2 for t₀
from −12.2 to +19.6) and the no-op with the random test net, which reports I/O v1 (`MAX_SUPPORTED_IO_VERSION`).
When step 2 raises that to 2, the same test runs the real compensation with the random net instead.

### 5.3 The v2 configs

`cpp/configs/training/selfplay_quoridor_v2.cfg` is `selfplay_quoridor.cfg` with:

- `maxPlies = 300`;
- `timeBonusPerPly = 0.05`, a **placeholder**: λ is a property of the training run and will be chosen by experiment
  (0 / 0.05 / 0.15 on a small net);
- `staticScoreUtilityFactor = 0` (dynamic score utility only; `dynamicScoreUtilityFactor = 0.30` as before): with
  λ > 0, `u` at the start is about λ·maxPlies (+15 for 0.05), which the dynamic utility re-centres;
- the randomizations and compensations of §5.1 and §5.2.

`gatekeeper_quoridor_v2.cfg` is `gatekeeper_quoridor.cfg` with the same `maxPlies`, `timeBonusPerPly` and
`staticScoreUtilityFactor = 0`. `gtp_quoridor.cfg` keeps the standard rules, with a comment that `timeBonusPerPly`
must be the λ the net was trained with.

### 5.4 Gatekeeper

- A game without a winner (the `maxPlies` draw, or a game cut off early) scores **exactly 0.5** for each side
  (`PlayUtils::whitePointsOfGame`), no longer `0.5·noResultUtilityForWhite + 0.5` (this settles the §7 question).
  Rule draws, cut-off games and Black's wins are counted and logged separately:
  `Game stats for A vs B: N games, D decisive (K black wins), R draws by maxPlies, C cut off, avg game length L`.
- Gatekeeper games are always the **standard game** (komi −0.5, 10/10 walls): the gatekeeper refuses a config that
  can create anything else (`GameInitializer::mayCreateNonStandardGames`: komi or fence randomization, komi noise,
  `komiAuto`, a non-standard `komiMean` or initial walls).

### 5.5 Self-play stats

Every `logGamesEvery` started games, and at a net's final cleanup, self-play logs per model, next to the
started / finished / cutoff line:

    Quoridor stats for <model>: completed N, draws D (rate r), avg plies P, black win rate in normal standard games
    b (k/n), by komi (10/10 walls): -1.5 0.778 (21/27) -0.5 0.622 (102/164) +0.5 0.385 (10/26) ..., fence handicap
    f (k/n)

All completed games count, including those not written to training data (draws while
`Play::DISCARD_MAX_PLIES_DRAWS` is true). "Normal" games are `FinishedGameData::MODE_NORMAL` (not forks); plies
are counted from the real game start. Training data is unchanged by step 3.

### 5.6 Arena and viewer

- `python/quoridor_arena`: komi and initial walls per game, globally (roster `"rules"`, `--komi`,
  `--black-walls`, `--white-walls`) or per pair (roster `"pair_rules"`), sent through QTP to both engines and the
  arbiter. Engines without support (SimpleQuoridor; roster `"supports_rules"`) only play standard games. SGFs take
  `KM` / `WB` / `WW` / `RE` from the arbiter's `printsgf` (the `KM[0]` / integer-margin workaround of §7 is gone),
  and an arbiter rule draw is a draw. Details in [Evaluation.md](Evaluation.md#komi-and-fence-handicap).
- `python/sgfs_viewer`: komi and initial walls in the game list and info panel when not standard, walls counters
  from the initial walls, and win-rate slices by komi and by fence handicap on the stats page.

### 5.7 Still to do (after merging step 2)

- **λ experiments:** self-play runs with `timeBonusPerPly` 0 / 0.05 / 0.15 on a small v2 net, the gatekeeper with the
  same λ per run; compare draw rate and average plies vs weak opponents (arena), first-player win rate by komi
  (self-play stats), and Elo (arena, standard game). Retune `dynamicScoreCenterScale` if the score utility is too
  weak or too strong with λ > 0 (§7).
- Check the compensation end to end with a v2 net (the log should no longer warn, fork games should end nearer
  50% by komi bucket).
- If draws are written to training data (step 2), the self-play stats need no change (they already count draws).

## 6. Step 1: where things live

### 6.1 Rules (`cpp/game/rules.{h,cpp}`)

| Field | Default | Config / QTP / JSON key | Changeable mid-game |
|---|---|---|---|
| `komi` | −0.5 | `komi` (config), QTP `komi`, SGF `KM` | yes (`komi`) |
| `maxPlies` | 300 | `maxPlies` | only before the first move |
| `timeBonusPerPly` | 0 | `timeBonusPerPly` | yes |
| `blackInitialFences` | 10 | `blackInitialWalls`, SGF `WB` | only before the first move |
| `whiteInitialFences` | 10 | `whiteInitialWalls`, SGF `WW` | only before the first move |

- `Rules::isValidKomi` (half-integer, `|komi| <= Rules::MAX_KOMI` = 20.5) and `Rules::roundKomi` (to the nearest
  valid komi) are the single definition of a legal komi.
- `Rules::toString()` is `Quoridor` for the standard game, and `Quoridor:key=value,...` listing the non-default
  keys otherwise (e.g. `Quoridor:maxPlies=400,timeBonusPerPly=0.05`); `toStringNoKomi()` leaves komi out and is
  what SGF `RU` holds. JSON uses the keys above (`kata-get-rules` prints the JSON without komi). Parsing accepts
  both forms plus `quoridor` / `default`; unknown keys are errors, the Go keys (`ko`, `scoring`, …) are accepted
  and ignored in JSON input. The Go rule fields remain as inert stubs.
- `Rules::validateOrThrow` checks the ranges (komi as above, `maxPlies` 1..100000, λ 0..1, walls 0..10).
- `Rules::operator==` compares all Quoridor fields (it used to return `true`).
- Config: `Setup::loadQuoridorRuleKeys` reads `maxPlies`, `timeBonusPerPly`, `blackInitialWalls`,
  `whiteInitialWalls` (gtp and every tool through `Setup::loadSingleRules`, self-play through `GameInitializer`).

### 6.2 Results (`cpp/game/boardhistory.{h,cpp}`)

- `BoardHistory::makeBoardMoveAssumeLegal` checks, after each move: goal reached → win (`scoreGameEndedAtGoal`),
  else `getCurrentTurnNumber() >= maxPlies` → draw. `BoardHistory::isDraw()` and `pliesUntilDraw()` report it.
- Stored results: `finalWhiteMinusBlackScore = u` (what search and training use as the score) and
  `finalWhiteLead = s`.
- `BoardHistory::setKomi` validates the komi and re-scores a finished game (the winner may flip), like upstream.
- `BoardHistory::getSituationRulesAndKoHash` (used by both the graph-search hash and the NN cache) folds in komi,
  `maxPlies`, λ and the ply count. With λ > 0 or near the ply limit, the same position at a different ply has a
  different value, so transpositions must not merge across plies.
- The only change in `cpp/search/`: at a finished game's terminal node, search takes `lead` from
  `finalWhiteLead` instead of copying `scoreMean` (`Search::playoutDescend`).
- `Board::setFencesLeft` applies the initial walls; `Board::checkConsistency` allows fewer walls on the board than
  the players have spent from 10 each.

### 6.3 Upstream komi helpers

| Helper | Go meaning | KataQuoridor |
|---|---|---|
| `Rules::komiIsIntOrHalfInt` | komi on the 0.5 grid | Upstream implementation restored (it was a stub returning `true`); user input is validated with `Rules::isValidKomi` instead. |
| `Rules::gameResultWillBeInteger` | integer komi → draws possible | `false`: the score never decides a draw. |
| `BoardHistory::whiteKomiAdjustmentForDraws` | shifts scores by `drawEquivalentWinsForWhite − 0.5` when draws are possible | Upstream implementation; 0 since `gameResultWillBeInteger()` is false. The draw's value goes through `whiteWinsOfWinner` only. |
| `ScoreValue::whiteScoreDrawAdjust` | score + the above | Unchanged (adds 0). |
| `ScoreValue::whiteScoreMeanSqOfScoreGridded` | spreads an integer (drawable) score over ±0.5 | **No gridding for Quoridor**: returns `score²`. Terminal scores are exact (`u` is not on a 0.5 grid once λ > 0, and a draw's 0 is an exact score, not a tie of a jittered score), so the terminal variance is 0. |
| `BoardHistory::currentSelfKomi` | komi from `pla`'s view (+ draw adjustment) | Upstream implementation: `±komi`. Written to training data and planned as the v2 komi input. |
| `BoardHistory::setKomi` | set komi, re-score a finished game | Implemented (was a no-op); throws on an invalid Quoridor komi. |
| `Search::setKomiIfNew` | | Unchanged (upstream search). |
| `PlayUtils::roundAndClipKomi` | round to 0.5, clip to ±(20 + area) | Rounds to the nearest valid Quoridor komi, clips to ±20.5. |
| `PlayUtils::setKomiWithNoise` / `setKomiWithoutNoise` | random / nearest rounding to the 0.5 grid | With noise: random rounding between the two neighbouring valid komis (so `komiAllowIntegerProb` has no effect). Without: the nearest valid komi. |
| `PlayUtils::adjustKomiToEven`, `computeLead` | binary search on the 0.5 grid; area-scoring parity smoothing | Binary search on the Quoridor komi grid (steps of 1); the final komi is randomly rounded between neighbouring valid komis; no parity smoothing in `computeLead`. Unused until step 3 (and only meaningful with nets that see komi). |
| `CompactSgf::getRulesOrWarn` / `getRulesOrFailAllowUnspecified` | missing `KM` → the caller's default komi | Missing `KM` → −0.5 (the SGF describes a standard game); `WB`/`WW` read. |
| QTP `ignoreGTPAndForceKomi` (config) | forced komi | Validated with `isValidKomi`. |
| `PlaySettings` `dynamicSelfKomiBonus*`, `fancyKomiVarying` | Go self-play komi tricks | Unchanged and off in the Quoridor configs; any komi they produce goes through `setKomi`, which throws on an invalid one. |
| `SgfNode::getKomiOrDefault` | parses `KM`, Go-server quirks | Validates with `isValidKomi`; `KM[0]` (written by 0.1.0) reads as −0.5; Go quirks removed. |

### 6.4 Self-play, gatekeeper and match

- `maxMovesPerGame` (and its alias `cutoffMoves`) must not disagree with `Rules::maxPlies`. **Resolution:**
  `maxPlies` (config key, default 300) is the source of truth. `maxMovesPerGame` defaults to it; if both are set
  and differ, `selfplay`, `gatekeeper` and `match` fail at startup.
- **Draw games are training data** (since step 2, see §4 and §8): written with value 0.5 / 0.5, score 0 and no lead
  target, and reported as draws (not "cutoff") in the self-play stats.
- The self-play `komiMean` must be a valid komi (standard −0.5, also the default); the old `komiMean = 0` is
  rejected at startup. `GameInitializer` starts every game from the config's rules and applies the initial walls.
- `Play::runGame` writes the final value targets' lead as `finalWhiteLead` (it used to copy the score).

### 6.5 QTP and SGF

- QTP: `komi <k>`, `get_komi`; `kata-get-rules` / `kata-set-rule(s)` expose `maxPlies`, `timeBonusPerPly`,
  `blackInitialWalls`, `whiteInitialWalls` (`maxPlies` and the walls only before the first move, since they define
  the game from its start; komi and λ any time, and a komi change re-scores a finished game). `showboard` prints
  `Ply: N (draw at M, K left)`, and komi / λ when not standard. `winner` reports `Draw` after a `maxPlies` draw.
- SGF: `KM` (komi), `WB` / `WW` (initial walls of Black / White, named like `PB` / `PW`, always written), `RU`
  (`Rules::toStringNoKomi()`), and `RE` = `B+|s|` / `W+|s|` (e.g. `W+0.5`, `B+2.5`), `0` for a draw. **`RE` changed
  from integer margins to half-integers.** SGFs without `KM` load as komi −0.5 (and `KM[0]`, which 0.1.0 wrote for
  the standard game, too), without `WB`/`WW` as 10 walls. `python/sgfs_viewer` bins half-integer results on the
  half grid.
- `Board::numStonesOnBoard()` (upstream's lower bound on the turn number of a setup position) returns the number of
  walls on the board, so SGF loads and `set_position` start counting plies at 0 (it used to return 2, the pawns).

## 7. Open questions

- **Ply count of set-up positions.** `set_position` and SGF setup (`AB`/`AW`) carry no ply number; plies then
  count from the number of walls on the board. Self-play SGFs of fork games record `initTurnNum` only in a comment.
  Fine for now; revisit if v2 trains from SGF start positions.
- **Positions already past `maxPlies`** (e.g. a start position at ply 350) are not ended at construction; the next
  move ends the game as a draw.
- **Score-utility scale with λ > 0.** `u` at the start is about λ·maxPlies. With the static score utility the
  offset is harmless (a constant), with the dynamic one it is re-centred. To be measured in step 3.
- **v1 nets and the NN cache.** The NN cache hash now includes the ply count, so a v1 net re-evaluates the same
  position at a different ply (e.g. after a pawn goes back and forth). Measure the cost if it matters.
- *Resolved in steps 2 and 3 (kept here as a record):* the v1 training writer's targets and the draw rows (step 2,
  §8); gatekeeper draws now count exactly 0.5 and the arena writes the arbiter's `KM` / `RE` (step 3, §5, §9).

## 8. Step 2: where things live

| Part | Where |
|---|---|
| I/O versions, input layout, v2 channel constants, symmetry | `cpp/neuralnet/quoridornn.{h,cpp}` (`fillRow`, `applyInputSymmetry`, `TRAINING_IO_VERSION`, `SPATIAL_LEGAL_*_V2`, `GLOBAL_*_V2`) |
| Geometric wall legality | `Board::isGeometricallyLegalWallPlacement` (`cpp/game/board.{h,cpp}`); `isLegalWallPlacement` = fences left + it |
| Model loading | `cpp/neuralnet/desc.cpp`: option D in 1..2 and the input channel counts must match it; `nneval.cpp` picks the inputs by the net's version (unchanged) |
| Training rows, targets, draw rows | `TrainingWriteBuffers::addRow` and `fillQuoridorInputRow` (`cpp/dataio/trainingwrite.{h,cpp}`; the Go target paths were removed) |
| Draws in self-play, stats | `Play::runGame` (`cpp/program/play.cpp`), `SelfplayManager` (`gamesDrawnCount`, `gameStatsSummary`) |
| Tools | `writesampletrainquoridor` (`cpp/command/misc.cpp`), `dumpnninputs -io-version`, `evalnnparity` (`cpp/command/nnparity.cpp`), `scripts/cuda_parity.sh` |
| C++ tests | `runtests nninputs` (legal-wall planes, mirror symmetry, globals), `quoridorscore` (outcome targets of won games and a draw), `trainingwrite` |
| Presets, channel counts | `python/katago/train/modelconfigs.py` (`get_quoridor_io_version`, `QUORIDOR_NUM_*_INPUT_FEATURES`, `*_quoridor_v2`) |
| Heads, postprocessing | `QuoridorValueHead`, `Model.postprocess_single_heads_output` (`python/katago/train/model_pytorch.py`) |
| Losses, metric keys | `Metrics.metrics_dict_batchwise_single_heads_output_quoridor` (`python/katago/train/metrics_pytorch.py`) |
| Loader, symmetry | `python/katago/train/data_processing_pytorch.py` (`QUORIDOR_WALL_ANCHOR_CHANNELS`, channel-count check); v1 key rename in `load_model.py` |
| Exporter | `python/export_model_pytorch.py` (option D, scoreValue mapping, ONNX channel counts) |
| Python tests | `tests/test_quoridor_model.py` (shapes, gradients, draw rows, symmetries), `tests/test_nn_parity.py` (v1 and v2 features and parity), `tests/test_end_to_end_training.py` |
| v1 only | `python/model_viewer` (refuses v2 nets) |

Open points after step 2:

- **Loss weights** of the new terms (lead as the old margin, remaining plies × 1.0 with δ = 0.25 in units of 300
  plies) are first guesses; check the `smloss` / `leadloss` / `rtloss` scales in the first retrain.
- **Short-term score error multiplier** (4, from v1) assumes `|u|` of a few moves; with λ > 0 the short-term score
  error is larger, so the predicted error saturates higher. Revisit with the λ experiments of step 3.
- **Draw share.** A fresh net draws most games (a random net drew 98% at `maxPlies = 40` in the smoke run); at 300
  plies expect many draws early in the retrain. Draw rows have no lead target, so the lead head learns from
  decisive games only.
- **CUDA parity for v2** was not run in step 2 (Eigen FP32 only); `scripts/cuda_parity.sh` covers v1 and v2.

## 9. Step 3: where things live

| What | Where |
|---|---|
| Komi / fence-handicap randomization, config keys | `GameInitializer::initShared`, `createGameSharedUnsynchronized` (`cpp/program/play.{h,cpp}`) |
| Gatekeeper's standard-game check | `GameInitializer::mayCreateNonStandardGames`; `cpp/command/gatekeeper.cpp` |
| Compensation gate for I/O v1 nets | `Play::runGame` (`canCompensateKomi`); `PlayUtils::nnEvalSeesKomi`, `MIN_QUORIDOR_IO_VERSION_SEEING_KOMI` |
| Fair-komi search, random rounding | `PlayUtils::findEvenKomi`, `roundKomiRandomly`, used by `adjustKomiToEven` / `computeLead` (`cpp/program/playutils.{h,cpp}`) |
| Gatekeeper points | `PlayUtils::whitePointsOfGame`; `NetAndStuff::runWriteDataLoop` (`cpp/command/gatekeeper.cpp`) |
| Self-play stats | `SelfplayManager::countQuoridorGameResult`, `quoridorStatsSummary` (`cpp/program/selfplaymanager.{h,cpp}`), called from `cpp/command/selfplay.cpp` |
| Configs | `cpp/configs/training/selfplay_quoridor_v2.cfg`, `gatekeeper_quoridor_v2.cfg`; λ note in `cpp/configs/gtp_quoridor.cfg` |
| C++ tests | `cpp/tests/testquoridorselfplay.cpp` (`runtests quoridorselfplay`) |
| Arena | `referee.py` (`make_rules`, `RulesState`, `play_game(rules=...)`, result from the arbiter's `RE`), `arena.py` (roster `rules` / `pair_rules` / `supports_rules`, `--komi` / `--black-walls` / `--white-walls`, rules in game ids), `elo.py` (Rules column); tests `tests/test_arena.py`, `tests/test_referee_rules.py`, mock engine `tests/mock_engine.py` |
| Viewer | `python/sgfs_viewer/js/core.js` (`rulesOf`), `viewer.js` (tags, info, walls counters), `stats.js` (win-rate slices) |
