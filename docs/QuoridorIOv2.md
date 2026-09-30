# Quoridor I/O v2: rules, scoring, inputs and self-play

*Design document, started 2026-09-30. It is the source of truth for the three steps of the I/O v2 retrain:*

1. *rules and scoring (game rules, terminal scores, komi, fence handicap, QTP/SGF), see
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
- `scoreMean` is the **utility score** `u`: it includes the time bonus, so with λ > 0 it can read around
  λ·maxPlies (e.g. +15 for λ = 0.05) at the start while the lead is about 0. That is expected.
- **GUIs should show `scoreLead`.** At terminal search nodes `lead = s` while `scoreMean = u`.
- In the standard game (komi −0.5) an equal race reads `scoreLead ≈ −0.5`, and a lead of +0.5 means "White wins by
  one tempo".

## 4. Step 2: Quoridor I/O v2 neural net (plan)

- **New inputs:**
  - legal-wall planes (2 × 8 × 8, from the lazy BFS; cheap);
  - moves until draw, `(maxPlies − ply) / maxPlies` from `BoardHistory::pliesUntilDraw()`;
  - komi, scaled (e.g. `komi / 10`, from the side to move's view: `BoardHistory::currentSelfKomi`).
  The NN cache hash already folds in the rules (komi, `maxPlies`, λ) and the ply count
  (`BoardHistory::getSituationRulesAndKoHash`), so these inputs need no hash change.
- **Heads:** separate heads for the utility score (`scoreMean`, target `u`) and the tempo lead (`lead`, target
  `s`). In v1 both come from one margin head.
- **New auxiliary target** `remaining_turn_num`: plies until the game ends (draws included), next to the existing
  `varTimeLeft`.
- **Draw games are written to training data** with value 0.5 / 0.5 and score 0 (today they are discarded, see
  §6.4). Rows of draw games need a sensible lead target (0) and weight.
- **Exporter, loader, parity:** new I/O version 2 in `quoridornn`, the model header, `export_model_pytorch.py`,
  the Python loader, and the parity test.
- **v1 support policy:** decide whether v1 nets keep being supported (inputs switch on the header's I/O version,
  as today) or are dropped after the retrain. Old v1 training data cannot be mixed with v2 (different targets).

## 5. Step 3: self-play (plan)

- **Komi randomization:** most games standard (−0.5), a fraction with a random half-integer komi.
- **Fence-handicap randomization:** most games 10/10, a fraction with fewer walls for one side.
- **`forkCompensateKomiProb`:** make fork / side positions fair with `PlayUtils::adjustKomiToEven`, which now works
  on the Quoridor komi grid (§6.3).
- **Score utility:** `staticScoreUtilityFactor = 0` (dynamic score utility only), and λ experiments:
  0 / 0.05 / 0.15 on a small net. The score-utility scale (`atan(u / (scale · 9))`) may need retuning once `u`
  includes a time bonus.
- **Metrics:** draw rate, average plies vs. weak opponents (does the engine finish?), first-player win rate, Elo.

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
  keys otherwise (e.g. `Quoridor:maxPlies=400,timeBonusPerPly=0.05`). This is what SGF `RU` holds. JSON uses the
  keys above. The Go rule fields remain as inert stubs; the Go keys are accepted and ignored in JSON input.
- `Rules::operator==` compares all Quoridor fields (it used to return `true`).

### 6.2 Results (`cpp/game/boardhistory.{h,cpp}`)

- `BoardHistory::makeBoardMoveAssumeLegal` checks, after each move: goal reached → win (`scoreGameEndedAtGoal`),
  else `getCurrentTurnNumber() >= maxPlies` → draw.
- Stored results: `finalWhiteMinusBlackScore = u` (what search and training use as the score) and
  `finalWhiteLead = s`.
- `BoardHistory::setKomi` validates the komi and re-scores a finished game (the winner may flip), like upstream.
- `BoardHistory::getSituationRulesAndKoHash` (used by both the graph-search hash and the NN cache) folds in komi,
  `maxPlies`, λ and the ply count. With λ > 0 or near the ply limit, the same position at a different ply has a
  different value, so transpositions must not merge across plies.

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
| `PlayUtils::setKomiWithNoise` / `setKomiWithoutNoise` | random rounding to the 0.5 grid | Random rounding between the two neighbouring valid komis. |
| `PlayUtils::adjustKomiToEven`, `computeLead` | binary search on the 0.5 grid; area-scoring parity smoothing | Binary search on the Quoridor komi grid (step 1); no parity smoothing. |
| `SgfNode::getKomiOrDefault` | parses `KM`, Go-server quirks | Validates with `isValidKomi`; `KM[0]` (written by 0.1.0) reads as −0.5; Go quirks removed. |

### 6.4 Self-play, gatekeeper and match

- `maxMovesPerGame` (and its alias `cutoffMoves`) must not disagree with `Rules::maxPlies`. **Resolution:**
  `maxPlies` (config key, default 300) is the source of truth. `maxMovesPerGame` defaults to it; if both are set
  and differ, `selfplay`, `gatekeeper` and `match` fail at startup.
- **Draw games are still discarded from training data**, as the 0.1.0 cutoff games were: `Play::runGame` marks a
  `maxPlies` draw as `hitTurnLimit` when `Play::DISCARD_MAX_PLIES_DRAWS` is true (the one switch; step 2 flips it).
  They still count as "cutoff" games in the self-play stats.
- The self-play `komiMean` must be a valid komi (standard −0.5); the old `komiMean = 0` is rejected at startup.

### 6.5 QTP and SGF

- QTP: `komi <k>`, `get_komi`; `kata-get-rules` / `kata-set-rule(s)` expose `maxPlies`, `timeBonusPerPly`,
  `blackInitialWalls`, `whiteInitialWalls`. `showboard` prints the ply count and plies until draw, and komi / λ when
  not standard. `winner` reports `Draw` after a `maxPlies` draw.
- SGF: `KM` (komi), `WB` / `WW` (initial walls of Black / White, always written), `RU` (`Rules::toString()`), and
  `RE` = `B+|s|` / `W+|s|` (e.g. `W+0.5`, `B+2.5`), `0` for a draw. **`RE` changed from integer margins to
  half-integers.** SGFs without `KM` load as komi −0.5, without `WB`/`WW` as 10 walls.
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
- **Gatekeeper draws** are counted as `0.5·noResultUtilityForWhite + 0.5` wins for White (unchanged from 0.1.0).
  With a rule draw, 0.5 would be cleaner.
- **The arena** keeps its own `--max-plies` (default 300) and asks the arbiter for the winner first, so a rule draw
  is reported by the arbiter. Its SGFs still write `KM[0]` and integer margins in `RE` (they load as standard
  games); switch them to the engine's `KM`/`RE` when the arena learns about komi.
- **v1 nets and the NN cache.** The NN cache hash now includes the ply count, so a v1 net re-evaluates the same
  position at a different ply (e.g. after a pawn goes back and forth). Measure the cost if it matters.
