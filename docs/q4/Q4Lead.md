# Q4 lead: the score terms of the search utility

Status: on by default in self-play, match and qtp from 2026-10 (the values of Duel's lead term). Effect check and
round report: [rounds/RLead.md](rounds/RLead.md).

## 1. Why

A seat whose own win probability is below 2% almost stops placing walls (run1 self-play: 8.8% of its moves, against
about 50% while it still has a chance), and the win/loss utility gives it no preference between moves. The decided
phase (the winner above 0.9) lasted 48 plies, 41% of the game. The lead gives a hopeless seat a reason to keep racing
and to slow down the likely winner, as Duel's lead term does for the losing side. No new head and no new training
target: the net already predicts each relative seat's final distance (misc slots 1–4, [Q4IO.md](Q4IO.md) §5.2).

## 2. The lead of a seat

From one evaluation (an NN leaf or a terminal node):

```
D_s     = 32 * misc[1 + rel(s)]     predicted walls-only distance of seat s at the end (cells); 0 for the winner
p_s     = value[s]                  win probability of seat s (absolute seats; the draw slot is not used)
lead_i  = sum_{j alive, j != i} w_j * (D_j - D_i),   w_j = p_j / sum_{k alive, k != i} p_k
          uniform over the other alive seats if that sum is below 1e-6
```

- A dead seat has no lead, and is not an opponent in the sum.
- With no other alive seat the lead is 0.
- Terminal nodes: `D` is the exact walls-only distance of each alive seat to the centre at the end (the winner 0;
  a draw: the distances at the end), `p` is one-hot (the winner) or the draw slot (then the weights are uniform).
  For a terminal the winner's lead is the mean distance gap to the others; each loser's lead is the winner's distance
  minus its own (negative, the winner takes all).
- Code: `Search::computeSeatLeads` (`cpp/q4/search/q4search.cpp`), the inputs of a stored NN output are
  `NNOutput::finalDistAbs` and `NNOutput::aliveMask` (`initNodeNNOutput`), the terminal distances in `playoutDescend`.

## 3. The utility

Added to the win/loss utility of each alive seat, the way Duel adds its score utility (`Search::getScoreUtility`):

```
U_i = computeSeatUtility(i)                                  (unchanged win/loss part)
    + staticScoreUtilityFactor  * f(lead_i, centre 0, scale 2.0)
    + dynamicScoreUtilityFactor * f(lead_i, centre recentLeadCenter_i, scale dynamicScoreCenterScale)
```

`f` is Duel's `ScoreValue::expectedWhiteScoreValue` with stdev 0 (atan-shaped, in (-1, 1)), and the board scale
`sqrt(NUM_CELLS)` = 11 (the 11 × 11 board, as Duel's `sqrtBoardArea`). The net predicts only the mean distance, so the
stdev of Duel's score term is 0 here. A dead seat gets no score term; its utility stays the loss value.

`utilityAvg` stores the full `U` (per seat, averaged over the children with the same weights as the win/loss
utility), so FPU, LCB, virtual loss and tree reuse read the new utility without change. The utility range (the sum of
the three factors) replaces the win/loss radius in the FPU floor, the LCB prior and the virtual loss; the FPU's loss
value stays `-winLossUtilityFactor`.

Not included: the uncertainty weight of a node uses the short-term win/loss error only (as before); Duel's score
uncertainty is not in Q4 (the net has no short-term score error).

## 4. Parameters and their Q4 meaning

| Key | Q4 meaning | Default in the Q4 configs (2026-10) |
|---|---|---|
| `staticScoreUtilityFactor` | weight of the lead against the centre 0 (the lead as it is) | 0.0 (off) |
| `dynamicScoreUtilityFactor` | weight of the lead against the centre of the root (the lead relative to the root's lead) | 0.30 (Duel) |
| `dynamicScoreCenterZeroWeight` | the centre moves this share of the way towards 0 from the root lead | 0.25 (Duel) |
| `dynamicScoreCenterScale` | scale of both terms: `scale * sqrt(NUM_CELLS)` = 5.5 cells at 0.5 | 0.50 (Duel) |

`recentLeadCenter[s]` is set at the start of each search as Duel sets `recentScoreCenter`: the root lead of the previous
tree (if it has visits), else the root's NN lead; then `(1 - zeroWeight) * lead`, capped at `±scale * sqrt(NUM_CELLS)`
around the lead.

The old check in `Search::checkParams` that refused the two factors for Q4 is gone. The configs that were forced to 0
when a key is missing (`q4qtp`, `q4tool`) still do so; `q4search_test.cfg` keeps 0 (the existing search tests).

## 5. History

- 2026-10: the lead is on (dynamic 0.30) in `q4_selfplay.cfg`, `q4_match_common.cfg` (the gatekeeper and the match
  configs) and `q4qtp.cfg`. Off in `q4search_test.cfg` and the test configs.
- The effect check (RLead.md) compares a 0.30 player with a 0 player of the same net; the results decide whether
  the value stays at 0.30 (if the lead costs win rate between equals, try 0.15 and report both).

## 6. The move comment

`lead=` after `weight=` in the move comment: the root leads of S, W, N, E, one decimal, in cells (absolute seats):

```
C[0.31 0.22 0.27 0.15 0.05 v=600 weight=1.00 lead=1.2,-0.4,-0.3,-0.5]
```

Written by self-play, q4match and the gatekeeper whenever a search ran, also with both factors 0, so the lead can be
measured before it is switched on. Optional for the readers: old comments have no `lead=` (C++ `Q4MoveComment::hasLead`,
Python and the viewer: no `lead` key). See [Q4Sgf.md](Q4Sgf.md) §4.

## 7. Q4 match players

`search:<model>@visits[:mode],dyn=0.3,static=0` overrides the two factors for one player (`dyn`, `static`), so one
table can hold players with and without the lead term.

## 8. Tests

- C++ `runtests q4search`: L1 (the lead formula on hand-made evaluations: weights, uniform fallback, eliminated seat,
  terminal, one alive seat); L2 (both factors 0 gives `computeSeatUtility` bit for bit; the dynamic order and bounds; a
  dead seat has no score term; the root centres from the NN; a search with the score terms gives finite leads).
- C++ `runtests q4board`: the move comment with and without `lead=`.
- Python `python/tests/test_q4_record.py`: the comment parser with and without `lead`.
- `node python/q4_sgfs_viewer/tests/check_core.js`: the viewer parser, and the fixture has games with leads.
- Bit-identity: with both factors 0 the games of a fixed q4match config equal those of the previous binary, move for
  move and value for value (the `lead=` field aside).
