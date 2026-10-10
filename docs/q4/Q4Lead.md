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
lead_i  = sum_{j alive, j != i} w_j * (D_j - D_i),   w_j = (p_j + eps) / sum_{k alive, k != i} (p_k + eps)
eps     = q4LeadWeightEps (0.02 in the Q4 configs)
```

- With eps = 0 the weights are the old `p_j / sum p_k`, uniform over the other alive seats if that sum is below 1e-6.
  The old weights jump there: for a seat about to win, the other three have p ≈ 0 and their normalised weights are
  set by tiny numbers (p = (3e-3, 2e-3, 1e-4) gives (0.59, 0.39, 0.02); p = (1e-3, 1e-3, 1e-3) gives thirds), and the
  weights jump to uniform below the fallback. That made the dominant seat's utility noisy and discontinuous (the
  terminal node has the uniform weights; RLeadEps.md).
- With eps > 0 the weights are continuous in p. A seat at p = 0 gets eps / (p + eps) = 1/3 of the weight of a seat at
  p = 0.1 and 2% of the weight of one at p = 1 (eps = 0.02). Terminal nodes use the same formula with their one-hot or
  draw p, so a terminal is consistent with a leaf that has the same p.
- Size of eps (the sweep: the second place's p from 0 to 0.2, the others at 0, distances 0/10/20/30):
  the largest lead change per 0.001 of p is 10 cells for eps = 0 (the jump at the fallback), 0.33 for eps = 0.01,
  0.17 for eps = 0.02 and 0.07 for eps = 0.05. The rigorous bound is `dp * 4S / (9 eps)` (S = the spread of the
  distances), 0.67 cells per 0.001 at eps = 0.02. For the hopeless case (mover at 0.01, winner at 0.97, the other
  opponent at 0.01) the winner's weight is 0.980 with eps = 0, 0.961 (0.01), 0.943 (0.02), 0.895 (0.05). eps = 0.05
  moves a case that already works by 8.5 points, so it is too large; 0.01 leaves a 0.33 step. 0.02 is the smallest
  value with a step under 0.2 cells per 0.001 and a hopeless-seat weight change under 4 points.
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
| `q4LeadWeightEps` | smoothing of the lead's weights (§2): `(p + eps) / sum (p + eps)`; 0 = the old weights | 0.02 (0 in `q4search_test.cfg`) |

`recentLeadCenter[s]` is set at the start of each search as Duel sets `recentScoreCenter`: the root lead of the previous
tree (if it has visits), else the root's NN lead; then `(1 - zeroWeight) * lead`, capped at `±scale * sqrt(NUM_CELLS)`
around the lead.

The old check in `Search::checkParams` that refused the two factors for Q4 is gone. The configs that were forced to 0
when a key is missing (`q4qtp`, `q4tool`) still do so; `q4search_test.cfg` keeps 0 (the existing search tests).

## 5. History

- 2026-10: the lead is on (dynamic 0.30) in `q4_selfplay.cfg`, `q4_match_common.cfg` (the gatekeeper and the match
  configs) and `q4qtp.cfg`. Off in `q4search_test.cfg` and the test configs.
- 2026-10 (RLeadEps.md): `q4LeadWeightEps` = 0.02 in the same configs, a required key like `q4OpponentMode`. The
  old weights (eps 0) are discontinuous at the fallback and at the terminal node; the change was made to fix the
  noise of the dominant seat's lead, not to change the outcome. The GPU noise table and the effect check at eps 0.02
  are pending (RLeadEps.md).
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
  terminal, one alive seat); L1b (eps: eps = 0 equal to the old code on a grid of p and masks; continuity with a stated
  bound; the hopeless seat's weight; terminal vs leaf; eliminated seat); L2 (both factors 0 gives `computeSeatUtility` bit for bit; the dynamic order and bounds; a
  dead seat has no score term; the root centres from the NN; a search with the score terms gives finite leads).
- C++ `runtests q4board`: the move comment with and without `lead=`.
- Python `python/tests/test_q4_record.py`: the comment parser with and without `lead`.
- `node python/q4_sgfs_viewer/tests/check_core.js`: the viewer parser, and the fixture has games with leads.
- Bit-identity: with both factors 0 the games of a fixed q4match config equal those of the previous binary, move for
  move and value for value (the `lead=` field aside).
