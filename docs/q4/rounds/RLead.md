# RLead: the lead in the search utility (effect check)

Design and formula: [../Q4Lead.md](../Q4Lead.md). Branch `q4-lead` (commits `a9431cf2`, `365ec8bd`, `f26e561a`,
`cae9501d`, and the report commit). Binary: a CUDA build of that branch, RelWithDebInfo, the same flags as `cpp/katago`.

## 1. What was built

- Search: each seat's lead from the net's predicted final distances (misc 1..4 × 32) and the win probabilities;
  terminal nodes use the exact distances. The utility adds the static and dynamic score terms (Duel's
  `getScoreUtility`, stdev 0). Dynamic centres per seat as Duel's `recentScoreCenter`. FPU, LCB, virtual loss use the
  full utility range.
- Move comment: `lead=` (root leads of S, W, N, E, one decimal), written by self-play, q4match and the gatekeeper.
- Configs: `dynamicScoreUtilityFactor = 0.30`, `dynamicScoreCenterZeroWeight = 0.25`, `dynamicScoreCenterScale = 0.50`
  (Duel's values) in `q4_selfplay.cfg`, `q4_match_common.cfg` (gatekeeper, match) and `q4qtp.cfg`. Static 0.
- Tools: q4match player spec `search:<model>@visits,dyn=..,static=..`; `python/q4/lead_effect.py`; viewer lead display
  and graph mode.

Changes from the prompt: none in the formula or the parameters. Two details differ from the prompt's wording:
the lead has stdev 0 in the utility (the net predicts only the mean distance, so Duel's score stdev has no source),
and the node keeps `leadAvg` but not `leadSqAvg`, because nothing uses the variance.

## 2. Tests

- `runtests q4board` (comment with and without `lead=`), `q4search` (L1 formula, L2 utility/centres/search), `q4match`
  and `q4selfplay` (all pass with the new configs).
- Bit-identity: with both factors 0, a q4match run (2 players, 4 games, 40 visits, run1 net) reproduces the previous
  binary's games move for move and value for value (the `lead=` field aside), and the same run twice with the
  previous binary is identical.
- Python: `test_q4_record.py` (run directly; pytest in this environment also collects unrelated modules),
  `test_q4_lead_effect.py`. Viewer: `check_core.js` passes with lead-bearing fixture games.

## 3. Effect check

Net: the newest run1 model `q4f-s7411456-d1516463` (copied to the scratch directory). Players of the same net at 100
visits: `on` = dyn 0.30 / static 0, `off` = dyn 0 / static 0. 1200 games in five tables (100 openings each; rotations
as q4match gives them: 2 for ABAB, 4 for the mixed bot tables, 1 for four identical players). Win rates with Wilson 95%
intervals (per table: the whole match tool's output, `match_report.py`, with bootstrap intervals over openings, agrees).

| Table | Players | Result | Wilson 95% |
|---|---|---|---|
| t1 (200 games) | on, off, on, off | on 112 wins: **56.0%** (copies combined); off 88: 44.0% | on [49.1, 62.7]; off [37.3, 50.9] |
| t2on (400) | on, grudge>0, basher, greedy | on 27.3%, grudge 13.5%, basher 30.2%, greedy 29.0% | on [23.1, 31.8]; grudge [10.5, 17.2] |
| t2off (400) | off, grudge>0, basher, greedy | off 29.5%, grudge 17.0%, basher 27.0%, greedy 26.5% | off [25.2, 34.1]; grudge [13.6, 21.0] |
| t3on (100) | on ×4 | self-play, seat wins 23/23/25/29 | — |
| t3off (100) | off ×4 | self-play, seat wins 26/22/26/26 | — |

Reading: between equals, the lead player won 56% of 200 games (p ≈ 0.09, two-sided, not significant), so the lead
did not cost win rate. Against the blockers (table 2) the lead player did not do better than the no-lead player:
27.3% vs 29.5%, intervals overlapping. The grudge bot won less against the lead player (13.5% vs 17.0%), which is the
expected direction, but both differences are within the noise of 400 games. The expected large gain against blockers
was not observed.

### Hopeless seats (`python/q4/lead_effect.py`, from the move comments)

A hopeless move: the mover's own win probability below 2% and walls in hand.

| Table | Seat player | Hopeless moves | Wall share | Self-hurting walls (lengthen own path) | Decided phase (winner > 0.9): plies, share of game |
|---|---|---:|---:|---:|---|
| t3off (self-play-like) | off | 2871 | 7.7% | 29.3% | 36.7 plies, 28.1% |
| t3on (self-play-like) | on | 2778 | 7.5% | 24.0% | 42.2 plies, 33.4% |
| t1 | off | 2875 | 7.3% | 27.3% | 39.1 plies, 30.0% (all games) |
| t1 | on | 2505 | 8.3% | 24.0% | (same table) |
| t2off | off | 2153 | 8.7% | 18.7% | 49.4 plies, 42.3% |
| t2on | on | 2095 | 10.5% | 18.1% | 44.5 plies, 39.1% |

The run1 baseline (8.8% wall share, 28% self-hurting, 48 plies / 41%) matches the no-lead self-play-like table
(t3off), so the population is comparable. The lead did **not** produce the expected clear rise in the wall share of
hopeless seats: 7.5% vs 7.7% in the self-play-like tables; 10.5% vs 8.7% against the blockers (the on-seat is the
only one that walls more), with more than 2000 hopeless moves per side. Self-hurting walls fell from 29% to 24% (self
play) and from 19% to 18% (blockers): the lead player's hopeless walls hurt itself less, which is the opposite of the
prediction's mechanism and is not explained yet. The decided phase is longer with the lead in self-play (42 vs 37
plies) and shorter against blockers (44.5 vs 49.4). None of these is a clean effect, and one run of 100 games per
self-play table is short for the decided-phase numbers.

## 4. Verdict and what to do next

- No measured cost between equals (56% vs 44%, not significant), no measured gain against blockers, no measured change
  in the hopeless-seat wall share. The mechanism the design relies on did not show up at this sample size.
- The defaults in the configs are on (dynamic 0.30) as the prompt asked. The data do not justify that yet; the
  switch is one line per config if you want it off (`dynamicScoreUtilityFactor = 0.0`) before the next run.
- Not run because the prompt's trigger did not occur: the 0.15 variant (it is needed only for a large loss between
  equals). A second look at the hopeless-seat numbers would need more games (the decided-phase sample is 100 games per
  table) and a look at why self-hurting walls fell.

Files: the 1200 games (`effect.sgfs`) and the JSON of the stats are in the session scratch directory, not in the
repository.
