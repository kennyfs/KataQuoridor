# RLeadEps: continuous opponent weights in the seat lead

Design and formula: [../Q4Lead.md](../Q4Lead.md) §2 and §4. Branch `q4-lead-eps`, on top of `q4-four-at-a-table` (41f2cfa2).
Commits: `b16beee4` (feature), `5a797c41` (L1b tests, L1 before the net), `fa5825e9` (selfplay test configs pin eps 0),
`e18db303` (docs). All signed.

Status: the CPU part is done (formula, plumbing, tests, the eps study). The GPU part (the noise table on the run1 net
and the effect check) is **not run yet**: the GPU was at 100 % with a training loop, and no CPU substitute was run.

## 1. What changed

- `w_j = (p_j + eps) / sum_{k alive, k != i} (p_k + eps)`, eps = `q4LeadWeightEps`. eps = 0 is the old code, bit for bit
  (the same branch, including the 1e-6 uniform fallback). Terminal nodes use the same function.
- `Search::setLeadWeightEps` (Q4-only, outside `SearchParams`; clears the tree on change); `loadLeadWeightEps` reads the
  required key. Set wherever a Q4 search is built from a config: q4match, q4qtp, self-play and play.
- Configs: 0.02 in `q4_selfplay.cfg`, `q4_match_common.cfg`, `q4qtp.cfg`; 0 in `q4search_test.cfg`. The selfplay test
  configs pin 0.

## 2. Choice of eps

Prompt default 0.05; the study (pure formula, `dp` sweep of the second place from 0 to 0.2, the others at 0, distances
0/10/20/30 cells):

| eps | largest lead change per 0.001 of p | hopeless seat: winner's weight (old 0.980) |
|---|---|---|
| 0 (old) | 10.0 cells (the jump at the fallback) | 0.980 |
| 0.01 | 0.33 cells | 0.961 |
| **0.02** | **0.17 cells** | **0.943** |
| 0.05 | 0.07 cells | 0.895 |

The hopeless case: mover p = 0.01, winner 0.97, another opponent 0.01. At 0.05 the weight on the winner drops 8.5
points in a case the old code already handles, which conflicts with the rule "no change where it works today". 0.02 is
the smallest eps with a step below 0.2 cells per 0.001 and a hopeless-seat change below 4 points. Chosen: **0.02**.

## 3. Unit tests (`runtests q4search`, passes)

- L1 (old hand cases) and L1b: eps = 0 equals a copy of the old code bit for bit over 16 masks and a 7⁴ grid of p
  values; continuity (bound `dp * 4S / (9 eps)`, measured max step 0.083 cells per 0.0005, bound 0.333 per 0.0005,
  old code: 10 cells between p = 0 and 1e-4); the hopeless seat (|Δweight| < 0.04); terminal equals the leaf with the
  same p; a leaf at (0.999, 0.0003 …) is within 0.05 cells of the terminal; eliminated seat.
- Other suites on the branch, all pass: `q4board`, `q4selfplay`, `q4match`. `q4search` (full, including T19–T30 and L2)
  passes after a clean rebuild of `cpp/build-eigen`.

Note: the first q4search run failed in T19 (`bad_alloc`, then an assert in `deleteAllTableNodesMulithreaded`). The
parent commit passes; a clean rebuild of the branch passes. The cause was stale objects in the build directory, not the
change. The L1 tests now run before the evaluator is built (they need no net).

## 4. Pending (GPU)

- Noise table: 100 self-play positions with the mover at p > 0.97 and 100 hopeless, from
  `~/q4/run1/selfplay/q4f-s8162304-d1651071/sgfs/*.sgfs`, with the newest run1 net `q4f-s9026048-d1785547` (by mtime,
  read-only). Std of `lead_i` over the 8 symmetries (`q4tool evalnn`, `sym` 0–7) for eps 0 vs 0.02 (and 0.05).
- Effect check, table 1 of [RLead.md](RLead.md) only: eps 0.02 vs eps 0 at `dyn = 0.3`, 100 openings, Wilson intervals.

Both wait for the GPU. Run them when it is free.
