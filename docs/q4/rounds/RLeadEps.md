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

## 4. Noise on the run1 net (GPU, `q4tool evalnn` on the CUDA build)

Net: the newest run1 model `q4f-s9026048-d1785547` (by mtime). Positions: 100 with the mover's own p > 0.97 and 100 with
p < 0.02, sampled (seed 20261010) from the self-play record `q4f-s8162304-d1651071/sgfs/72264716F37540A6.sgfs`, only
search moves before any elimination. Each position in 8 symmetries (the net is not symmetric, so these are the
"small perturbations"; `nnRandomize` was not used). `lead_i` of the mover, population of 8 values per position, sample
std (ddof 1), averaged over the 100 positions of each class.

| Class | eps | mean std | median std | mean lead_i |
|---|---|---|---|---|
| winner (p > 0.97) | 0 | 0.359 | 0.314 | 8.49 |
| winner | 0.02 | 0.357 | 0.305 | 11.65 |
| winner | 0.05 | 0.330 | 0.301 | 11.88 |
| winner | 0.10 | 0.316 | 0.287 | 11.99 |
| hopeless (p < 0.02) | 0 | 0.358 | 0.335 | -12.23 |
| hopeless | 0.02 | 0.353 | 0.327 | -11.92 |
| hopeless | 0.05 | 0.346 | 0.321 | -11.51 |
| hopeless | 0.10 | 0.337 | 0.321 | -10.96 |

**The prediction did not hold.** The winner's lead noise does not drop with eps 0.02 (0.359 → 0.357). Diagnosis: the
old weights are unstable only when the other seats' p sum is tiny. For the winners the median sum of the others' p is
4.3e-3 (min 5e-4), and only 14 % of the positions have it below 1e-3. In the rest the old weights are already smooth
across symmetries, and the spread of `lead_i` comes from the distance outputs (misc), which eps does not touch. The
hopeless seat's noise is unchanged, as expected. The mean lead of the winner moves from 8.5 to 11.6 cells with eps:
the eps 0 mean is pulled by the discontinuous cases, so the eps 0.02 values are the more faithful average.

So the continuity fix is correct in the formula (§3) but does not reduce noise on this sample. Its benefit, if any, is
at the fallback and the terminal node, which this table does not measure.

## 5. Effect check (GPU), table 1 of RLead

Table `t1` = on02, on00, on02, on00 (`q4match` 4-seat table, 100 openings × 2 rotations = 200 games, 100 visits,
`nnRandomize = true`, the newest run1 net). Both players: `dyn = 0.3`, `static = 0`; they differ only in
`q4LeadWeightEps`: on02 = 0.02, on00 = 0. (The per-player key `eps=` is new in q4match, §7 of Q4Lead.md.)

| Player | Wins / 200 | Win rate | Wilson 95 % |
|---|---|---|---|
| eps 0.02 | 95 | 47.5 % | [40.7, 54.4] |
| eps 0 | 105 | 52.5 % | [45.6, 59.3] |

Difference −5 points, z = −0.71, two-sided p ≈ 0.48: no measured loss between equals. Seat wins 39 / 61 / 42 / 58; mean
plies 126.7; no draws. The sample is one table of 200 games, so intervals are about ±7 points: a loss of 5 points
cannot be ruled out (the upper end of the eps-0.02 interval is 54 %, the lower end 41 %, so a loss up to about 7 points
or a gain of about 7 points are both within the noise). This is the table the prompt asked for; no more tables were run.

## 6. Conclusion

- The continuity change is implemented and unit-tested; eps 0.02 costs no measurable win rate at this sample size.
- The noise reduction that motivated it is **not** visible on the run1 self-play positions (§4). Do not claim it.
- Keep the default at 0.02 (it is the continuous version of the same weights, and it is what the tests check), or set
  0 to restore the old behaviour exactly. Any decision on the default is the user's.
