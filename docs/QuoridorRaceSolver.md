# Quoridor pawn race solver

*Added 2026-10-18. Code: `cpp/game/racesolver.{h,cpp}`; tests: `cpp/tests/testquoridorracesolver.cpp`
(`katago runtests quoridorracesolver`); measurement tool: `katago racecheck`.*

Once a player has no walls left, the rest of the game is partly or fully a **pawn race** that can be solved exactly.
[QuoridorIOv2.md §1](QuoridorIOv2.md#1-motivation) ruled out hand-coded endgame logic; that rule is relaxed for this
one case, because a distance rule is not good enough (pawns block each other in corridors, and straight and diagonal
jumps change the race: with both players at 0 walls and equal distances, "B to move wins" was wrong in 11% of the
run4 self-play games), while the real game is small enough to solve exactly.

## 1. The algorithm

With the walls fixed, a game of pawn moves only is a game on the states (Black's square, White's square, side to
move): 2 × 81 × 81 = 13122 states (12960 are real; same square or a pawn on its goal row are not). Moves are the real
pawn rules: a step, a straight jump over the adjacent opponent, a diagonal jump when the straight jump is blocked by
a wall or the edge (`RaceSolution::genMoves`, cross-checked against `Board::getLegalPawnDestinations` on random walls
and pawns by the tests).

A move that puts a pawn on its goal row ends the game. Its value is the lead `s = t + komi` of
[QuoridorIOv2.md §2](QuoridorIOv2.md#2-definitions), computed exactly as `BoardHistory::scoreGameEndedAtGoal` does,
from the loser's walls-only distance (the BFS map `Board::calDistMap`, which ignores pawns). Infinite play is a draw
(`s = 0`), which matches the repetition rule and the `maxPlies` rule when enough plies are left.

Exact minimax of `s` with cycles is solved by retrograde analysis, one boolean game per lead threshold:

- `P(k)` (k > 0): the states from which White can force the game to end with `2s >= k` in finitely many plies. This is
  the attractor of the terminal moves with `2s >= k`, computed with move counters (White states need one winning move,
  Black states need all moves winning). Thresholds are the distinct terminal values, tried from the largest.
- `N(k)` (k < 0): the same for Black forcing `2s <= k`, tried from the most negative.
- A state in `P(k)` for the largest such `k` has value `k` (White wins by `k/2`); in `N(k)` for the most negative
  `k`, value `k`; in neither: draw. By determinacy of reachability games this is the exact value.

Per state the solver stores three numbers (`RaceSolution`):

| | Meaning |
|---|---|
| `val2` | `2s`: odd for a win / loss, 0 for a draw |
| `plies` | plies the game lasts when both play optimally: the value is kept, the winner finishes as fast as possible, the loser delays as long as possible |
| `forced` | plies within which the winner can force a result at least as good as `val2` against *any* play, including a loser who gives up lead to stall (≥ `plies`) |

A pawn with no legal move (boxed in by walls and the other pawn; never seen in the run4 games) makes the state a draw
by convention; nothing is proven there.

**Cost:** about **0.8 ms** per wall configuration on random walls (mean over 100 configurations, max 1.1 ms; 1.4 ms
mean on the wall configurations of real run4 games, which have more walls). 1 solve gives every pawn placement.

## 2. Cache

`RaceSolver::getCached`: a global cache keyed by (placed horizontal and vertical wall bitmaps, 2 × komi), exact keys,
not hashes. It does not depend on the pawns, the walls left, the plies, or the repetition rule. It is thread safe (a
mutex around a map; solving is outside the lock, so two threads may solve the same configuration at once, which is
harmless) and bounded: first in, first out, `quoridorRaceSolverCacheEntries` entries (default 1024, 52 KB each).
Replaying the 10000 run4 self-play games, the hit rate is 92% (218881 hits, 18898 misses). In a 1000-visit search from a
one-sided race position with 4 walls to place it was 89% (3649 hits, 456 misses, 0.45 s of solving).

## 3. Validity (`RaceSolver::evaluate`)

A solved value is used only when:

- the plies fit: `forced <= BoardHistory::pliesUntilDraw()` (a goal on the last ply still wins). Draws are always valid:
  a horizon only turns wins into draws, never a draw into a result;
- the repetition rule cannot spoil it. A winning strategy that always takes the minimal `forced` rank never visits a
  state twice, so the future of a solved line has no repetitions of its own. The only way to a draw is to re-enter a
  state of the *history* since the last wall that has already occurred `repetitionDrawCount - 1` times. **Choice: such
  positions are left unsolved** (the net and search handle them), not re-solved with those states as draw sinks. The
  check sorts the hashes of `positionsSinceLastWall` (without the current position) and looks for a hash with
  `count >= N - 1`. In the run4 replay only 20 of 63615 plies with 0 walls on both sides were left unsolved by the plies-left or repetition checks (no split between the two was recorded).

## 4. Where it is used

| What | Config key | Default |
|---|---|---|
| Search treats solved nodes as terminal leaves | `quoridorRaceSolver` | off |
| ... including proven wins of the only player with walls | `quoridorRaceSolverOneSided` | on (when the solver is on) |
| Size of the solution cache | `quoridorRaceSolverCacheEntries` | 1024 |
| Self-play finishes the game with solver moves | `quoridorRaceSelfplayFinish` | off |

All off by default for old configs. On in `selfplay_quoridor_v2.cfg` (both solver and finish),
`gatekeeper_quoridor_v2.cfg` and `gtp_quoridor.cfg` (solver); `python/quoridor_arena/kq_ladder.py` writes
`quoridorRaceSolver = true` into its match configs; set `"race_solver": false` in the ladder JSON to compare without it.

### Search (`Search::addRaceSolverLeafIfSolved`, called from `playoutDescend` for non-root nodes)

A node is a solved leaf when

- **both players have 0 walls**: exact win / loss / draw and lead;
- **only one player (B) has walls** and B wins the pure race with B restricted to pawn moves: B's win is proven (B can
  only do better with walls; the opponent has no walls). **Lead bound:** the leaf carries the *race lead* as the lead
  and score, a lower bound of B's real lead. B could gain margin with walls, but the win is the win. A race that B
  loses or draws is not proven (B may win with walls), and neither is anything with walls on both sides.

The leaf value is built like a finished game: win / loss from the winner, `scoreMean` = lead plus the time bonus for
the solved plies, `lead` = the lead. The root is never a solved leaf (as `forceNonTerminal`), because a solved root
must still choose a move. Subtree reuse is safe: a solved leaf has no NN output, so it is never promoted to a root.
Graph search shares nodes by the graph hash, which includes the ply, the rules and the position, and chains the path
hash after pawn moves, so a solved value is never shared across different plies or repetition histories.

Solved leaves are exact in value but have no preference among equal values, so a winner could dawdle. Therefore
`Search::getChosenMoveLoc` also restricts the chosen move at an *exactly solved root* to `RaceSolver::optimalMoves`
(moves that keep the value; the winner takes the minimal `plies`, the loser the maximal one), choosing among them by the
search's own selection values. The tests check that the chosen move is optimal on random races.

### Self-play (`Play::runGame`, `quoridorRaceSelfplayFinish`)

Once both players have 0 walls and the position is exactly solved and valid, the game is finished with
`optimalMoves` (ties at random), without search or NN evals. The game still ends with its real moves, so every
end-of-game target stays consistent. For each solved ply the per-turn arrays get an entry with **target weight 0** (no
row is written) that carries the exact value, so the TD value targets of earlier rows see the exact values.
`FinishedGameData::numRaceSolverPlies` counts them. The test plays games with and without the finish and checks the
rows written equal the searched plies and the final targets agree.

### GTP, match, gatekeeper

Search uses the solver, so they get it with no further change. On a solved position `kata-analyze` shows winrate 0 / 1
and the exact lead: from the start position with 0 walls each, after `play B e8` and `play W e2`, run4-s20387072 gives
the same sign as the exact answer, winrate 0 and `scoreLead -0.5` for the side to move (exact), where search without
the solver says winrate 0.003 and lead -0.69. Note that with the solver the root's children are solved, so visits
accumulate very fast (millions per second); use `maxVisits`.

## 5. Measurements

### Self-play vs the exact value (`katago racecheck`, run4-s20387072-d8403664, 10000 games, 6694 standard komi)

Run: `katago racecheck ~/q1_run/run4/selfplay/run4-s20387072-d8403664/sgfs/D22530C258F301AF.sgfs`
(about 30 s on one CPU core).

| | all 10000 games | standard komi, 6694 games |
|---|---:|---:|
| plies with both players at 0 walls | 63615 (9.93% of plies) | 38910 (9.22%) |
| of them solved exactly (valid) | 63595 | 38890 (20 not solved: plies left or repetition history) |
| games with such a position | 4836 (48.4%) | 2971 (44.4%) |
| moves that lost the result class (win → draw / loss, draw → loss) | **0** | **0** |
| moves that kept the result but lost lead | 14 (0.022% of moves, in 11 games) | 3 (0.008%) |
| game result differs from the exact result of its first exact position | **0 of 4836** | 0 of 2971 |
| search win/loss value in the move comment differs from exact by > 0.2 | 1 of 63505 | 1 of 38849 |
| one-sided proven wins: plies / games | 30700 (4.8%) / 2632 | 14178 (3.4%) / 1297 |
| proven winner lost or drew the game | **0 of 2632** | 0 of 1297 |
| search comment differs from the proven win by > 0.2 | 11 of 30537 (0.036%) | 7 of 14088 (0.05%) |

**Self-play is exact at 99.98% on the exact endgames** (like KataCDC's 99.98%): the mistakes are tiny lead losses, never
a result. So the earlier 11% was the distance rule being wrong, not self-play mistakes, and the solver mostly saves
time in self-play; it adds strength mainly where a search is short of visits (low-visit matches) and removes the
stalling risk.

**Savings of the self-play finish**, per 10000 games: **63595 plies** (9.9% of all plies; 13.2 per game in the 48% of
games that have such a position) are no longer searched. These plies had spent 15.1
million search visits (an upper bound of the NN evals saved, fewer with the NN cache and reduced visits), and
carried a training row weight of 1272 (those rows are no longer written).

### Solver cost and cache
See §1 and §2: 0.8 ms mean per configuration; hit rate 92% on game replay.

### Ladder
A short `kq_ladder` (v256, solver on vs off) was **skipped**: the GPU was in use by a training loop (100%
utilization).

## 6. Tests
`katago runtests quoridorracesolver`: move generation cross-check (16918 position / player pairs, 7904 jump moves);
hand-built races (straight jump, diagonal jump at a wall, a corridor block, a race won by the player behind on
distance because of a jump); the solved value equals a brute-force minimax over full `BoardHistory` play (175
positions including 25 draw states, horizon 9); optimal play on `BoardHistory` ends with the solved result in exactly
the solved number of plies; `evaluate()` (exact, one-sided, plies left, repetition history); the cache; search (root
not terminal, best move keeps the win, chosen move optimal, cache stats); self-play finish.
