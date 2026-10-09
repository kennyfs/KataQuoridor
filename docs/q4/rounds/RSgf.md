# Report: Q4 game records as Duel-style SGF, and the Q4 viewer (Prompt P)

**Branch:** `gemini/q4-sgf-records` (from `gemini/q4-four-at-a-table` at `b5ab16d6`); nothing was pushed, merged or committed
to `gemini/q4-four-at-a-table` or `main`. **Machine:** the dev box (RTX 3070 8 GB), CUDA build of `cpp/katago`.
Format: [Q4Sgf.md](../Q4Sgf.md).

## 1. Commits

1. `feat(q4): write game records as Duel-style SGF lines` (C++: `Q4Record::toSgfLine/fromSgfLine`, the writers, tests).
2. `fix(q4): the gatekeeper no longer plays the placeholder table of its config probe`.
3. `feat(q4): Python SGF record reader/writer; rating, match report and tests read .sgfs`.
4. `feat(q4): Q4 SGF viewer (python/q4_sgfs_viewer); the Duel viewer refuses GM[Q4] files`.
5. this report + `Q4Sgf.md` + doc edits.

All signed (GPG).

## 2. The format as implemented (changes from the prompt)

As the prompt, with these differences:

- `weight` is the **unrounded** target weight (`targetWeightByTurnUnrounded`, e.g. `0.53`, as Duel shows), written when the
  game ends, not the stochastically integerized one (which would give 0 / 1 / 2 only).
- Result mapping is done in the writer only; `Q4Record` keeps the old result strings (`"1+"`, `"Draw"`, `"none"`).
- `DR[unfinished]` is written whenever `RE[?]`; for a self-play draw `DR` is derived from the plies (`maxPlies` or
  `repetition`).
- `type<i>` for a self-play learner seat is `selfplay`; for a population seat the seat kind (`weak`, `snapshot`,
  `greedy`, `randomPawn`, `basher`, `grudge`); `PS..PE` for those seats is the kind or the snapshot net name.
- `gtype`: `normal`, `fork` (the game's `MODE_FORK`), or `mixed` (a population game with a non-learner seat). Q4 has no
  `side` game: side positions are rows, not games.
- `matchOpeningPlies` is gone: `startTurnIdx` is used for self-play (`startHist`) and for match openings alike.
- The Q4 reader (C++) rejects a move whose property is not the seat to move; the SGF has no node for a skipped seat.
- Values in the root `C` must not contain a comma (a model path with a comma would break `net<i>`); not escaped.

## 3. Which command writes what

See Q4Sgf.md §5. `q4selfplay`: `<model>/sgfs/*.sgfs`; `q4gatekeeper`: `<candidate>/games.sgfs`; `q4match -output`: SGF lines;
`q4_formal_eval.sh`: `eval/ladder.sgfs`; `q4tool popgames`, `q4qtp printrecord/loadrecord`: SGF. `q4_synchronous_loop.sh`,
`q4_formal_run.sh`, the shuffle and cleanup scripts do not read the records directory (checked by grep). `q4tool style`
reads SGF lines; its JSON `{"rules","events"[,"initialBoard"]}` input is kept only for the symmetry tests, which
build non-played event lists.
Match and gatekeeper games now carry the search comments (root values of the moving search seat, `weight=1.00`).
Existing `.jsonl` files are not converted.

## 4. The viewer (`python/q4_sgfs_viewer/`)

Written fresh (the Duel one is too tied to two players), reusing its CSS, `serve.py`, layout and interaction code
(resizable list, wheel / swipe / keyboard stepping, collapsible list on a phone).

- **List:** result badge (seat colour, D, ?), plies, tags (`4×selfplay` or `S:basher W:weak ...`, rule only if the file's games
  differ, walls if not 7/7/7/7, gtype if not normal, `table #opening.rotation`, `elim n`, draw reason). Headers are parsed
  from the root node only at load; a background pass (30 ms slices) parses each game once for its summary.
- **Filters:** result (S/W/N/E, any draw, repetition, maxPlies, unfinished), rule (any / on / off / N), walls, gtype, table,
  player type of any seat, model of any seat, eliminations, plies range; chips show the active ones, shared with the stats.
- **Sorts:** file order, plies (both ways), winner, largest swing, decided early (earliest ply a seat exceeds 0.9), draws first,
  and a direction toggle.
- **Board:** 11×11, columns a..k rows 1..11 y up, `R` rotates by 90° (any seat at the bottom; edge labels follow the
  rotation), four Okabe-Ito seat colours with the seat letter, the centre f6 marked, walls drawn in the colour of the
  placing seat, the last move outlined, eliminated seats greyed in their card and their pawn removed (walls stay), cards with
  walls left, steps to the centre, and the seat to move highlighted. `P` shows all shortest-path cells per seat (one corner of
  the cell per seat, BFS on the cell graph with the walls, pawns ignored: same rule as `distToCenter`).
- **Evaluation:** stacked bar + numbers (S, W, N, E, draw) of the comment of the move about to be played, visits, weight, "cheap"
  when the visits are below the file's full-search visits; a 5-line graph (the winner's line thick, the draw dashed grey,
  lines broken where no comment exists, click to jump); the move list (`1. S f2`, visits, opening plies separated,
  **blunder** = the largest drop of a seat's own probability from one of its comments to its next, marked ⚠).
- **Statistics:** games, mean / median / histogram of plies, wins by seat (share of finished games, draw separate) and a
  score by seat (draw = ¼ each; the labels say so), draws by reason, results per repetition rule, the winner's mean
  probability by ply, calibration (seat to move at every 10th ply, draw = ¼), full vs cheap searches and visits,
  comebacks (winner < 10% at some ply); for match files a per-player table (appearances, wins, score with a Wilson 95%
  interval, wins by seat), per table and per table × rotation.
- **Skipped:** per-opening tables, blunder statistics, learning curves across files, a play button (as asked).
- **Duel viewer:** `python/sgfs_viewer/js/viewer.js` got a 4-line refusal of files with `GM[Q4]` (the prompt asked for the
  refusal and also not to change that viewer; this is the minimal change; no Duel logic touched).

## 5. Tests

| test | result |
|---|---|
| `katago runtests q4board q4match q4selfplay` (C++; round trip incl. comments, elimination, maxPlies draw, match metadata, escapes, 8 malformed lines) | pass |
| `python/tests/test_q4_record.py` (new; C++ → Python → same line on 80 population games; hand-written game) | 3 pass |
| `test_q4_selfplay.py` (11, reads `sgfs/*.sgfs`, observer-row values vs the comment) | 11 pass (tolerance 6e-3: comments have 2 decimals) |
| `test_q4_training.py`, `test_q4_style.py`, `test_q4_qtp.py`, `test_q4_rating.py` | 14, 4, 5, 7 pass |
| `test_q4_viewer.py` (new: node `check_core.js` + the viewer's BFS distances equal `reference.py` at every position of the 12 fixture games) | 2 pass |

Each pytest file in its own process (`-o addopts=""`); `Q4_EIGEN_BIN` / `KATAGO_BIN` pointed at `cpp/katago` (a CUDA build).
`test_q4_selfplay.py::test_c1_c2_writer_vs_replay` died with SIGSEGV in two full-file runs that overlapped with other GPU
jobs of mine (a self-play run and a browser session), and passed in two isolated runs, in the 3-test prefix, and in a
quiet full-file run; I did not find the cause (not reproduced with the same command line outside pytest).
Not run: `q4_formal_eval.sh` end to end (needs a formal run directory); only its paths were changed.
`tests/fixture.sgfs` = 12 real games: self-play (normal, mixed, fork; two with eliminations; a maxPlies draw; a repetition
draw), 3 q4match games (a search net and bots), 1 gatekeeper game. Nothing is hand-written.

## 6. How the viewer was checked

Served `fixture.sgfs` with `serve.py`, in the app's browser pane at 1500 px (desktop) and 375 × 812 (phone): game list, a
selfplay fork draw, a match game rotated (`R`) at a ply with a comment (bar, numbers, distances, blunder mark), the stats tab
(match tables; all 8 cards), the phone layout (collapsed list, board, controls bar). One bug found and fixed this way:
`scrollIntoView` scrolled the whole page away from the board on a phone; list rows now scroll inside their box. No console errors.
Not checked by eye: the file-open dialog and drag-and-drop (same code as Duel's), swipe on a real touch device.

## 7. Things that looked wrong

- **q4gatekeeper** (fixed, commit 2): `mc.tables` kept the placeholder table `t` (`x,x,x,x`, 1 opening) of the config probe, so
  every candidate also played one cand×4 game, which the callback counted into the benchmark *base* tally of opening 0. It only
  mattered with `gateBenchOpenings >= 2`.
- The old match records had no comments and the self-play ones used a different format (`v=[...] visits= cheap=` plus an
  `elim=N` comment as an event): all gone.
- `Q4Rules::toString()` prints `initialWalls=a,b,c,d` with commas inside a comma list; the SGF `RU` avoids it (walls are `WS..WE`).
- Duel viewer: `loadText` would try `headerOf` on any `(;` line; with a Q4 file it produced garbage or an exception, hence the refusal.
