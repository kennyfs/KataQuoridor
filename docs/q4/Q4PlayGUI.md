# Q4 play and analysis GUI

A local web app to play Four-at-a-Table (Q4) and, above all, to analyse Q4 positions with a Q4 net. You move for any
seat, the net (or a bot) moves for the others, and the engine shows what it thinks of the position. Code:
`python/q4_play_gui/` (`serve.py`, `game.py`, `engine.py`, `evaluation.py`, `index.html`, `js/core.js`, `js/board.js`,
`js/app.js`, `style.css`, `tests/`). It is a rewrite for four seats of the Duel GUI ([../PlayGUI.md](../PlayGUI.md)), and
the page looks and behaves like the Duel page (same layout, palette, board drawing, evaluation card, candidate table,
move list, shortcut line, dialogs); the Duel GUI and viewer are unchanged. Differences that Q4 needs: four seat cards
(player setting, eliminate), a five-way evaluation bar, rotation by 90°.

**Layout.** Header: New game, Undo, Redo, ▶ Analysis (+ *max* visits), AI (*move*, *until human*), turn / seat at the
bottom, PV, Evaluation, SGF, Load. Left: the board, the four seat cards, a status line, the move box. Right: Evaluation
(five-way bar, text, chart over the game), Analysis (candidate table), Moves, Search settings (folded), the shortcut
line. Under 981 px the columns stack; at phone width the seat cards stack too.

The engine is `katago q4qtp` (`cpp/q4/command/q4qtp.cpp`). The rules come only from the engine (`play`, `legal_moves`,
`eliminate`, `undo`, `winner`, `q4-position`); the GUI draws what the engine answers.

## Launch

```bash
python3 python/q4_play_gui/serve.py --katago cpp/build/katago --model path/to/model.bin.gz
```

| option | default | meaning |
|---|---|---|
| `--katago` | (required) | a KataQuoridor binary with the Q4 commands (`katago q4qtp`) |
| `--model` | (required) | the Q4 net, `model.bin.gz` (`~/q4/run1/models/<newest>/model.bin.gz` in the training layout) |
| `--config` | `cpp/configs/q4/q4qtp.cfg` | the QTP config (search parameters); `numSearchThreads` is overridden by `--threads` |
| `--threads` | 2 | search threads of the engine (keep it low next to a training loop) |
| `--visits` | 400 | default visits of an AI net seat (a seat can have its own) |
| `--interval-cs` | 25 | analysis report interval, centiseconds |
| `--port` | 8767 | the page is `http://127.0.0.1:8767/` |
| `--host` | 127.0.0.1 | listen address; anything else makes the game reachable from your network |
| `--no-browser` | off | do not open the page |
| `--move-timeout` | 600 | seconds before a search counts as hung (the engine is then restarted on request) |
| `--log-dir` | a temp dir | keep `q4qtp.stderr.log` here |

Python 3 standard library only. One game at a time, and one browser tab at a time: a second tab gets "another browser
tab controls this game" until the first one has been quiet for 15 s; a reload of the same tab takes the game back.

## Seats

Four seats, S W N E (Q4Rules §2: S = player 1 starting at f1, W = a6, N = f11, E = k6). The seat cards sit under the
board (the Duel page's player cards, four of them). Each has:

- **kind**: *Human* (you play its moves), *AI net* (the Q4 net searches; visits per move, empty = the default), or
  *Bot* (`greedy`, `basher`, `randomPawn`, `random`, `grudge`; `grudge` needs a target seat). Changes apply at once, also
  in the middle of a game. All four Human = *analysis mode* (the header says so).
- the **walls left** (as bars, like the Duel card), the **distance to the centre** (walls-only, from the engine), alive /
  eliminated, and the seat to move (the card has the seat colour as an outline, and the status line names it).
- **eliminate** (✕): asks first, in a dialog. With two seats alive the confirmation says that the game ends at once (the engine ends it,
  undo reopens it). Eliminating a seat keeps its walls on the board; its unplaced walls are lost (engine rule), and undo
  gives them back.

*AI plays now* plays the seat to move (if it is an AI seat). *AI plays until a human seat* plays AI seats one after the
other (20 moves per request, repeated until a human seat is to move; *Stop* ends it).

## Board

- 11 × 11, drawn like the Duel board: rounded cells with grooves between them, row 11 at the top, the centre f6 marked.
  **No coordinates or move names are drawn on the board** (no text in the cells, none on candidates or on a hovered
  wall); only the file / rank letters outside the four edges (they follow the rotation), the seat letter in each pawn
  and the numbers of the candidates and of a variation. The move list, the move box and the tables use the notation.
- Pawns are discs in the seat colour (the Q4 viewer's set: S blue, W orange, N pink, E green); the seat to move has a
  dashed ring; the last pawn move shows where it came from.
- **Rotate** (R, *turn*, or the S W N E buttons in the header): puts that seat at the bottom.
- Legal pawn destinations of the seat to move are dots (jumps included, with the diagonal steps around a pawn that has a pawn behind it, `Q4Rules.md` 4.3). Loading a record without `jump=diag` (played under the old jump rule) replays it with the new rule: an old two-pawn jump that is no longer legal is rejected with the engine's message at that event. Clicking a pawn highlights its seat's legal
  moves (a read from the engine for a seat that is not to move); clicking a dot plays it.
- **Walls**: hover a groove (the band between cells) to see whether the wall is legal for the seat to move. A legal
  wall is a preview in the seat colour and is placed by a click; an illegal one is red and not clickable (a click says
  so in a message). Legality is the engine's `legal_moves`, not a JS rule. On a touch screen the first tap on a wall
  previews it and a second tap on the same wall places it.
- Walls are drawn in the colour of the seat that placed them (dimmed once that seat is eliminated); the last wall has
  a white glow.
- Keyboard move entry: type `f2`, `e5h`, `j10v` in the box and press Enter (the engine checks it; a wrong entry shows
  a message).
- **Wheel over the board**: up = undo one event, down = redo one event (an elimination is one event) in analysis mode;
  see *Moves and review* for a game against the AI.

## Analysis

Press **Start analysis** (Space). While it runs, the engine searches the position and reports every interval
(`q4-search_analyze_cancellable`, see below); the board and the tables update. Nothing is played.

- **Root line**: the five win probabilities (S W N E draw) as a bar under the board (`search`), the root visits, utilities
  and leads (in cells). When analysis is off, the stored value of this position or the raw net is shown.
- **Raw net**: `net, no search`: the net's own value of the position and the plies to the end (`q4-rawnn`).
- **Candidates**: at most **5** moves, on the board and in the table: those whose visits are **more than 1%** of all
  searched children's visits, best LCB first. The table (as Duel's) shows the rank, the move, the mover's win
  probability, visits (and their share), policy, utility and LCB (sort by clicking win%, visits, policy, utility or lcb;
  the choice is remembered), and under each line the five win probabilities after the move as a stacked bar and the
  principal variation with the seat letters. Click a line, or the candidate on the board, to play it; hover either
  to see the variation on the board (numbered; the **PV** button / P turns this off). On the board the best is light
  blue and the others green, fading with visits; a pawn move shows the mover's win chance and the visits, a wall
  its win chance.
- **Max visits** (empty = unlimited) caps the search; at the cap the analysis reports *done*. The **interval** is in
  centiseconds (25 = a report every 0.25 s).
- The search keeps its tree while the position stays the same: a second search on the same position starts from the
  visits of the first (`test_engine_real`: its first report has more root visits than the first search's last one).

A position's evaluation (the root values of the search when it ran) is kept with the position; it is shown in the move
list (the mover's win chance before each move) and written as the comment of that move in the `.sgfs` download.

## Moves and review

- The move list: a row per event with the seat colour, the move (`f2`, `e5h`; an AI move has the tag `AI`), `EL N` for an
  elimination, and the mover's win chance before the move. Undone events stay in the list, dimmed. The wheel over the
  list scrolls it. Click a row to go to that position.
- **Analysis mode** (all four seats Human): the wheel over the board, ← and →, Home and End, a row of the list and the
  chart go through the engine: ← / wheel up = undo one event, → / wheel down = redo one. One step per engine answer:
  wheel events that arrive while the engine is answering are dropped.
- **A game against the AI** (at least one seat is not Human): the wheel, ← →, Home / End, the list and the chart *review*
  read-only: the position is rebuilt in the page from the event list (pawns, walls, who is out, walls left; distances
  and legal moves are not known there) and the game is not touched; *Back to current* (or Esc, End) returns. The
  **Undo** and **Redo** buttons (U, Backspace) still go through the engine.
- **Undo** takes back exactly one event: a move or an elimination. **Redo** replays the next event. Playing a move after
  an undo drops the events after the position (the redo line), as a branch does.
- An elimination that leaves one seat ends the game (*game over: X wins*); undo reopens it.
- **Download .sgfs**: the game up to the position shown (the engine's `printrecord`), with the evaluation of every
  searched position as the comment of the move after it (`C[pS pW pN pE pD v=.. weight=1.00 lead=..]`, see
  [Q4Sgf.md](Q4Sgf.md) §4). The seat types of the root say what each seat was (`human`, `search` for a net, or the bot's
  name). The file opens in `python3 python/q4_sgfs_viewer/serve.py FILE.sgfs`.
- **Load / paste**: a Q4 record line (`.sgfs`, a file or pasted) or a move list such as `S f2, W b6, EL N, E j10v`
  (move numbers are allowed). The game is replayed into the engine: every move and elimination is checked against the
  rules and the seat to move, and an error names the event. *Stop at ply* leaves the game at that position, so you can
  continue analysing from there (any later move branches).
- **New game…** (a dialog like Duel's): *You play* (keep the seats, one seat, random, AI only, analysis = all human),
  the strength of the AI seats (Easy 1, Normal 16, Hard 256, Max 800 visits, or custom) and, folded, the rules
  (`maxPlies`, `repetitionDrawCount`, initial walls per seat). With *Keep seats* the seat settings stay.

## Search settings

The *Search settings* panel sets, for the engine (`q4-set-param`; each change clears the tree):

- **opponent mode**: `maxn` (every seat an own seat), `style` (the other seats play like a population: their policy is
  the style policy), `expect` (argmax on the style policy). The own seat for `style` / `expect` is "me": by default the
  human seats; choose one seat as "me" to make only that seat own.
- visits (the default for AI seats), the lead factor (`dynamicScoreUtilityFactor`), the lead smoothing
  (`q4LeadWeightEps`), `cpuctExploration`, `rootPolicyTemperature`.
- *Show effective engine values* prints `q4-search-params`.

The header shows the opponent mode. The style and expect modes run on the run1 net (checked: a `style` search gives values and a PV); the GUI does not
vet a net for them.

## Keys

| key | action |
|---|---|
| U / Backspace | undo one event (engine) |
| ← / → | analysis mode: undo / redo one event; against the AI: review back / forward (read-only) |
| wheel over the board | the same as ← / → |
| Home / End | start / latest position |
| Space | start / pause the analysis |
| R | rotate the board by 90° |
| N | New game |
| P / E | PV on hover on / off; evaluation on / off |
| Esc | clear a selected pawn, or back from a review |
| Enter (in the move box) | play the move typed |

## Engine commands added (`cpp/q4/command/q4qtp.cpp`)

| command | what |
|---|---|
| `q4-search_analyze_cancellable [interval centis] [visit cap]` | Searches the seat to move's position; every interval prints `info root visits N value a,b,c,d,e utility .. lead ..` and one `info move <move> visits N prior P utility U lcb L value a,b,c,d,e pv S <move> W <move> ..` per searched child. A line on stdin cancels it (`= play cancelled`, and that line is the next command); the cap ends it (`= done`). No move is made and the tree is kept. Same framing as Duel's `kata-search_analyze_cancellable`: a `=` header, then the info lines, then the answer line. |
| `q4-set-param <key> <value>` | keys `maxVisits`, `q4OpponentMode` (maxn/style/expect), `q4OwnSeats` (bit mask 0..15: bit s = seat s+1 is own, besides the seat to move), `dynamicScoreUtilityFactor`, `q4LeadWeightEps` ([0,1]), `cpuctExploration`, `rootPolicyTemperature`. Unknown keys and out-of-range values are errors. A change clears the tree. |
| `q4-genmove-bot <bot> [target seat]` | the move of a bot for the seat to move (`greedy`, `basher`, `randomPawn`, `random`, `grudge`; grudge needs the target). Bots are made once per process and kept. |
| `q4-position` | the position as one JSON object: `plies`, `toMove` (0..3), `finished`, `winner` (0..3 or -1), `draw`, `alive`, `pawns` (cells, `""` if out), `wallsLeft`, `dist`, `walls` (QTP notation). |
| `q4-search-params` | now also prints `dynamicScoreUtilityFactor`, `q4OpponentMode`, `q4OwnSeats`, `q4LeadWeightEps`. |

Seat numbers in QTP are 1..4 (`play 1 f2` is S); the GUI uses S W N E in its own JSON and converts.

Stdin is read on its own thread (`StdinLines`), so a line that arrives during a search is seen at once and is not hidden
in a buffer. A running `q4qtp` that receives `quit` while its stdin is still open does not exit (also true for the
binary before this change): the GUI closes stdin and kills the process instead.

## Tests

```bash
cd python/q4_play_gui/tests
/home/kenny/ml_venv/bin/python -m unittest test_evaluation test_game_mock test_server   # no engine needed
node core_check.js                                                                     # the pure JS helpers
Q4_KATAGO=/path/katago Q4_MODEL=/path/model.bin.gz /home/kenny/ml_venv/bin/python -m unittest test_engine_real
```

- `test_evaluation`: the analysis lines, the `q4-rawnn` text, the move-list parser.
- `test_game_mock`: eliminate / undo / redo flows (one elimination, two undone twice, eliminate-play-undo-undo, the
  last-but-one confirmation and the game end), review and branching, seat changes mid-game, AI-all, settings, the SGF
  round trip through the Q4 reader, malformed input, engine death and restart. These run on `mock_qtp.py`, a mock of
  the QTP commands with simple rules (the engine's rules are tested by `test_engine_real`).
- `test_server`: the HTTP layer on the mock: one tab, JSON only, status codes, the download, the static page.
- `test_engine_real` (needs the binary and a net): eliminate/undo state equality with the real rules (
    `q4-position`, `walls`, `legal_moves`, `to_move`), walls restored by undo, the cancellable analysis (newline
  cancel, another command cancels and is run next, the tree is kept, the visit cap gives `done`, the PV has seat
  letters), `q4-set-param` keys and errors, bots and grudge.
- `core_check.js`: notation, the rotation of the board (each seat at the bottom for rot 0..3), wall grooves for every
  anchor, the wall rectangles, formatting, the seat of each wall.

## Known limits

- No C++ unit test group was added: the engine commands are tested through the protocol (`test_engine_real`).
- The tree is kept while the position is the same. Undo, redo and review go through the engine's undo, which clears the
  tree (as `undo` always has), so a review followed by a return re-searches from scratch.
- Not done (nice-to-haves): the policy heat map (`q4-rawnn` gives the top 10 only), the shortest-path and "delays seat
  X" overlays, a position editor (`set_position` does not exist), compare mode, the repetition indicator.
- The raw net and the search settings are one net per process; to compare two nets, start two servers on two ports.
- Phone layout: checked at 375 px width (the board, the four seat cards, the evaluation, the candidate table, the list
  and the settings stack; the candidate table has narrower columns there). Not checked on a real phone.
- Read-only review in a game against the AI shows no distances (`? steps`), no analysis and no legal moves: the
  engine only knows its current position.
- The page is light only (the Duel palette); the earlier dark theme is gone.
