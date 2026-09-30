# Play GUI: KataQuoridor in the browser

`python/play_gui/` is a small local web app for playing Quoridor Duel against KataQuoridor with the mouse.
It needs only Python 3 (standard library) and a KataQuoridor `katago` binary; the page works offline.

## Launch

```bash
python3 python/play_gui/serve.py --katago cpp/build-cuda/katago --model /path/to/model.bin.gz
```

Options: `--config` (default `cpp/configs/gtp_quoridor.cfg`), `--port` (default 8766), `--host`
(default 127.0.0.1), `--no-browser`, `--override-config numSearchThreads=2,...` (extra katago overrides),
`--hint-visits` (default 400), `--move-timeout` (default 600 s), `--demo-delay` (AI vs AI, default 1 s),
`--log-dir` (keep engine logs; by default they go to a temporary directory removed at exit).

The server starts one engine process with `logAllGTPCommunication=false, ponderingEnabled=false,
allowResignation=false, reportAnalysisWinratesAs=SIDETOMOVE` and opens the page. One game at a time.

## Playing

- **New game** (N): your side (Black moves first, White, Random, or an AI vs AI demo) and strength:
  Easy = 1 visit (raw policy net), Normal = 16, Hard = 256, Max = 800, or a custom visit count.
  The last choice is remembered in the browser.
- **Pawn moves:** legal destinations (jumps included) are marked with dots; click one. Clicking your own
  pawn highlights them.
- **Walls:** hover a groove between cells. A horizontal groove gives an `h` wall, a vertical one a `v` wall,
  centred on the nearest groove crossing. Legal previews are in your colour, illegal ones red (not clickable).
- **Keyboard fallback:** type a move in QTP notation (`e8`, `e2h`, `d5v`) and press Enter.
- **Undo** (U) takes back your last move and the AI's reply. **Hint** (H) searches your position and shows
  the top 3 moves on the board and in the side panel (click a line to play it). **Flip** (F) turns the board.
  **Evaluation** (E) shows or hides the win chances. **SGF** downloads the game as `.sgfs`, which
  `python/sgfs_viewer` opens (with the evaluation of every position in the move comments).
- Click a move in the list, or use ← / →, to review earlier positions (read-only); Esc or End goes back.

The board is drawn as seen from your side: as White, row 1 is at the bottom and column a on the left; as
Black, the board is rotated 180°. (sgfs_viewer draws a mirrored view instead, so the two can look different.)

## Notation reminder

Columns `a`–`i` left to right, rows `1`–`9` bottom to top (standard view). Black starts on `e9` and heads to
row 1; White starts on `e1` and heads to row 9. A wall is its anchor cell (column `a`–`h`, row `1`–`8`)
plus `h` or `v`: `e2h` lies between rows 2 and 3 under columns e and f; `e2v` lies between columns e and f
beside rows 2 and 3. See [GameRules.md](../GameRules.md).

## Evaluation

After every move the panel shows Black's and White's win probability and the expected margin in moves
("Black leads by 1.2 moves"). After an AI move with search it is the search result (visits and best line);
after your move, or with Easy (1 visit, no search output), it is the raw net (`kata-raw-nn`), later replaced
by the AI's root search when it has one. A finished game shows the exact result.

## Troubleshooting

- **`katago binary not found` / engine stopped at start:** check `--katago` and `--model`. The red banner
  shows the engine's last stderr lines. OpenCL/TensorRT/Metal builds refuse Quoridor nets; use CUDA or Eigen.
- **CUDA vs Eigen:** both work. Eigen (CPU) is much slower: prefer Easy/Normal, and consider
  `--override-config numSearchThreads=<cores>`.
- **The first move is slow:** the engine loads the net (and on CUDA initialises the GPU) at start; the page
  shows "Loading the engine…" until it is ready.
- **The engine crashed:** use **Restart engine** in the banner. The game is replayed into a new process.
- **Port in use:** `--port 8767`.

## Tests

```bash
cd python
python -m pytest -o addopts="" play_gui/tests
```
