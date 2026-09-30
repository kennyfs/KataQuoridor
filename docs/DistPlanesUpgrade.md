# S8–S11 fix: upgrading a running training directory

## The bug

Spatial input channels 8–11 (`QuoridorNN::fillRow`) are BFS distance fields, encoded as
`d < 0 ? 1.0 : min(1.0, d/32)`:

| ch | meaning |
|----|---------|
| 8  | distance to my goal row |
| 9  | distance to the opponent's goal row |
| 10 | distance from my pawn |
| 11 | distance from the opponent's pawn |

Training data stored every spatial plane through `packBits` (`(uint8_t)value`). That truncated these channels to
1 only where a cell is unreachable or `d >= 32`, and to 0 everywhere else. So every trained net saw S8–S11 as
nearly all-zero, while inference in C++ fed the continuous values. On average this shifted the win rate by
5.6 percentage points, and it changed the top move in 17% of positions.

## The fix (still Quoridor I/O v1, one code path)

- The C++ writer adds the npz array `spatialDistNCHW`: `uint8 [N,4,9,9]`, raw BFS distances for ch8–11 in
  canonical (side-to-move) orientation, with 255 for unreachable cells.
  - The packed bit planes 8–11 are written as 0.
  - A `testAssert` checks that every value passed to `packBits` is exactly 0 or 1.
  - The BFS lives in one place, `QuoridorNN::fillDistances`, shared by `fillRow` and the writer.
- The loader (`data_processing_pytorch.py`) rebuilds ch8–11 as `where(d == 255, 1, min(1, d/32))` right after
  `unpackbits`, before the symmetry is applied. Channels 8–11 use the plain x-flip, like the pawn planes.
- `shuffle.py` carries `spatialDistNCHW` through.
- Files that lack `spatialDistNCHW` fail loudly (`KeyError`) in both the shuffler and the loader.
- Existing data is upgraded exactly: `python/quoridor_add_dist_planes.py` recomputes the distances from planes
  that every row already stores. These are the pawns (ch1/ch2) and the canonical blocked edges (ch3–6). The
  goal rows are canonical row 0 for the side to move and row 8 for the opponent.
- Existing nets are migrated by `python/quoridor_migrate_dist_planes.py`. It zeroes
  `conv_spatial.weight[:, 8:12]` in the model, the SWA model and the matching optimizer state. The migrated net
  ignores ch8–11, as it effectively did during training, and then learns to use the real distances.
  - Side effect: the old net had seen the rare "unreachable" 1-bits in these channels. That small signal is
    dropped, and it is re-learned from the real distances.

## Operator runbook

Set these once (adjust to your setup):

```bash
REPO=/path/to/KataQuoridor         # the repo the loop runs from (synchronous_loop.sh copies cpp/katago from it)
BASEDIR=/path/to/run/basedir       # contains selfplay/, shuffleddata/, train/, models/, ...
TRAININGNAME=<name>                # the TRAININGNAME passed to synchronous_loop.sh (BASEDIR/train/<name>)
PY=/path/to/venv/bin/python        # a Python with torch and numpy
```

### 1. Stop the loop

Stop `synchronous_loop.sh` (Ctrl-C in its terminal, or kill it). Make sure no `katago selfplay`, `shuffle.py` or
`train.py` process is still running:

```bash
pgrep -af "katago selfplay|katago gatekeeper|shuffle.py|train.py" || echo "all stopped"
```

### 2. Build the new binary

This is an incremental build in your existing build directory, and the loop picks up `cpp/katago`:

```bash
cd $REPO && git fetch origin && git checkout claude/quoridor-perft-tests-fupqdv   # or merge it into your branch
cd $REPO/cpp/build && make -j$(nproc) 2>&1 | tail -1
cp $REPO/cpp/build/katago $REPO/cpp/katago
```

Optionally sanity-check the new build (takes about 1 minute):

```bash
cd $REPO/python && $PY -m pytest -q tests/test_dist_planes.py 2>&1 | tail -1
```

### 3. Convert `selfplay/` (in place, atomic per file, resumable)

```bash
cd $REPO/python && $PY quoridor_add_dist_planes.py $BASEDIR/selfplay -num-processes 8
```

- The script skips files that already have the key, so it is safe to re-run after an interruption.
- Each file is cross-checked while it is converted. The old truncated bits and globals 13/14 must agree with the
  recomputed distances; the script stops with an assertion if they don't.
- **Time:** about 20k rows/s per process on sample data, where the time goes on npz decompression and
  recompression. About 1.5M rows takes roughly 1–2 minutes with 8 processes, and under 5 minutes with 1.
- **Disk:** each file is rewritten one at a time, so you only need free space for a few files at once.

### 4. Delete `shuffleddata/`

It is regenerated from `selfplay/` on every cycle, and the old shuffled files lack `spatialDistNCHW`, so the
loader would refuse them:

```bash
rm -rf $BASEDIR/shuffleddata $BASEDIR/shufflescratch
```

### 5. Migrate the training checkpoint

```bash
cd $REPO/python && $PY quoridor_migrate_dist_planes.py -checkpoint $BASEDIR/train/$TRAININGNAME/checkpoint.ckpt
```

- The script writes a backup first (`checkpoint.ckpt.pre_distplanes`) and is idempotent: a marker is kept in
  `train_state`.
- It takes a few seconds.
- The `checkpoint_prev*.ckpt` files are only fallbacks. Migrate one with the same command if you ever resume
  from it.

### 6. Migrate and re-export the model(s) selfplay and gating use

This step is needed. Without it, selfplay keeps running the unmigrated net, which receives continuous S8–S11 that
it never trained on: that is the bug itself. Gating would also compare a migrated candidate against an unmigrated
incumbent.

Migrate the newest accepted model in `models/`, plus anything still waiting in `modelstobetested/`:

```bash
cd $REPO/python
for d in $(ls -td $BASEDIR/models/*/ | head -1) $BASEDIR/modelstobetested/*/; do
  [ -f "$d/model.ckpt" ] && $PY quoridor_migrate_dist_planes.py -export-model-dir "$d"
done
# Checkpoints exported by train.py but not yet converted by export_model_for_selfplay.sh:
for c in $BASEDIR/torchmodels_toexport/*/model.ckpt; do
  [ -f "$c" ] && $PY quoridor_migrate_dist_planes.py -checkpoint "$c"
done
```

- `-export-model-dir` migrates `model.ckpt`, then re-exports `model.bin.gz` under the same model name with
  `export_model_pytorch.py -use-swa`, exactly as `export_model_for_selfplay.sh` does. It keeps
  `model.bin.gz.pre_distplanes` and `model.ckpt.pre_distplanes` as backups.
- Re-running it is a no-op, and it resumes correctly if it was interrupted between the checkpoint step and the
  export step.
- Each model takes about 10–30 s.
- Older models in `models/` are no longer used by selfplay, so leave them alone.

### 7. Restart the loop

Start `synchronous_loop.sh` exactly as before, with the same arguments. It copies the new scripts and
`cpp/katago` into a fresh dated archive.

What to expect:

- The first cycle's shuffle rebuilds `shuffleddata/` from the converted `selfplay/`.
- Training resumes from the migrated checkpoint.
- The first new models learn to use the real distances.
- Early on, gatekeeper results may be noisy for a cycle or two.

## Tests

- `python/tests/test_dist_planes.py`:
  - **Train↔inference parity:** 600 random-game positions from `katago dumpnninputs` are encoded with the
    writer's own `TrainingWriteBuffers::fillQuoridorInputRow` and decoded by the loader's
    `decode_binary_input`. They must equal `fillRow`'s output exactly, on all 17 channels and all 15 globals.
  - **Converter:** fresh rows from the new writer (writer sample rows and random-game rows) are turned back into
    old-format rows (key removed, truncated bits restored). The converter must reproduce the writer's arrays bit
    for bit, and a second run must skip every file.
  - **Migration:** a tiny random checkpoint (model, SWA model and SGD momentum) is migrated. It must give
    identical outputs on inputs whose ch8–11 are zero, ignore ch8–11 entirely, leave every other tensor
    unchanged, and be idempotent even after further training.
- The e2e test (`test_end_to_end_training.py`) also checks the shape and dtype of `spatialDistNCHW`.
