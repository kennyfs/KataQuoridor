# Round 8b Report (R8b): self-play without a model ("random" net)

**Branch:** `gemini/q4-four-at-a-table`, base `773e8eba`. **Machine:** 12th Gen Intel(R) Core(TM) i5-12500 (6 cores / 12 threads), 30 GB RAM.

---

## 1. Cause

When `models/` is empty, KataGo and Duel self-play use a "random" net:
`LoadModel::findLatestModel` returns `modelName = "random"`, `modelFile = "/dev/null"`, and `Setup::initializeNNEvaluator` constructs an `NNEvaluator` with `debugSkipNeuralNet = true`.

Before this round, `katago q4selfplay` with an empty `models/` directory failed with:
```
ERROR: game loop loop thread failed: KataQuoridor: model random is not a Q4 (four-player) network (option D 3, expected a value >= 100) and cannot be evaluated on the Q4 path
```
Reproduction command:
```bash
mkdir -p /tmp/q4rnd/models
./cpp/build-eigen/katago q4selfplay -models-dir /tmp/q4rnd/models -output-dir /tmp/q4rnd/out \
  -config cpp/configs/q4/training/q4_selfplay.cfg -max-games-total 4 -seed r \
  -override-config numGameThreads=2,logToStdout=true
```

Causes:
1. `NNEvaluator::NNEvaluator`: the constructor sets `inputsVersion = QuoridorNN::MAX_SUPPORTED_IO_VERSION` (Duel value 3) when `debugSkipNeuralNet` is true (`nneval.cpp`). `NNEvaluator::evaluateQ4Raw` threw because `inputsVersion` was not a Q4 version (expected $\ge 100$).
2. In the server loop, the `debugSkipNeuralNet` block filled only the Duel fields of `NNOutput` (`policyProbs`, `whiteWinProb`, etc.), leaving `q4Raw` null (`std::shared_ptr<Q4RawNNOutput>`). On the Q4 path, `Q4NN::evaluate` requires `buf.result->q4Raw`.

---

## 2. Changes

- **`cpp/neuralnet/nneval.h` (`struct NNResultBuf`)**:
  - Added `bool isQ4Row;` next to `quoridorSymmetry`.
- **`cpp/neuralnet/nneval.cpp`**:
  - `NNResultBuf::NNResultBuf()`: initialized `isQ4Row(false)`.
  - `NNEvaluator::evaluateQ4Raw()`:
    - Guarded the version check with `if(!debugSkipNeuralNet && !Q4NNConst::isQ4IOVersion(inputsVersion))` so the random evaluator is accepted.
    - Set `buf.isQ4Row = true;` before `queryQueue.forcePush(&buf)`.
  - `NNEvaluator::evaluate()` (Duel path):
    - Set `buf.isQ4Row = false;` explicitly before `queryQueue.forcePush(&buf)`.
  - `NNEvaluator::serve()`:
    - In `if(debugSkipNeuralNet)`, branched on `if(resultBuf->isQ4Row)`. For Q4 rows, allocated `NNOutput` with `q4Raw = std::make_shared<Q4RawNNOutput>()`:
      - `policyLogits`: samples from $N(0, 1)$ (`rand.nextGaussian()`).
      - `valueLogits`: samples from $N(0, 0.2)$ (`rand.nextGaussian() * 0.20`).
      - `miscValues`: zeroed with `miscValues[5] = -20.0f` (short-term value error $\approx 0$).
      - `trajectoryLogits`: zeroed.
    - Left the existing Duel branch byte-for-byte unchanged in `else`.

No changes were needed in `cpp/q4/command/q4selfplay.cpp` as it already handles `modelName == "random"`.

---

## 3. Tests & Deliberate Breaks

### 3.1 New Tests

1. **C++ (`cpp/q4/tests/testq4selfplay.cpp` -> `testR1RandomNet()`, group `q4selfplay`)**:
   - Initialized evaluator with `modelName = "random"`, `modelFile = "/dev/null"`, `POS_LEN x POS_LEN`.
   - `Q4NN::evaluate` on start position: policy probabilities over legal actions are finite, positive, and sum to 1 ($\Delta = 0.0$); 5 values are finite and sum to 1 ($\Delta = 0.0$); short-term error is finite and $< 0.01$ ($2.27 \times 10^{-5} < 0.01$).
   - `Q4S::Search` with 100 visits from start position finishes and returns a legal action.
   - `Q4GameRunner::runGame`: completes a self-play game and writes rows to `Q4TrainingDataWriter`.
2. **Python (`python/tests/test_q4_selfplay.py` -> `test_r1_random_net_selfplay`)**:
   - Runs `q4selfplay` in `tmp_path` with empty `models/`, `-max-games-total 4`, `numGameThreads=2,logToStdout=true,maxVisits=100,cheapSearchVisits=25`.
   - Verifies exit code 0, at least one `.npz` file under `out/random/tdata/`, exact Q4 keys present including `metadataInputNC` with shape `(N, 192)`, rows with $C_{26} > 0$ exist, and value vectors $C_{0..4}$ and $C_{20..24}$ sum to 1 within $10^{-4}$.

### 3.2 Deliberate Breaks

1. **Break 1: Disable `isQ4Row` branch in server loop (`if(false && resultBuf->isQ4Row)`)**:
   - Command:
     ```bash
     cmake --build cpp/build-eigen -j$(nproc) && KATAGO_TEST_PYTHON=/home/kenny/ml_venv/bin/python ./cpp/build-eigen/katago runtests q4selfplay
     ```
   - Failure:
     ```
     terminate called after throwing an instance of 'StringError'
       what():  Q4NN::evaluate: the evaluator returned no Q4 raw output (not a Q4 model?)
     ```
2. **Break 2: Restore unconditional Q4 version check (`if(!Q4NNConst::isQ4IOVersion(inputsVersion))`)**:
   - C++ test:
     ```
     terminate called after throwing an instance of 'StringError'
       what():  KataQuoridor: model random is not a Q4 (four-player) network (option D 3, expected a value >= 100) and cannot be evaluated on the Q4 path
     ```
   - Pytest (`/home/kenny/ml_venv/bin/python -m pytest -o addopts="" tests/test_q4_selfplay.py -k test_r1_random_net_selfplay -v`):
     ```
     FAILED tests/test_q4_selfplay.py::test_r1_random_net_selfplay - AssertionError: assert -6 == 0
     ...
     stderr="terminate called after throwing an instance of 'StringError'\n  what():  KataQuoridor: model random is not a Q4 (four-player) network (option D 3, expected a value >= 100) and cannot be evaluated on the Q4 path\n"
     ```

---

## 4. Regressions

All regression suites passed with 0 errors:

1. **`q4selfplay`**:
   ```bash
   KATAGO_TEST_PYTHON=/home/kenny/ml_venv/bin/python ./cpp/build-eigen/katago runtests q4selfplay
   ```
   Output: `All Q4 selfplay C++ tests passed!`, `All tests passed` (exit code 0).

2. **`q4search`**:
   ```bash
   KATAGO_TEST_PYTHON=/home/kenny/ml_venv/bin/python ./cpp/build-eigen/katago runtests q4search
   ```
   Output: `All Q4 Search tests PASSED!`, `All tests passed` (exit code 0).

3. **Full `runtests` suite (including Duel suite)**:
   ```bash
   KATAGO_TEST_PYTHON=/home/kenny/ml_venv/bin/python ./cpp/build-eigen/katago runtests
   ```
   Output: `All tests passed` (exit code 0).

4. **Pytest `test_r1_random_net_selfplay`**:
   ```bash
   cd python && /home/kenny/ml_venv/bin/python -m pytest -o addopts="" tests/test_q4_selfplay.py -k test_r1_random_net_selfplay -v
   ```
   Output: `1 passed, 10 deselected in 3.23s` (exit code 0).

---

## 5. Speed Comparison (5-minute runs, CPU Release build)

**Environment:** Intel i5-12500 (12 threads), `cpp/build-eigen-release` (`-DCMAKE_BUILD_TYPE=Release -DUSE_BACKEND=EIGEN -DUSE_AVX2=1`).
**Settings:** `numGameThreads=24,numEigenThreadsPerModel=12,nnMaxBatchSize=16,nnCacheSizePowerOfTwo=18,maxVisits=600,cheapSearchVisits=100`.

### 5.1 Random net (empty `models/`)

Command:
```bash
timeout -s INT 300s ./cpp/build-eigen-release/katago q4selfplay -models-dir /tmp/q4bench_rand/models -output-dir /tmp/q4bench_rand/out -config cpp/configs/q4/training/q4_selfplay.cfg -seed r1_rand -override-config "numGameThreads=24,numEigenThreadsPerModel=12,nnMaxBatchSize=16,nnCacheSizePowerOfTwo=18,maxVisits=600,cheapSearchVisits=100,logToStdout=true"
```
Row count:
```bash
/home/kenny/ml_venv/bin/python -c '
import glob, numpy as np
files = glob.glob("/tmp/q4bench_rand/out/random/tdata/*.npz")
total = sum(np.load(f)["globalTargetsNC"].shape[0] for f in files)
print(f"Random net: {len(files)} files, {total} rows over 300.537 s -> {total / 300.537:.2f} rows/s")
'
```
Output:
`Random net: 103 files, 102555 rows over 300.537 s -> 341.24 rows/s` (valid games completed: 3,735).

### 5.2 Evaluated net (`b1c32_q4-meta`, without `--scale-heads`, seed 1)

Model generated via:
```bash
/home/kenny/ml_venv/bin/python python/q4/make_random_model.py b1c32_q4-meta /tmp/q4bench_meta/models --model-name q4rand --seed 1
```
Command:
```bash
timeout -s INT 300s ./cpp/build-eigen-release/katago q4selfplay -models-dir /tmp/q4bench_meta/models -output-dir /tmp/q4bench_meta/out -config cpp/configs/q4/training/q4_selfplay.cfg -seed r1_meta -override-config "numGameThreads=24,numEigenThreadsPerModel=12,nnMaxBatchSize=16,nnCacheSizePowerOfTwo=18,maxVisits=600,cheapSearchVisits=100,logToStdout=true"
```
Row count:
```bash
/home/kenny/ml_venv/bin/python -c '
import glob, numpy as np
files = glob.glob("/tmp/q4bench_meta/out/q4rand/tdata/*.npz")
total = sum(np.load(f)["globalTargetsNC"].shape[0] for f in files)
print(f"b1c32_q4-meta: {len(files)} files, {total} rows over 300.59 s -> {total / 300.59:.2f} rows/s")
'
```
Output:
`b1c32_q4-meta: 22 files, 21041 rows over 300.59 s -> 70.00 rows/s` (valid games completed: 967).

### Summary

| Model | Valid Games | NPZ Files | Written Rows | Wall Time | Rows / Second | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| `b1c32_q4-meta` (Eigen) | 967 | 22 | 21,041 | 300.59 s | **70.00** | 1.00× |
| **Random net** (`models/` empty) | **3,735** | **103** | **102,555** | 300.54 s | **341.24** | **4.88×** |

---

## 6. Command for Generating Random-Net Data on a 32-CPU Machine

```bash
taskset -c 0-31 ./cpp/build-eigen-release/katago q4selfplay -models-dir <empty dir> -output-dir <out> \
  -config cpp/configs/q4/training/q4_selfplay.cfg -max-rows-total 500000 -seed q4rand1 \
  -override-config "numGameThreads=96,numEigenThreadsPerModel=32,nnMaxBatchSize=16,nnCacheSizePowerOfTwo=18,maxVisits=600,cheapSearchVisits=100"
```
