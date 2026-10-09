#include "q4trainingwrite.h"

#include <cmath>
#include <fstream>
#include <iostream>

#include "../../core/fileutils.h"
#include "../../core/makedir.h"
#include "../../core/test.h"

namespace Q4Play {

Q4ValueTargets::Q4ValueTargets() {
  for(int i = 0; i < 5; i++)
    value[i] = 0.0f;
}

Q4SidePosition::Q4SidePosition()
  : state(),
    toMove(0),
    unreducedNumVisits(0),
    policyTarget(),
    policySurprise(0.0),
    policyEntropy(0.0),
    searchEntropy(0.0),
    valueTargets(),
    nnRawStats(),
    targetWeight(0.0f),
    targetWeightUnrounded(0.0f),
    numNeuralNetChangesSoFar(0)
{}

Q4SidePosition::Q4SidePosition(const Q4PlayState& s, int numNNChangesSoFar)
  : state(s),
    toMove(s.board.toMove),
    unreducedNumVisits(0),
    policyTarget(),
    policySurprise(0.0),
    policyEntropy(0.0),
    searchEntropy(0.0),
    valueTargets(),
    nnRawStats(),
    targetWeight(0.0f),
    targetWeightUnrounded(0.0f),
    numNeuralNetChangesSoFar(numNNChangesSoFar)
{}

Q4FinishedGameData::Q4FinishedGameData()
  : modelName(),
    startState(),
    endHist(),
    gameHash(),
    hitTurnLimit(false),
    mode(MODE_NORMAL),
    hasFullData(false),
    targetWeightByTurn(),
    targetWeightByTurnUnrounded(),
    policyTargetsByTurn(),
    policySurpriseByTurn(),
    policyEntropyByTurn(),
    searchEntropyByTurn(),
    valueTargetsByTurn(),
    nnRawStatsByTurn(),
    valueSurpriseByTurn(),
    wasCheapSearchByTurn(),
    seatKindByTurn(),
    seatInfo(),
    sidePositions(),
    changedNeuralNets(),
    comments(),
    trainingWeight(1.0),
    startPly(0),
    rules()
{}

Q4FinishedGameData::~Q4FinishedGameData() {
  for(size_t i = 0; i < policyTargetsByTurn.size(); i++)
    delete policyTargetsByTurn[i].policyTargets;
  policyTargetsByTurn.clear();
  for(size_t i = 0; i < sidePositions.size(); i++)
    delete sidePositions[i];
  sidePositions.clear();
  for(size_t i = 0; i < changedNeuralNets.size(); i++)
    delete changedNeuralNets[i];
  changedNeuralNets.clear();
}

// Copy floats that are all 0-1 into bits, packing 8 to a byte, big-endian-style within each byte.
static void packBits(const float* binaryFloats, int len, uint8_t* bits) {
  for(int i = 0; i < len; i += 8) {
    if(i + 8 <= len) {
      bits[i >> 3] =
        ((uint8_t)binaryFloats[i + 0] << 7) |
        ((uint8_t)binaryFloats[i + 1] << 6) |
        ((uint8_t)binaryFloats[i + 2] << 5) |
        ((uint8_t)binaryFloats[i + 3] << 4) |
        ((uint8_t)binaryFloats[i + 4] << 3) |
        ((uint8_t)binaryFloats[i + 5] << 2) |
        ((uint8_t)binaryFloats[i + 6] << 1) |
        ((uint8_t)binaryFloats[i + 7] << 0);
    }
    else {
      bits[i >> 3] = 0;
      for(int di = 0; i + di < len; di++) {
        bits[i >> 3] |= ((uint8_t)binaryFloats[i + di] << (7 - di));
      }
    }
  }
}

int Q4TrainingWriteBuffers::actionToPolicySlot(int action) {
  if(Q4Board::isPawnAction(action)) {
    return action;
  }
  else if(Q4Board::isVWallAction(action)) {
    int anchor = action - 121;
    int ay = Q4Board::anchorY(anchor);
    int ax = Q4Board::anchorX(anchor);
    return 1 * POS_AREA + ay * POS_LEN + ax;
  }
  else if(Q4Board::isHWallAction(action)) {
    int anchor = action - 221;
    int ay = Q4Board::anchorY(anchor);
    int ax = Q4Board::anchorX(anchor);
    return 2 * POS_AREA + ay * POS_LEN + ax;
  }
  return -1;
}

static void zeroPolicyTarget(int policySize, int16_t* target) {
  for(int pos = 0; pos < policySize; pos++)
    target[pos] = 0;
}

static void uniformPolicyTarget(int policySize, int16_t* target) {
  for(int pos = 0; pos < policySize; pos++)
    target[pos] = 1;
}

static void fillPolicyTargetQ4(const std::vector<Q4PolicyTargetMove>& moves, int16_t* target) {
  zeroPolicyTarget(Q4TrainingWriteBuffers::POLICY_SIZE, target);
  for(const auto& m : moves) {
    int slot = Q4TrainingWriteBuffers::actionToPolicySlot(m.action);
    if(slot >= 0 && slot < Q4TrainingWriteBuffers::POLICY_SIZE) {
      target[slot] = m.policyTarget;
    }
  }
}

void Q4TrainingWriteBuffers::fillValueTDTargets(
  const std::vector<Q4ValueTargets>& valueTargetsByTurn,
  int idx,
  int toMove,
  double nowFactor,
  float* buf
) {
  double target[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  double weightLeft = 1.0;
  for(size_t i = idx; i < valueTargetsByTurn.size(); i++) {
    double weightNow;
    if(i == valueTargetsByTurn.size() - 1) {
      weightNow = weightLeft;
      weightLeft = 0.0;
    }
    else {
      weightNow = weightLeft * nowFactor;
      weightLeft *= (1.0 - nowFactor);
    }
    const Q4ValueTargets& vt = valueTargetsByTurn[i];
    for(int k = 0; k < 4; k++) {
      int s = (toMove + k) % 4;
      target[k] += weightNow * vt.value[s];
    }
    target[4] += weightNow * vt.value[4];
  }
  for(int k = 0; k < 5; k++) {
    buf[k] = (float)target[k];
  }
}

Q4TrainingWriteBuffers::Q4TrainingWriteBuffers(int maxRws)
  : maxRows(maxRws),
    curRows(0),
    binaryInputNCHWPacked({maxRws, NUM_BINARY_CHANNELS, PACKED_BOARD_AREA}),
    spatialDistNCHW({maxRws, NUM_DIST_CHANNELS, POS_LEN, POS_LEN}),
    globalInputNC({maxRws, NUM_GLOBAL_CHANNELS}),
    policyTargetsNCMove({maxRws, POLICY_NUM_CHANNELS, POLICY_SIZE}),
    globalTargetsNC({maxRws, GLOBAL_TARGET_NUM_CHANNELS}),
    scoreDistrN({maxRws, SCORE_DISTR_LEN}),
    valueTargetsNCHW({maxRws, VALUE_TARGET_CHANNELS, POS_LEN, POS_LEN})
{}

void Q4TrainingWriteBuffers::clear() {
  curRows = 0;
}

void Q4TrainingWriteBuffers::writeToZipFile(const std::string& fileName) {
  ZipFile zipFile(fileName);

  uint64_t numBytes = binaryInputNCHWPacked.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("binaryInputNCHWPacked", binaryInputNCHWPacked.dataIncludingHeader, numBytes);

  numBytes = spatialDistNCHW.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("spatialDistNCHW", spatialDistNCHW.dataIncludingHeader, numBytes);

  numBytes = globalInputNC.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("globalInputNC", globalInputNC.dataIncludingHeader, numBytes);

  numBytes = policyTargetsNCMove.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("policyTargetsNCMove", policyTargetsNCMove.dataIncludingHeader, numBytes);

  numBytes = globalTargetsNC.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("globalTargetsNC", globalTargetsNC.dataIncludingHeader, numBytes);

  numBytes = scoreDistrN.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("scoreDistrN", scoreDistrN.dataIncludingHeader, numBytes);

  numBytes = valueTargetsNCHW.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("valueTargetsNCHW", valueTargetsNCHW.dataIncludingHeader, numBytes);

  zipFile.close();
}

void Q4TrainingWriteBuffers::addRow(
  const Q4PlayState& state,
  int turnIdx,
  float targetWeight,
  int64_t unreducedNumVisits,
  const std::vector<Q4PolicyTargetMove>* policyTarget0,
  const std::vector<Q4PolicyTargetMove>* policyTargetNext,
  int actionPlayedThisTurn,
  double policySurprise,
  double policyEntropy,
  double searchEntropy,
  const std::vector<Q4ValueTargets>& valueTargetsByTurn,
  int valueTargetsIdx,
  float valueTargetWeight,
  float tdValueTargetWeight,
  const Q4NNRawStats& nnRawStats,
  bool isSidePosition,
  Hash128 gameHash,
  int gameMode,
  int startPly,
  bool hitTurnLimit,
  int numAliveAtRow,
  int maxPlies,
  int repetitionDrawCount,
  int finalGamePlies,
  const uint8_t finalDistToCenter[4],
  const bool seatEliminatedBeforeEnd[4],
  const std::vector<Q4Board>& boardHistoryFromTurnToEnd,
  const std::vector<int>& actionsPlayedFromTurnToEnd,
  const std::vector<int>& actionSeatsFromTurnToEnd,
  int seatKind
) {
  testAssert(curRows < maxRows);

  float rowSpatialScratch[NUM_BINARY_CHANNELS * POS_AREA];
  float rowGlobalScratch[NUM_GLOBAL_CHANNELS];
  Q4NN::RawDistances rawDist;
  Q4NN::fillRow(state, false, rowSpatialScratch, rowGlobalScratch, &rawDist);

  // Distances in channels 10..14 are written as 0 in binary planes
  for(int ch = 10; ch <= 14; ch++) {
    for(int c = 0; c < POS_AREA; c++) {
      rowSpatialScratch[ch * POS_AREA + c] = 0.0f;
    }
  }

  // Pack binary channels
  uint8_t* rowBinPacked = binaryInputNCHWPacked.data + curRows * NUM_BINARY_CHANNELS * PACKED_BOARD_AREA;
  for(int ch = 0; ch < NUM_BINARY_CHANNELS; ch++) {
    packBits(rowSpatialScratch + ch * POS_AREA, POS_AREA, rowBinPacked + ch * PACKED_BOARD_AREA);
  }

  // Spatial raw distances [5, 11, 11]
  uint8_t* rowDist = spatialDistNCHW.data + curRows * NUM_DIST_CHANNELS * POS_AREA;
  for(int ch = 0; ch < NUM_DIST_CHANNELS; ch++) {
    std::copy(rawDist.d[ch], rawDist.d[ch] + POS_AREA, rowDist + ch * POS_AREA);
  }

  // Global inputs [28]
  float* rowGlobalInput = globalInputNC.data + curRows * NUM_GLOBAL_CHANNELS;
  std::copy(rowGlobalScratch, rowGlobalScratch + NUM_GLOBAL_CHANNELS, rowGlobalInput);

  // Policy targets [3, 363]
  int16_t* rowPolicy = policyTargetsNCMove.data + curRows * POLICY_NUM_CHANNELS * POLICY_SIZE;
  float* rowGlobal = globalTargetsNC.data + curRows * GLOBAL_TARGET_NUM_CHANNELS;
  std::fill_n(rowGlobal, GLOBAL_TARGET_NUM_CHANNELS, 0.0f);

  // C0: Search policy target. Rows of non-learner seats (observer rows) have none: zeros and weight 0.
  if(seatKind != 0) {
    testAssert(policyTarget0 == nullptr || policyTarget0->empty());
    zeroPolicyTarget(POLICY_SIZE, rowPolicy + 0 * POLICY_SIZE);
    rowGlobal[26] = 0.0f;
  }
  else if(policyTarget0 != nullptr) {
    fillPolicyTargetQ4(*policyTarget0, rowPolicy + 0 * POLICY_SIZE);
    rowGlobal[26] = 1.0f;
  }
  else {
    uniformPolicyTarget(POLICY_SIZE, rowPolicy + 0 * POLICY_SIZE);
    rowGlobal[26] = 0.0f;
  }

  // C1: Action actually played this turn
  zeroPolicyTarget(POLICY_SIZE, rowPolicy + 1 * POLICY_SIZE);
  if(actionPlayedThisTurn >= 0) {
    int slot = actionToPolicySlot(actionPlayedThisTurn);
    if(slot >= 0 && slot < POLICY_SIZE) {
      rowPolicy[1 * POLICY_SIZE + slot] = 1;
    }
  }
  rowGlobal[29] = 1.0f;

  // C2: Next seat's search policy target
  if(policyTargetNext != nullptr) {
    fillPolicyTargetQ4(*policyTargetNext, rowPolicy + 2 * POLICY_SIZE);
    rowGlobal[28] = 1.0f;
  }
  else {
    uniformPolicyTarget(POLICY_SIZE, rowPolicy + 2 * POLICY_SIZE);
    rowGlobal[28] = 0.0f;
  }

  // Global targets [64]
  int toMove = state.board.toMove;
  // C0–4: Final result
  fillValueTDTargets(valueTargetsByTurn, valueTargetsIdx, toMove, 0.0, rowGlobal + 0);
  // C5–9: TD value target, nowFactor = 1/(1 + 121·0.176)
  fillValueTDTargets(valueTargetsByTurn, valueTargetsIdx, toMove, 1.0 / (1.0 + 121.0 * 0.176), rowGlobal + 5);
  // C10–14: TD value target, nowFactor = 1/(1 + 121·0.056)
  fillValueTDTargets(valueTargetsByTurn, valueTargetsIdx, toMove, 1.0 / (1.0 + 121.0 * 0.056), rowGlobal + 10);
  // C15–19: TD value target, nowFactor = 1/(1 + 121·0.016)
  fillValueTDTargets(valueTargetsByTurn, valueTargetsIdx, toMove, 1.0 / (1.0 + 121.0 * 0.016), rowGlobal + 15);
  // C20–24: This turn's search value (nowFactor = 1.0)
  fillValueTDTargets(valueTargetsByTurn, valueTargetsIdx, toMove, 1.0, rowGlobal + 20);

  rowGlobal[25] = targetWeight;
  rowGlobal[27] = valueTargetWeight;
  rowGlobal[30] = (float)policySurprise;
  rowGlobal[31] = (float)policyEntropy;
  rowGlobal[32] = (float)searchEntropy;
  rowGlobal[33] = (float)(1.0f - tdValueTargetWeight);
  rowGlobal[34] = (float)(1.0f - valueTargetWeight);
  rowGlobal[35] = (float)((finalGamePlies - state.plies) * valueTargetWeight);

  for(int k = 0; k < 4; k++) {
    int s = (toMove + k) % 4;
    rowGlobal[36 + k] = (float)finalDistToCenter[s];
    rowGlobal[40 + k] = seatEliminatedBeforeEnd[s] ? 0.0f : valueTargetWeight;
  }

  rowGlobal[44] = (float)(gameHash.hash0 & 0x3FFFFF);
  rowGlobal[45] = (float)((gameHash.hash0 >> 22) & 0x3FFFFF);
  rowGlobal[46] = (float)((gameHash.hash0 >> 44) & 0xFFFFF);
  rowGlobal[47] = (float)(gameHash.hash1 & 0x3FFFFF);
  rowGlobal[48] = (float)((gameHash.hash1 >> 22) & 0x3FFFFF);
  rowGlobal[49] = (float)((gameHash.hash1 >> 44) & 0xFFFFF);

  rowGlobal[50] = (float)state.plies;
  rowGlobal[51] = (float)startPly;
  rowGlobal[52] = (float)gameMode;
  rowGlobal[53] = (float)unreducedNumVisits;
  rowGlobal[54] = (float)nnRawStats.rawNNUtility;
  rowGlobal[55] = (float)numAliveAtRow;
  rowGlobal[56] = (float)repetitionDrawCount;
  rowGlobal[57] = (float)maxPlies;
  rowGlobal[58] = hitTurnLimit ? 1.0f : 0.0f;
  rowGlobal[59] = (!isSidePosition) ? 1.0f : 0.0f;
  rowGlobal[60] = 2.0f;
  rowGlobal[61] = (float)seatKind;
  rowGlobal[62] = 0.0f;
  rowGlobal[63] = 0.0f;

  // scoreDistrN [1]
  int8_t* rowScoreDistr = scoreDistrN.data + curRows * SCORE_DISTR_LEN;
  rowScoreDistr[0] = 0;

  // valueTargetsNCHW [12, 11, 11]
  int8_t* rowValueTargets = valueTargetsNCHW.data + curRows * VALUE_TARGET_CHANNELS * POS_AREA;
  std::fill_n(rowValueTargets, VALUE_TARGET_CHANNELS * POS_AREA, (int8_t)0);

  if(valueTargetWeight > 0.0f) {
    for(int k = 0; k < 4; k++) {
      int s = (toMove + k) % 4;
      if(!state.board.isAlive(s))
        continue;

      // C0–3: Cells visited by the pawn of relative seat k
      int8_t* pathChannel = rowValueTargets + k * POS_AREA;
      int curPawn = state.board.pawn[s];
      if(curPawn >= 0 && curPawn < POS_AREA)
        pathChannel[curPawn] = 1;
      for(const auto& b : boardHistoryFromTurnToEnd) {
        if(b.isAlive(s)) {
          int p = b.pawn[s];
          if(p >= 0 && p < POS_AREA)
            pathChannel[p] = 1;
        }
      }

      // C4–11: Walls placed from this row on by relative seat k
      int8_t* vWallChannel = rowValueTargets + (4 + 2 * k) * POS_AREA;
      int8_t* hWallChannel = rowValueTargets + (5 + 2 * k) * POS_AREA;
      for(size_t i = 0; i < actionsPlayedFromTurnToEnd.size(); i++) {
        if(actionSeatsFromTurnToEnd[i] == s) {
          int act = actionsPlayedFromTurnToEnd[i];
          if(Q4Board::isVWallAction(act)) {
            int anchor = act - 121;
            int ay = Q4Board::anchorY(anchor);
            int ax = Q4Board::anchorX(anchor);
            vWallChannel[ay * POS_LEN + ax] = 1;
          }
          else if(Q4Board::isHWallAction(act)) {
            int anchor = act - 221;
            int ay = Q4Board::anchorY(anchor);
            int ax = Q4Board::anchorX(anchor);
            hWallChannel[ay * POS_LEN + ax] = 1;
          }
        }
      }
    }
  }

  curRows++;
}

Q4TrainingDataWriter::Q4TrainingDataWriter(
  const std::string& outDir,
  int maxRows,
  double firstRandProp,
  uint64_t randSeed
) : outputDir(outDir),
    maxRowsPerTrainFile(maxRows),
    firstFileRandMinProp(firstRandProp),
    rand(randSeed),
    writeMutex(),
    writeBuffers(std::make_unique<Q4TrainingWriteBuffers>(maxRows)),
    totalRowsWritten(0),
    totalGamesWritten(0)
{
  MakeDir::make(outputDir);
  if(firstFileRandMinProp > 0.0 && firstFileRandMinProp < 1.0) {
    currentFileLimit = (int)round(rand.nextDouble(firstFileRandMinProp, 1.0) * maxRowsPerTrainFile);
    if(currentFileLimit < 1) currentFileLimit = 1;
  }
  else {
    currentFileLimit = maxRowsPerTrainFile;
  }
}

Q4TrainingDataWriter::~Q4TrainingDataWriter() {
  flushIfNonempty();
}

void Q4TrainingDataWriter::flushLocked() {
  if(writeBuffers->numRows() <= 0)
    return;

  std::string resultingFilename = outputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".npz";
  std::string tmpFilename = resultingFilename + ".tmp";
  writeBuffers->writeToZipFile(tmpFilename);
  totalRowsWritten.fetch_add(writeBuffers->numRows(), std::memory_order_relaxed);
  writeBuffers->clear();
  FileUtils::rename(tmpFilename, resultingFilename);
  currentFileLimit = maxRowsPerTrainFile;
}

void Q4TrainingDataWriter::flushIfNonempty() {
  std::lock_guard<std::mutex> lock(writeMutex);
  flushLocked();
}

void Q4TrainingDataWriter::writeGame(const Q4FinishedGameData& data) {
  std::lock_guard<std::mutex> lock(writeMutex);

  // Collect move actions and seats from events
  std::vector<Q4PlayState> statesBeforeMove;
  std::vector<int> actionsPlayed;
  std::vector<int> actionSeats;
  std::vector<Q4Board> allBoardsAfterStart;

  Q4PlayState stateReplay = data.startState;
  allBoardsAfterStart.push_back(stateReplay.board);

  size_t startEventIdx = data.startHist.events.size();
  for(size_t ei = startEventIdx; ei < data.endHist.events.size(); ei++) {
    const auto& ev = data.endHist.events[ei];
    if(ev.isElimination) {
      stateReplay.eliminate(ev.eliminatedSeat);
      allBoardsAfterStart.push_back(stateReplay.board);
    }
    else {
      statesBeforeMove.push_back(stateReplay);
      actionsPlayed.push_back(ev.action);
      actionSeats.push_back(stateReplay.board.toMove);
      stateReplay.playAssumeLegal(ev.action);
      allBoardsAfterStart.push_back(stateReplay.board);
    }
  }

  int numTurns = (int)statesBeforeMove.size();
  testAssert((int)data.targetWeightByTurn.size() == numTurns);
  testAssert((int)data.policyTargetsByTurn.size() == numTurns);
  testAssert((int)data.valueTargetsByTurn.size() == numTurns + 1);
  testAssert((int)data.nnRawStatsByTurn.size() == numTurns);
  testAssert((int)data.seatKindByTurn.size() == numTurns);

  // Final game properties
  int finalGamePlies = stateReplay.plies;
  int winnerSeat = data.endHist.winnerSeat;
  uint8_t finalDistToCenter[4];
  bool seatEliminatedBeforeEnd[4];

  for(int s = 0; s < 4; s++) {
    if(!stateReplay.board.isAlive(s)) {
      seatEliminatedBeforeEnd[s] = true;
      finalDistToCenter[s] = 255;
    }
    else {
      seatEliminatedBeforeEnd[s] = false;
      if(s == winnerSeat) {
        finalDistToCenter[s] = 0;
      }
      else {
        finalDistToCenter[s] = stateReplay.board.distToCenter[stateReplay.board.pawn[s]];
      }
    }
  }

  // Write main game rows
  for(int t = 0; t < numTurns; t++) {
    float targetWeight = data.targetWeightByTurn[t];
    if(targetWeight <= 0.0f)
      continue;

    const Q4PlayState& st = statesBeforeMove[t];
    int64_t unreducedVisits = data.policyTargetsByTurn[t].unreducedNumVisits;
    const auto* policy0 = data.policyTargetsByTurn[t].policyTargets;
    const bool nextIsObserver = (t + 1 < numTurns) && data.seatKindByTurn[t + 1] != 0;
    const auto* policyNext =
      (t + 1 < numTurns && !nextIsObserver) ? data.policyTargetsByTurn[t + 1].policyTargets : nullptr;
    int actionPlayed = actionsPlayed[t];
    double polSurprise = data.policySurpriseByTurn[t];
    double polEntropy = data.policyEntropyByTurn[t];
    double srchEntropy = data.searchEntropyByTurn[t];
    const auto& nnRaw = data.nnRawStatsByTurn[t];

    float valTargetWeight = 1.0f;
    float tdValWeight = 1.0f;

    // Board history and actions from turn t to end
    std::vector<Q4Board> boardsFromTToEnd;
    for(int j = t + 1; j < (int)allBoardsAfterStart.size(); j++) {
      boardsFromTToEnd.push_back(allBoardsAfterStart[j]);
    }
    std::vector<int> actsFromTToEnd;
    std::vector<int> seatsFromTToEnd;
    for(int j = t; j < numTurns; j++) {
      actsFromTToEnd.push_back(actionsPlayed[j]);
      seatsFromTToEnd.push_back(actionSeats[j]);
    }

    writeBuffers->addRow(
      st,
      t,
      targetWeight,
      unreducedVisits,
      policy0,
      policyNext,
      actionPlayed,
      polSurprise,
      polEntropy,
      srchEntropy,
      data.valueTargetsByTurn,
      t,
      valTargetWeight,
      tdValWeight,
      nnRaw,
      false, // isSidePosition
      data.gameHash,
      data.mode,
      data.startPly,
      data.hitTurnLimit,
      st.board.getNumAlive(),
      data.rules.maxPlies,
      data.rules.repetitionDrawCount,
      finalGamePlies,
      finalDistToCenter,
      seatEliminatedBeforeEnd,
      boardsFromTToEnd,
      actsFromTToEnd,
      seatsFromTToEnd,
      data.seatKindByTurn[t]
    );

    if(writeBuffers->numRows() >= currentFileLimit) {
      flushLocked();
    }
  }

  // Write side positions
  for(size_t i = 0; i < data.sidePositions.size(); i++) {
    const Q4SidePosition* sp = data.sidePositions[i];
    if(sp->targetWeight <= 0.0f)
      continue;

    std::vector<Q4ValueTargets> sideValueTargets = {sp->valueTargets};
    std::vector<Q4Board> emptyBoards;
    std::vector<int> emptyActs;
    std::vector<int> emptySeats;
    uint8_t zeroDist[4] = {0, 0, 0, 0};
    bool elimFlags[4] = {true, true, true, true};

    writeBuffers->addRow(
      sp->state,
      sp->state.plies,
      sp->targetWeight,
      sp->unreducedNumVisits,
      &(sp->policyTarget),
      nullptr, // policyNext
      Q4Board::NULL_ACTION,
      sp->policySurprise,
      sp->policyEntropy,
      sp->searchEntropy,
      sideValueTargets,
      0,
      0.0f, // valueTargetWeight = 0 for side position
      0.0f, // tdValueTargetWeight = 0
      sp->nnRawStats,
      true, // isSidePosition
      data.gameHash,
      data.mode,
      data.startPly,
      data.hitTurnLimit,
      sp->state.board.getNumAlive(),
      data.rules.maxPlies,
      data.rules.repetitionDrawCount,
      finalGamePlies,
      zeroDist,
      elimFlags,
      emptyBoards,
      emptyActs,
      emptySeats,
      0
    );

    if(writeBuffers->numRows() >= currentFileLimit) {
      flushLocked();
    }
  }

  totalGamesWritten.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace Q4Play
