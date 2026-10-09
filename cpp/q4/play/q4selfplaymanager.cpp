#include "q4selfplaymanager.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "../../core/test.h"

namespace Q4Play {

Q4SelfPlayManager::ModelData::ModelData(
  const std::string& name,
  NNEvaluator* neval,
  int maxDataQueueSize,
  Q4TrainingDataWriter* tdWriter,
  std::ofstream* recOut,
  double initialLastReleaseTime,
  bool hasDataLoop
) : modelName(name),
    nnEval(neval),
    gameStartedCount(0),
    gamesFinishedCount(0),
    movesPlayedCount(0),
    gamesCutoffCount(0),
    movesPlayedCutoffCount(0),
    q4Stats(),
    lastReleaseTime(initialLastReleaseTime),
    hasDataWriteLoop(hasDataLoop),
    finishedGameQueue(maxDataQueueSize),
    acquireCount(0),
    tdataWriter(tdWriter),
    recordsOut(recOut)
{}

Q4SelfPlayManager::ModelData::~ModelData() {
  if(tdataWriter != nullptr) {
    tdataWriter->flushIfNonempty();
    delete tdataWriter;
    tdataWriter = nullptr;
  }
  if(recordsOut != nullptr) {
    recordsOut->close();
    delete recordsOut;
    recordsOut = nullptr;
  }
  if(nnEval != nullptr) {
    delete nnEval;
    nnEval = nullptr;
  }
}

Q4SelfPlayManager::Q4SelfPlayManager(
  int maxQueueSize,
  Logger* log,
  int64_t logGamesEv,
  bool autoCleanup
) : maxDataQueueSize(maxQueueSize),
    logger(log),
    logGamesEvery(logGamesEv),
    autoCleanupAllButLatestIfUnused(autoCleanup),
    timer(),
    managerMutex(),
    modelDatas(),
    numDataWriteLoopsActive(0),
    dataWriteLoopsAreDone(),
    totalNumRowsProcessed(0)
{}

// Per-evaluator NN totals (the GPU batch-size benchmark of R8 Part C: learners vs snapshot evaluators)
static void logEvaluatorTotals(Logger* logger, const std::string& name, const NNEvaluator* nnEval) {
  if(logger == nullptr || nnEval == nullptr)
    return;
  logger->write(
    "NN evaluator totals " + name + ": " + Global::uint64ToString(nnEval->numRowsProcessed()) + " rows " +
    Global::uint64ToString(nnEval->numBatchesProcessed()) + " batches, average batch " +
    Global::doubleToString(nnEval->averageProcessedBatchSize())
  );
}

Q4SelfPlayManager::~Q4SelfPlayManager() {
  std::unique_lock<std::mutex> lock(managerMutex);
  for(size_t i = 0; i < modelDatas.size(); i++) {
    testAssert(modelDatas[i]->acquireCount == 0);
    logEvaluatorTotals(logger, modelDatas[i]->modelName, modelDatas[i]->nnEval);
    modelDatas[i]->finishedGameQueue.setReadOnly();
    totalNumRowsProcessed += modelDatas[i]->nnEval->numRowsProcessed();
    if(!modelDatas[i]->hasDataWriteLoop) {
      delete modelDatas[i];
    }
  }
  modelDatas.clear();
  while(numDataWriteLoopsActive > 0) {
    dataWriteLoopsAreDone.wait(lock);
  }
}

uint64_t Q4SelfPlayManager::getTotalNumRowsProcessed() const {
  std::lock_guard<std::mutex> lock(managerMutex);
  uint64_t total = totalNumRowsProcessed;
  for(size_t i = 0; i < modelDatas.size(); i++) {
    total += modelDatas[i]->nnEval->numRowsProcessed();
  }
  return total;
}

static void dataWriteLoopFunc(Q4SelfPlayManager* manager, Q4SelfPlayManager::ModelData* modelData) {
  manager->runDataWriteLoop(modelData);
}

void Q4SelfPlayManager::loadModelAndStartDataWriting(
  NNEvaluator* nnEval,
  Q4TrainingDataWriter* tdataWriter,
  std::ofstream* recordsOut
) {
  std::lock_guard<std::mutex> lock(managerMutex);
  std::string modelName = nnEval->getModelName();
  for(size_t i = 0; i < modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      throw StringError("Model with name " + modelName + " already loaded into SelfPlayManager");
    }
  }

  ModelData* modelData = new ModelData(
    modelName, nnEval, maxDataQueueSize, tdataWriter, recordsOut, timer.getSeconds(), true
  );
  modelDatas.push_back(modelData);
  maybeAutoCleanupAlreadyLocked();

  numDataWriteLoopsActive++;
  std::thread dataWriteLoopThread(dataWriteLoopFunc, this, modelData);
  dataWriteLoopThread.detach();
}

void Q4SelfPlayManager::loadModelNoDataWritingLoop(
  NNEvaluator* nnEval,
  Q4TrainingDataWriter* tdataWriter,
  std::ofstream* recordsOut
) {
  std::lock_guard<std::mutex> lock(managerMutex);
  std::string modelName = nnEval->getModelName();
  for(size_t i = 0; i < modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      throw StringError("Model with name " + modelName + " already loaded into SelfPlayManager");
    }
  }

  ModelData* modelData = new ModelData(
    modelName, nnEval, maxDataQueueSize, tdataWriter, recordsOut, timer.getSeconds(), false
  );
  modelDatas.push_back(modelData);
  maybeAutoCleanupAlreadyLocked();
}

void Q4SelfPlayManager::maybeAutoCleanupAlreadyLocked() {
  if(autoCleanupAllButLatestIfUnused && modelDatas.size() > 1) {
    for(size_t i = 0; i < modelDatas.size() - 1; i++) {
      ModelData* foundData = modelDatas[i];
      if(foundData->acquireCount <= 0) {
        testAssert(foundData->acquireCount == 0);
        foundData->finishedGameQueue.setReadOnly();
        totalNumRowsProcessed += foundData->nnEval->numRowsProcessed();
        if(!foundData->hasDataWriteLoop) {
          delete foundData;
        }
        modelDatas.erase(modelDatas.begin() + i);
        i--;
      }
    }
  }
}

void Q4SelfPlayManager::cleanupUnusedModelsOlderThan(double seconds) {
  std::lock_guard<std::mutex> lock(managerMutex);
  double now = timer.getSeconds();
  for(size_t i = 0; i < modelDatas.size(); i++) {
    ModelData* foundData = modelDatas[i];
    if(foundData->acquireCount <= 0 && now - foundData->lastReleaseTime > seconds) {
      testAssert(foundData->acquireCount == 0);
      if(logger != nullptr)
        logger->write("Unloading network that hasn't been used in a while: " + foundData->modelName);
      foundData->finishedGameQueue.setReadOnly();
      totalNumRowsProcessed += foundData->nnEval->numRowsProcessed();
      if(!foundData->hasDataWriteLoop) {
        delete foundData;
      }
      modelDatas.erase(modelDatas.begin() + i);
      i--;
    }
  }
}

void Q4SelfPlayManager::unloadUnusedModelsNotIn(const std::vector<std::string>& keep) {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i < modelDatas.size(); i++) {
    ModelData* foundData = modelDatas[i];
    bool wanted = std::find(keep.begin(), keep.end(), foundData->modelName) != keep.end();
    if(!wanted && foundData->acquireCount <= 0) {
      if(logger != nullptr)
        logger->write("Unloading snapshot network: " + foundData->modelName);
      logEvaluatorTotals(logger, foundData->modelName, foundData->nnEval);
      foundData->finishedGameQueue.setReadOnly();
      totalNumRowsProcessed += foundData->nnEval->numRowsProcessed();
      if(!foundData->hasDataWriteLoop) {
        delete foundData;
      }
      modelDatas.erase(modelDatas.begin() + i);
      i--;
    }
  }
}

void Q4SelfPlayManager::clearUnusedModelCaches() {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i < modelDatas.size(); i++) {
    ModelData* foundData = modelDatas[i];
    if(foundData->acquireCount <= 0) {
      foundData->nnEval->clearCache();
    }
  }
}

std::vector<std::string> Q4SelfPlayManager::modelNames() const {
  std::lock_guard<std::mutex> lock(managerMutex);
  std::vector<std::string> names;
  for(size_t i = 0; i < modelDatas.size(); i++) {
    names.push_back(modelDatas[i]->modelName);
  }
  return names;
}

std::string Q4SelfPlayManager::getLatestModelName() const {
  std::lock_guard<std::mutex> lock(managerMutex);
  if(modelDatas.empty())
    return std::string();
  return modelDatas.back()->modelName;
}

bool Q4SelfPlayManager::hasModel(const std::string& modelName) const {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i < modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName)
      return true;
  }
  return false;
}

size_t Q4SelfPlayManager::numModels() const {
  std::lock_guard<std::mutex> lock(managerMutex);
  return modelDatas.size();
}

NNEvaluator* Q4SelfPlayManager::acquireModelAlreadyLocked(ModelData* foundData) {
  testAssert(foundData != nullptr);
  foundData->acquireCount++;
  return foundData->nnEval;
}

NNEvaluator* Q4SelfPlayManager::acquireModel(const std::string& modelName) {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i < modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      return acquireModelAlreadyLocked(modelDatas[i]);
    }
  }
  return nullptr;
}

NNEvaluator* Q4SelfPlayManager::acquireLatest() {
  std::lock_guard<std::mutex> lock(managerMutex);
  if(modelDatas.empty())
    return nullptr;
  return acquireModelAlreadyLocked(modelDatas.back());
}

void Q4SelfPlayManager::releaseAlreadyLocked(ModelData* foundData) {
  testAssert(foundData != nullptr);
  foundData->acquireCount--;
  testAssert(foundData->acquireCount >= 0);
  if(foundData->acquireCount == 0) {
    foundData->lastReleaseTime = timer.getSeconds();
  }
}

void Q4SelfPlayManager::release(const std::string& modelName) {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i < modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      releaseAlreadyLocked(modelDatas[i]);
      return;
    }
  }
  testAssert(false);
}

void Q4SelfPlayManager::release(NNEvaluator* nnEval) {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i < modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      releaseAlreadyLocked(modelDatas[i]);
      return;
    }
  }
  testAssert(false);
}

void Q4SelfPlayManager::countOneGameStarted(NNEvaluator* nnEval) {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i < modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      modelDatas[i]->gameStartedCount++;
      return;
    }
  }
}

void Q4SelfPlayManager::countOneGameHitCutoff(NNEvaluator* nnEval, int64_t numMoves) {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i < modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      modelDatas[i]->gamesCutoffCount.fetch_add(1, std::memory_order_relaxed);
      modelDatas[i]->movesPlayedCutoffCount.fetch_add(numMoves, std::memory_order_relaxed);
      return;
    }
  }
}

std::string Q4SelfPlayManager::q4StatsSummary(ModelData* modelData) {
  std::lock_guard<std::mutex> lock(modelData->q4Stats.mutex);
  const auto& s = modelData->q4Stats;
  if(s.games <= 0) return std::string();

  std::ostringstream out;
  double avgPlies = (s.games > 0) ? (double)s.plies / s.games : 0.0;
  out << "Model " << modelData->modelName << " stats (" << s.games << " games): "
      << "avg plies = " << std::fixed << std::setprecision(1) << avgPlies
      << " | win rates: S0=" << std::setprecision(1) << (100.0 * s.winsBySeat[0] / s.games) << "%"
      << " S1=" << (100.0 * s.winsBySeat[1] / s.games) << "%"
      << " S2=" << (100.0 * s.winsBySeat[2] / s.games) << "%"
      << " S3=" << (100.0 * s.winsBySeat[3] / s.games) << "%"
      << " | draw rate = " << (100.0 * (s.maxPliesDraws + s.repetitionDraws) / s.games) << "%"
      << " (maxPlies: " << (100.0 * s.maxPliesDraws / s.games) << "%"
      << ", rep: " << (100.0 * s.repetitionDraws / s.games) << "%)"
      << " | eliminations = " << s.eliminations
      << " | rows written = " << s.rowsWritten;
  // Learner seats' win rate (points per learner seat; 0.25 is parity at a four-learner table) by number of learners
  // and by the kind of the other seats in the game.
  bool anyMixed = false;
  for(int n = 1; n <= 3; n++)
    anyMixed = anyMixed || s.popGames[n] > 0;
  if(anyMixed) {
    out << "\n  learner win rate by number of learners:";
    for(int n = 1; n <= 4; n++) {
      if(s.popGames[n] <= 0) continue;
      out << " " << n << ": " << std::setprecision(1) << (100.0 * s.popLearnerPoints[n] / s.popLearnerSeats[n])
          << "% per learner seat (" << s.popGames[n] << " games)";
    }
    out << "\n  by opponent kind (learner / opponent points per seat in games containing that kind):";
    for(int k = SEAT_WEAK; k < NUM_SEAT_KINDS; k++) {
      if(s.kindGames[k] <= 0) continue;
      out << " " << seatKindName(k) << ": " << std::setprecision(1)
          << (100.0 * s.kindLearnerPoints[k] / std::max<int64_t>(1, s.kindLearnerSeats[k])) << "% / "
          << (100.0 * s.kindPoints[k] / s.kindSeats[k]) << "% (" << s.kindGames[k] << " games)";
    }
  }
  return out.str();
}

void Q4SelfPlayManager::countQ4GameResult(NNEvaluator* nnEval, const Q4FinishedGameData& gameData) {
  ModelData* targetData = nullptr;
  {
    std::lock_guard<std::mutex> lock(managerMutex);
    for(size_t i = 0; i < modelDatas.size(); i++) {
      if(modelDatas[i]->nnEval == nnEval) {
        targetData = modelDatas[i];
        break;
      }
    }
  }
  if(targetData == nullptr) return;

  bool shouldLog = false;
  std::string summaryLine;
  {
    std::lock_guard<std::mutex> lock(targetData->q4Stats.mutex);
    auto& s = targetData->q4Stats;
    s.games++;
    s.plies += gameData.endHist.plies;

    if(gameData.endHist.winnerSeat >= 0 && gameData.endHist.winnerSeat < 4) {
      s.winsBySeat[gameData.endHist.winnerSeat]++;
    }
    else if(gameData.endHist.isDraw) {
      if(gameData.endHist.plies >= gameData.rules.maxPlies) {
        s.maxPliesDraws++;
      }
      else {
        s.repetitionDraws++;
      }
    }
    else if(gameData.hitTurnLimit) {
      s.cutoffGames++;
    }

    if(!gameData.hitTurnLimit) {
      // points of the seats: the winner 1, a draw 1/numAlive for each alive seat
      double points[4] = {0, 0, 0, 0};
      if(gameData.endHist.winnerSeat >= 0 && gameData.endHist.winnerSeat < 4) {
        points[gameData.endHist.winnerSeat] = 1.0;
      }
      else if(gameData.endHist.isDraw) {
        int numAlive = gameData.endHist.currentBoard.getNumAlive();
        for(int seat = 0; seat < 4; seat++)
          if(gameData.endHist.currentBoard.isAlive(seat))
            points[seat] = 1.0 / numAlive;
      }
      int numLearners = 0;
      double learnerPoints = 0.0;
      int kindCount[NUM_SEAT_KINDS] = {0, 0, 0, 0, 0, 0, 0};
      double kindPts[NUM_SEAT_KINDS] = {0, 0, 0, 0, 0, 0, 0};
      for(int seat = 0; seat < 4; seat++) {
        int kind = gameData.seatInfo[seat].kind;
        kindCount[kind]++;
        kindPts[kind] += points[seat];
        if(kind == SEAT_LEARNER) {
          numLearners++;
          learnerPoints += points[seat];
        }
      }
      s.popGames[numLearners]++;
      s.popLearnerSeats[numLearners] += numLearners;
      s.popLearnerPoints[numLearners] += learnerPoints;
      for(int kind = SEAT_WEAK; kind < NUM_SEAT_KINDS; kind++) {
        if(kindCount[kind] <= 0) continue;
        s.kindGames[kind]++;
        s.kindSeats[kind] += kindCount[kind];
        s.kindPoints[kind] += kindPts[kind];
        s.kindLearnerSeats[kind] += numLearners;
        s.kindLearnerPoints[kind] += learnerPoints;
      }
    }

    for(const auto& ev : gameData.endHist.events) {
      if(ev.isElimination) {
        s.eliminations++;
        break;
      }
    }

    for(float w : gameData.targetWeightByTurn) {
      s.rowsWritten += (int64_t)std::round(w);
    }
    for(const auto* sp : gameData.sidePositions) {
      s.rowsWritten += (int64_t)std::round(sp->targetWeight);
    }

    if(logGamesEvery > 0 && s.games % logGamesEvery == 0) {
      shouldLog = true;
    }
  }

  if(shouldLog) {
    summaryLine = q4StatsSummary(targetData);
    if(!summaryLine.empty() && logger != nullptr) {
      logger->write(summaryLine);
    }
  }
}

void Q4SelfPlayManager::enqueueDataToWrite(const std::string& modelName, Q4FinishedGameData* gameData) {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i < modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      bool suc = modelDatas[i]->finishedGameQueue.waitPush(gameData);
      testAssert(suc);
      return;
    }
  }
  testAssert(false);
}

void Q4SelfPlayManager::enqueueDataToWrite(NNEvaluator* nnEval, Q4FinishedGameData* gameData) {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i < modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      bool suc = modelDatas[i]->finishedGameQueue.waitPush(gameData);
      testAssert(suc);
      return;
    }
  }
  testAssert(false);
}

void Q4SelfPlayManager::withDataWriters(
  NNEvaluator* nnEval,
  const std::function<void(Q4TrainingDataWriter* tdataWriter, std::ofstream* recordsOut)>& f
) {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i < modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      f(modelDatas[i]->tdataWriter, modelDatas[i]->recordsOut);
      return;
    }
  }
  testAssert(false);
}

void Q4SelfPlayManager::runDataWriteLoop(ModelData* modelData) {
  runDataWriteLoopImpl(modelData);
  delete modelData;
  {
    std::lock_guard<std::mutex> lock(managerMutex);
    numDataWriteLoopsActive--;
    if(numDataWriteLoopsActive == 0) {
      dataWriteLoopsAreDone.notify_all();
    }
  }
}

void Q4SelfPlayManager::runDataWriteLoopImpl(ModelData* modelData) {
  Q4FinishedGameData* gameData = nullptr;
  while(modelData->finishedGameQueue.waitPop(gameData)) {
    modelData->gamesFinishedCount.fetch_add(1, std::memory_order_relaxed);
    modelData->movesPlayedCount.fetch_add(gameData->endHist.plies, std::memory_order_relaxed);

    // Write record to recordsOut
    if(modelData->recordsOut != nullptr) {
      Q4Record rec;
      rec.rules = gameData->rules;
      rec.players.clear();
      for(int s = 0; s < 4; s++) {
        const Q4SeatInfo& info = gameData->seatInfo[s];
        Q4PlayerInfo p;
        if(info.kind == SEAT_LEARNER) {
          p.name = modelData->modelName;
          p.type = "selfplay";
          p.net = modelData->modelName;
        }
        else {
          p.name = info.net.empty() ? std::string(seatKindName(info.kind)) : info.net;
          if(info.kind == SEAT_GRUDGE)
            p.name += ">" + std::to_string(info.grudgeTarget);
          p.type = seatKindName(info.kind);
          p.net = info.net;
          p.visits = info.visits;
        }
        rec.players.push_back(p);
      }
      if(gameData->hitTurnLimit) {
        rec.result = "none";
      }
      else if(gameData->endHist.isDraw) {
        rec.result = "Draw";
      }
      else if(gameData->endHist.winnerSeat >= 0 && gameData->endHist.winnerSeat < 4) {
        rec.result = std::to_string(gameData->endHist.winnerSeat + 1) + "+";
      }
      else {
        rec.result = "none";
      }
      rec.events = gameData->endHist.events;
      rec.hasGameHash = true;
      rec.gameHash0 = gameData->gameHash.hash0;
      rec.gameHash1 = gameData->gameHash.hash1;
      rec.startTurnIdx = (int)gameData->startHist.events.size();
      bool anyOther = false;
      for(int s = 0; s < 4; s++)
        anyOther = anyOther || gameData->seatInfo[s].kind != SEAT_LEARNER;
      rec.gtype = gameData->mode == Q4FinishedGameData::MODE_FORK ? "fork" : (anyOther ? "mixed" : "normal");
      if(gameData->endHist.isDraw)
        rec.drawReason = gameData->endHist.plies >= gameData->rules.maxPlies ? "maxPlies" : "repetition";
      for(size_t k = 0; k < gameData->comments.size(); k++)
        rec.setMoveComment(rec.startTurnIdx + k, gameData->comments[k]);
      *modelData->recordsOut << rec.toSgfLine() << "\n";
      modelData->recordsOut->flush();
    }

    // Write training rows
    if(!gameData->hitTurnLimit && modelData->tdataWriter != nullptr) {
      modelData->tdataWriter->writeGame(*gameData);
    }

    delete gameData;
  }

  if(modelData->tdataWriter != nullptr) {
    modelData->tdataWriter->flushIfNonempty();
  }

  // End of this net's cycle: the final statistics (including the population win rates)
  std::string summary = q4StatsSummary(modelData);
  if(!summary.empty() && logger != nullptr)
    logger->write("Final " + summary);
}

}  // namespace Q4Play
