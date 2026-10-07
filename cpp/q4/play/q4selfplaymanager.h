#ifndef Q4_SELFPLAYMANAGER_H_
#define Q4_SELFPLAYMANAGER_H_

#include <atomic>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "../../core/threadsafequeue.h"
#include "../../core/timer.h"
#include "../../neuralnet/nneval.h"
#include "../dataio/q4trainingwrite.h"
#include "../q4record.h"
#include "q4play.h"

namespace Q4Play {

class Q4SelfPlayManager {
 public:
  Q4SelfPlayManager(
    int maxDataQueueSize,
    Logger* logger,
    int64_t logGamesEvery,
    bool autoCleanupAllButLatestIfUnused
  );
  ~Q4SelfPlayManager();

  Q4SelfPlayManager(const Q4SelfPlayManager&) = delete;
  Q4SelfPlayManager& operator=(const Q4SelfPlayManager&) = delete;

  void loadModelAndStartDataWriting(
    NNEvaluator* nnEval,
    Q4TrainingDataWriter* tdataWriter,
    std::ofstream* recordsOut
  );
  void loadModelNoDataWritingLoop(
    NNEvaluator* nnEval,
    Q4TrainingDataWriter* tdataWriter,
    std::ofstream* recordsOut
  );

  uint64_t getTotalNumRowsProcessed() const;

  std::vector<std::string> modelNames() const;
  std::string getLatestModelName() const;
  bool hasModel(const std::string& modelName) const;
  size_t numModels() const;

  NNEvaluator* acquireModel(const std::string& modelName);
  NNEvaluator* acquireLatest();
  void release(const std::string& modelName);
  void release(NNEvaluator* nnEval);

  void cleanupUnusedModelsOlderThan(double seconds);
  void clearUnusedModelCaches();

  void countOneGameStarted(NNEvaluator* nnEval);
  void countOneGameHitCutoff(NNEvaluator* nnEval, int64_t numMoves);
  void countQ4GameResult(NNEvaluator* nnEval, const Q4FinishedGameData& gameData);

  void enqueueDataToWrite(const std::string& modelName, Q4FinishedGameData* gameData);
  void enqueueDataToWrite(NNEvaluator* nnEval, Q4FinishedGameData* gameData);

  void withDataWriters(
    NNEvaluator* nnEval,
    const std::function<void(Q4TrainingDataWriter* tdataWriter, std::ofstream* recordsOut)>& f
  );

  struct ModelData {
    std::string modelName;
    NNEvaluator* nnEval;
    int64_t gameStartedCount;
    std::atomic<int64_t> gamesFinishedCount;
    std::atomic<int64_t> movesPlayedCount;
    std::atomic<int64_t> gamesCutoffCount;
    std::atomic<int64_t> movesPlayedCutoffCount;

    struct Q4Stats {
      std::mutex mutex;
      int64_t games = 0;
      int64_t plies = 0;
      int64_t winsBySeat[4] = {0, 0, 0, 0};
      int64_t maxPliesDraws = 0;
      int64_t repetitionDraws = 0;
      int64_t cutoffGames = 0;
      int64_t eliminations = 0;
      int64_t rowsWritten = 0;
    };
    Q4Stats q4Stats;

    double lastReleaseTime;
    bool hasDataWriteLoop;

    ThreadSafeQueue<Q4FinishedGameData*> finishedGameQueue;
    int acquireCount;

    Q4TrainingDataWriter* tdataWriter;
    std::ofstream* recordsOut;

    ModelData(
      const std::string& name,
      NNEvaluator* neval,
      int maxDataQueueSize,
      Q4TrainingDataWriter* tdWriter,
      std::ofstream* recOut,
      double initialLastReleaseTime,
      bool hasDataWriteLoop
    );
    ~ModelData();
  };

 private:
  const int maxDataQueueSize;
  Logger* logger;
  const int64_t logGamesEvery;
  const bool autoCleanupAllButLatestIfUnused;

  const ClockTimer timer;

  mutable std::mutex managerMutex;
  std::vector<ModelData*> modelDatas;
  int numDataWriteLoopsActive;
  std::condition_variable dataWriteLoopsAreDone;

  uint64_t totalNumRowsProcessed;

  NNEvaluator* acquireModelAlreadyLocked(ModelData* foundData);
  void releaseAlreadyLocked(ModelData* foundData);
  void maybeAutoCleanupAlreadyLocked();
  void runDataWriteLoopImpl(ModelData* modelData);
  static std::string q4StatsSummary(ModelData* modelData);

 public:
  void runDataWriteLoop(ModelData* modelData);
};

}  // namespace Q4Play

#endif  // Q4_SELFPLAYMANAGER_H_
