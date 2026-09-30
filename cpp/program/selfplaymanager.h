#ifndef PROGRAM_SELFPLAYMANAGER_H_
#define PROGRAM_SELFPLAYMANAGER_H_

#include <atomic>
#include <map>
#include <mutex>

#include "../core/threadsafequeue.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../dataio/trainingwrite.h"
#include "../neuralnet/nneval.h"

class SelfplayManager {
 public:
  SelfplayManager(
    int maxDataQueueSize,
    Logger* logger,
    int64_t logGamesEvery,
    bool autoCleanupAllButLatestIfUnused
  );
  ~SelfplayManager();

  SelfplayManager(const SelfplayManager& other);
  SelfplayManager& operator=(const SelfplayManager& other);
  SelfplayManager(SelfplayManager&& other);
  SelfplayManager& operator=(SelfplayManager&& other);

  //All below functions are internally synchronized and thread-safe.

  //SelfplayManager takes responsibility for deleting the data writers and closing and deleting sgfOut.
  //loadModelNoDataWritingLoop is for the manual writing interface
  void loadModelAndStartDataWriting(
    NNEvaluator* nnEval,
    TrainingDataWriter* tdataWriter,
    std::ofstream* sgfOut
  );
  void loadModelNoDataWritingLoop(
    NNEvaluator* nnEval,
    TrainingDataWriter* tdataWriter,
    std::ofstream* sgfOut
  );

  //NN queries summed across all the models managed by this manager over all time.
  uint64_t getTotalNumRowsProcessed() const;

  //For all of the below, model names are simply from nnEval->getModelName().

  //Models that aren't cleaned up yet are in the order from earliest to latest
  std::vector<std::string> modelNames() const;
  std::string getLatestModelName() const;
  bool hasModel(const std::string& modelName) const;
  size_t numModels() const;

  //Returns NULL if acquire failed (such as if that model was scheduled to be cleaned up or already cleaned up,).
  //Must call release when done, and cease using the NNEvaluator after that.
  NNEvaluator* acquireModel(const std::string& modelName);
  NNEvaluator* acquireLatest();
  //Release a model either by name or by the nnEval object that was returned.
  void release(const std::string& modelName);
  void release(NNEvaluator* nnEval);

  //Clean up any currently-unused models if their last usage was older than this many seconds ago.
  void cleanupUnusedModelsOlderThan(double seconds);
  //Clear the evaluation caches of any models that are currently unused.
  void clearUnusedModelCaches();

  //====================================================================================
  //These should only be called by a thread that has currently acquired the model.

  //Increment a counter and maybe log some stats
  void countOneGameStarted(NNEvaluator* nnEval);
  //Count a game that hit maxMovesPerGame and was discarded rather than written (numMoves = its length).
  //Games finished normally are counted by the data write loop (gamesFinishedCount).
  void countOneGameHitCutoff(NNEvaluator* nnEval, int64_t numMoves);
  //Quoridor I/O v2 step 3: count the result of every completed game (written or not) for the Quoridor stats line
  //(draw rate, first-player win rates, average plies).
  void countQuoridorGameResult(NNEvaluator* nnEval, const FinishedGameData& gameData);

  //SelfplayManager takes responsibility for deleting the gameData once written.
  //Use these only if loadModelAndStartDataWriting was used to start the model.
  void enqueueDataToWrite(const std::string& modelName, FinishedGameData* gameData);
  void enqueueDataToWrite(NNEvaluator* nnEval, FinishedGameData* gameData);

  //Use these if loadModelNoDataWritingLoop was used to start the model.
  void withDataWriters(
    NNEvaluator* nnEval,
    const std::function<void(TrainingDataWriter* tdataWriter, std::ofstream* sgfOut)>& f
  );

  //====================================================================================

  //For internal use
  struct ModelData {
    std::string modelName;
    NNEvaluator* nnEval;
    int64_t gameStartedCount;
    // Counted at game-finish in the data write loop (lock-free), read cross-thread for logging.
    std::atomic<int64_t> gamesFinishedCount;
    std::atomic<int64_t> movesPlayedCount;
    // Games that hit the move cutoff, and the moves in them. These games are not written or counted above.
    std::atomic<int64_t> gamesCutoffCount;
    std::atomic<int64_t> movesPlayedCutoffCount;
    // Quoridor results of all completed games (see countQuoridorGameResult), guarded by quoridorStats.mutex.
    struct QuoridorStats {
      std::mutex mutex;
      int64_t games = 0;
      int64_t draws = 0;
      int64_t plies = 0;
      // Normal games (not forks etc.) with the standard komi and 10/10 walls.
      int64_t standardGames = 0;
      int64_t standardBlackWins = 0;
      // Normal games with 10/10 walls by komi: (games, Black wins).
      std::map<float,std::pair<int64_t,int64_t>> byKomi;
      // Normal games with a fence handicap: (games, Black wins).
      std::pair<int64_t,int64_t> fenceHandicap = {0,0};
    };
    QuoridorStats quoridorStats;
    double lastReleaseTime;
    bool hasDataWriteLoop;

    ThreadSafeQueue<FinishedGameData*> finishedGameQueue;
    int acquireCount;

    TrainingDataWriter* tdataWriter;
    std::ofstream* sgfOut;

    ModelData(
      const std::string& name, NNEvaluator* neval, int maxDataQueueSize,
      TrainingDataWriter* tdWriter, std::ofstream* sOut,
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

  NNEvaluator* acquireModelAlreadyLocked(SelfplayManager::ModelData* foundData);
  void releaseAlreadyLocked(SelfplayManager::ModelData* foundData);
  void maybeAutoCleanupAlreadyLocked();
  void runDataWriteLoopImpl(ModelData* modelData);
  //One summary line: games started / finished normally / hit cutoff, cutoff rate, average game length.
  static std::string gameStatsSummary(const ModelData* modelData, int64_t gameStartedCount);
  //One line of Quoridor result stats: draw rate, average plies, first-player win rates (standard, by komi, handicap).
  static std::string quoridorStatsSummary(ModelData* modelData);

 public:
  //For internal use
  void runDataWriteLoop(ModelData* modelData);

};

#endif //PROGRAM_SELFPLAYMANAGER_H_
