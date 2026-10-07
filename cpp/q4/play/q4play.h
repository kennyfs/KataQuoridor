#ifndef Q4_PLAY_H_
#define Q4_PLAY_H_

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../../core/config_parser.h"
#include "../../core/global.h"
#include "../../core/logger.h"
#include "../../core/multithread.h"
#include "../../core/rand.h"
#include "../dataio/q4trainingwrite.h"
#include "../q4board.h"
#include "../q4history.h"
#include "../q4playstate.h"
#include "../q4rules.h"
#include "../search/q4search.h"
#include "q4playsettings.h"

namespace Q4Play {

struct Q4InitialPosition {
  Q4History history;
  bool isPlainFork;
  double trainingWeight;
  Q4Rules rules;

  Q4InitialPosition();
  Q4InitialPosition(const Q4History& h, bool plainFork, double weight, const Q4Rules& r);
  ~Q4InitialPosition() = default;
};

struct Q4ForkData {
  std::mutex mutex;
  std::vector<const Q4InitialPosition*> forks;

  Q4ForkData();
  ~Q4ForkData();

  void add(const Q4InitialPosition* pos);
  const Q4InitialPosition* get(Rand& rand);
};

struct Q4OtherGameProperties {
  bool allowPolicyInit = true;
  bool isFork = false;
  double trainingWeight = 1.0;
  int startPly = 0;
};

class Q4GameInitializer {
 public:
  Q4GameInitializer(ConfigParser& cfg, Logger& logger);
  Q4GameInitializer(ConfigParser& cfg, Logger& logger, const std::string& randSeed);
  ~Q4GameInitializer() = default;

  Q4GameInitializer(const Q4GameInitializer&) = delete;
  Q4GameInitializer& operator=(const Q4GameInitializer&) = delete;

  void createGame(
    Q4PlayState& state,
    Q4History& hist,
    const Q4InitialPosition* initialPosition,
    const Q4PlaySettings& playSettings,
    Q4OtherGameProperties& otherGameProps
  );

  Q4Rules createRules();
  const Q4Rules& getBaseRules() const { return baseRules; }

 private:
  void initShared(ConfigParser& cfg, Logger& logger);
  void createGameSharedUnsynchronized(
    Q4PlayState& state,
    Q4History& hist,
    const Q4InitialPosition* initialPosition,
    const Q4PlaySettings& playSettings,
    Q4OtherGameProperties& otherGameProps
  );

  std::mutex createGameMutex;
  Rand rand;
  Q4Rules baseRules;

  bool q4RepetitionDrawRandom;
  double q4RepetitionDrawProb;
  std::vector<int> q4RepetitionDrawCounts;
  std::vector<double> q4RepetitionDrawCountWeights;
};

namespace Play {
  int loadMaxMovesPerGame(ConfigParser& cfg);

  void extractPolicyTarget(
    std::vector<Q4PolicyTargetMove>& buf,
    const Q4S::Search* toMoveBot,
    const Q4S::SearchNode* node,
    std::vector<int>& actionsBuf,
    std::vector<double>& playSelectionValuesBuf
  );

  void initializeGameUsingPolicy(
    Q4S::Search* bot,
    Q4PlayState& state,
    Q4History& hist,
    Rand& gameRand,
    double proportionOfBoardArea,
    double policyInitGammaShape,
    double temperature
  );

  Q4FinishedGameData* runGame(
    const std::string& searchRandSeed,
    Q4S::Search* bot,
    const Q4PlayState& startState,
    const Q4History& startHist,
    bool clearBotBeforeSearch,
    Logger& logger,
    bool logSearchInfo,
    bool logMoves,
    int maxMovesPerGame,
    const std::function<bool()>& shouldStop,
    const WaitableFlag* shouldPause,
    const Q4PlaySettings& playSettings,
    const Q4OtherGameProperties& otherGameProps,
    Rand& gameRand,
    const std::function<NNEvaluator*()>& checkForNewNNEval,
    const std::function<void(const Q4PlayState&, int, const Q4S::Search*)>& onEachMove
  );

  void maybeForkGame(
    const Q4FinishedGameData* finishedGameData,
    Q4ForkData* forkData,
    const Q4PlaySettings& playSettings,
    Rand& gameRand,
    Q4S::Search* bot
  );
}

class Q4GameRunner {
  bool logSearchInfo;
  bool logMoves;
  int maxMovesPerGame;
  bool clearBotBeforeSearch;
  Q4PlaySettings playSettings;
  std::unique_ptr<Q4GameInitializer> gameInit;

 public:
  struct BotSpec {
    int botIdx = 0;
    std::string botName;
    NNEvaluator* nnEval = nullptr;
    SearchParams baseParams;
  };

  Q4GameRunner(ConfigParser& cfg, const Q4PlaySettings& playSettings, Logger& logger);
  Q4GameRunner(ConfigParser& cfg, const std::string& gameInitRandSeed, const Q4PlaySettings& playSettings, Logger& logger);
  ~Q4GameRunner() = default;

  Q4GameRunner(const Q4GameRunner&) = delete;
  Q4GameRunner& operator=(const Q4GameRunner&) = delete;

  Q4FinishedGameData* runGame(
    const std::string& seed,
    const BotSpec& botSpec,
    Q4ForkData* forkData,
    Logger& logger,
    const std::function<bool()>& shouldStop,
    const WaitableFlag* shouldPause,
    const std::function<NNEvaluator*()>& checkForNewNNEval,
    const std::function<void(const BotSpec&, Q4S::Search*)>& afterInitialization,
    const std::function<void(const Q4PlayState&, int, const Q4S::Search*)>& onEachMove
  );

  const Q4GameInitializer* getGameInitializer() const { return gameInit.get(); }
};

}  // namespace Q4Play

#endif  // Q4_PLAY_H_
