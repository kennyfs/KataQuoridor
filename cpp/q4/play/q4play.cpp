#include "q4play.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "../../core/fileutils.h"
#include "../../core/makedir.h"
#include "../../core/timer.h"
#include "../nn/q4nn.h"

namespace Q4Play {

//---------------------------------------------------------------------------------------------------------
// Initial position & Fork data
//---------------------------------------------------------------------------------------------------------

Q4InitialPosition::Q4InitialPosition()
  : history(), isPlainFork(false), trainingWeight(1.0), rules() {}

Q4InitialPosition::Q4InitialPosition(const Q4History& h, bool plainFork, double weight, const Q4Rules& r)
  : history(h), isPlainFork(plainFork), trainingWeight(weight), rules(r) {}

Q4ForkData::Q4ForkData() : mutex(), forks() {}

Q4ForkData::~Q4ForkData() {
  for(const auto* f : forks)
    delete f;
  forks.clear();
}

void Q4ForkData::add(const Q4InitialPosition* pos) {
  std::lock_guard<std::mutex> lock(mutex);
  forks.push_back(pos);
}

const Q4InitialPosition* Q4ForkData::get(Rand& rand) {
  std::lock_guard<std::mutex> lock(mutex);
  if(forks.empty())
    return nullptr;
  size_t idx = rand.nextUInt((uint32_t)forks.size());
  const Q4InitialPosition* pos = forks[idx];
  forks.erase(forks.begin() + idx);
  return pos;
}

//---------------------------------------------------------------------------------------------------------
// Game Initializer
//---------------------------------------------------------------------------------------------------------

Q4GameInitializer::Q4GameInitializer(ConfigParser& cfg, Logger& logger)
  : createGameMutex(), rand(), baseRules() {
  initShared(cfg, logger);
}

Q4GameInitializer::Q4GameInitializer(ConfigParser& cfg, Logger& logger, const std::string& randSeed)
  : createGameMutex(), rand(randSeed), baseRules() {
  initShared(cfg, logger);
}

void Q4GameInitializer::initShared(ConfigParser& cfg, Logger& logger) {
  (void)logger;
  baseRules = Q4Rules();
  if(cfg.contains("maxPlies"))
    baseRules.maxPlies = cfg.getInt("maxPlies", 1, 100000);
  if(cfg.contains("repetitionDrawCount"))
    baseRules.repetitionDrawCount = cfg.getInt("repetitionDrawCount", 0, Q4Rules::MAX_REPETITION_DRAW_COUNT);

  q4RepetitionDrawRandom = cfg.contains("q4RepetitionDrawProb");
  q4RepetitionDrawProb = q4RepetitionDrawRandom ? cfg.getDouble("q4RepetitionDrawProb", 0.0, 1.0) : 0.0;
  q4RepetitionDrawCounts = cfg.contains("q4RepetitionDrawCounts") ?
    cfg.getInts("q4RepetitionDrawCounts", 2, Q4Rules::MAX_REPETITION_DRAW_COUNT) : std::vector<int>({3});
  q4RepetitionDrawCountWeights = cfg.contains("q4RepetitionDrawCountWeights") ?
    cfg.getDoubles("q4RepetitionDrawCountWeights", 0.0, 1e10) : std::vector<double>(q4RepetitionDrawCounts.size(), 1.0);

  if(q4RepetitionDrawRandom) {
    double sum = 0.0;
    for(double w : q4RepetitionDrawCountWeights) sum += w;
    if(q4RepetitionDrawCounts.empty() || q4RepetitionDrawCountWeights.size() != q4RepetitionDrawCounts.size() || !(sum > 0.0)) {
      throw StringError("q4RepetitionDrawCountWeights must give one non-negative weight per q4RepetitionDrawCounts entry, with positive sum");
    }
  }
}

Q4Rules Q4GameInitializer::createRules() {
  std::lock_guard<std::mutex> lock(createGameMutex);
  Q4Rules rules = baseRules;
  if(q4RepetitionDrawRandom) {
    if(rand.nextBool(q4RepetitionDrawProb)) {
      rules.repetitionDrawCount = q4RepetitionDrawCounts[rand.nextUInt(q4RepetitionDrawCountWeights.data(), q4RepetitionDrawCountWeights.size())];
    }
    else {
      rules.repetitionDrawCount = 0;
    }
  }
  return rules;
}

void Q4GameInitializer::createGameSharedUnsynchronized(
  Q4PlayState& state,
  Q4History& hist,
  const Q4InitialPosition* initialPosition,
  const Q4PlaySettings& playSettings,
  Q4OtherGameProperties& otherGameProps
) {
  (void)playSettings;
  if(initialPosition != nullptr) {
    hist = initialPosition->history;
    state = hist.getState();
    otherGameProps.isFork = initialPosition->isPlainFork;
    otherGameProps.trainingWeight = initialPosition->trainingWeight;
    otherGameProps.startPly = hist.plies;
    otherGameProps.allowPolicyInit = false;
  }
  else {
    Q4Rules rules = baseRules;
    if(q4RepetitionDrawRandom) {
      if(rand.nextBool(q4RepetitionDrawProb)) {
        rules.repetitionDrawCount = q4RepetitionDrawCounts[rand.nextUInt(q4RepetitionDrawCountWeights.data(), q4RepetitionDrawCountWeights.size())];
      }
      else {
        rules.repetitionDrawCount = 0;
      }
    }
    state = Q4PlayState(rules);
    hist = Q4History(rules);
    otherGameProps.isFork = false;
    otherGameProps.trainingWeight = 1.0;
    otherGameProps.startPly = 0;
    otherGameProps.allowPolicyInit = true;
  }
}

void Q4GameInitializer::createGame(
  Q4PlayState& state,
  Q4History& hist,
  const Q4InitialPosition* initialPosition,
  const Q4PlaySettings& playSettings,
  Q4OtherGameProperties& otherGameProps
) {
  std::lock_guard<std::mutex> lock(createGameMutex);
  createGameSharedUnsynchronized(state, hist, initialPosition, playSettings, otherGameProps);
}

//---------------------------------------------------------------------------------------------------------
// Play namespace implementation
//---------------------------------------------------------------------------------------------------------

namespace Play {

int loadMaxMovesPerGame(ConfigParser& cfg) {
  int maxPlies = cfg.contains("maxPlies") ? cfg.getInt("maxPlies", 1, 100000) : 400;
  for(const std::string& key : {std::string("maxMovesPerGame"), std::string("cutoffMoves")}) {
    if(cfg.contains(key)) {
      int val = cfg.getInt(key, 0, 1 << 30);
      if(val == 0) return 0;
      if(val != maxPlies) {
        throw IOError(
          key + " = " + Global::intToString(val) + " disagrees with maxPlies = " + Global::intToString(maxPlies) +
          " in " + cfg.getFileName() + ". Set maxPlies instead or make them equal."
        );
      }
    }
  }
  return maxPlies;
}

void extractPolicyTarget(
  std::vector<Q4PolicyTargetMove>& buf,
  const Q4S::Search* toMoveBot,
  const Q4S::SearchNode* node,
  std::vector<int>& actionsBuf,
  std::vector<double>& playSelectionValuesBuf
) {
  double scaleMaxToAtLeast = 10.0;
  testAssert(node != nullptr);
  const bool allowDirectPolicyMoves = false;
  double lcbBuf[Q4Board::NUM_ACTIONS];
  double radiusBuf[Q4Board::NUM_ACTIONS];
  bool success = toMoveBot->getPlaySelectionValues(
    *node, actionsBuf, playSelectionValuesBuf, nullptr, scaleMaxToAtLeast, allowDirectPolicyMoves, false, false,
    lcbBuf, radiusBuf
  );
  testAssert(success);
  testAssert(actionsBuf.size() == playSelectionValuesBuf.size());

  double maxValue = 0.0;
  for(double val : playSelectionValuesBuf) {
    testAssert(std::isfinite(val) && val >= 0.0);
    if(val > maxValue) maxValue = val;
  }
  double factor = 1.0;
  if(maxValue > 30000.0)
    factor = 30000.0 / maxValue;

  buf.clear();
  buf.reserve(actionsBuf.size());
  for(size_t i = 0; i < actionsBuf.size(); i++) {
    double val = playSelectionValuesBuf[i] * factor;
    buf.emplace_back(actionsBuf[i], (int16_t)std::round(val));
  }
}

void initializeGameUsingPolicy(
  Q4S::Search* bot,
  Q4PlayState& state,
  Q4History& hist,
  Rand& gameRand,
  double proportionOfBoardArea,
  double policyInitGammaShape,
  double temperature
) {
  if(state.isFinished)
    return;
  double mean = 121.0 * proportionOfBoardArea;
  int numInitialMovesToPlay;
  if(policyInitGammaShape != 1.0) {
    numInitialMovesToPlay = (int)std::floor(gameRand.nextGamma(policyInitGammaShape) * (mean / policyInitGammaShape));
  }
  else {
    numInitialMovesToPlay = (int)std::floor(gameRand.nextExponential() * mean);
  }

  NNResultBuf buf;
  for(int i = 0; i < numInitialMovesToPlay; i++) {
    if(state.isFinished)
      break;
    std::vector<int> legalActions;
    state.getLegalActions(legalActions);
    if(legalActions.empty())
      break;

    Q4NN::Eval eval;
    Q4NN::evaluate(*bot->nnEvaluator, buf, hist, 0, false, eval, &gameRand);

    std::vector<double> relProbs;
    relProbs.reserve(legalActions.size());
    for(int act : legalActions) {
      double prob = eval.policyProbs[0][act];
      relProbs.push_back(std::pow(std::max(prob, 1e-30), 1.0 / temperature));
    }

    uint32_t chosenIdx;
    if(gameRand.nextBool(0.0002)) {
      chosenIdx = gameRand.nextUInt((uint32_t)legalActions.size());
    }
    else {
      chosenIdx = gameRand.nextUInt(relProbs.data(), relProbs.size());
    }
    int chosenAction = legalActions[chosenIdx];

    testAssert(state.isLegalAction(chosenAction));
    hist.play(chosenAction);
    state.playAssumeLegal(chosenAction);
  }
}

static int chooseRandomForkingMove(
  const float* policyProbs,
  const Q4PlayState& state,
  Rand& gameRand,
  int banAction
) {
  std::vector<int> legalActions;
  state.getLegalActions(legalActions);
  if(legalActions.empty())
    return Q4Board::NULL_ACTION;
  if(legalActions.size() == 1)
    return legalActions[0];

  std::vector<int> candidateActions;
  for(int a : legalActions) {
    if(a != banAction) candidateActions.push_back(a);
  }
  if(candidateActions.empty())
    candidateActions = legalActions;

  double r = gameRand.nextDouble();
  if(r < 0.70) {
    std::vector<double> probs;
    for(int a : candidateActions) probs.push_back(std::max((double)policyProbs[a], 1e-30));
    return candidateActions[gameRand.nextUInt(probs.data(), probs.size())];
  }
  else if(r < 0.95) {
    std::vector<double> probs;
    for(int a : candidateActions) probs.push_back(std::pow(std::max((double)policyProbs[a], 1e-30), 0.5));
    return candidateActions[gameRand.nextUInt(probs.data(), probs.size())];
  }
  else {
    return candidateActions[gameRand.nextUInt((uint32_t)candidateActions.size())];
  }
}

static double valueSurpriseKL(const double target[5], const float rawNN[5]) {
  double kl = 0.0;
  for(int c = 0; c < 5; c++) {
    if(target[c] > 1e-100) {
      double p = target[c];
      double q = std::max((double)rawNN[c], 1e-100);
      kl += p * (std::log(p) - std::log(q));
    }
  }
  return std::min(std::max(0.0, kl), 1.0);
}

static void computeValueSurpriseByTurn(
  std::vector<double>& valueSurpriseByTurn,
  const std::vector<Q4ValueTargets>& valueTargetsByTurn,
  const std::vector<std::vector<float>>& rawNNValuesByTurn,
  bool useSearchValueSurprise
) {
  size_t numTurns = rawNNValuesByTurn.size();
  testAssert(valueTargetsByTurn.size() == numTurns + 1);
  valueSurpriseByTurn.resize(numTurns);

  if(useSearchValueSurprise) {
    for(size_t i = 0; i < numTurns; i++) {
      double target[5];
      for(int c = 0; c < 5; c++) target[c] = valueTargetsByTurn[i].value[c];
      valueSurpriseByTurn[i] = valueSurpriseKL(target, rawNNValuesByTurn[i].data());
    }
    return;
  }

  double nowFactor = 1.0 / (1.0 + 121.0 * 0.016);
  double smoothed[5];
  for(int c = 0; c < 5; c++) smoothed[c] = valueTargetsByTurn[numTurns].value[c];

  for(int i = (int)numTurns - 1; i >= 0; i--) {
    for(int c = 0; c < 5; c++) {
      smoothed[c] = smoothed[c] + nowFactor * (valueTargetsByTurn[i].value[c] - smoothed[c]);
    }
    valueSurpriseByTurn[i] = valueSurpriseKL(smoothed, rawNNValuesByTurn[i].data());
  }
}

struct SearchLimitsThisMove {
  int64_t numAlterVisits = 0;
  bool doAlterVisits = false;
  bool removeRootNoise = false;
  float targetWeight = 1.0f;
  bool isCheapSearch = false;
};

static SearchLimitsThisMove getSearchLimitsThisMove(
  Q4S::Search* bot,
  const Q4PlaySettings& playSettings,
  Rand& gameRand,
  const std::vector<double>& historicalMaxWinrates
) {
  SearchLimitsThisMove limits;
  limits.numAlterVisits = bot->searchParams.maxVisits;
  limits.targetWeight = 1.0f;

  if(playSettings.cheapSearchProb > 0.0 && gameRand.nextBool(playSettings.cheapSearchProb)) {
    limits.doAlterVisits = true;
    limits.isCheapSearch = true;
    limits.numAlterVisits = std::min(limits.numAlterVisits, (int64_t)playSettings.cheapSearchVisits);
    limits.targetWeight *= playSettings.cheapSearchTargetWeight;
    if(playSettings.cheapSearchTargetWeight <= 0.0f) {
      limits.removeRootNoise = true;
    }
  }
  else if(playSettings.reduceVisits) {
    size_t lookback = (size_t)playSettings.reduceVisitsThresholdLookback;
    if(historicalMaxWinrates.size() >= lookback && lookback > 0) {
      double minWin = 1e9;
      for(size_t j = 0; j < lookback; j++) {
        double w = historicalMaxWinrates[historicalMaxWinrates.size() - 1 - j];
        if(w < minWin) minWin = w;
      }
      if(minWin > playSettings.reduceVisitsThreshold) {
        double prop = (minWin - playSettings.reduceVisitsThreshold) / (1.0 - playSettings.reduceVisitsThreshold);
        prop = std::min(1.0, std::max(0.0, prop));
        double visitReductionProp = prop * prop;
        limits.doAlterVisits = true;
        limits.numAlterVisits = (int64_t)std::round(
          limits.numAlterVisits + visitReductionProp * ((double)playSettings.reducedVisitsMin - (double)limits.numAlterVisits)
        );
        limits.targetWeight = (float)(
          limits.targetWeight + visitReductionProp * (playSettings.reducedVisitsWeight - limits.targetWeight)
        );
        limits.numAlterVisits = std::max(limits.numAlterVisits, (int64_t)playSettings.reducedVisitsMin);
      }
    }
  }

  return limits;
}

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
  const std::function<void(const Q4PlayState&, int, const Q4S::Search*)>& onEachMove,
  Q4GameSeats* seats
) {
  (void)searchRandSeed;
  (void)clearBotBeforeSearch;
  (void)logSearchInfo;
  (void)logMoves;

  Q4FinishedGameData* gameData = new Q4FinishedGameData();
  gameData->rules = startHist.rules;
  gameData->gameHash.hash0 = gameRand.nextUInt64();
  gameData->gameHash.hash1 = gameRand.nextUInt64();
  gameData->startPly = otherGameProps.startPly;
  gameData->trainingWeight = otherGameProps.trainingWeight;
  gameData->mode = otherGameProps.isFork ? Q4FinishedGameData::MODE_FORK : Q4FinishedGameData::MODE_NORMAL;

  Q4PlayState state = startState;
  Q4History hist = startHist;

  // Policy-initialized opening if applicable
  if(playSettings.initGamesWithPolicy && otherGameProps.allowPolicyInit && !state.isFinished) {
    initializeGameUsingPolicy(
      bot, state, hist, gameRand,
      playSettings.policyInitAreaProp,
      playSettings.policyInitGammaShape,
      playSettings.policyInitAreaTemperature
    );
  }

  gameData->startState = state;
  gameData->startHist = hist;

  // The searches of the game: the learners' one (tree reuse across all their plies) and one per weak / snapshot seat.
  // After every ply every search gets makeMove.
  std::vector<Q4S::Search*> allSearches = {bot};
  std::vector<Q4S::Search*> weakSearches;
  if(seats != nullptr) {
    for(int s = 0; s < 4; s++) {
      gameData->seatInfo[s] = seats->info[s];
      Q4S::Search* other = seats->players[s].search;
      if(other != nullptr) {
        allSearches.push_back(other);
        if(seats->players[s].kind == SEAT_WEAK)
          weakSearches.push_back(other);
      }
    }
  }
  for(Q4S::Search* search : allSearches)
    search->setPosition(hist);

  // Elimination setup (prompt B4): with prob q4EliminationProb, at ply in [1, min(maxPlies, 200)]
  bool willEliminate = gameRand.nextBool(playSettings.q4EliminationProb);
  int maxElimPly = std::min(maxMovesPerGame, 200);
  int elimPly = (maxElimPly >= 1) ? gameRand.nextInt(1, maxElimPly) : 1;
  bool eliminationDone = false;

  std::vector<int> actionsBuf;
  std::vector<double> playSelectionValuesBuf;
  std::vector<Q4SidePosition*> sidePositionsToSearch;
  std::vector<double> historicalMaxWinrates;
  std::vector<std::vector<float>> rawNNValuesByTurn;

  auto maybeCheckForNewNNEval = [&bot, &weakSearches, &checkForNewNNEval, &gameRand, &gameData](int nextTurnIdx) {
    if(checkForNewNNEval != nullptr && gameRand.nextBool(0.1)) {
      NNEvaluator* newNNEval = checkForNewNNEval();
      if(newNNEval != nullptr) {
        bot->setNNEval(newNNEval);
        for(Q4S::Search* weak : weakSearches)
          weak->setNNEval(newNNEval);  // "the current net"
        gameData->changedNeuralNets.push_back(new ChangedNeuralNet(newNNEval->getModelName(), nextTurnIdx));
      }
    }
  };

  // Main play loop
  for(int i = 0; i < maxMovesPerGame; i++) {
    if(state.isFinished)
      break;
    if(shouldPause != nullptr)
      shouldPause->waitUntilFalse();
    if(shouldStop != nullptr && shouldStop())
      break;

    // Elimination hook
    if(willEliminate && !eliminationDone && state.plies == elimPly) {
      std::vector<int> alive;
      for(int s = 0; s < 4; s++) {
        if(state.board.isAlive(s)) alive.push_back(s);
      }
      if(alive.size() > 1) {
        int sElim = alive[gameRand.nextUInt((uint32_t)alive.size())];
        hist.eliminate(sElim);
        state.eliminate(sElim);
        for(Q4S::Search* search : allSearches) {
          search->clearSearch();
          search->setPosition(hist);
        }
        eliminationDone = true;
        gameData->comments.push_back("elim=" + std::to_string(sElim + 1));
        if(state.isFinished)
          break;
      }
    }

    int toMove = state.board.toMove;
    float turnStyle[Q4StyleTracker::NUM_FEATURES];
    hist.getStyleFeatures(turnStyle);
    gameData->styleByTurn.insert(gameData->styleByTurn.end(), turnStyle, turnStyle + Q4StyleTracker::NUM_FEATURES);
    // Who plays this ply. A non-learner seat's ply is observed by the learners' search (cheap-search settings, no
    // root noise) so that the ply has its root value vector; the seat's own player then picks the move.
    const int seatKind = (seats != nullptr) ? seats->players[toMove].kind : (int)SEAT_LEARNER;
    const bool isObserverTurn = seatKind != SEAT_LEARNER;
    SearchLimitsThisMove limits;
    if(!isObserverTurn) {
      limits = getSearchLimitsThisMove(bot, playSettings, gameRand, historicalMaxWinrates);
    }
    else {
      limits.numAlterVisits = std::min((int64_t)bot->searchParams.maxVisits, (int64_t)playSettings.cheapSearchVisits);
      limits.doAlterVisits = true;
      limits.removeRootNoise = true;
      limits.isCheapSearch = true;
      limits.targetWeight = 1.0f;
    }

    int action = Q4Board::NULL_ACTION;
    if(limits.doAlterVisits) {
      SearchParams oldParams = bot->searchParams;
      bot->searchParams.maxVisits = limits.numAlterVisits;
      bot->searchParams.maxPlayouts = limits.numAlterVisits;
      if(limits.removeRootNoise) {
        bot->searchParams.rootNoiseEnabled = false;
        bot->searchParams.rootPolicyTemperature = 1.0;
        bot->searchParams.rootPolicyTemperatureEarly = 1.0;
        bot->searchParams.rootDesiredPerChildVisitsCoeff = 0.0;
        bot->searchParams.rootNumSymmetriesToSample = 1;
      }
      action = bot->runWholeSearchAndGetMove();
      bot->searchParams = oldParams;
    }
    else {
      action = bot->runWholeSearchAndGetMove();
    }

    if(isObserverTurn) {
      // The observer search only provides the targets of this ply; the move comes from the seat's own player.
      Q4SeatPlayer& player = seats->players[toMove];
      if(player.search != nullptr)
        action = player.search->runWholeSearchAndGetMove();
      else
        action = player.bot->getMove(state.board);
    }

    if(action == Q4Board::NULL_ACTION || !state.isLegalAction(action)) {
      std::ostringstream sout;
      sout << "Q4Play::runGame illegal move produced: " << action << " by seat " << toMove;
      logger.write(sout.str());
      throw StringError(sout.str());
    }

    int64_t unreducedVisits = bot->getRootVisits();
    Q4S::ReportedSearchValues values;
    bot->getRootValues(values);

    Q4ValueTargets vt;
    double maxWin = 0.0;
    for(int c = 0; c < 5; c++) {
      vt.value[c] = (float)values.value[c];
      if(c < 4 && values.value[c] > maxWin) maxWin = values.value[c];
    }
    gameData->valueTargetsByTurn.push_back(vt);
    historicalMaxWinrates.push_back(maxWin);

    // Extract search policy target
    auto* ptBuf = new std::vector<Q4PolicyTargetMove>();
    if(!isObserverTurn)
      extractPolicyTarget(*ptBuf, bot, bot->rootNode, actionsBuf, playSelectionValuesBuf);
    gameData->policyTargetsByTurn.emplace_back(ptBuf, unreducedVisits);
    gameData->seatKindByTurn.push_back(seatKind);

    // NN raw values and stats
    const Q4S::NNOutput* rootNNOutput = (bot->rootNode != nullptr) ? bot->rootNode->getNNOutput() : nullptr;
    std::vector<float> rawNN(5, 0.2f);
    double policyEntropy = 0.0;
    double searchEntropy = 0.0;
    double policySurprise = 0.0;

    if(rootNNOutput != nullptr) {
      for(int c = 0; c < 5; c++) rawNN[c] = rootNNOutput->valueAbs[c];
    }
    // As KataGo's extractSearchTargetsThisTurn
    bool surpriseSuccess = bot->getPolicySurpriseAndEntropy(policySurprise, searchEntropy, policyEntropy);
    testAssert(surpriseSuccess);
    (void)surpriseSuccess;
    rawNNValuesByTurn.push_back(rawNN);

    Q4NNRawStats nnRaw;
    double valAbsDouble[5];
    for(int c = 0; c < 5; c++) valAbsDouble[c] = (double)rawNN[c];
    nnRaw.rawNNUtility = Q4S::Search::computeSeatUtility(toMove, valAbsDouble, state.board.getNumAlive(), true, 1.0);
    nnRaw.policyEntropy = policyEntropy;
    gameData->nnRawStatsByTurn.push_back(nnRaw);

    if(isObserverTurn)
      policySurprise = 0.0;  // the search did not produce the policy of this seat
    gameData->policySurpriseByTurn.push_back(std::max(0.0, policySurprise));
    gameData->policyEntropyByTurn.push_back(policyEntropy);
    gameData->searchEntropyByTurn.push_back(std::max(0.0, searchEntropy));
    gameData->targetWeightByTurn.push_back(limits.targetWeight);
    gameData->wasCheapSearchByTurn.push_back(limits.isCheapSearch);

    // Record move comment
    std::ostringstream cmt;
    cmt << "v=[";
    for(int vi = 0; vi < 5; vi++) {
      cmt << (vi > 0 ? ", " : "") << std::fixed << std::setprecision(4) << vt.value[vi];
    }
    cmt << "] visits=" << unreducedVisits << " cheap=" << (limits.isCheapSearch ? 1 : 0);
    gameData->comments.push_back(cmt.str());

    // Candidate side position
    if(!isObserverTurn && playSettings.sidePositionProb > 0.0 && gameRand.nextBool(playSettings.sidePositionProb) && rootNNOutput != nullptr) {
      int sideAct = chooseRandomForkingMove(rootNNOutput->getPolicyProbsMaybeNoised(), state, gameRand, action);
      if(sideAct != Q4Board::NULL_ACTION && state.isLegalAction(sideAct)) {
        Q4PlayState sideState = state;
        sideState.playAssumeLegal(sideAct);
        if(!sideState.isFinished) {
          Q4SidePosition* sidePos = new Q4SidePosition(sideState, (int)gameData->changedNeuralNets.size());
          Q4History sideHist = hist;
          sideHist.play(sideAct);
          sideHist.getStyleFeatures(sidePos->style);
          sidePositionsToSearch.push_back(sidePos);
        }
      }
    }

    if(onEachMove != nullptr)
      onEachMove(state, action, bot);

    // Advance searches and game
    for(Q4S::Search* search : allSearches)
      search->makeMove(action);
    hist.play(action);
    state.playAssumeLegal(action);
    // The tree is reused, so the searches keep the old features until they are set for the new real position
    hist.getStyleFeatures(turnStyle);
    for(Q4S::Search* search : allSearches)
      search->setStyleFeatures(turnStyle);

    maybeCheckForNewNNEval(state.plies);
  }

  gameData->hitTurnLimit = !hist.isFinished;
  gameData->endHist = hist;
  gameData->hasFullData = true;

  // Final value target (game result)
  Q4ValueTargets finalResultTarget;
  if(hist.winnerSeat >= 0 && hist.winnerSeat < 4) {
    finalResultTarget.value[hist.winnerSeat] = 1.0f;
  }
  else if(hist.isDraw) {
    finalResultTarget.value[4] = 1.0f;
  }
  gameData->valueTargetsByTurn.push_back(finalResultTarget);

  // Value surprise calculation
  computeValueSurpriseByTurn(
    gameData->valueSurpriseByTurn,
    gameData->valueTargetsByTurn,
    rawNNValuesByTurn,
    playSettings.useSearchValueSurprise
  );

  // Surprise weighting on main rows
  if(playSettings.policySurpriseDataWeight > 0.0 || playSettings.valueSurpriseDataWeight > 0.0) {
    size_t numWeights = gameData->targetWeightByTurn.size();
    double sumWeights = 0.0;
    double sumPolicySurpriseWeighted = 0.0;
    double sumValueSurpriseWeighted = 0.0;

    for(size_t k = 0; k < numWeights; k++) {
      float w = gameData->targetWeightByTurn[k];
      sumWeights += w;
      sumPolicySurpriseWeighted += gameData->policySurpriseByTurn[k] * w;
      sumValueSurpriseWeighted += gameData->valueSurpriseByTurn[k] * w;
    }

    if(sumWeights >= 1.0) {
      double avgPolSurprise = sumPolicySurpriseWeighted / sumWeights;
      double avgValSurprise = sumValueSurpriseWeighted / sumWeights;
      double valSurpriseWeight = playSettings.valueSurpriseDataWeight;
      if(avgValSurprise < 0.010) valSurpriseWeight *= (avgValSurprise / 0.010);

      double thresholdToIncludeReduced = avgPolSurprise * 1.5;
      auto policyPropVal = [&](size_t idx) {
        float tw = gameData->targetWeightByTurn[idx];
        double ps = gameData->policySurpriseByTurn[idx];
        double excess = std::max(0.0, ps - thresholdToIncludeReduced);
        return tw * ps + (1.0 - tw) * excess;
      };
      auto valuePropVal = [&](size_t idx) {
        return gameData->targetWeightByTurn[idx] * gameData->valueSurpriseByTurn[idx];
      };

      double sumPolProp = 0.0;
      double sumValProp = 0.0;
      for(size_t k = 0; k < numWeights; k++) {
        sumPolProp += policyPropVal(k);
        sumValProp += valuePropVal(k);
      }
      sumPolProp = std::max(sumPolProp, 1e-10);
      sumValProp = std::max(sumValProp, 1e-10);

      for(size_t k = 0; k < numWeights; k++) {
        float tw = gameData->targetWeightByTurn[k];
        double newWeight =
          (1.0 - playSettings.policySurpriseDataWeight - valSurpriseWeight) * tw +
          playSettings.policySurpriseDataWeight * policyPropVal(k) * sumWeights / sumPolProp +
          valSurpriseWeight * valuePropVal(k) * sumWeights / sumValProp;
        gameData->targetWeightByTurn[k] = (float)newWeight;
      }
    }
  }

  // Search side positions if game finished without hitting cutoff
  if(!gameData->hitTurnLimit) {
    for(auto* sp : sidePositionsToSearch) {
      if(shouldPause != nullptr) shouldPause->waitUntilFalse();
      if(shouldStop != nullptr && shouldStop()) {
        delete sp;
        continue;
      }

      bot->setPosition(sp->state);
      bot->setStyleFeatures(sp->style);
      bot->runWholeSearchAndGetMove();

      extractPolicyTarget(sp->policyTarget, bot, bot->rootNode, actionsBuf, playSelectionValuesBuf);

      Q4S::ReportedSearchValues sideVals;
      bot->getRootValues(sideVals);
      for(int c = 0; c < 5; c++) sp->valueTargets.value[c] = (float)sideVals.value[c];

      const Q4S::NNOutput* sideNN = (bot->rootNode != nullptr) ? bot->rootNode->getNNOutput() : nullptr;
      if(sideNN != nullptr) {
        double dVal[5];
        for(int c = 0; c < 5; c++) dVal[c] = (double)sideNN->valueAbs[c];
        sp->nnRawStats.rawNNUtility = Q4S::Search::computeSeatUtility(sp->toMove, dVal, sp->state.board.getNumAlive(), true, 1.0);
      }

      double pSurprise = 0.0, pEntropy = 0.0, sEntropy = 0.0;
      bot->getPolicySurpriseAndEntropy(pSurprise, sEntropy, pEntropy);
      sp->policySurprise = pSurprise;
      sp->policyEntropy = pEntropy;
      sp->searchEntropy = sEntropy;

      sp->targetWeight = 1.0f;
      sp->unreducedNumVisits = bot->getRootVisits();
      gameData->sidePositions.push_back(sp);
    }
  }
  else {
    for(auto* sp : sidePositionsToSearch) delete sp;
  }
  sidePositionsToSearch.clear();

  // Scale weights
  if(playSettings.scaleDataWeight != 1.0) {
    for(size_t k = 0; k < gameData->targetWeightByTurn.size(); k++)
      gameData->targetWeightByTurn[k] = (float)(playSettings.scaleDataWeight * gameData->targetWeightByTurn[k]);
    for(auto* sp : gameData->sidePositions)
      sp->targetWeight = (float)(playSettings.scaleDataWeight * sp->targetWeight);
  }

  // Unrounded copy
  gameData->targetWeightByTurnUnrounded = gameData->targetWeightByTurn;
  for(auto* sp : gameData->sidePositions) sp->targetWeightUnrounded = sp->targetWeight;

  // Stochastic integerization
  if(!playSettings.noResolveTargetWeights) {
    auto resolveWeight = [&gameRand](float w) -> float {
      if(w <= 0.0f) return 0.0f;
      float floored = std::floor(w);
      float excess = w - floored;
      return gameRand.nextBool(excess) ? floored + 1.0f : floored;
    };
    for(size_t k = 0; k < gameData->targetWeightByTurn.size(); k++)
      gameData->targetWeightByTurn[k] = resolveWeight(gameData->targetWeightByTurn[k]);
    for(auto* sp : gameData->sidePositions)
      sp->targetWeight = resolveWeight(sp->targetWeight);
  }

  return gameData;
}

void maybeForkGame(
  const Q4FinishedGameData* finishedGameData,
  Q4ForkData* forkData,
  const Q4PlaySettings& playSettings,
  Rand& gameRand,
  Q4S::Search* bot
) {
  if(forkData == nullptr) return;
  bool earlyFork = gameRand.nextBool(playSettings.earlyForkGameProb);
  bool lateFork = !earlyFork && (playSettings.forkGameProb > 0.0) && gameRand.nextBool(playSettings.forkGameProb);
  if(!earlyFork && !lateFork) return;

  int totalEvents = (int)finishedGameData->endHist.events.size();
  if(totalEvents <= 0) return;

  int moveIdx;
  if(earlyFork) {
    moveIdx = (int)std::floor(gameRand.nextExponential() * (playSettings.earlyForkGameExpectedMoveProp * 121.0));
    if(moveIdx >= totalEvents) moveIdx = totalEvents - 1;
    if(moveIdx < 0) moveIdx = 0;
  }
  else {
    moveIdx = gameRand.nextInt(0, totalEvents - 1);
  }

  Q4History forkHist(finishedGameData->rules);
  for(int i = 0; i < moveIdx && i < totalEvents; i++) {
    const auto& ev = finishedGameData->endHist.events[i];
    if(ev.isElimination) forkHist.eliminate(ev.eliminatedSeat);
    else forkHist.play(ev.action);
  }
  if(forkHist.isFinished) return;

  std::vector<int> legalActions;
  forkHist.getState().getLegalActions(legalActions);
  if(legalActions.empty()) return;

  int maxChoices = earlyFork ? playSettings.earlyForkGameMaxChoices : playSettings.forkGameMaxChoices;
  int numChoices = gameRand.nextInt(playSettings.forkGameMinChoices, maxChoices);
  numChoices = std::min((int)legalActions.size(), std::max(1, numChoices));

  // Sample candidate legal moves
  std::vector<int> candidates;
  std::vector<int> pool = legalActions;
  for(int c = 0; c < numChoices && !pool.empty(); c++) {
    size_t pick = gameRand.nextUInt((uint32_t)pool.size());
    candidates.push_back(pool[pick]);
    pool.erase(pool.begin() + pick);
  }

  int pla = forkHist.getState().board.toMove;
  int bestMove = Q4Board::NULL_ACTION;
  double bestScore = -1e9;
  NNResultBuf buf;

  for(int candAction : candidates) {
    Q4PlayState copyState = forkHist.getState();
    copyState.playAssumeLegal(candAction);
    double score = 0.0;
    if(copyState.isFinished) {
      if(copyState.winnerSeat == pla) score = 1.0;
      else if(copyState.isDraw) score = 0.0;
      else score = -1.0;
    }
    else {
      Q4NN::Eval eval;
      Q4History candHist = forkHist;
      candHist.play(candAction);
      Q4NN::evaluate(*bot->nnEvaluator, buf, candHist, 0, false, eval, &gameRand);
      double valAbs[5];
      for(int k = 0; k < 5; k++) valAbs[k] = (double)eval.valueAbsMasked[k];
      score = Q4S::Search::computeSeatUtility(pla, valAbs, copyState.board.getNumAlive(), true, 1.0);
    }

    if(bestMove == Q4Board::NULL_ACTION || score > bestScore) {
      bestMove = candAction;
      bestScore = score;
    }
  }

  if(bestMove != Q4Board::NULL_ACTION && forkHist.getState().isLegalAction(bestMove)) {
    forkHist.play(bestMove);
    if(!forkHist.isFinished) {
      forkData->add(new Q4InitialPosition(forkHist, true, finishedGameData->trainingWeight, finishedGameData->rules));
    }
  }
}

}  // namespace Play

//---------------------------------------------------------------------------------------------------------
// GameRunner implementation
//---------------------------------------------------------------------------------------------------------

Q4GameRunner::Q4GameRunner(ConfigParser& cfg, const Q4PlaySettings& pSettings, Logger& logger)
  : logSearchInfo(cfg.getBool("logSearchInfo")),
    logMoves(cfg.getBool("logMoves")),
    maxMovesPerGame(Play::loadMaxMovesPerGame(cfg)),
    clearBotBeforeSearch(cfg.contains("clearBotBeforeSearch") ? cfg.getBool("clearBotBeforeSearch") : false),
    playSettings(pSettings),
    gameInit(std::make_unique<Q4GameInitializer>(cfg, logger))
{}

Q4GameRunner::Q4GameRunner(ConfigParser& cfg, const std::string& gameInitRandSeed, const Q4PlaySettings& pSettings, Logger& logger)
  : logSearchInfo(cfg.getBool("logSearchInfo")),
    logMoves(cfg.getBool("logMoves")),
    maxMovesPerGame(Play::loadMaxMovesPerGame(cfg)),
    clearBotBeforeSearch(cfg.contains("clearBotBeforeSearch") ? cfg.getBool("clearBotBeforeSearch") : false),
    playSettings(pSettings),
    gameInit(std::make_unique<Q4GameInitializer>(cfg, logger, gameInitRandSeed))
{}

Q4FinishedGameData* Q4GameRunner::runGame(
  const std::string& seed,
  const BotSpec& botSpec,
  Q4ForkData* forkData,
  Logger& logger,
  const std::function<bool()>& shouldStop,
  const WaitableFlag* shouldPause,
  const std::function<NNEvaluator*()>& checkForNewNNEval,
  const std::function<void(const BotSpec&, Q4S::Search*)>& afterInitialization,
  const std::function<void(const Q4PlayState&, int, const Q4S::Search*)>& onEachMove,
  const Q4SnapshotSource* snapshotSource
) {
  Rand gameRand(seed);

  const Q4InitialPosition* initialPosition = nullptr;
  if(forkData != nullptr) {
    initialPosition = forkData->get(gameRand);
  }

  Q4PlayState startState;
  Q4History startHist;
  Q4OtherGameProperties otherGameProps;
  gameInit->createGame(startState, startHist, initialPosition, playSettings, otherGameProps);

  if(initialPosition != nullptr) {
    delete initialPosition;
    initialPosition = nullptr;
  }

  std::unique_ptr<Q4S::Search> bot = std::make_unique<Q4S::Search>(
    botSpec.baseParams, botSpec.nnEval, &logger, seed + ":search"
  );

  if(afterInitialization != nullptr) {
    afterInitialization(botSpec, bot.get());
  }

  // The composition of the table. Drawn from its own stream (seeded from the game seed) so that the game RNG, and
  // therefore every all-learner game, does not depend on the population settings.
  Rand popRand(seed + ":population");
  std::vector<Q4SnapshotRef> snapshots;
  if(snapshotSource != nullptr && playSettings.population.usesSnapshots())
    snapshots = snapshotSource->acquireAll();
  if((int)snapshots.size() > playSettings.population.numSnapshots)
    snapshots.resize(playSettings.population.numSnapshots);
  Q4Composition comp = sampleComposition(popRand, playSettings.population, (int)snapshots.size());

  std::unique_ptr<Q4GameSeats> seats = std::make_unique<Q4GameSeats>();
  std::vector<std::unique_ptr<Q4S::Search>> otherSearches;
  for(int s = 0; s < 4; s++) {
    const Q4SeatSpec& spec = comp.seats[s];
    Q4SeatPlayer& player = seats->players[s];
    Q4SeatInfo& info = seats->info[s];
    player.kind = spec.kind;
    info.kind = spec.kind;
    info.net = botSpec.botName;
    if(spec.kind == SEAT_LEARNER) {
      info.visits = botSpec.baseParams.maxVisits;
      continue;
    }
    if(spec.kind == SEAT_WEAK || spec.kind == SEAT_SNAPSHOT) {
      SearchParams params = botSpec.baseParams;
      NNEvaluator* eval = botSpec.nnEval;
      if(spec.kind == SEAT_WEAK) {
        params.maxVisits = spec.weakVisits;
        params.chosenMoveTemperatureEarly = spec.weakTemperature;
        params.chosenMoveTemperature = spec.weakTemperature;
        info.temperature = spec.weakTemperature;
      }
      else {
        const Q4SnapshotRef& snap = snapshots.at(spec.snapshotIdx);
        eval = snap.nnEval;
        info.net = snap.name;
        params.maxVisits = playSettings.population.snapshotVisits;
        params.chosenMoveTemperatureEarly = playSettings.population.snapshotTemperatureEarly;
        params.chosenMoveTemperature = playSettings.population.snapshotTemperature;
        info.temperature = playSettings.population.snapshotTemperature;
      }
      params.maxPlayouts = params.maxVisits;
      // No root exploration for the other players: this is a net playing, not generating data
      params.rootNoiseEnabled = false;
      params.rootPolicyTemperature = 1.0;
      params.rootPolicyTemperatureEarly = 1.0;
      params.rootDesiredPerChildVisitsCoeff = 0.0;
      params.rootNumSymmetriesToSample = 1;
      Q4S::Search::checkParams(params);
      otherSearches.push_back(std::make_unique<Q4S::Search>(params, eval, &logger, seed + ":seat" + Global::intToString(s)));
      player.search = otherSearches.back().get();
      info.visits = params.maxVisits;
    }
    else {
      Rand botRand(seed + ":bot" + Global::intToString(s));
      player.bot = Q4Bots::makeBot(seatKindName(spec.kind), botRand.nextUInt64(), spec.grudgeTarget);
      info.grudgeTarget = spec.grudgeTarget;
      info.net = "";
    }
  }

  Q4FinishedGameData* finishedGameData = Play::runGame(
    seed,
    bot.get(),
    startState,
    startHist,
    clearBotBeforeSearch,
    logger,
    logSearchInfo,
    logMoves,
    maxMovesPerGame,
    shouldStop,
    shouldPause,
    playSettings,
    otherGameProps,
    gameRand,
    checkForNewNNEval,
    onEachMove,
    comp.numLearners() == 4 ? nullptr : seats.get()
  );
  if(finishedGameData != nullptr && comp.numLearners() == 4) {
    for(int s = 0; s < 4; s++)
      finishedGameData->seatInfo[s] = seats->info[s];
  }

  if(finishedGameData != nullptr && forkData != nullptr) {
    Play::maybeForkGame(finishedGameData, forkData, playSettings, gameRand, bot.get());
  }

  otherSearches.clear();
  if(snapshotSource != nullptr && !snapshots.empty())
    snapshotSource->releaseAll(snapshots);
  return finishedGameData;
}

}  // namespace Q4Play
