#include "../core/config_parser.h"
#include "../core/global.h"
#include "../core/rand.h"
#include "../external/nlohmann_json/json.hpp"
#include "../main.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"
#include "nn/q4nn.h"
#include "q4board.h"
#include "q4bots.h"
#include "q4history.h"
#include "q4record.h"
#include "q4rules.h"
#include "../search/q4search.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

struct PlayerSpec {
  bool isSearch = false;
  std::string rawSpec = "";
  std::string botType = "";
  std::string modelPath = "";
  int visits = 200;
};

PlayerSpec parsePlayerSpec(const std::string& spec) {
  PlayerSpec ps;
  ps.rawSpec = spec;
  if(spec.rfind("search:", 0) == 0) {
    ps.isSearch = true;
    std::string rest = spec.substr(7);
    size_t atPos = rest.rfind('@');
    if(atPos != std::string::npos) {
      ps.modelPath = rest.substr(0, atPos);
      ps.visits = Global::stringToInt(rest.substr(atPos + 1));
    }
    else {
      ps.modelPath = rest;
      ps.visits = 200;
    }
  }
  else {
    ps.botType = spec;
  }
  return ps;
}

struct MatchPlayer {
  std::unique_ptr<Q4Bot> bot = nullptr;
  std::unique_ptr<Q4S::Search> search = nullptr;

  int getMove(const Q4History& history) {
    if(search != nullptr) {
      search->setPosition(history);
      return search->runWholeSearchAndGetMove();
    }
    return bot->getMove(history.currentBoard);
  }

  void updateMove(int act) {
    if(search != nullptr) {
      search->makeMove(act);
    }
  }

  void clearTree(const Q4History& history) {
    if(search != nullptr) {
      search->clearSearch();
      search->setPosition(history);
    }
  }
};

}  // namespace

int MainCmds::q4match(const std::vector<std::string>& args) {
  std::vector<std::string> subArgs = args;
  if(!subArgs.empty() && subArgs[0] == "q4match")
    subArgs = std::vector<std::string>(subArgs.begin() + 1, subArgs.end());

  int numGames = 1000;
  std::vector<std::string> botSpecs = {"random", "random", "random", "random"};
  std::string configFile = "";
  std::string outputFile = "";
  uint64_t masterSeed = 12345;
  int maxPlies = 400;
  bool rotateSeats = false;

  for(size_t i = 0; i < subArgs.size(); i++) {
    if(subArgs[i] == "-games" && i + 1 < subArgs.size()) {
      numGames = Global::stringToInt(subArgs[++i]);
    }
    else if((subArgs[i] == "-bots" || subArgs[i] == "-players") && i + 1 < subArgs.size()) {
      botSpecs = Global::split(subArgs[++i], ',');
    }
    else if(subArgs[i] == "-config" && i + 1 < subArgs.size()) {
      configFile = subArgs[++i];
    }
    else if(subArgs[i] == "-output" && i + 1 < subArgs.size()) {
      outputFile = subArgs[++i];
    }
    else if(subArgs[i] == "-seed" && i + 1 < subArgs.size()) {
      masterSeed = (uint64_t)Global::stringToInt64(subArgs[++i]);
    }
    else if((subArgs[i] == "-maxplies" || subArgs[i] == "-max-plies") && i + 1 < subArgs.size()) {
      maxPlies = Global::stringToInt(subArgs[++i]);
    }
    else if(subArgs[i] == "-rotate") {
      rotateSeats = true;
    }
  }

  while(botSpecs.size() < 4) {
    botSpecs.push_back("random");
  }

  ConfigParser cfg;
  if(!configFile.empty()) {
    try {
      cfg.initialize(configFile);
    }
    catch(const std::exception& e) {
      std::cerr << "Warning: Could not read config file " << configFile << ": " << e.what() << std::endl;
    }
  }

  // Fallback defaults for SearchParams
  if(!cfg.contains("maxVisits")) cfg.overrideKey("maxVisits", "800");
  if(!cfg.contains("numSearchThreads")) cfg.overrideKey("numSearchThreads", "1");
  if(!cfg.contains("winLossUtilityFactor")) cfg.overrideKey("winLossUtilityFactor", "1.0");
  if(!cfg.contains("cpuctExploration")) cfg.overrideKey("cpuctExploration", "1.1");
  if(!cfg.contains("cpuctExplorationLog")) cfg.overrideKey("cpuctExplorationLog", "0.0");
  if(!cfg.contains("fpuReductionMax")) cfg.overrideKey("fpuReductionMax", "0.2");
  if(!cfg.contains("rootFpuReductionMax")) cfg.overrideKey("rootFpuReductionMax", "0.0");
  if(!cfg.contains("fpuParentWeightByVisitedPolicy")) cfg.overrideKey("fpuParentWeightByVisitedPolicy", "true");
  if(!cfg.contains("fpuParentWeightByVisitedPolicyPow")) cfg.overrideKey("fpuParentWeightByVisitedPolicyPow", "2.0");
  if(!cfg.contains("valueWeightExponent")) cfg.overrideKey("valueWeightExponent", "0.5");
  if(!cfg.contains("useUncertainty")) cfg.overrideKey("useUncertainty", "true");
  if(!cfg.contains("uncertaintyExponent")) cfg.overrideKey("uncertaintyExponent", "1.0");
  if(!cfg.contains("uncertaintyCoeff")) cfg.overrideKey("uncertaintyCoeff", "0.25");
  if(!cfg.contains("useLcbForSelection")) cfg.overrideKey("useLcbForSelection", "true");
  if(!cfg.contains("lcbStdevs")) cfg.overrideKey("lcbStdevs", "5.0");
  if(!cfg.contains("minVisitPropForLCB")) cfg.overrideKey("minVisitPropForLCB", "0.15");
  if(!cfg.contains("useNonBuggyLcb")) cfg.overrideKey("useNonBuggyLcb", "true");
  if(!cfg.contains("rootNoiseEnabled")) cfg.overrideKey("rootNoiseEnabled", "false");
  if(!cfg.contains("rootDirichletNoiseTotalConcentration")) cfg.overrideKey("rootDirichletNoiseTotalConcentration", "10.83");
  if(!cfg.contains("rootDirichletNoiseWeight")) cfg.overrideKey("rootDirichletNoiseWeight", "0.25");
  if(!cfg.contains("rootDesiredPerChildVisitsCoeff")) cfg.overrideKey("rootDesiredPerChildVisitsCoeff", "2");
  if(!cfg.contains("rootPolicyTemperatureEarly")) cfg.overrideKey("rootPolicyTemperatureEarly", "1.25");
  if(!cfg.contains("rootPolicyTemperature")) cfg.overrideKey("rootPolicyTemperature", "1.1");
  if(!cfg.contains("rootNumSymmetriesToSample")) cfg.overrideKey("rootNumSymmetriesToSample", "8");
  if(!cfg.contains("chosenMoveTemperatureEarly")) cfg.overrideKey("chosenMoveTemperatureEarly", "0.75");
  if(!cfg.contains("chosenMoveTemperatureHalflife")) cfg.overrideKey("chosenMoveTemperatureHalflife", "38");
  if(!cfg.contains("chosenMoveTemperature")) cfg.overrideKey("chosenMoveTemperature", "0.15");
  if(!cfg.contains("chosenMoveSubtract")) cfg.overrideKey("chosenMoveSubtract", "0");
  if(!cfg.contains("chosenMovePrune")) cfg.overrideKey("chosenMovePrune", "1");
  if(!cfg.contains("staticScoreUtilityFactor")) cfg.overrideKey("staticScoreUtilityFactor", "0.0");
  if(!cfg.contains("dynamicScoreUtilityFactor")) cfg.overrideKey("dynamicScoreUtilityFactor", "0.0");
  if(!cfg.contains("policyOptimism")) cfg.overrideKey("policyOptimism", "0.0");
  if(!cfg.contains("rootPolicyOptimism")) cfg.overrideKey("rootPolicyOptimism", "0.0");
  if(!cfg.contains("useGraphSearch")) cfg.overrideKey("useGraphSearch", "false");
  if(!cfg.contains("useEvalCache")) cfg.overrideKey("useEvalCache", "false");
  if(!cfg.contains("subtreeValueBiasFactor")) cfg.overrideKey("subtreeValueBiasFactor", "0.0");
  if(!cfg.contains("avoidRepeatedPatternUtility")) cfg.overrideKey("avoidRepeatedPatternUtility", "0.0");
  if(!cfg.contains("antiMirror")) cfg.overrideKey("antiMirror", "false");
  if(!cfg.contains("playoutDoublingAdvantage")) cfg.overrideKey("playoutDoublingAdvantage", "0.0");
  if(!cfg.contains("visitCapContempt")) cfg.overrideKey("visitCapContempt", "0");
  if(!cfg.contains("rootSymmetryPruning")) cfg.overrideKey("rootSymmetryPruning", "false");
  if(!cfg.contains("conservativePass")) cfg.overrideKey("conservativePass", "false");
  if(!cfg.contains("enablePassingHacks")) cfg.overrideKey("enablePassingHacks", "false");
  if(!cfg.contains("enableMorePassingHacks")) cfg.overrideKey("enableMorePassingHacks", "false");
  if(!cfg.contains("fillDameBeforePass")) cfg.overrideKey("fillDameBeforePass", "false");
  if(!cfg.contains("rootEndingBonusPoints")) cfg.overrideKey("rootEndingBonusPoints", "0.0");
  if(!cfg.contains("rootPruneUselessMoves")) cfg.overrideKey("rootPruneUselessMoves", "false");
  if(!cfg.contains("ignorePreRootHistory")) cfg.overrideKey("ignorePreRootHistory", "false");
  if(!cfg.contains("ignoreAllHistory")) cfg.overrideKey("ignoreAllHistory", "false");

  std::vector<PlayerSpec> playerSpecs(4);
  for(int p = 0; p < 4; p++) {
    playerSpecs[p] = parsePlayerSpec(botSpecs[p]);
  }

  std::cout << "=== KataQuoridor Q4 Match ===" << std::endl;
  std::cout << "Games: " << numGames << std::endl;
  std::cout << "Players: P1=" << botSpecs[0] << ", P2=" << botSpecs[1]
            << ", P3=" << botSpecs[2] << ", P4=" << botSpecs[3] << std::endl;
  std::cout << "Seat rotation: " << (rotateSeats ? "enabled" : "disabled") << std::endl;
  std::cout << "Max plies: " << maxPlies << std::endl;

  Logger logger;
  std::map<std::string, std::unique_ptr<NNEvaluator>> evaluators;
  for(const auto& ps : playerSpecs) {
    if(ps.isSearch && evaluators.find(ps.modelPath) == evaluators.end()) {
      if(!cfg.contains("nnCacheSizePowerOfTwo")) cfg.overrideKey("nnCacheSizePowerOfTwo", "16");
      if(!cfg.contains("nnMutexPoolSizePowerOfTwo")) cfg.overrideKey("nnMutexPoolSizePowerOfTwo", "12");
      Rand evalSeed(masterSeed + 777);
      NNEvaluator* nnEval = Setup::initializeNNEvaluator(
        ps.modelPath, ps.modelPath, "", cfg, logger, evalSeed, 1, Q4NNConst::POS_LEN, Q4NNConst::POS_LEN,
        Setup::MaxBatchSizeRequest::explicitSize(16), true, false, Setup::SETUP_FOR_GTP
      );
      evaluators[ps.modelPath] = std::unique_ptr<NNEvaluator>(nnEval);
    }
  }

  std::ofstream outStream;
  if(!outputFile.empty()) {
    outStream.open(outputFile);
    if(!outStream.is_open()) {
      std::cerr << "Error: Could not open output file " << outputFile << std::endl;
      return 1;
    }
  }

  int seatWinCounts[4] = {0, 0, 0, 0};
  int playerWinCounts[4] = {0, 0, 0, 0};
  int drawCount = 0;
  int minPlies = 999999;
  int maxObservedPlies = 0;
  int64_t totalPlies = 0;

  Q4Rules rules;
  rules.maxPlies = maxPlies;

  Rand masterRand(masterSeed);

  for(int g = 0; g < numGames; g++) {
    int playerOfSeat[4];
    for(int p = 0; p < 4; p++) {
      int s = rotateSeats ? (p + g) % 4 : p;
      playerOfSeat[s] = p;
    }

    std::unique_ptr<MatchPlayer> players[4];
    for(int s = 0; s < 4; s++) {
      int p = playerOfSeat[s];
      const PlayerSpec& ps = playerSpecs[p];
      uint64_t bSeed = masterRand.nextUInt64();
      players[s] = std::make_unique<MatchPlayer>();

      if(ps.isSearch) {
        SearchParams params = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_GTP);
        params.maxVisits = ps.visits;
        params.numThreads = 1;
        Q4S::Search::checkParams(params);
        players[s]->search = std::make_unique<Q4S::Search>(
          params, evaluators[ps.modelPath].get(), &logger, "match_" + std::to_string(bSeed)
        );
      }
      else {
        players[s]->bot = Q4Bots::makeBot(ps.botType, bSeed, (s + 1) % 4);
      }
    }

    Q4History history(rules);
    for(int s = 0; s < 4; s++) {
      if(players[s]->search)
        players[s]->search->setPosition(history);
    }

    while(!history.isFinished) {
      int toMove = history.currentBoard.toMove;
      int act = players[toMove]->getMove(history);
      if(act == Q4Board::NULL_ACTION)
        break;
      history.play(act);
      for(int s = 0; s < 4; s++)
        players[s]->updateMove(act);
    }

    if(history.winnerSeat >= 0 && history.winnerSeat < 4) {
      seatWinCounts[history.winnerSeat]++;
      int winningPlayer = playerOfSeat[history.winnerSeat];
      playerWinCounts[winningPlayer]++;
    }
    else {
      drawCount++;
    }

    int pl = history.plies;
    totalPlies += pl;
    if(pl < minPlies) minPlies = pl;
    if(pl > maxObservedPlies) maxObservedPlies = pl;

    if(outStream.is_open()) {
      Q4Record rec;
      rec.rules = rules;
      rec.events = history.events;
      rec.result = history.getResultString();
      for(int s = 0; s < 4; s++) {
        int p = playerOfSeat[s];
        rec.players[s].name = "P" + Global::intToString(p + 1) + "_" + playerSpecs[p].rawSpec;
        rec.players[s].type = playerSpecs[p].rawSpec;
      }
      outStream << rec.toJsonLine() << "\n";
    }

    if((g + 1) % 25 == 0 || g + 1 == numGames) {
      std::cout << "Played " << (g + 1) << "/" << numGames << " games..." << std::endl;
    }
  }

  std::cout << "\n--- Match Results (" << numGames << " games) ---" << std::endl;
  std::cout << "By Player:" << std::endl;
  for(int p = 0; p < 4; p++) {
    double pct = 100.0 * playerWinCounts[p] / numGames;
    std::cout << "Player " << (p + 1) << " (" << playerSpecs[p].rawSpec << "): "
              << playerWinCounts[p] << " wins (" << std::fixed << std::setprecision(1) << pct << "%)" << std::endl;
  }
  std::cout << "By Seat:" << std::endl;
  for(int s = 0; s < 4; s++) {
    double pct = 100.0 * seatWinCounts[s] / numGames;
    std::cout << "Seat " << (s + 1) << ": "
              << seatWinCounts[s] << " wins (" << std::fixed << std::setprecision(1) << pct << "%)" << std::endl;
  }
  double drawPct = 100.0 * drawCount / numGames;
  std::cout << "Draws: " << drawCount << " (" << std::fixed << std::setprecision(1) << drawPct << "%)" << std::endl;
  std::cout << "Game lengths (plies): min=" << minPlies
            << ", avg=" << std::setprecision(1) << ((double)totalPlies / numGames)
            << ", max=" << maxObservedPlies << std::endl;

  return 0;
}
