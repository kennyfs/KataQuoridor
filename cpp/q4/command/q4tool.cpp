#include "../core/global.h"
#include "../core/rand.h"
#include "../core/timer.h"
#include "../external/nlohmann_json/json.hpp"
#include "../main.h"
#include "command/q4json.h"
#include "command/q4nntool.h"
#include "q4board.h"
#include "q4bots.h"
#include "q4history.h"
#include "q4notation.h"
#include "../core/config_parser.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"
#include "nn/q4nn.h"
#include "../search/q4search.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

uint64_t runPerft(const Q4Board& board, int depth) {
  if(board.isFinished() || depth == 0)
    return 1;
  std::vector<int> actions;
  board.getLegalActions(board.toMove, actions);
  if(depth == 1)
    return actions.size();
  uint64_t total = 0;
  for(int act : actions) {
    Q4Board nextBoard = board;
    nextBoard.applyAction(act);
    total += runPerft(nextBoard, depth - 1);
  }
  return total;
}

}  // namespace

Q4Board parseBoardFromJson(const json& j) {
  Q4Board board;
  board.alive = 0;
  std::fill(std::begin(board.occupant), std::end(board.occupant), -1);

  // Pawns
  if(j.contains("pawns") && j["pawns"].is_array()) {
    auto pArr = j["pawns"];
    for(size_t s = 0; s < 4 && s < pArr.size(); s++) {
      if(pArr[s].is_null()) {
        board.pawn[s] = -1;
      }
      else if(pArr[s].is_array()) {
        int x = pArr[s][0].get<int>();
        int y = pArr[s][1].get<int>();
        if(x >= 0 && y >= 0) {
          int c = Q4Board::cellOf(x, y);
          board.pawn[s] = c;
          board.occupant[c] = (int)s;
        }
        else {
          board.pawn[s] = -1;
        }
      }
      else if(pArr[s].is_string()) {
        std::string str = pArr[s].get<std::string>();
        if(str == "null" || str == "-1" || str.empty()) {
          board.pawn[s] = -1;
        }
        else {
          int c = Q4Notation::stringToCell(str);
          board.pawn[s] = c;
          board.occupant[c] = (int)s;
        }
      }
      else if(pArr[s].is_number_integer()) {
        int c = pArr[s].get<int>();
        board.pawn[s] = c;
        if(c >= 0) board.occupant[c] = (int)s;
      }
    }
  }

  // Alive
  if(j.contains("alive") && j["alive"].is_array()) {
    auto aArr = j["alive"];
    for(size_t s = 0; s < 4 && s < aArr.size(); s++) {
      if(aArr[s].get<bool>())
        board.alive |= (1 << s);
    }
  }
  else {
    for(int s = 0; s < 4; s++) {
      if(board.pawn[s] >= 0)
        board.alive |= (1 << s);
    }
  }

  // Walls left
  if(j.contains("wallsLeft") && j["wallsLeft"].is_array()) {
    auto wArr = j["wallsLeft"];
    for(size_t s = 0; s < 4 && s < wArr.size(); s++)
      board.wallsLeft[s] = wArr[s].get<int>();
  }

  // Placed walls
  if(j.contains("hwalls") && j["hwalls"].is_array()) {
    for(const auto& item : j["hwalls"]) {
      if(item.is_array()) {
        int ax = item[0].get<int>();
        int ay = item[1].get<int>();
        int a = Q4Board::anchorOf(ax, ay);
        board.hWalls.set(a);
        board.applyWall(ax, ay, true);
        // applyWall changes toMove and decrements wallsLeft, so reset below
      }
      else if(item.is_string()) {
        std::string s = item.get<std::string>();
        int act = Q4Notation::stringToAction(s);
        int a = act - 221;
        board.hWalls.set(a);
      }
    }
  }

  if(j.contains("vwalls") && j["vwalls"].is_array()) {
    for(const auto& item : j["vwalls"]) {
      if(item.is_array()) {
        int ax = item[0].get<int>();
        int ay = item[1].get<int>();
        int a = Q4Board::anchorOf(ax, ay);
        board.vWalls.set(a);
      }
      else if(item.is_string()) {
        std::string s = item.get<std::string>();
        int act = Q4Notation::stringToAction(s);
        int a = act - 121;
        board.vWalls.set(a);
      }
    }
  }

  // Re-initialize grid boundaries and apply all walls to blocked
  Q4Board fresh;
  fresh.alive = board.alive;
  for(int s = 0; s < 4; s++) {
    fresh.pawn[s] = board.pawn[s];
    fresh.wallsLeft[s] = board.wallsLeft[s];
  }
  std::copy(std::begin(board.occupant), std::end(board.occupant), std::begin(fresh.occupant));

  for(int ay = 0; ay < Q4Board::NUM_ANCHORS; ay++) {
    for(int ax = 0; ax < Q4Board::NUM_ANCHORS; ax++) {
      int a = Q4Board::anchorOf(ax, ay);
      if(board.hWalls.test(a)) {
        fresh.hWalls.set(a);
        fresh.applyWall(ax, ay, true);
      }
      if(board.vWalls.test(a)) {
        fresh.vWalls.set(a);
        fresh.applyWall(ax, ay, false);
      }
    }
  }

  // Restore wallsLeft and toMove
  if(j.contains("wallsLeft") && j["wallsLeft"].is_array()) {
    auto wArr = j["wallsLeft"];
    for(size_t s = 0; s < 4 && s < wArr.size(); s++)
      fresh.wallsLeft[s] = wArr[s].get<int>();
  }
  if(j.contains("toMove")) {
    fresh.toMove = j["toMove"].get<int>();
  }
  else {
    fresh.toMove = 0;
  }

  fresh.recomputeDistancesToCenter();
  fresh.hash = fresh.getHashFromScratch();
  return fresh;
}

json boardToJson(const Q4Board& board) {
  json j;
  json pawns = json::array();
  for(int s = 0; s < 4; s++) {
    if(board.pawn[s] >= 0) {
      pawns.push_back({Q4Board::cellX(board.pawn[s]), Q4Board::cellY(board.pawn[s])});
    } else {
      pawns.push_back(nullptr);
    }
  }
  j["pawns"] = pawns;

  json alive = json::array();
  for(int s = 0; s < 4; s++) {
    alive.push_back(board.isAlive(s));
  }
  j["alive"] = alive;

  json wallsLeft = json::array();
  for(int s = 0; s < 4; s++) {
    wallsLeft.push_back(board.wallsLeft[s]);
  }
  j["wallsLeft"] = wallsLeft;

  json hwalls = json::array();
  json vwalls = json::array();
  for(int ay = 0; ay < Q4Board::NUM_ANCHORS; ay++) {
    for(int ax = 0; ax < Q4Board::NUM_ANCHORS; ax++) {
      int a = Q4Board::anchorOf(ax, ay);
      if(board.hWalls.test(a)) hwalls.push_back({ax, ay});
      if(board.vWalls.test(a)) vwalls.push_back({ax, ay});
    }
  }
  j["hwalls"] = hwalls;
  j["vwalls"] = vwalls;
  j["toMove"] = board.toMove;
  return j;
}

Q4History parseHistoryFromJson(const json& j) {
  Q4Rules rules;
  if(j.contains("rules")) {
    const auto& r = j["rules"];
    if(r.contains("maxPlies")) rules.maxPlies = r["maxPlies"].get<int>();
    if(r.contains("repetitionDrawCount")) rules.repetitionDrawCount = r["repetitionDrawCount"].get<int>();
    if(r.contains("initialWalls") && r["initialWalls"].is_array()) {
      for(int s = 0; s < 4; s++) rules.initialWalls[s] = r["initialWalls"][s].get<int>();
    }
  }
  else {
    if(j.contains("maxPlies")) rules.maxPlies = j["maxPlies"].get<int>();
    if(j.contains("repetitionDrawCount")) rules.repetitionDrawCount = j["repetitionDrawCount"].get<int>();
  }
  Q4History history(rules);
  if(j.contains("initialBoard")) {
    Q4Board initB = parseBoardFromJson(j["initialBoard"]);
    history.clear(initB, rules);
  }
  else if(j.contains("board") && (!j.contains("events") || j["events"].empty())) {
    Q4Board b = parseBoardFromJson(j["board"]);
    history.clear(b, rules);
    return history;
  }
  else if(!j.contains("events")) {
    Q4Board b = parseBoardFromJson(j);
    history.clear(b, rules);
    return history;
  }

  if(j.contains("events") && j["events"].is_array()) {
    for(const auto& ev : j["events"]) {
      if(ev.contains("elim")) {
        history.eliminate(ev["elim"].get<int>());
      }
      else if(ev.contains("a") || ev.contains("action")) {
        std::string aStr = ev.contains("a") ? ev["a"].get<std::string>() : ev["action"].get<std::string>();
        history.play(Q4Notation::stringToAction(aStr));
      }
    }
  }
  return history;
}

int MainCmds::q4tool(const std::vector<std::string>& args) {
  std::vector<std::string> subArgs = args;
  if(!subArgs.empty() && subArgs[0] == "q4tool")
    subArgs = std::vector<std::string>(subArgs.begin() + 1, subArgs.end());

  if(subArgs.empty()) {
    std::cout << "Usage: katago q4tool <subcommand> [options]\n"
              << "Subcommands:\n"
              << "  legal                 Read JSON positions from stdin, output legal action names\n"
              << "  dumpinputs            Dump NN inputs for positions (all 8 symmetries)\n"
              << "  evalnn                Evaluate positions with NN model (-model <file>)\n"
              << "  symavg                Invariance of the 8-symmetry average (-model <file>)\n"
              << "  nncache               NN cache hits and misses (-model <file>)\n"
              << "  nnbench               Input-fill time, NN evaluations per second (-model <file>)\n"
              << "  perft <depth>         Run perft from start position\n"
              << "  bench                 Run performance benchmarks (movegen and playouts)\n";
    return 0;
  }

  std::string subcmd = subArgs[0];

  if(subcmd == "legal") {
    std::string line;
    while(std::getline(std::cin, line)) {
      line = Global::trim(line);
      if(line.empty()) continue;
      try {
        json j = json::parse(line);
        Q4Board board;
        bool isFinished = false;
        int winner = -1;
        bool isDraw = false;

        if(j.contains("events") && j["events"].is_array()) {
          Q4Rules rules;
          if(j.contains("rules")) {
            const auto& r = j["rules"];
            if(r.contains("maxPlies")) rules.maxPlies = r["maxPlies"].get<int>();
            if(r.contains("repetitionDrawCount")) rules.repetitionDrawCount = r["repetitionDrawCount"].get<int>();
            if(r.contains("initialWalls") && r["initialWalls"].is_array()) {
              for(int s = 0; s < 4; s++) rules.initialWalls[s] = r["initialWalls"][s].get<int>();
            }
          }
          Q4History history(rules);
          if(j.contains("initialBoard")) {
            Q4Board initB = parseBoardFromJson(j["initialBoard"]);
            history.clear(initB, rules);
          }

          for(const auto& ev : j["events"]) {
            if(ev.contains("elim")) {
              history.eliminate(ev["elim"].get<int>());
            }
            else if(ev.contains("a") || ev.contains("action")) {
              std::string aStr = ev.contains("a") ? ev["a"].get<std::string>() : ev["action"].get<std::string>();
              history.play(Q4Notation::stringToAction(aStr));
            }
          }
          board = history.currentBoard;
          isFinished = history.isFinished;
          winner = history.winnerSeat;
          isDraw = history.isDraw;
        }
        else {
          board = parseBoardFromJson(j);
          isFinished = board.isFinished();
          winner = board.getWinner();
          isDraw = false;
        }

        std::vector<int> actions;
        if(!isFinished) {
          board.getLegalActions(board.toMove, actions);
        }
        std::vector<std::string> names;
        for(int act : actions)
          names.push_back(Q4Notation::actionToString(act));
        std::sort(names.begin(), names.end());

        // Output JSON response with legal actions, distances, and terminal status
        json resp;
        resp["legal"] = names;
        json dists = json::array();
        for(int s = 0; s < 4; s++) {
          if(board.isAlive(s) && board.pawn[s] >= 0)
            dists.push_back((int)board.distToCenter[board.pawn[s]]);
          else
            dists.push_back(255);
        }
        resp["dist"] = dists;

        json fullDist = json::array();
        for(int c = 0; c < Q4Board::NUM_CELLS; c++)
          fullDist.push_back((int)board.distToCenter[c]);
        resp["distToCenter"] = fullDist;

        resp["isFinished"] = isFinished;
        resp["winner"] = winner;
        resp["isDraw"] = isDraw;
        std::cout << resp.dump() << "\n" << std::flush;
      }
      catch(const std::exception& e) {
        std::cerr << "Error parsing position in q4tool legal: " << e.what() << "\n";
      }
    }
    return 0;
  }
  else if(subcmd == "dumpinputs" || subcmd == "evalnn" || subcmd == "symavg" || subcmd == "nncache" || subcmd == "nnbench") {
    return Q4NNTool::run(subcmd, subArgs);
  }
  else if(subcmd == "perft") {
    int depth = 1;
    if(subArgs.size() > 1)
      depth = Global::stringToInt(subArgs[1]);

    Q4Board board;
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t count = runPerft(board, depth);
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Q4 Perft Depth " << depth << ": " << count
              << " nodes in " << secs << " s ("
              << (uint64_t)(count / std::max(secs, 1e-6)) << " nps)\n";
    return 0;
  }
  else if(subcmd == "bench") {
    std::cout << "=== Running Q4 Performance Benchmarks ===\n";

    // 1. Move generation on start position
    Q4Board startBoard;
    std::vector<int> actions;
    int movegenIters = 20000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < movegenIters; i++) {
      startBoard.getLegalActions(startBoard.toMove, actions);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double secsStart = std::chrono::duration<double>(t1 - t0).count();
    double rateStart = (double)movegenIters / secsStart;
    std::cout << "Legal-move generations (start position): "
              << (uint64_t)rateStart << " gens/sec (" << movegenIters << " iterations in "
              << secsStart << " s)\n";

    // 2. Move generation on a mid-game position with walls
    Q4Board midBoard;
    midBoard.applyAction(Q4Board::actionOfPawn(Q4Board::cellOf(5, 1))); // South moves to f2
    midBoard.applyAction(Q4Board::actionOfPawn(Q4Board::cellOf(1, 5))); // West moves to b6
    midBoard.applyAction(Q4Board::actionOfPawn(Q4Board::cellOf(5, 9))); // North moves to f10
    midBoard.applyAction(Q4Board::actionOfPawn(Q4Board::cellOf(9, 5))); // East moves to j6
    // Place some walls around
    midBoard.applyAction(Q4Board::actionOfHWall(Q4Board::anchorOf(4, 4))); // e5h
    midBoard.applyAction(Q4Board::actionOfVWall(Q4Board::anchorOf(6, 4))); // g5v
    midBoard.applyAction(Q4Board::actionOfHWall(Q4Board::anchorOf(4, 6))); // e7h
    midBoard.applyAction(Q4Board::actionOfVWall(Q4Board::anchorOf(3, 4))); // d5v

    t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < movegenIters; i++) {
      midBoard.getLegalActions(midBoard.toMove, actions);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double secsMid = std::chrono::duration<double>(t1 - t0).count();
    double rateMid = (double)movegenIters / secsMid;
    std::cout << "Legal-move generations (mid-game position): "
              << (uint64_t)rateMid << " gens/sec (" << movegenIters << " iterations in "
              << secsMid << " s)\n";

    // 3. Random playouts
    int playoutGames = 500;
    Rand rand(12345);
    t0 = std::chrono::high_resolution_clock::now();
    int totalPlies = 0;
    for(int g = 0; g < playoutGames; g++) {
      Q4History history;
      while(!history.isFinished) {
        history.currentBoard.getLegalActions(history.currentBoard.toMove, actions);
        if(actions.empty()) break;
        int act = actions[rand.nextUInt((uint32_t)actions.size())];
        history.play(act);
      }
      totalPlies += history.plies;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double secsPlayout = std::chrono::duration<double>(t1 - t0).count();
    double gamesPerSec = (double)playoutGames / secsPlayout;
    double pliesPerSec = (double)totalPlies / secsPlayout;

    std::cout << "Random-playout games: " << (uint64_t)gamesPerSec << " games/sec ("
              << playoutGames << " games in " << secsPlayout << " s, avg "
              << (double)totalPlies / playoutGames << " plies/game, "
              << (uint64_t)pliesPerSec << " plies/sec)\n";

    return 0;
  }

  if(subcmd == "searchbench") {
    std::string modelPath = "";
    int numThreads = 1;
    int numVisits = 1000;
    int numPositions = 20;
    uint64_t seed = 42;
    std::string configFile = "";

    for(size_t i = 0; i < subArgs.size(); i++) {
      if(subArgs[i] == "-model" && i + 1 < subArgs.size()) {
        modelPath = subArgs[++i];
      }
      else if(subArgs[i] == "-threads" && i + 1 < subArgs.size()) {
        numThreads = Global::stringToInt(subArgs[++i]);
      }
      else if(subArgs[i] == "-visits" && i + 1 < subArgs.size()) {
        numVisits = Global::stringToInt(subArgs[++i]);
      }
      else if(subArgs[i] == "-positions" && i + 1 < subArgs.size()) {
        numPositions = Global::stringToInt(subArgs[++i]);
      }
      else if(subArgs[i] == "-seed" && i + 1 < subArgs.size()) {
        seed = (uint64_t)Global::stringToInt64(subArgs[++i]);
      }
      else if(subArgs[i] == "-config" && i + 1 < subArgs.size()) {
        configFile = subArgs[++i];
      }
    }

    if(modelPath.empty()) {
      std::cerr << "Usage: katago q4tool searchbench -model <path> [-threads 1|4] [-visits 1000] [-positions 20] [-config <cfg>]" << std::endl;
      return 1;
    }

    Logger logger;
    ConfigParser cfg;
    if(!configFile.empty()) {
      try {
        cfg.initialize(configFile);
      }
      catch(const std::exception& e) {
        std::cerr << "Warning: Could not read config file " << configFile << ": " << e.what() << std::endl;
      }
    }

    if(!cfg.contains("nnCacheSizePowerOfTwo")) cfg.overrideKey("nnCacheSizePowerOfTwo", "16");
    if(!cfg.contains("nnMutexPoolSizePowerOfTwo")) cfg.overrideKey("nnMutexPoolSizePowerOfTwo", "12");
    if(!cfg.contains("maxVisits")) cfg.overrideKey("maxVisits", std::to_string(numVisits));
    if(!cfg.contains("numSearchThreads")) cfg.overrideKey("numSearchThreads", std::to_string(numThreads));
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

    Rand evalSeed(seed + 100);
    int maxBatchSize = std::max(32, numThreads);
    NNEvaluator* nnEval = Setup::initializeNNEvaluator(
      modelPath, modelPath, "", cfg, logger, evalSeed, 1, Q4NNConst::POS_LEN, Q4NNConst::POS_LEN,
      Setup::MaxBatchSizeRequest::explicitSize(maxBatchSize), true, false, Setup::SETUP_FOR_BENCHMARK
    );

    SearchParams params = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_BENCHMARK);
    params.maxVisits = numVisits;
    params.numThreads = numThreads;
    Q4S::Search::checkParams(params);

    // Sample distinct positions
    std::vector<Q4PlayState> positions;
    Rand rand(seed);
    Q4Rules rules;
    while((int)positions.size() < numPositions) {
      Q4History hist(rules);
      int targetPlies = rand.nextInt(0, 32);
      for(int p = 0; p < targetPlies && !hist.isFinished; p++) {
        std::vector<int> acts;
        hist.currentBoard.getLegalActions(hist.currentBoard.toMove, acts);
        if(acts.empty()) break;
        hist.play(acts[rand.nextUInt((uint32_t)acts.size())]);
      }
      if(!hist.isFinished) {
        positions.push_back(hist.state);
      }
    }

    Q4S::Search search(params, nnEval, &logger, "bench_" + std::to_string(seed));

    // Warmup on position 0
    search.setPosition(positions[0]);
    search.runWholeSearch();
    search.clearSearch();

    int64_t totalVisits = 0;
    search.resetSearchThreadCpuTimeNs();
    auto t0 = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < positions.size(); i++) {
      search.setPosition(positions[i]);
      search.runWholeSearch();
      totalVisits += search.getRootVisits();
      search.clearSearch();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double pps = (double)totalVisits / secs;
    int64_t totalCpuNs = search.getSearchThreadCpuTimeNs();
    double cpuUsPerPlayout = totalVisits > 0 ? ((double)totalCpuNs / (totalVisits * 1000.0)) : 0.0;

    std::cout << "=== Q4 Search Benchmark ===" << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Threads: " << numThreads << std::endl;
    std::cout << "Visits per search: " << numVisits << std::endl;
    std::cout << "Positions: " << positions.size() << std::endl;
    std::cout << "Total visits: " << totalVisits << " in " << std::fixed << std::setprecision(3) << secs << " s" << std::endl;
    std::cout << "Playouts per second: " << std::fixed << std::setprecision(1) << pps << " visits/sec" << std::endl;
    std::cout << "Search thread CPU time per playout: " << std::fixed << std::setprecision(1) << cpuUsPerPlayout << " us" << std::endl;

    delete nnEval;
    return 0;
  }

  std::cerr << "Unknown q4tool subcommand: " << subcmd << "\n";
  return 1;
}
