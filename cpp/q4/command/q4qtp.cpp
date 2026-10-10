#include "../../game/board.h"
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
#include "q4notation.h"
#include "q4record.h"
#include "q4rules.h"
#include "../search/q4search.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

namespace {

void respondSuccess(const std::string& msg = "") {
  std::cout << "= " << msg << "\n\n" << std::flush;
}

void respondError(const std::string& msg) {
  std::cout << "? " << msg << "\n\n" << std::flush;
}

// Reads stdin on its own thread, so that a line sent while a search runs is seen at once (it cancels
// q4-search_analyze_cancellable) and is not hidden in the stdio buffer. Lines are queued in order.
class StdinLines {
public:
  StdinLines() : st(std::make_shared<State>()) {
    std::shared_ptr<State> s = st;
    std::thread([s]() {
      std::string l;
      while(std::getline(std::cin, l)) {
        std::lock_guard<std::mutex> lock(s->mutex);
        s->lines.push_back(l);
        s->cv.notify_all();
      }
      std::lock_guard<std::mutex> lock(s->mutex);
      s->eof = true;
      s->cv.notify_all();
    }).detach();
  }

  // Blocks until a line is queued; false at the end of input with nothing queued.
  bool next(std::string& out) {
    std::unique_lock<std::mutex> lock(st->mutex);
    st->cv.wait(lock, [this]() { return !st->lines.empty() || st->eof; });
    if(st->lines.empty())
      return false;
    out = st->lines.front();
    st->lines.pop_front();
    return true;
  }

  // Waits up to timeoutMs for a line or the end of input; true if one of them came (nothing is consumed).
  bool arrivedWithin(int timeoutMs) {
    std::unique_lock<std::mutex> lock(st->mutex);
    st->cv.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this]() { return !st->lines.empty() || st->eof; });
    return !st->lines.empty() || st->eof;
  }

private:
  struct State {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
  };
  std::shared_ptr<State> st;
};

const char* SEAT_LETTERS = "SWNE";

std::string joinDoubles(const double* v, int n, int prec) {
  std::ostringstream ss;
  for(int i = 0; i < n; i++) {
    if(i > 0) ss << ",";
    ss << std::fixed << std::setprecision(prec) << v[i];
  }
  return ss.str();
}

// The keys q4-set-param accepts.
const char* SET_PARAM_KEYS = "maxVisits, q4OpponentMode, q4OwnSeats, dynamicScoreUtilityFactor, "
                             "q4LeadWeightEps, cpuctExploration, rootPolicyTemperature";

}  // namespace

int MainCmds::q4qtp(const std::vector<std::string>& args) {
  // NNEvaluator's transformer warmup builds a Duel Board, which needs the Duel Zobrist tables.
  Board::initHash();
  std::vector<std::string> subArgs = args;
  if(!subArgs.empty() && subArgs[0] == "q4qtp")
    subArgs = std::vector<std::string>(subArgs.begin() + 1, subArgs.end());

  std::string configFile = "";
  std::string modelFile = "";
  std::string botType = "greedy";
  double temperature = 0.0;
  uint64_t seed = 42;

  for(size_t i = 0; i < subArgs.size(); i++) {
    if(subArgs[i] == "-config" && i + 1 < subArgs.size()) {
      configFile = subArgs[++i];
    }
    else if(subArgs[i] == "-model" && i + 1 < subArgs.size()) {
      modelFile = subArgs[++i];
    }
    else if(subArgs[i] == "-bot" && i + 1 < subArgs.size()) {
      botType = subArgs[++i];
    }
    else if(subArgs[i] == "-temp" && i + 1 < subArgs.size()) {
      temperature = Global::stringToDouble(subArgs[++i]);
    }
    else if(subArgs[i] == "-seed" && i + 1 < subArgs.size()) {
      seed = (uint64_t)Global::stringToInt64(subArgs[++i]);
    }
  }

  ConfigParser cfg;
  if(!configFile.empty()) {
    try {
      cfg.initialize(configFile);
      if(cfg.contains("bot"))
        botType = cfg.getString("bot");
      if(cfg.contains("model"))
        modelFile = cfg.getString("model");
      if(cfg.contains("temperature"))
        temperature = cfg.getDouble("temperature");
      if(cfg.contains("seed"))
        seed = (uint64_t)cfg.getInt64("seed");
    }
    catch(const std::exception& e) {
      std::cerr << "Warning: Could not read config file " << configFile << ": " << e.what() << std::endl;
    }
  }

  if((botType == "nnpolicy" || botType == "search") && modelFile.empty()) {
    std::cerr << "Error: bot = " << botType << " needs a model (-model <file> or model = <file> in the config)" << std::endl;
    return 1;
  }

  // The evaluator keeps a pointer to its logger until it is deleted, so the logger lives as long as the command.
  Logger logger;
  NNEvaluator* nnEval = nullptr;
  if(!modelFile.empty()) {
    if(!cfg.contains("nnCacheSizePowerOfTwo")) cfg.overrideKey("nnCacheSizePowerOfTwo", "16");
    if(!cfg.contains("nnMutexPoolSizePowerOfTwo")) cfg.overrideKey("nnMutexPoolSizePowerOfTwo", "12");
    Rand seedRand(seed);
    nnEval = Setup::initializeNNEvaluator(
      modelFile, modelFile, "", cfg, logger, seedRand, 1, Q4NNConst::POS_LEN, Q4NNConst::POS_LEN,
      Setup::MaxBatchSizeRequest::explicitSize(16), true, false, Setup::SETUP_FOR_GTP
    );
  }

  Rand rand(seed);
  Q4Rules rules;
  Q4History history(rules);
  std::unique_ptr<Q4Bot> bot = nullptr;
  std::unique_ptr<Q4S::Search> search = nullptr;
  if(botType == "search") {
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
    SearchParams searchParams = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_GTP);
    Q4S::Search::checkParams(searchParams);
    search = std::make_unique<Q4S::Search>(searchParams, nnEval, &logger, "q4qtp_" + std::to_string(seed));
    search->setOpponentModel(Q4S::loadOpponentMode(cfg), 0);   // own seat: the seat to move
    search->setLeadWeightEps(Q4S::loadLeadWeightEps(cfg));
    search->setPosition(history);
  }
  else if(botType != "nnpolicy") {
    bot = Q4Bots::makeBot(botType, seed);
  }

  // Bots other than the one of -bot (q4-genmove-bot), made on first use and kept, so a grudge bot keeps its memory.
  std::map<std::string, std::unique_ptr<Q4Bot>> botCache;

  StdinLines input;
  std::string line;
  while(input.next(line)) {
    line = Global::trim(line);
    if(line.empty() || line[0] == '#')
      continue;

    // Remove GTP ID if present
    std::string id = "";
    std::vector<std::string> tokens = Global::split(line, ' ');
    if(tokens.empty())
      continue;

    size_t cmdIdx = 0;
    if(!tokens[0].empty() && std::isdigit(tokens[0][0])) {
      id = tokens[0] + " ";
      cmdIdx = 1;
    }
    if(cmdIdx >= tokens.size())
      continue;

    std::string cmd = Global::toLower(tokens[cmdIdx]);
    std::vector<std::string> cmdArgs(tokens.begin() + cmdIdx + 1, tokens.end());

    try {
      if(cmd == "protocol_version") {
        respondSuccess("2");
      }
      else if(cmd == "name") {
        respondSuccess("KataQuoridor Q4");
      }
      else if(cmd == "version") {
        respondSuccess("1.18.2");
      }
      else if(cmd == "known_command") {
        if(cmdArgs.empty()) {
          respondSuccess("false");
        }
        else {
          std::string c = Global::toLower(cmdArgs[0]);
          bool known = (c == "protocol_version" || c == "name" || c == "version" ||
                        c == "known_command" || c == "list_commands" || c == "quit" ||
                        c == "clear_board" || c == "play" || c == "legal_moves" ||
                        c == "eliminate" || c == "undo" || c == "showboard" ||
                        c == "winner" || c == "walls" || c == "dist" ||
                        c == "to_move" || c == "get_rules" || c == "set_rule" ||
                        c == "set_rules" || c == "printrecord" || c == "loadrecord" ||
                        c == "genmove" || c == "q4-rawnn" || c == "q4-set-temperature" ||
                        c == "q4-analyze" || c == "q4-search-params" ||
                        c == "q4-search_analyze_cancellable" || c == "q4-set-param" ||
                        c == "q4-genmove-bot" || c == "q4-position");
          respondSuccess(known ? "true" : "false");
        }
      }
      else if(cmd == "list_commands") {
        std::string list = "protocol_version\nname\nversion\nknown_command\nlist_commands\nquit\n"
                           "clear_board\nplay\nlegal_moves\neliminate\nundo\nshowboard\n"
                           "winner\nwalls\ndist\nto_move\nget_rules\nset_rule\nset_rules\n"
                           "printrecord\nloadrecord\ngenmove\nq4-rawnn\nq4-set-temperature\n"
                           "q4-analyze\nq4-search-params\nq4-search_analyze_cancellable\n"
                           "q4-set-param\nq4-genmove-bot\nq4-position";
        respondSuccess(list);
      }
      else if(cmd == "quit") {
        respondSuccess();
        break;
      }
      else if(cmd == "clear_board") {
        history.clear(Q4Board(rules), rules);
        if(search) { search->clearSearch(); search->setPosition(history); }
        respondSuccess();
      }
      else if(cmd == "play") {
        if(cmdArgs.size() < 2)
          throw StringError("Usage: play <seat> <action>");
        int seat = Q4Notation::stringToSeat(cmdArgs[0]);
        if(history.isFinished)
          throw StringError("Game is already finished");
        if(seat != history.currentBoard.toMove)
          throw StringError("Not seat " + Global::intToString(seat + 1) + "'s turn (it is seat " +
                            Global::intToString(history.currentBoard.toMove + 1) + "'s turn)");
        int action = Q4Notation::stringToAction(cmdArgs[1]);
        if(!history.currentBoard.isLegalAction(action, seat))
          throw StringError("Illegal move: " + cmdArgs[1]);
        history.play(action);
        if(search) search->makeMove(action);
        respondSuccess();
      }
      else if(cmd == "legal_moves") {
        int seat = history.currentBoard.toMove;
        if(!cmdArgs.empty())
          seat = Q4Notation::stringToSeat(cmdArgs[0]);
        std::vector<int> actions;
        history.currentBoard.getLegalActions(seat, actions);
        std::ostringstream ss;
        for(size_t i = 0; i < actions.size(); i++) {
          if(i > 0) ss << " ";
          ss << Q4Notation::actionToString(actions[i]);
        }
        respondSuccess(ss.str());
      }
      else if(cmd == "eliminate") {
        if(cmdArgs.empty())
          throw StringError("Usage: eliminate <seat>");
        int seat = Q4Notation::stringToSeat(cmdArgs[0]);
        if(!history.currentBoard.isAlive(seat))
          throw StringError("Seat " + Global::intToString(seat + 1) + " is already eliminated");
        history.eliminate(seat);
        if(search) { search->clearSearch(); search->setPosition(history); }
        respondSuccess();
      }
      else if(cmd == "undo") {
        if(!history.undo())
          throw StringError("Cannot undo: already at start of game");
        if(search) { search->clearSearch(); search->setPosition(history); }
        respondSuccess();
      }
      else if(cmd == "showboard") {
        respondSuccess("\n" + Q4Notation::renderBoardAscii(history.currentBoard, history.plies));
      }
      else if(cmd == "winner") {
        respondSuccess(history.getResultString());
      }
      else if(cmd == "walls") {
        if(cmdArgs.empty()) {
          std::ostringstream ss;
          for(int s = 0; s < Q4Board::NUM_SEATS; s++) {
            if(s > 0) ss << " ";
            ss << (int)history.currentBoard.wallsLeft[s];
          }
          respondSuccess(ss.str());
        }
        else {
          int s = Q4Notation::stringToSeat(cmdArgs[0]);
          respondSuccess(Global::intToString((int)history.currentBoard.wallsLeft[s]));
        }
      }
      else if(cmd == "dist") {
        if(cmdArgs.empty()) {
          std::ostringstream ss;
          for(int s = 0; s < Q4Board::NUM_SEATS; s++) {
            if(s > 0) ss << " ";
            if(history.currentBoard.isAlive(s) && history.currentBoard.pawn[s] >= 0)
              ss << (int)history.currentBoard.distToCenter[history.currentBoard.pawn[s]];
            else
              ss << "255";
          }
          respondSuccess(ss.str());
        }
        else {
          int s = Q4Notation::stringToSeat(cmdArgs[0]);
          if(history.currentBoard.isAlive(s) && history.currentBoard.pawn[s] >= 0)
            respondSuccess(Global::intToString((int)history.currentBoard.distToCenter[history.currentBoard.pawn[s]]));
          else
            respondSuccess("255");
        }
      }
      else if(cmd == "to_move") {
        respondSuccess(Global::intToString(history.currentBoard.toMove + 1));
      }
      else if(cmd == "get_rules") {
        respondSuccess(rules.toString());
      }
      else if(cmd == "set_rule") {
        if(history.plies > 0)
          throw StringError("Cannot set rules after game has started");
        if(cmdArgs.size() < 2)
          throw StringError("Usage: set_rule <key> <value>");
        std::string kv = cmdArgs[0] + "=" + cmdArgs[1];
        rules = Q4Rules::parse(kv);
        history.clear(Q4Board(rules), rules);
        if(search) { search->clearSearch(); search->setPosition(history); }
        respondSuccess();
      }
      else if(cmd == "set_rules") {
        if(history.plies > 0)
          throw StringError("Cannot set rules after game has started");
        if(cmdArgs.empty())
          throw StringError("Usage: set_rules <json or Q4:k=v,...>");
        std::string rest = cmdArgs[0];
        for(size_t i = 1; i < cmdArgs.size(); i++)
          rest += " " + cmdArgs[i];
        rules = Q4Rules::parseRulesOrJson(rest);
        history.clear(Q4Board(rules), rules);
        if(search) { search->clearSearch(); search->setPosition(history); }
        respondSuccess();
      }
      else if(cmd == "printrecord") {
        Q4Record rec;
        rec.rules = history.rules;
        rec.events = history.events;
        rec.result = history.getResultString();
        respondSuccess(rec.toSgfLine());
      }
      else if(cmd == "loadrecord") {
        if(cmdArgs.empty())
          throw StringError("Usage: loadrecord <file> [n]");
        std::string file = cmdArgs[0];
        int targetLine = 1;
        if(cmdArgs.size() >= 2)
          targetLine = Global::stringToInt(cmdArgs[1]);

        std::ifstream in(file);
        if(!in.is_open())
          throw StringError("Could not open file: " + file);
        std::string rline;
        int currentLine = 0;
        bool found = false;
        while(std::getline(in, rline)) {
          rline = Global::trim(rline);
          if(rline.empty()) continue;
          currentLine++;
          if(currentLine == targetLine) {
            found = true;
            break;
          }
        }
        if(!found)
          throw StringError("Record number " + Global::intToString(targetLine) + " not found in " + file);

        Q4Record rec = Q4Record::fromSgfLine(rline);
        rules = rec.rules;
        rec.replay(history);
        if(search) { search->clearSearch(); search->setPosition(history); }
        respondSuccess();
      }
      else if(cmd == "genmove") {
        if(history.isFinished)
          throw StringError("Game is already finished");
        int seat = history.currentBoard.toMove;
        if(!cmdArgs.empty()) {
          int reqSeat = Q4Notation::stringToSeat(cmdArgs[0]);
          if(reqSeat != seat)
            throw StringError("Requested seat " + Global::intToString(reqSeat + 1) +
                              " is not to move (seat " + Global::intToString(seat + 1) + " is)");
        }
        int action = Q4Board::NULL_ACTION;
        if(botType == "search") {
          search->setPosition(history);
          action = search->runWholeSearchAndGetMove();
        }
        else if(botType == "nnpolicy") {
          // The policy (variant 0, symmetry 0) over the legal actions: the argmax, or a sample at the temperature.
          NNResultBuf buf;
          Q4NN::Eval eval;
          Q4NN::evaluate(*nnEval, buf, history, 0, false, eval);
          std::vector<int> legalActions;
          history.currentBoard.getLegalActions(seat, legalActions);
          if(legalActions.empty())
            throw StringError("No legal move available");
          std::vector<float> probs(Q4Board::NUM_ACTIONS);
          Q4NN::softmaxLegal(eval.policyLogits[0], legalActions, probs.data(), (float)temperature);
          if(temperature <= 1e-4) {
            float best = -1.0f;
            for(int a : legalActions) {
              if(probs[a] > best) {
                best = probs[a];
                action = a;
              }
            }
          }
          else {
            float r = (float)rand.nextDouble();
            float cumulative = 0.0f;
            for(size_t i = 0; i < legalActions.size(); i++) {
              cumulative += probs[legalActions[i]];
              if(r <= cumulative || i + 1 == legalActions.size()) {
                action = legalActions[i];
                break;
              }
            }
          }
        }
        else {
          action = bot->getMove(history.currentBoard);
        }
        if(action == Q4Board::NULL_ACTION)
          throw StringError("No legal move available");
        history.play(action);
        if(search) search->makeMove(action);
        respondSuccess(Q4Notation::actionToString(action));
      }
      else if(cmd == "q4-set-temperature") {
        if(cmdArgs.empty())
          throw StringError("Usage: q4-set-temperature <temperature>");
        temperature = Global::stringToDouble(cmdArgs[0]);
        respondSuccess();
      }
      else if(cmd == "q4-rawnn") {
        if(nnEval == nullptr)
          throw StringError("No NN model loaded (use -model <file>)");
        int sym = 0;
        if(!cmdArgs.empty()) {
          sym = Global::stringToInt(cmdArgs[0]);
          if(sym < 0 || sym >= Q4Symmetry::NUM_SYMMETRIES)
            throw StringError("Symmetry must be in 0..7");
        }
        if(history.isFinished)
          throw StringError("Game is already finished");
        NNResultBuf buf;
        Q4NN::Eval eval;
        Q4NN::evaluate(*nnEval, buf, history, sym, true, eval);
        std::vector<int> legalActions;
        history.currentBoard.getLegalActions(history.currentBoard.toMove, legalActions);
        respondSuccess(Q4NN::formatEval(eval, legalActions, 10));
      }
      else if(cmd == "q4-analyze") {
        if(search == nullptr)
          throw StringError("Search not enabled (bot must be search and model provided)");
        int visits = search->searchParams.maxVisits;
        if(!cmdArgs.empty())
          visits = Global::stringToInt(cmdArgs[0]);
        search->setPosition(history);
        SearchParams origParams = search->searchParams;
        SearchParams tempParams = origParams;
        tempParams.maxVisits = visits;
        search->setParamsNoClearing(tempParams);
        search->runWholeSearch();
        search->setParamsNoClearing(origParams);

        Q4S::ReportedSearchValues rootVals;
        search->getRootValues(rootVals);
        std::vector<Q4S::AnalysisData> analysis;
        search->getAnalysisData(analysis);

        std::ostringstream ss;
        ss << "rootValues: [";
        for(int k = 0; k < 5; k++) {
          if(k > 0) ss << ", ";
          ss << std::fixed << std::setprecision(4) << rootVals.value[k];
        }
        ss << "] rootUtilities: [";
        for(int s = 0; s < 4; s++) {
          if(s > 0) ss << ", ";
          ss << std::fixed << std::setprecision(4) << rootVals.utility[s];
        }
        ss << "]\n";
        for(size_t i = 0; i < analysis.size(); i++) {
          const auto& ad = analysis[i];
          if(i > 0) ss << "\n";
          ss << "move " << Q4Notation::actionToString(ad.move)
             << " visits " << ad.numVisits
             << " prior " << std::fixed << std::setprecision(4) << ad.policyPrior
             << " utility " << std::fixed << std::setprecision(4) << ad.utility
             << " lcb " << std::fixed << std::setprecision(4) << ad.lcb
             << " value [";
          for(int k = 0; k < 5; k++) {
            if(k > 0) ss << ", ";
            ss << std::fixed << std::setprecision(4) << ad.valueAvg[k];
          }
          ss << "] pv ";
          ad.writePV(ss);
        }
        respondSuccess(ss.str());
      }
      else if(cmd == "q4-position") {
        // The whole position as one JSON object (seats 0..3 = S W N E; cells and walls in QTP notation).
        const Q4Board& b = history.currentBoard;
        json j;
        j["plies"] = history.plies;
        j["toMove"] = b.toMove;
        j["finished"] = history.isFinished;
        j["winner"] = history.winnerSeat;
        j["draw"] = history.isDraw;
        j["alive"] = json::array();
        j["pawns"] = json::array();
        j["wallsLeft"] = json::array();
        j["dist"] = json::array();
        for(int s = 0; s < Q4Board::NUM_SEATS; s++) {
          j["alive"].push_back(b.isAlive(s));
          j["pawns"].push_back(b.pawn[s] >= 0 ? Q4Notation::cellToString(b.pawn[s]) : "");
          j["wallsLeft"].push_back((int)b.wallsLeft[s]);
          j["dist"].push_back(b.isAlive(s) && b.pawn[s] >= 0 ? (int)b.distToCenter[b.pawn[s]] : 255);
        }
        j["walls"] = json::array();
        for(int a = 0; a < Q4Board::NUM_WALL_ANCHORS; a++) {
          int ax = Q4Board::anchorX(a), ay = Q4Board::anchorY(a);
          if(b.hWalls[a])
            j["walls"].push_back(Q4Notation::wallToString(ax, ay, true));
          if(b.vWalls[a])
            j["walls"].push_back(Q4Notation::wallToString(ax, ay, false));
        }
        respondSuccess(j.dump());
      }
      else if(cmd == "q4-set-param") {
        // Changing a value clears the tree (the stored values depend on it), as the setters of Q4S::Search do.
        if(search == nullptr)
          throw StringError("Search not enabled (bot must be search and model provided)");
        if(cmdArgs.size() < 2)
          throw StringError(std::string("Usage: q4-set-param <key> <value>; keys: ") + SET_PARAM_KEYS);
        const std::string& key = cmdArgs[0];
        const std::string& value = cmdArgs[1];
        if(key == "maxVisits") {
          SearchParams p = search->searchParams;
          p.maxVisits = Global::stringToInt64(value);
          search->setParams(p);
        }
        else if(key == "dynamicScoreUtilityFactor" || key == "cpuctExploration" || key == "rootPolicyTemperature") {
          SearchParams p = search->searchParams;
          double v = Global::stringToDouble(value);
          if(key == "dynamicScoreUtilityFactor") p.dynamicScoreUtilityFactor = v;
          else if(key == "cpuctExploration") p.cpuctExploration = v;
          else p.rootPolicyTemperature = v;
          search->setParams(p);
        }
        else if(key == "q4OpponentMode") {
          search->setOpponentModel(Q4S::parseOpponentMode(value), search->ownSeatMask);
        }
        else if(key == "q4OwnSeats") {
          int64_t mask = Global::stringToInt64(value);
          if(mask < 0 || mask > 15)
            throw StringError("q4OwnSeats must be a bit mask 0..15 (bit s = seat s+1 is own)");
          search->setOpponentModel(search->opponentMode, (uint8_t)mask);
        }
        else if(key == "q4LeadWeightEps") {
          double eps = Global::stringToDouble(value);
          if(!(eps >= 0.0 && eps <= 1.0))
            throw StringError("q4LeadWeightEps must be in [0, 1]");
          search->setLeadWeightEps(eps);
        }
        else {
          throw StringError("Unknown key: " + key + " (keys: " + SET_PARAM_KEYS + ")");
        }
        respondSuccess();
      }
      else if(cmd == "q4-genmove-bot") {
        // The move of a bot for the seat to move (the -bot of the process is the seat-independent default only).
        if(history.isFinished)
          throw StringError("Game is already finished");
        if(cmdArgs.empty())
          throw StringError("Usage: q4-genmove-bot <bot> [target seat]");
        std::string type = cmdArgs[0];
        int target = -1;
        if(cmdArgs.size() >= 2)
          target = Q4Notation::stringToSeat(cmdArgs[1]);
        if(Global::toLower(type) == "grudge" && target < 0)
          throw StringError("grudge needs a target seat: q4-genmove-bot grudge <seat>");
        std::string key = Global::toLower(type) + "/" + std::to_string(target);
        auto it = botCache.find(key);
        if(it == botCache.end())
          it = botCache.emplace(key, Q4Bots::makeBot(type, seed, target)).first;
        int action = it->second->getMove(history.currentBoard);
        if(action == Q4Board::NULL_ACTION)
          throw StringError("No legal move available");
        history.play(action);   // throws on an illegal move
        if(search) search->makeMove(action);
        respondSuccess(Q4Notation::actionToString(action));
      }
      else if(cmd == "q4-search_analyze_cancellable") {
        // q4-search_analyze_cancellable [interval centis, default 25] [visit cap, default maxVisits]: searches the seat to
        // move's position and streams "info root ..." and "info move ..." lines every interval until a line arrives on
        // stdin (the answer line is "play cancelled") or the cap is reached ("done"). No move is made and the tree is kept
        // for the next search (no clearing). Output: "=", info lines, the answer line, an empty line.
        if(search == nullptr)
          throw StringError("Search not enabled (bot must be search and model provided)");
        if(history.isFinished)
          throw StringError("Game is already finished");
        int intervalCs = 25;
        if(!cmdArgs.empty())
          intervalCs = Global::stringToInt(cmdArgs[0]);
        if(intervalCs < 1)
          throw StringError("Interval must be at least 1 centisecond");
        const int intervalMs = intervalCs * 10;
        // The visit cap of this search (default: maxVisits); the parameters are restored afterwards, without clearing the tree.
        const SearchParams origParams = search->searchParams;
        SearchParams cappedParams = origParams;
        if(cmdArgs.size() >= 2)
          cappedParams.maxVisits = Global::stringToInt64(cmdArgs[1]);
        search->setParamsNoClearing(cappedParams);

        auto printAnalysis = [&]() {
          Q4S::ReportedSearchValues rootVals;
          std::ostringstream ss;
          if(search->getRootValues(rootVals)) {
            ss << "info root visits " << search->getRootVisits()
               << " value " << joinDoubles(rootVals.value, 5, 4)
               << " utility " << joinDoubles(rootVals.utility, 4, 4)
               << " lead " << joinDoubles(rootVals.lead, 4, 4) << "\n";
          }
          else {
            ss << "info root visits " << search->getRootVisits() << "\n";
          }
          std::vector<Q4S::AnalysisData> analysis;
          search->getAnalysisData(analysis);
          for(const auto& ad : analysis) {
            if(ad.numVisits <= 0)
              continue;   // a candidate without a visit yet says nothing
            // The PV with the seat of each move: "S f2 W b6 ...". The first move is the candidate itself.
            std::ostringstream pv;
            Q4PlayState s = search->getRootState();
            for(size_t i = 0; i < ad.pv.size() && !s.isFinished; i++) {
              if(i > 0) pv << " ";
              pv << SEAT_LETTERS[s.board.toMove] << " " << Q4Notation::actionToString(ad.pv[i]);
              s.playAssumeLegal(ad.pv[i]);
            }
            ss << "info move " << Q4Notation::actionToString(ad.move)
               << " visits " << ad.numVisits
               << " prior " << std::fixed << std::setprecision(4) << ad.policyPrior
               << " utility " << std::fixed << std::setprecision(4) << ad.utility
               << " lcb " << std::fixed << std::setprecision(4) << ad.lcb
               << " value " << joinDoubles(ad.valueAvg, 5, 4)
               << " pv " << pv.str() << "\n";
          }
          std::cout << ss.str() << std::flush;
        };

        // As Duel's kata-search_analyze_cancellable: the "=" header first, the "info" lines, then the answer line
        // (no second "="), so that a client that streams the info lines sees them inside the response.
        std::cout << "=\n" << std::flush;

        std::atomic<bool> stopNow(false);
        std::atomic<bool> finished(false);
        std::exception_ptr workerError = nullptr;
        std::thread worker([&]() {
          try {
            search->runWholeSearch(&stopNow);
          }
          catch(...) {
            workerError = std::current_exception();
          }
          finished.store(true);
        });

        bool cancelled = false;
        while(true) {
          if(input.arrivedWithin(intervalMs)) {
            cancelled = true;   // the line stays queued: it is the next command
            break;
          }
          if(finished.load())
            break;
          printAnalysis();
        }
        stopNow.store(true);
        worker.join();
        search->setParamsNoClearing(origParams);
        if(workerError) {
          std::string what = "search failed";
          try {
            std::rethrow_exception(workerError);
          }
          catch(const std::exception& e) {
            what = e.what();
          }
          std::cout << "error: " << what << "\n\n" << std::flush;
          continue;
        }
        printAnalysis();
        std::cout << (cancelled ? "play cancelled" : "done") << "\n\n" << std::flush;
      }
      else if(cmd == "q4-search-params") {
        if(search == nullptr)
          throw StringError("Search not enabled (bot must be search and model provided)");
        std::ostringstream ss;
        ss << "maxVisits = " << search->searchParams.maxVisits << "\n";
        ss << "numSearchThreads = " << search->searchParams.numThreads << "\n";
        ss << "rootNumSymmetriesToSample = " << search->searchParams.rootNumSymmetriesToSample << "\n";
        ss << "cpuctExploration = " << search->searchParams.cpuctExploration << "\n";
        ss << "cpuctExplorationLog = " << search->searchParams.cpuctExplorationLog << "\n";
        ss << "fpuReductionMax = " << search->searchParams.fpuReductionMax << "\n";
        ss << "rootFpuReductionMax = " << search->searchParams.rootFpuReductionMax << "\n";
        ss << "useLcbForSelection = " << (search->searchParams.useLcbForSelection ? "true" : "false") << "\n";
        ss << "lcbStdevs = " << search->searchParams.lcbStdevs << "\n";
        ss << "minVisitPropForLCB = " << search->searchParams.minVisitPropForLCB << "\n";
        ss << "winLossUtilityFactor = " << search->searchParams.winLossUtilityFactor << "\n";
        ss << "rootNoiseEnabled = " << (search->searchParams.rootNoiseEnabled ? "true" : "false") << "\n";
        ss << "rootDirichletNoiseTotalConcentration = " << search->searchParams.rootDirichletNoiseTotalConcentration << "\n";
        ss << "rootDirichletNoiseWeight = " << search->searchParams.rootDirichletNoiseWeight << "\n";
        ss << "rootPolicyTemperature = " << search->searchParams.rootPolicyTemperature << "\n";
        ss << "chosenMoveTemperature = " << search->searchParams.chosenMoveTemperature << "\n";
        ss << "dynamicScoreUtilityFactor = " << search->searchParams.dynamicScoreUtilityFactor << "\n";
        ss << "q4OpponentMode = " << Q4S::opponentModeName(search->opponentMode) << "\n";
        ss << "q4OwnSeats = " << (int)search->ownSeatMask << "\n";
        ss << "q4LeadWeightEps = " << search->leadWeightEps;
        respondSuccess(ss.str());
      }
      else {
        respondError("unknown command: " + cmd);
      }
    }
    catch(const std::exception& e) {
      respondError(e.what());
    }
  }

  delete nnEval;
  return 0;
}
