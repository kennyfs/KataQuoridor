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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

void respondSuccess(const std::string& msg = "") {
  std::cout << "= " << msg << "\n\n" << std::flush;
}

void respondError(const std::string& msg) {
  std::cout << "? " << msg << "\n\n" << std::flush;
}

}  // namespace

int MainCmds::q4qtp(const std::vector<std::string>& args) {
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

  if(botType == "nnpolicy" && modelFile.empty()) {
    std::cerr << "Error: bot = nnpolicy needs a model (-model <file> or model = <file> in the config)" << std::endl;
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
  if(botType != "nnpolicy")
    bot = Q4Bots::makeBot(botType, seed);

  std::string line;
  while(std::getline(std::cin, line)) {
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
                        c == "genmove" || c == "q4-rawnn" || c == "q4-set-temperature");
          respondSuccess(known ? "true" : "false");
        }
      }
      else if(cmd == "list_commands") {
        std::string list = "protocol_version\nname\nversion\nknown_command\nlist_commands\nquit\n"
                           "clear_board\nplay\nlegal_moves\neliminate\nundo\nshowboard\n"
                           "winner\nwalls\ndist\nto_move\nget_rules\nset_rule\nset_rules\n"
                           "printrecord\nloadrecord\ngenmove\nq4-rawnn\nq4-set-temperature";
        respondSuccess(list);
      }
      else if(cmd == "quit") {
        respondSuccess();
        break;
      }
      else if(cmd == "clear_board") {
        history.clear(Q4Board(rules), rules);
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
        respondSuccess();
      }
      else if(cmd == "undo") {
        if(!history.undo())
          throw StringError("Cannot undo: already at start of game");
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
        respondSuccess();
      }
      else if(cmd == "printrecord") {
        Q4Record rec;
        rec.rules = history.rules;
        rec.events = history.events;
        rec.result = history.getResultString();
        respondSuccess(rec.toJsonLine());
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

        Q4Record rec = Q4Record::fromJsonLine(rline);
        rules = rec.rules;
        rec.replay(history);
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
        if(botType == "nnpolicy") {
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
