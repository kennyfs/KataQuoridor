#include "q4styletool.h"

#include "../../core/global.h"
#include "../../core/rand.h"
#include "../../external/nlohmann_json/json.hpp"
#include "../q4bots.h"
#include "../q4history.h"
#include "../q4record.h"
#include "../q4style.h"
#include "../q4notation.h"
#include "q4json.h"

#include <cstdio>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

bool isQ4StyleCommand(const std::string& subcmd) {
  return subcmd == "style" || subcmd == "popgames";
}

static int runStyle(const std::vector<std::string>& args) {
  std::string inputFile;
  for(size_t i = 0; i < args.size(); i++) {
    if(args[i] == "-input" && i + 1 < args.size())
      inputFile = args[++i];
    else
      throw StringError("q4tool style: unknown argument " + args[i]);
  }
  std::ifstream fileIn;
  if(!inputFile.empty()) {
    fileIn.open(inputFile);
    if(!fileIn.is_open())
      throw StringError("q4tool style: could not open " + inputFile);
  }
  std::istream& in = inputFile.empty() ? std::cin : fileIn;

  std::string line;
  int game = 0;
  while(std::getline(in, line)) {
    if(Global::trim(line).empty())
      continue;
    // A game record (an SGF line), or the test input {"rules": {...}, "initialBoard": {...}?, "events": [{"a": "e5h"} |
    // {"elim": seat}]} (events of the symmetry tests that need not be a played game, or a start position that is not
    // the standard one; elim counts seats from 1 without initialBoard, as the former JSON records did).
    Q4Record rec;
    Q4History hist;
    // A game record is an SGF line; the JSON form below is only an input of the symmetry tests.
    const bool isJson = Global::trim(line)[0] == '{';
    json j = isJson ? json::parse(line) : json::object();
    if(isJson) {
      Q4Rules rules;
      if(j.contains("rules")) {
        const json& r = j["rules"];
        rules.maxPlies = r.value("maxPlies", rules.maxPlies);
        rules.repetitionDrawCount = r.value("repetitionDrawCount", rules.repetitionDrawCount);
      }
      hist.clear(j.contains("initialBoard") ? parseBoardFromJson(j["initialBoard"]) : Q4Board(rules), rules);
      for(const json& item : j["events"]) {
        Q4Event ev;
        ev.isElimination = item.contains("elim");
        ev.action = ev.isElimination ? Q4Board::NULL_ACTION : Q4Notation::stringToAction(item["a"].get<std::string>());
        ev.eliminatedSeat = ev.isElimination ? item["elim"].get<int>() - (j.contains("initialBoard") ? 0 : 1) : -1;
        rec.events.push_back(ev);
      }
    }
    else {
      rec = Q4Record::fromSgfLine(Global::trim(line));
      hist.clear(Q4Board(rec.rules), rec.rules);
    }
    std::vector<int> perspectives;
    std::string features = "[";
    int numPositions = 0;
    auto emit = [&]() {
      float f[Q4StyleTracker::NUM_FEATURES];
      hist.getStyleFeatures(f);
      perspectives.push_back(hist.currentBoard.toMove);
      for(int k = 0; k < Q4StyleTracker::NUM_FEATURES; k++) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.9g", (double)f[k]);
        if(numPositions > 0 || k > 0)
          features += ",";
        features += buf;
      }
      numPositions++;
    };
    emit();
    for(const Q4Event& ev : rec.events) {
      if(ev.isElimination)
        hist.eliminate(ev.eliminatedSeat);
      else
        hist.play(ev.action);
      emit();
    }
    features += "]";
    std::cout << "{\"game\":" << game << ",\"numPositions\":" << numPositions << ",\"perspectives\":"
              << json(perspectives).dump() << ",\"features\":" << features << "}\n";
    game++;
  }
  return 0;
}

static int runPopGames(const std::vector<std::string>& args) {
  int n = 100;
  uint64_t seed = 1;
  int maxPlies = 120;
  double elimProb = 0.3;
  std::vector<std::string> fixedBots, fixedTargets;
  for(size_t i = 0; i < args.size(); i++) {
    const bool hasNext = i + 1 < args.size();
    if(args[i] == "-n" && hasNext)
      n = Global::stringToInt(args[++i]);
    else if(args[i] == "-seed" && hasNext)
      seed = (uint64_t)Global::stringToInt64(args[++i]);
    else if(args[i] == "-maxplies" && hasNext)
      maxPlies = Global::stringToInt(args[++i]);
    else if(args[i] == "-elimprob" && hasNext)
      elimProb = Global::stringToDouble(args[++i]);
    else if(args[i] == "-bots" && hasNext)
      fixedBots = Global::split(args[++i], ',');
    else if(args[i] == "-targets" && hasNext)
      fixedTargets = Global::split(args[++i], ',');
    else
      throw StringError("q4tool popgames: unknown or incomplete argument " + args[i]);
  }
  static const char* const BOTS[5] = {"random", "randomPawn", "greedy", "basher", "grudge"};
  for(int g = 0; g < n; g++) {
    Rand rand("q4popgames:" + Global::uint64ToString(seed) + ":" + Global::intToString(g));
    Q4Rules rules;
    rules.maxPlies = maxPlies;
    rules.repetitionDrawCount = rand.nextBool(0.5) ? 3 : 0;
    Q4History hist(rules);
    Q4Record rec;
    rec.rules = rules;
    std::unique_ptr<Q4Bot> bots[4];
    rec.players.resize(4);
    for(int s = 0; s < 4; s++) {
      std::string type = BOTS[rand.nextUInt(5)];
      int target = (s + 1 + (int)rand.nextUInt(3)) % 4;
      if(fixedBots.size() == 4)
        type = Global::trim(fixedBots[s]);
      if(fixedTargets.size() == 4)
        target = Global::stringToInt(Global::trim(fixedTargets[s]));
      bots[s] = Q4Bots::makeBot(type, rand.nextUInt64(), target);
      rec.players[s].name = "Seat " + Global::intToString(s);
      rec.players[s].type = type;
    }
    const bool willEliminate = rand.nextBool(elimProb);
    const int elimPly = 1 + (int)rand.nextUInt((uint32_t)std::max(1, maxPlies - 1));
    bool eliminated = false;
    while(!hist.isFinished) {
      if(willEliminate && !eliminated && hist.plies >= elimPly) {
        std::vector<int> alive;
        for(int s = 0; s < 4; s++)
          if(hist.currentBoard.isAlive(s))
            alive.push_back(s);
        if(alive.size() > 1) {
          hist.eliminate(alive[rand.nextUInt((uint32_t)alive.size())]);
          eliminated = true;
          continue;
        }
      }
      int act = bots[hist.currentBoard.toMove]->getMove(hist.currentBoard);
      if(act == Q4Board::NULL_ACTION)
        break;
      hist.play(act);
    }
    rec.events = hist.events;
    rec.result = hist.getResultString();
    std::cout << rec.toSgfLine() << "\n";
  }
  return 0;
}

int runQ4StyleCommand(const std::string& subcmd, const std::vector<std::string>& args) {
  if(subcmd == "style")
    return runStyle(args);
  if(subcmd == "popgames")
    return runPopGames(args);
  throw StringError("Unknown q4 style command " + subcmd);
}
