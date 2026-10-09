#include "../../core/config_parser.h"
#include "../../core/datetime.h"
#include "../../core/fileutils.h"
#include "../../core/global.h"
#include "../../core/logger.h"
#include "../../core/makedir.h"
#include "../../core/rand.h"
#include "../../dataio/loadmodel.h"
#include "../../external/nlohmann_json/json.hpp"
#include "../../game/board.h"
#include "../../main.h"
#include "../../program/setup.h"
#include "q4gating.h"
#include "q4matchengine.h"

#include <chrono>
#include <csignal>
#include <fstream>
#include <map>
#include <thread>

using json = nlohmann::json;

namespace {

std::atomic<bool> shouldStop(false);
void signalHandler(int signal) {
  if(signal == SIGINT || signal == SIGTERM)
    shouldStop.store(true);
}

// As Duel's gatekeeper: a model is a directory (or file) inside the test-models dir; accepting or rejecting moves it.
void moveModel(
  const std::string& modelName, const std::string& modelFile, const std::string& modelDir,
  const std::string& testModelsDir, const std::string& intoDir, Logger& logger
) {
  if(FileUtils::weaklyCanonical(modelDir) == FileUtils::weaklyCanonical(testModelsDir)) {
    std::string renameDest = intoDir + "/" + modelName;
    logger.write("Moving " + modelFile + " to " + renameDest);
    FileUtils::rename(modelFile, renameDest);
  }
  else if(Global::isPrefix(FileUtils::weaklyCanonical(modelDir), FileUtils::weaklyCanonical(testModelsDir))) {
    std::string renameDest = intoDir + "/" + modelName;
    logger.write("Moving " + modelDir + " to " + renameDest);
    FileUtils::rename(modelDir, renameDest);
  }
  else {
    throw StringError("Model " + modelDir + " does not appear to be a subdir of " + testModelsDir + ", can't move it");
  }
}

}  // namespace

// katago q4gatekeeper -config gate.cfg -test-models-dir D -accepted-models-dir A -rejected-models-dir R
//   -output-dir O [-required-candidate-score 0.5] [-no-autoreject-old-models] [-quit-if-no-nets-to-test]
// Port of Duel's gatekeeper (cpp/command/gatekeeper.cpp): the same directories and model moves, but the candidate is
// tested in Q4 tables (q4matchengine) with the decision of q4gating.h. Config keys (besides the search and NN keys):
//   gateVisits, gateOpenings (ABAB tables), gateBenchOpenings, gateBenchStrong ("search:<model>[@visits]", optional: no
//   benchmark without it), gateBenchWeak ("greedy,randomPawn"), numGameThreads, maxPlies, openingPlies{Min,Max}.
int MainCmds::q4gatekeeper(const std::vector<std::string>& args) {
  Board::initHash();  // NNEvaluator's transformer warmup builds a Duel Board
  std::vector<std::string> subArgs = args;
  if(!subArgs.empty() && subArgs[0] == "q4gatekeeper")
    subArgs = std::vector<std::string>(subArgs.begin() + 1, subArgs.end());

  std::string configFile, testModelsDir, acceptedModelsDir, rejectedModelsDir, outputDir, overrides;
  double requiredScore = 0.5;
  bool noAutoRejectOldModels = false;
  bool quitIfNoNetsToTest = false;
  for(size_t i = 0; i < subArgs.size(); i++) {
    const bool hasNext = i + 1 < subArgs.size();
    if(subArgs[i] == "-config" && hasNext) configFile = subArgs[++i];
    else if(subArgs[i] == "-override-config" && hasNext) overrides = subArgs[++i];
    else if(subArgs[i] == "-test-models-dir" && hasNext) testModelsDir = subArgs[++i];
    else if(subArgs[i] == "-accepted-models-dir" && hasNext) acceptedModelsDir = subArgs[++i];
    else if(subArgs[i] == "-rejected-models-dir" && hasNext) rejectedModelsDir = subArgs[++i];
    else if((subArgs[i] == "-output-dir" || subArgs[i] == "-sgf-output-dir") && hasNext) outputDir = subArgs[++i];
    else if(subArgs[i] == "-required-candidate-score" && hasNext) requiredScore = Global::stringToDouble(subArgs[++i]);
    else if(subArgs[i] == "-no-autoreject-old-models") noAutoRejectOldModels = true;
    else if(subArgs[i] == "-quit-if-no-nets-to-test") quitIfNoNetsToTest = true;
    else throw StringError("q4gatekeeper: unknown or incomplete argument " + subArgs[i]);
  }
  if(configFile.empty() || testModelsDir.empty() || acceptedModelsDir.empty() || rejectedModelsDir.empty() || outputDir.empty())
    throw StringError("q4gatekeeper needs -config, -test-models-dir, -accepted-models-dir, -rejected-models-dir and -output-dir");

  ConfigParser cfg(configFile);
  for(const std::string& kv : Global::split(overrides, ',')) {
    size_t eq = kv.find('=');
    if(Global::trim(kv).empty())
      continue;
    if(eq == std::string::npos)
      throw StringError("q4gatekeeper: -override-config expects key=value pairs, got " + kv);
    cfg.overrideKey(Global::trim(kv.substr(0, eq)), Global::trim(kv.substr(eq + 1)));
  }

  MakeDir::make(testModelsDir);
  MakeDir::make(acceptedModelsDir);
  MakeDir::make(rejectedModelsDir);
  MakeDir::make(outputDir);

  Rand seedRand;
  Logger logger(&cfg);
  logger.addFile(outputDir + "/log" + DateTime::getCompactDateTimeString() + "-" + Global::uint64ToHexString(seedRand.nextUInt64()) + ".log");
  logger.write("Q4 gatekeeper starting; required candidate score " + Global::doubleToString(requiredScore));

  const int gateVisits = cfg.getInt("gateVisits", 1, 1000000);
  const int gateOpenings = cfg.getInt("gateOpenings", 1, 1000000);
  const int gateBenchOpenings = cfg.contains("gateBenchOpenings") ? cfg.getInt("gateBenchOpenings", 0, 1000000) : 0;
  const std::string benchStrong = cfg.contains("gateBenchStrong") ? Global::trim(cfg.getString("gateBenchStrong")) : "";
  const std::string benchWeak = cfg.contains("gateBenchWeak") ? cfg.getString("gateBenchWeak") : "greedy,randomPawn";
  Q4Match::MatchConfig base;  // numGameThreads, maxPlies, opening settings, seed
  {
    ConfigParser matchCfg = cfg;
    matchCfg.overrideKey("players", "x");
    matchCfg.overrideKey("player_x", "random");
    matchCfg.overrideKey("tables", "t");
    matchCfg.overrideKey("table_t", "x,x,x,x");
    matchCfg.overrideKey("table_t_openings", "1");
    base = Q4Match::loadMatchConfig(matchCfg);
  }
  // As Duel's gatekeeper: SETUP_FOR_OTHER, the same as self-play.
  SearchParams searchParams = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_OTHER);

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting will not work");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  while(!shouldStop.load()) {
    std::string testName, testFile, testDir;
    time_t testTime;
    bool found = LoadModel::findLatestModel(testModelsDir, logger, testName, testFile, testDir, testTime);
    if(!found || testFile == "/dev/null") {
      if(quitIfNoNetsToTest)
        break;
      std::this_thread::sleep_for(std::chrono::seconds(4));
      continue;
    }
    logger.write("Found new candidate neural net " + testName);

    std::string accName, accFile, accDir;
    time_t accTime;
    if(!LoadModel::findLatestModel(acceptedModelsDir, logger, accName, accFile, accDir, accTime)) {
      logger.write("Error: no accepted model found in " + acceptedModelsDir);
      break;
    }
    if(accTime > testTime && !noAutoRejectOldModels) {
      logger.write("Rejecting " + testName + " automatically since older than best accepted model");
      moveModel(testName, testFile, testDir, testModelsDir, rejectedModelsDir, logger);
      continue;
    }

    Q4Match::MatchConfig mc = base;
    auto net = [&](const std::string& name, const std::string& file) {
      return Q4Match::parsePlayerSpec(name, "search:" + file + "@" + Global::intToString(gateVisits));
    };
    mc.players = {net("cand", testFile), net("base", accFile)};
    auto addPlayer = [&](const std::string& name, const std::string& spec) {
      mc.players.push_back(Q4Match::parsePlayerSpec(name, spec));
      return (int)mc.players.size() - 1;
    };
    auto table = [&](const std::string& name, std::vector<int> p, int openings) {
      Q4Match::TableSpec t;
      t.name = name;
      for(int k = 0; k < 4; k++)
        t.slots[k].player = p[k];
      t.numOpenings = openings;
      mc.tables.push_back(t);
    };
    table("abab", {0, 1, 0, 1}, gateOpenings);
    const bool benchmark = !benchStrong.empty() && gateBenchOpenings >= 2;
    if(benchmark) {
      int strong = addPlayer("strong", benchStrong);
      std::vector<std::string> weak = Global::split(benchWeak, ',');
      if(weak.size() != 2)
        throw StringError("gateBenchWeak needs exactly 2 bots, got " + benchWeak);
      int w1 = addPlayer("weak1", Global::trim(weak[0]));
      int w2 = addPlayer("weak2", Global::trim(weak[1]));
      table("benchCand", {0, strong, w1, w2}, gateBenchOpenings);
      table("benchBase", {1, strong, w1, w2}, gateBenchOpenings);
    }

    std::vector<Q4Match::GameSpec> games;
    for(size_t t = 0; t < mc.tables.size(); t++) {
      std::vector<Q4Match::GameSpec> g = Q4Match::scheduleTable(mc.tables[t], (int)t, mc.seed);
      games.insert(games.end(), g.begin(), g.end());
    }
    logger.write("Testing " + testName + " against " + accName + ": " + Global::uint64ToString(games.size()) + " games");

    const std::string outThis = outputDir + "/" + testName;
    MakeDir::make(outThis);
    std::ofstream gamesOut(outThis + "/games.jsonl");
    double ababPoints = 0.0;
    int ababGames = 0;
    std::map<int, double> benchCandPoints, benchBasePoints;   // opening -> summed points of the tested net
    std::map<int, int> benchCandGames, benchBaseGames;
    Q4Match::runGames(mc, games, cfg, searchParams, logger, Setup::SETUP_FOR_OTHER, [&](const Q4Match::GameResult& r) {
      gamesOut << Q4Match::resultToJsonLine(mc, r) << "\n";
      const std::string& tname = mc.tables[r.spec.table].name;
      if(tname == "abab") {
        ababGames++;
        if(r.winnerSeat < 0)
          ababPoints += 0.5;
        else if(mc.players[r.spec.seatPlayer[r.winnerSeat]].name == "cand")
          ababPoints += 1.0;
      }
      else {
        const bool isCand = tname == "benchCand";
        double pts = r.winnerSeat < 0 ? 0.25 : (mc.players[r.spec.seatPlayer[r.winnerSeat]].name == (isCand ? "cand" : "base") ? 1.0 : 0.0);
        (isCand ? benchCandPoints : benchBasePoints)[r.spec.opening] += pts;
        (isCand ? benchCandGames : benchBaseGames)[r.spec.opening] += 1;
      }
    });
    if(shouldStop.load())
      break;

    std::vector<double> candPerOpening, basePerOpening;
    for(const auto& kv : benchCandPoints) {
      auto it = benchBasePoints.find(kv.first);
      if(it == benchBasePoints.end())
        continue;
      candPerOpening.push_back(kv.second / benchCandGames[kv.first]);
      basePerOpening.push_back(it->second / benchBaseGames[kv.first]);
    }
    Q4Gate::Decision d = Q4Gate::decide(ababPoints, ababGames, requiredScore, candPerOpening, basePerOpening);
    json dj;
    dj["candidate"] = testName;
    dj["current"] = accName;
    dj["accept"] = d.accept;
    dj["reason"] = d.reason;
    dj["ababScore"] = d.ababScore;
    dj["ababGames"] = ababGames;
    dj["benchChecked"] = d.benchChecked;
    dj["benchDiff"] = d.benchDiff;
    dj["benchDiffHi"] = d.benchDiffHi;
    std::ofstream(outThis + "/decision.json") << dj.dump(2) << "\n";
    logger.write(
      Global::strprintf(
        "Candidate %s: ABAB score %.3f in %d games, benchmark difference %.3f (upper %.3f, %s): %s %s",
        testName.c_str(), d.ababScore, ababGames, d.benchDiff, d.benchDiffHi,
        d.benchChecked ? "checked" : "not run", d.accept ? "accepting" : "rejecting", d.reason.c_str()
      )
    );
    if(d.accept)
      std::this_thread::sleep_for(std::chrono::seconds(2));
    moveModel(testName, testFile, testDir, testModelsDir, d.accept ? acceptedModelsDir : rejectedModelsDir, logger);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  logger.write("Q4 gatekeeper done");
  return 0;
}
