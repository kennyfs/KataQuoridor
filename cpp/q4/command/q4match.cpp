#include "../../core/config_parser.h"
#include "../../core/global.h"
#include "../../core/rand.h"
#include "../../external/nlohmann_json/json.hpp"
#include "../../game/board.h"
#include "../../main.h"
#include "../../program/setup.h"
#include "q4matchengine.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

struct PlayerTally {
  int games = 0;
  int wins = 0;
  int slotsInTable = 0;   // copies of this player at the table
};

struct TableTally {
  int games = 0;
  int seatWins[4] = {0, 0, 0, 0};
  int drawsMaxPlies = 0;
  int drawsRepetition = 0;
  int drawsOther = 0;
  int64_t totalPlies = 0;
  std::map<std::string, PlayerTally> players;
};

}  // namespace

// katago q4match -config match.cfg [-override-config k=v,k=v] [-output games.sgfs] [-summary summary.json] [-list-games]
// Config mode: players / player_<name> / tables / table_<t> / table_<t>_openings (see q4matchengine.h and
// cpp/configs/q4/match/). Quick mode (the Round 3 interface): -players spec,spec,spec,spec -games N [-rotate] [-maxplies N].
int MainCmds::q4match(const std::vector<std::string>& args) {
  Board::initHash();  // NNEvaluator's transformer warmup builds a Duel Board
  std::vector<std::string> subArgs = args;
  if(!subArgs.empty() && subArgs[0] == "q4match")
    subArgs = std::vector<std::string>(subArgs.begin() + 1, subArgs.end());

  int numGames = 1000;
  std::vector<std::string> botSpecs;
  std::string configFile = "";
  std::string overrides = "";
  std::string outputFile = "";
  std::string summaryFile = "";
  uint64_t masterSeed = 12345;
  bool seedGiven = false;
  int maxPlies = 400;
  bool maxPliesGiven = false;
  bool rotateSeats = false;
  bool listGames = false;

  for(size_t i = 0; i < subArgs.size(); i++) {
    const bool hasNext = i + 1 < subArgs.size();
    if(subArgs[i] == "-games" && hasNext)
      numGames = Global::stringToInt(subArgs[++i]);
    else if((subArgs[i] == "-bots" || subArgs[i] == "-players") && hasNext)
      botSpecs = Global::split(subArgs[++i], ',');
    else if(subArgs[i] == "-config" && hasNext)
      configFile = subArgs[++i];
    else if(subArgs[i] == "-override-config" && hasNext)
      overrides = subArgs[++i];
    else if(subArgs[i] == "-output" && hasNext)
      outputFile = subArgs[++i];
    else if(subArgs[i] == "-summary" && hasNext)
      summaryFile = subArgs[++i];
    else if(subArgs[i] == "-seed" && hasNext) {
      masterSeed = (uint64_t)Global::stringToInt64(subArgs[++i]);
      seedGiven = true;
    }
    else if((subArgs[i] == "-maxplies" || subArgs[i] == "-max-plies") && hasNext) {
      maxPlies = Global::stringToInt(subArgs[++i]);
      maxPliesGiven = true;
    }
    else if(subArgs[i] == "-rotate")
      rotateSeats = true;
    else if(subArgs[i] == "-list-games")
      listGames = true;
    else
      throw StringError("q4match: unknown or incomplete argument " + subArgs[i]);
  }

  ConfigParser cfg;
  if(!configFile.empty())
    cfg.initialize(configFile);
  for(const std::string& kv : Global::split(overrides, ',')) {
    if(Global::trim(kv).empty())
      continue;
    size_t eq = kv.find('=');
    if(eq == std::string::npos)
      throw StringError("q4match: -override-config expects key=value pairs, got " + kv);
    cfg.overrideKey(Global::trim(kv.substr(0, eq)), Global::trim(kv.substr(eq + 1)));
  }

  const bool quickMode = !botSpecs.empty();
  Q4Match::MatchConfig mc;
  std::vector<Q4Match::GameSpec> games;
  if(quickMode) {
    while(botSpecs.size() < 4)
      botSpecs.push_back("random");
    mc.seed = masterSeed;
    mc.maxPlies = maxPlies;
    mc.repetitionDrawCount = cfg.getInt("repetitionDrawCount", 0, 100);
    mc.openingPliesMin = mc.openingPliesMax = 0;
    Q4Match::TableSpec table;
    table.name = "cli";
    for(int p = 0; p < 4; p++) {
      mc.players.push_back(Q4Match::parsePlayerSpec("P" + Global::intToString(p + 1) + "_" + botSpecs[p], botSpecs[p]));
      table.slots[p].player = p;
    }
    int rotations = rotateSeats ? (int)Q4Match::distinctRotations(table).size() : 1;
    table.numOpenings = (numGames + rotations - 1) / rotations;
    mc.tables.push_back(table);
    mc.numGameThreads = cfg.contains("numGameThreads") ? cfg.getInt("numGameThreads", 1, 4096) : 1;
    std::vector<Q4Match::GameSpec> all = Q4Match::scheduleTable(table, 0, mc.seed);
    for(const Q4Match::GameSpec& g : all) {
      if(!rotateSeats && g.rotation != 0)
        continue;
      if((int)games.size() < numGames)
        games.push_back(g);
    }
  }
  else {
    if(configFile.empty())
      throw StringError("q4match: give -config <file> (tables) or -players spec,spec,spec,spec (quick mode)");
    mc = Q4Match::loadMatchConfig(cfg);
    if(seedGiven)
      mc.seed = masterSeed;
    if(maxPliesGiven)
      mc.maxPlies = maxPlies;
    for(size_t t = 0; t < mc.tables.size(); t++) {
      std::vector<Q4Match::GameSpec> g = Q4Match::scheduleTable(mc.tables[t], (int)t, mc.seed);
      games.insert(games.end(), g.begin(), g.end());
    }
  }

  std::cout << "=== KataQuoridor Q4 Match ===" << std::endl;
  std::cout << "Games: " << games.size() << std::endl;
  for(size_t t = 0; t < mc.tables.size(); t++) {
    std::cout << "Table " << mc.tables[t].name << ":";
    for(int k = 0; k < 4; k++) {
      const Q4Match::Slot& s = mc.tables[t].slots[k];
      std::cout << " " << mc.players[s.player].name << (s.grudgeTargetSlot >= 0 ? ">" + Global::intToString(s.grudgeTargetSlot) : "");
    }
    std::cout << " (" << mc.tables[t].numOpenings << " openings x " << Q4Match::distinctRotations(mc.tables[t]).size()
              << " rotations)" << std::endl;
  }
  if(quickMode)
    std::cout << "Seat rotation: " << (rotateSeats ? "enabled" : "disabled") << std::endl;
  std::cout << "Max plies: " << mc.maxPlies << std::endl;

  if(listGames) {
    for(const Q4Match::GameSpec& g : games) {
      std::cout << "game table=" << mc.tables[g.table].name << " opening=" << g.opening << " rotation=" << g.rotation << " seats=";
      for(int s = 0; s < 4; s++)
        std::cout << (s ? "," : "") << mc.players[g.seatPlayer[s]].name;
      std::cout << std::endl;
    }
    return 0;
  }

  Logger logger;
  // As Duel's match: SETUP_FOR_MATCH. The config carries every search value; a table of bots needs none.
  bool anySearchPlayer = false;
  for(const Q4Match::PlayerSpec& p : mc.players)
    anySearchPlayer = anySearchPlayer || p.isSearch;
  SearchParams searchParams;
  if(anySearchPlayer)
    searchParams = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_MATCH);

  std::ofstream outStream;
  if(!outputFile.empty()) {
    outStream.open(outputFile);
    if(!outStream.is_open())
      throw StringError("q4match: could not open output file " + outputFile);
  }

  std::map<std::string, TableTally> tallies;
  for(const Q4Match::TableSpec& t : mc.tables) {
    TableTally& tt = tallies[t.name];
    for(int k = 0; k < 4; k++)
      tt.players[mc.players[t.slots[k].player].name].slotsInTable++;
  }
  size_t played = 0;
  const size_t total = games.size();
  Q4Match::runGames(mc, games, cfg, searchParams, logger, Setup::SETUP_FOR_MATCH, [&](const Q4Match::GameResult& r) {
    TableTally& tt = tallies[mc.tables[r.spec.table].name];
    tt.games++;
    tt.totalPlies += r.plies;
    for(int s = 0; s < 4; s++)
      tt.players[mc.players[r.spec.seatPlayer[s]].name].games++;
    if(r.winnerSeat >= 0) {
      tt.seatWins[r.winnerSeat]++;
      tt.players[mc.players[r.spec.seatPlayer[r.winnerSeat]].name].wins++;
    }
    else if(r.drawReason == "maxPlies")
      tt.drawsMaxPlies++;
    else if(r.drawReason == "repetition")
      tt.drawsRepetition++;
    else
      tt.drawsOther++;
    if(outStream.is_open())
      outStream << Q4Match::resultToSgfLine(mc, r) << "\n";
    played++;
    if(played % 25 == 0 || played == total)
      std::cout << "Played " << played << "/" << total << " games..." << std::endl;
  });

  json summary = json::object();
  std::cout << "\n--- Match Results ---" << std::endl;
  for(const auto& kv : tallies) {
    const TableTally& tt = kv.second;
    if(tt.games == 0)
      continue;
    json tj;
    tj["games"] = tt.games;
    tj["meanPlies"] = (double)tt.totalPlies / tt.games;
    tj["seatWins"] = {tt.seatWins[0], tt.seatWins[1], tt.seatWins[2], tt.seatWins[3]};
    tj["draws"] = {{"maxPlies", tt.drawsMaxPlies}, {"repetition", tt.drawsRepetition}, {"other", tt.drawsOther}};
    std::cout << "Table " << kv.first << " (" << tt.games << " games)" << std::endl;
    for(const auto& pk : tt.players) {
      // games counts seat appearances, so a player with k copies at the table has k x games/4... per table game
      double winRate = 100.0 * pk.second.wins / tt.games;
      std::cout << "  " << pk.first << " (x" << pk.second.slotsInTable << "): " << pk.second.wins << " wins ("
                << std::fixed << std::setprecision(1) << winRate << "%)" << std::endl;
      tj["players"][pk.first] = {{"wins", pk.second.wins}, {"slots", pk.second.slotsInTable}};
    }
    std::cout << "  seat wins: " << tt.seatWins[0] << " " << tt.seatWins[1] << " " << tt.seatWins[2] << " " << tt.seatWins[3]
              << "; draws: maxPlies " << tt.drawsMaxPlies << ", repetition " << tt.drawsRepetition << ", other "
              << tt.drawsOther << "; mean plies " << std::setprecision(1) << (double)tt.totalPlies / tt.games << std::endl;
    summary[kv.first] = tj;
  }
  if(!summaryFile.empty()) {
    std::ofstream s(summaryFile);
    if(!s.is_open())
      throw StringError("q4match: could not open summary file " + summaryFile);
    s << summary.dump(2) << "\n";
  }
  return 0;
}
