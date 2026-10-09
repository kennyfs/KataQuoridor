#include "q4matchengine.h"

#include "../../core/global.h"
#include "../../core/rand.h"
#include "../../neuralnet/nneval.h"
#include "../nn/q4nn.h"
#include "../q4bots.h"
#include "../q4history.h"
#include "../search/q4search.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <set>
#include <thread>

namespace Q4Match {

PlayerSpec parsePlayerSpec(const std::string& name, const std::string& spec) {
  PlayerSpec ps;
  ps.name = name;
  ps.rawSpec = Global::trim(spec);
  if(ps.rawSpec.rfind("search:", 0) == 0) {
    ps.isSearch = true;
    std::string rest = ps.rawSpec.substr(7);
    size_t atPos = rest.rfind('@');
    if(atPos != std::string::npos) {
      ps.modelPath = rest.substr(0, atPos);
      ps.visits = Global::stringToInt(rest.substr(atPos + 1));
    }
    else {
      ps.modelPath = rest;
      ps.visits = 200;
    }
    if(ps.modelPath.empty() || ps.visits < 1)
      throw StringError("Bad Q4 match player '" + name + "': expected search:<model path>[@visits], got " + spec);
  }
  else {
    ps.botType = ps.rawSpec;
    Q4Bots::makeBot(ps.botType, 0, 0);  // throws on an unknown bot type
  }
  return ps;
}

namespace {

// Seat s of rotation r is played by slot (s + r) % 4; slot t is therefore in seat (t - r) mod 4.
void fillSeats(const TableSpec& table, int rotation, GameSpec& g) {
  for(int s = 0; s < 4; s++) {
    int slot = (s + rotation) % 4;
    g.seatSlot[s] = slot;
    g.seatPlayer[s] = table.slots[slot].player;
    int t = table.slots[slot].grudgeTargetSlot;
    g.grudgeTargetSeat[s] = t < 0 ? -1 : ((t - rotation) % 4 + 4) % 4;
  }
}

bool sameSeating(const GameSpec& a, const GameSpec& b) {
  for(int s = 0; s < 4; s++)
    if(a.seatPlayer[s] != b.seatPlayer[s] || a.grudgeTargetSeat[s] != b.grudgeTargetSeat[s])
      return false;
  return true;
}

}  // namespace

std::vector<int> distinctRotations(const TableSpec& table) {
  std::vector<int> rotations;
  std::vector<GameSpec> kept;
  for(int r = 0; r < 4; r++) {
    GameSpec g;
    fillSeats(table, r, g);
    bool dup = false;
    for(const GameSpec& k : kept)
      dup = dup || sameSeating(k, g);
    if(!dup) {
      kept.push_back(g);
      rotations.push_back(r);
    }
  }
  return rotations;
}

std::vector<GameSpec> scheduleTable(const TableSpec& table, int tableIdx, uint64_t masterSeed) {
  std::vector<GameSpec> games;
  std::vector<int> rotations = distinctRotations(table);
  for(int o = 0; o < table.numOpenings; o++) {
    for(int r : rotations) {
      GameSpec g;
      g.table = tableIdx;
      g.opening = o;
      g.rotation = r;
      fillSeats(table, r, g);
      Rand seedRand(
        "q4match-game:" + Global::uint64ToString(masterSeed) + ":" + table.name + ":" + Global::intToString(o) + ":" +
        Global::intToString(r)
      );
      g.seed = seedRand.nextUInt64();
      games.push_back(g);
    }
  }
  return games;
}

std::vector<int> makeOpening(const Q4Rules& rules, uint64_t masterSeed, int opening, int numPlies, double wallProb) {
  Rand rand("q4match-opening:" + Global::uint64ToString(masterSeed) + ":" + Global::intToString(opening));
  Q4History hist(rules);
  std::vector<int> actions;
  std::vector<int> legal;
  std::vector<int> pawn;
  std::vector<int> walls;
  for(int i = 0; i < numPlies; i++) {
    int seat = hist.currentBoard.toMove;
    hist.currentBoard.getLegalActions(seat, legal);
    pawn.clear();
    walls.clear();
    for(int a : legal)
      (Q4Board::isPawnAction(a) ? pawn : walls).push_back(a);
    bool wall = !walls.empty() && (pawn.empty() || rand.nextDouble() < wallProb);
    const std::vector<int>& from = wall ? walls : pawn;
    if(from.empty())
      break;
    int a = from[rand.nextUInt((uint32_t)from.size())];
    hist.play(a);
    if(hist.isFinished) {
      hist.undo();
      break;
    }
    actions.push_back(a);
  }
  return actions;
}

namespace {

std::string requireString(ConfigParser& cfg, const std::string& key) {
  if(!cfg.contains(key))
    throw StringError("Q4 match config: missing key " + key);
  return Global::trim(cfg.getString(key));
}

Slot parseSlot(const std::string& text, const std::map<std::string, int>& playerIdx, const std::string& table) {
  Slot slot;
  std::string name = text;
  size_t gt = text.find('>');
  if(gt != std::string::npos) {
    name = Global::trim(text.substr(0, gt));
    int k;
    if(!Global::tryStringToInt(Global::trim(text.substr(gt + 1)), k) || k < 0 || k > 3)
      throw StringError("Q4 match config: table " + table + ": bad grudge target in '" + text + "', expected name>0..3");
    slot.grudgeTargetSlot = k;
  }
  auto it = playerIdx.find(Global::trim(name));
  if(it == playerIdx.end())
    throw StringError("Q4 match config: table " + table + ": unknown player '" + name + "'");
  slot.player = it->second;
  return slot;
}

}  // namespace

MatchConfig loadMatchConfig(ConfigParser& cfg) {
  MatchConfig mc;
  std::map<std::string, int> playerIdx;
  for(const std::string& name : Global::split(requireString(cfg, "players"), ',')) {
    std::string n = Global::trim(name);
    if(n.empty())
      continue;
    if(playerIdx.count(n))
      throw StringError("Q4 match config: player " + n + " listed twice");
    playerIdx[n] = (int)mc.players.size();
    mc.players.push_back(parsePlayerSpec(n, requireString(cfg, "player_" + n)));
  }
  if(mc.players.empty())
    throw StringError("Q4 match config: players is empty");

  for(const std::string& name : Global::split(requireString(cfg, "tables"), ',')) {
    std::string n = Global::trim(name);
    if(n.empty())
      continue;
    TableSpec t;
    t.name = n;
    std::vector<std::string> slots = Global::split(requireString(cfg, "table_" + n), ',');
    if(slots.size() != 4)
      throw StringError("Q4 match config: table_" + n + " needs 4 comma-separated slots, got " + Global::intToString((int)slots.size()));
    for(int k = 0; k < 4; k++) {
      t.slots[k] = parseSlot(slots[k], playerIdx, n);
      const PlayerSpec& p = mc.players[t.slots[k].player];
      bool isGrudge = !p.isSearch && Global::toLower(p.botType) == "grudge";
      if(isGrudge && t.slots[k].grudgeTargetSlot < 0)
        throw StringError("Q4 match config: table " + n + ": grudge player " + p.name + " needs a target slot (" + p.name + ">k)");
      if(!isGrudge && t.slots[k].grudgeTargetSlot >= 0)
        throw StringError("Q4 match config: table " + n + ": only grudge players take a target (" + slots[k] + ")");
    }
    t.numOpenings = cfg.getInt("table_" + n + "_openings", 1, 1000000);
    mc.tables.push_back(t);
  }
  if(mc.tables.empty())
    throw StringError("Q4 match config: tables is empty");

  mc.numGameThreads = cfg.contains("numGameThreads") ? cfg.getInt("numGameThreads", 1, 4096) : 1;
  mc.maxPlies = cfg.contains("maxPlies") ? cfg.getInt("maxPlies", 1, 100000) : 400;
  mc.repetitionDrawCount = cfg.getInt("repetitionDrawCount", 0, 100);
  mc.openingPliesMin = cfg.contains("openingPliesMin") ? cfg.getInt("openingPliesMin", 0, 100) : 4;
  mc.openingPliesMax = cfg.contains("openingPliesMax") ? cfg.getInt("openingPliesMax", 0, 100) : 8;
  if(mc.openingPliesMax < mc.openingPliesMin)
    throw StringError("Q4 match config: openingPliesMax < openingPliesMin");
  mc.openingWallProb = cfg.contains("openingWallProb") ? cfg.getDouble("openingWallProb", 0.0, 1.0) : 0.25;
  mc.seed = cfg.contains("seed") ? (uint64_t)cfg.getInt64("seed", 0, ((int64_t)1) << 62) : 12345;
  return mc;
}

namespace {

int openingLength(const MatchConfig& mc, int opening) {
  Rand r("q4match-openinglen:" + Global::uint64ToString(mc.seed) + ":" + Global::intToString(opening));
  return mc.openingPliesMin + (int)r.nextUInt((uint32_t)(mc.openingPliesMax - mc.openingPliesMin + 1));
}

struct SeatPlayer {
  std::unique_ptr<Q4Bot> bot;
  std::unique_ptr<Q4S::Search> search;
};

GameResult playGame(
  const MatchConfig& mc,
  const GameSpec& g,
  const SearchParams& baseParams,
  const std::map<std::string, std::unique_ptr<NNEvaluator>>& evaluators,
  Logger& logger
) {
  GameResult res;
  res.spec = g;
  Q4Rules rules;
  rules.maxPlies = mc.maxPlies;
  rules.repetitionDrawCount = mc.repetitionDrawCount;  // the self-play rules (Duel's gatekeeper: "same rules as selfplay")

  Q4History history(rules);
  std::vector<int> opening = makeOpening(rules, mc.seed, g.opening, openingLength(mc, g.opening), mc.openingWallProb);
  for(int a : opening)
    history.play(a);

  Rand rand(g.seed);
  SeatPlayer players[4];
  for(int s = 0; s < 4; s++) {
    const PlayerSpec& ps = mc.players[g.seatPlayer[s]];
    uint64_t seatSeed = rand.nextUInt64();
    if(ps.isSearch) {
      SearchParams params = baseParams;
      params.maxVisits = ps.visits;
      params.numThreads = 1;
      Q4S::Search::checkParams(params);
      players[s].search = std::make_unique<Q4S::Search>(
        params, evaluators.at(ps.modelPath).get(), &logger, "q4match_" + Global::uint64ToHexString(seatSeed)
      );
      players[s].search->setPosition(history);
    }
    else {
      players[s].bot = Q4Bots::makeBot(ps.botType, seatSeed, g.grudgeTargetSeat[s]);
    }
  }

  bool aborted = false;
  while(!history.isFinished) {
    int toMove = history.currentBoard.toMove;
    int act;
    if(players[toMove].search != nullptr)
      act = players[toMove].search->runWholeSearchAndGetMove();
    else
      act = players[toMove].bot->getMove(history.currentBoard);
    if(act == Q4Board::NULL_ACTION) {
      aborted = true;
      break;
    }
    history.play(act);
    float style[Q4StyleTracker::NUM_FEATURES];
    history.getStyleFeatures(style);
    for(int s = 0; s < 4; s++) {
      if(players[s].search != nullptr) {
        players[s].search->makeMove(act);
        players[s].search->setStyleFeatures(style);   // the tree is reused: the features of the new real position
      }
    }
  }

  res.plies = history.plies;
  res.winnerSeat = (history.winnerSeat >= 0 && history.winnerSeat < 4) ? history.winnerSeat : -1;
  if(aborted)
    res.drawReason = "unfinished";
  else if(res.winnerSeat < 0)
    res.drawReason = history.plies >= rules.maxPlies ? "maxPlies" : "repetition";

  Q4Record& rec = res.record;
  rec.rules = rules;
  rec.events = history.events;
  rec.result = aborted ? "none" : history.getResultString();
  rec.players.resize(4);
  for(int s = 0; s < 4; s++) {
    const PlayerSpec& ps = mc.players[g.seatPlayer[s]];
    rec.players[s].name = ps.name;
    rec.players[s].type = ps.isSearch ? "search" : ps.botType;
    rec.players[s].net = ps.modelPath;
    rec.players[s].visits = ps.isSearch ? ps.visits : 0;
  }
  rec.matchTable = mc.tables[g.table].name;
  rec.matchOpening = g.opening;
  rec.matchRotation = g.rotation;
  rec.matchOpeningPlies = (int)opening.size();
  rec.drawReason = res.drawReason;
  return res;
}

}  // namespace

void runGames(
  const MatchConfig& mc,
  const std::vector<GameSpec>& games,
  ConfigParser& cfg,
  const SearchParams& searchParams,
  Logger& logger,
  Setup::setup_for_t setupFor,
  const std::function<void(const GameResult&)>& onGameDone
) {
  std::map<std::string, std::unique_ptr<NNEvaluator>> evaluators;
  std::set<std::string> needed;
  for(const GameSpec& g : games)
    for(int s = 0; s < 4; s++)
      if(mc.players[g.seatPlayer[s]].isSearch)
        needed.insert(mc.players[g.seatPlayer[s]].modelPath);
  if(!needed.empty()) {
    Rand evalSeed(mc.seed + 777);
    for(const std::string& path : needed) {
      evaluators[path] = std::unique_ptr<NNEvaluator>(Setup::initializeNNEvaluator(
        path, path, "", cfg, logger, evalSeed, mc.numGameThreads, Q4NNConst::POS_LEN, Q4NNConst::POS_LEN,
        Setup::MaxBatchSizeRequest::requireFromConfig(), true, false, setupFor
      ));
    }
  }

  std::atomic<size_t> next(0);
  std::mutex mutex;
  std::exception_ptr error = nullptr;
  auto worker = [&]() {
    try {
      while(true) {
        size_t i = next.fetch_add(1);
        if(i >= games.size())
          break;
        {
          std::lock_guard<std::mutex> lock(mutex);
          if(error != nullptr)
            break;
        }
        GameResult r = playGame(mc, games[i], searchParams, evaluators, logger);
        std::lock_guard<std::mutex> lock(mutex);
        onGameDone(r);
      }
    }
    catch(...) {
      std::lock_guard<std::mutex> lock(mutex);
      if(error == nullptr)
        error = std::current_exception();
    }
  };
  int numThreads = std::max(1, std::min(mc.numGameThreads, (int)games.size()));
  std::vector<std::thread> threads;
  for(int t = 0; t < numThreads; t++)
    threads.emplace_back(worker);
  for(std::thread& t : threads)
    t.join();
  evaluators.clear();
  if(error != nullptr)
    std::rethrow_exception(error);
}

std::string resultToJsonLine(const MatchConfig& mc, const GameResult& r) {
  (void)mc;
  return r.record.toJsonLine();
}

}  // namespace Q4Match
