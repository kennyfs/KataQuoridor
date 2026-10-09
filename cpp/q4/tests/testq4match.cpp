#include "../../tests/tests.h"
#include "../../core/config_parser.h"
#include "../../core/logger.h"
#include "../command/q4gating.h"
#include "../command/q4matchengine.h"
#include "../q4history.h"
#include "../q4record.h"

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace std;
using namespace TestCommon;

namespace {

Q4Match::TableSpec makeTable(const string& name, vector<int> players, vector<int> grudgeTargets, int openings) {
  Q4Match::TableSpec t;
  t.name = name;
  for(int k = 0; k < 4; k++) {
    t.slots[k].player = players[k];
    t.slots[k].grudgeTargetSlot = grudgeTargets[k];
  }
  t.numOpenings = openings;
  return t;
}

// M1: over the rotations of one opening, every player sits in every seat equally often (counting copies).
void testM1SeatRotationBalance() {
  cout << "Running M1 seat rotation balance..." << endl;
  struct Case { vector<int> players; size_t expectedRotations; };
  const Case cases[] = {
    {{0, 1, 1, 1}, 4},  // candidate vs 3 x reference
    {{0, 1, 0, 1}, 2},  // ABAB
    {{0, 0, 1, 1}, 4},  // AABB
    {{0, 1, 2, 3}, 4},  // four different players
    {{0, 0, 0, 0}, 1},  // one player
    {{0, 1, 2, 1}, 4},
  };
  for(const Case& c : cases) {
    Q4Match::TableSpec t = makeTable("t", c.players, {-1, -1, -1, -1}, 3);
    vector<Q4Match::GameSpec> games = Q4Match::scheduleTable(t, 0, 99);
    testAssert(Q4Match::distinctRotations(t).size() == c.expectedRotations);
    testAssert(games.size() == 3 * c.expectedRotations);
    for(int o = 0; o < 3; o++) {
      map<int, int> seatCount[4];  // seat -> player -> count
      int n = 0;
      for(const Q4Match::GameSpec& g : games) {
        if(g.opening != o)
          continue;
        n++;
        for(int s = 0; s < 4; s++)
          seatCount[s][g.seatPlayer[s]]++;
      }
      testAssert((size_t)n == c.expectedRotations);
      // Every player appears the same number of times in each seat.
      for(int p = 0; p < 4; p++)
        for(int s = 1; s < 4; s++)
          testAssert(seatCount[s][p] == seatCount[0][p]);
    }
    // The slots in a game are a rotation of the table.
    for(const Q4Match::GameSpec& g : games)
      for(int s = 0; s < 4; s++) {
        testAssert(g.seatSlot[s] == (s + g.rotation) % 4);
        testAssert(g.seatPlayer[s] == c.players[g.seatSlot[s]]);
      }
  }
  cout << "M1 passed!" << endl;
}

// M2: a grudge slot's target is the seat the targeted slot sits in, in every rotation.
void testM2GrudgeTargets() {
  cout << "Running M2 grudge targets..." << endl;
  Q4Match::TableSpec t = makeTable("c", {0, 1, 1, 1}, {-1, 0, 0, 0}, 2);
  for(const Q4Match::GameSpec& g : Q4Match::scheduleTable(t, 0, 5)) {
    int candidateSeat = -1;
    for(int s = 0; s < 4; s++)
      if(g.seatSlot[s] == 0)
        candidateSeat = s;
    testAssert(g.seatPlayer[candidateSeat] == 0);
    for(int s = 0; s < 4; s++) {
      if(s == candidateSeat)
        testAssert(g.grudgeTargetSeat[s] == -1);
      else
        testAssert(g.grudgeTargetSeat[s] == candidateSeat);
    }
  }
  cout << "M2 passed!" << endl;
}

// M3: openings are paired: all rotations of an opening, and all tables, start from the same plies; different openings
// differ; the opening is legal and leaves the game running.
void testM3PairedOpenings() {
  cout << "Running M3 paired openings..." << endl;
  ConfigParser cfg;
  cfg.overrideKey("players", "g,r,b");
  cfg.overrideKey("player_g", "greedy");
  cfg.overrideKey("player_r", "random");
  cfg.overrideKey("player_b", "basher");
  cfg.overrideKey("tables", "one,two");
  cfg.overrideKey("table_one", "g,r,r,b");
  cfg.overrideKey("table_one_openings", "6");
  cfg.overrideKey("table_two", "b,g,b,g");
  cfg.overrideKey("table_two_openings", "6");
  cfg.overrideKey("numGameThreads", "3");
  cfg.overrideKey("maxPlies", "200");
  cfg.overrideKey("repetitionDrawCount", "3");
  cfg.overrideKey("openingPliesMin", "5");
  cfg.overrideKey("openingPliesMax", "9");
  cfg.overrideKey("seed", "77");
  Q4Match::MatchConfig mc = Q4Match::loadMatchConfig(cfg);

  vector<Q4Match::GameSpec> games;
  for(size_t t = 0; t < mc.tables.size(); t++) {
    vector<Q4Match::GameSpec> g = Q4Match::scheduleTable(mc.tables[t], (int)t, mc.seed);
    games.insert(games.end(), g.begin(), g.end());
  }
  testAssert(games.size() == 6 * 4 + 6 * 2);

  Logger logger;
  SearchParams params;
  map<int, set<vector<int>>> openingsByIndex;   // opening index -> distinct opening ply lists seen
  map<vector<int>, set<int>> indexesByOpening;
  int numGames = 0;
  Q4Match::runGames(mc, games, cfg, params, logger, Setup::SETUP_FOR_MATCH, [&](const Q4Match::GameResult& r) {
    numGames++;
    const Q4Record& rec = r.record;
    testAssert(rec.matchOpening == r.spec.opening && rec.matchRotation == r.spec.rotation);
    testAssert(rec.matchTable == mc.tables[r.spec.table].name);
    // The games use the self-play rules and the record says so.
    testAssert(rec.rules.repetitionDrawCount == 3 && rec.rules.maxPlies == 200);
    testAssert(Q4Record::fromJsonLine(rec.toJsonLine()).rules.repetitionDrawCount == 3);
    testAssert(rec.matchOpeningPlies >= 1 && rec.matchOpeningPlies <= 9);
    vector<int> opening;
    for(int i = 0; i < rec.matchOpeningPlies; i++)
      opening.push_back(rec.events[i].action);
    openingsByIndex[r.spec.opening].insert(opening);
    indexesByOpening[opening].insert(r.spec.opening);
    // The record replays to the recorded result and survives a JSON round trip with its match metadata.
    Q4Record back = Q4Record::fromJsonLine(rec.toJsonLine());
    testAssert(back.matchTable == rec.matchTable && back.matchOpening == rec.matchOpening);
    testAssert(back.matchRotation == rec.matchRotation && back.drawReason == rec.drawReason);
    Q4History h(rec.rules);
    back.replay(h);
    testAssert(h.getResultString() == rec.result);
    for(int s = 0; s < 4; s++)
      testAssert(rec.players[s].name == mc.players[r.spec.seatPlayer[s]].name);
  });
  testAssert(numGames == (int)games.size());
  for(const auto& kv : openingsByIndex)
    testAssert(kv.second.size() == 1);            // the same opening for every rotation and table
  testAssert(openingsByIndex.size() == 6);
  testAssert(indexesByOpening.size() == 6);       // different indices give different openings

  // The same schedule and seed give the same openings again; a different seed does not.
  Q4Rules rules;
  vector<int> a = Q4Match::makeOpening(rules, 77, 3, 7, 0.25);
  testAssert(a == Q4Match::makeOpening(rules, 77, 3, 7, 0.25));
  testAssert(a != Q4Match::makeOpening(rules, 78, 3, 7, 0.25));
  testAssert(a.size() == 7);
  cout << "M3 passed!" << endl;
}

// M4: config errors are hard errors.
void testM4ConfigErrors() {
  cout << "Running M4 config errors..." << endl;
  auto expectError = [](const vector<pair<string, string>>& kvs) {
    ConfigParser cfg;
    for(const auto& kv : kvs)
      cfg.overrideKey(kv.first, kv.second);
    bool caught = false;
    try {
      Q4Match::loadMatchConfig(cfg);
    }
    catch(const StringError&) {
      caught = true;
    }
    testAssert(caught);
  };
  const pair<string, string> rep("repetitionDrawCount", "3");
  const pair<string, string> players("players", "a,g"), pa("player_a", "greedy"), pg("player_g", "grudge"), tables("tables", "t");
  expectError({rep, players, pa, pg, tables, {"table_t", "a,a,a"}});                  // 3 slots
  expectError({rep, players, pa, pg, tables, {"table_t", "a,a,a,x"}});                // unknown player
  expectError({rep, players, pa, pg, tables, {"table_t", "a,g,g,g"}});                // grudge without a target
  expectError({rep, players, pa, pg, tables, {"table_t", "a>1,a,a,a"}});              // a target on a non-grudge bot
  expectError({rep, players, pa, {"player_g", "nobot"}, tables, {"table_t", "a,a,a,a"}});  // unknown bot
  expectError({rep, players, pa, pg, tables, {"table_t", "a,g>9,g>0,g>0"}});          // target out of range
  // The repetition rule of the games is a required key (no silent default).
  {
    ConfigParser cfg;
    for(const auto& kv : vector<pair<string, string>>{players, pa, pg, tables, {"table_t", "a,a,a,a"}, {"table_t_openings", "1"}})
      cfg.overrideKey(kv.first, kv.second);
    bool caught = false;
    try {
      Q4Match::loadMatchConfig(cfg);
    }
    catch(const StringError&) {
      caught = true;
    }
    testAssert(caught);
    cfg.overrideKey("repetitionDrawCount", "3");
    Q4Match::loadMatchConfig(cfg);
  }
  // A search player may set its own q4OpponentMode (Plan §12.3): search:<model>@<visits>:<mode>
  {
    Q4Match::PlayerSpec ps = Q4Match::parsePlayerSpec("e", "search:dir/model.bin.gz@50:expect");
    testAssert(ps.isSearch && ps.modelPath == "dir/model.bin.gz" && ps.visits == 50 && ps.opponentMode == "expect");
    ps = Q4Match::parsePlayerSpec("m", "search:dir/model.bin.gz@50");
    testAssert(ps.visits == 50 && ps.opponentMode.empty());
    bool caught = false;
    try { Q4Match::parsePlayerSpec("x", "search:dir/model.bin.gz@50:paranoid"); }
    catch(const StringError&) { caught = true; }
    testAssert(caught);
  }
  cout << "M4 passed!" << endl;
}

// M5: the gatekeeper decision on canned results.
void testM5GatekeeperDecision() {
  cout << "Running M5 gatekeeper decision..." << endl;
  using Q4Gate::decide;
  const vector<double> none;
  // ABAB: 40 of 80 points accepts (ties accept), 39.5 rejects.
  testAssert(decide(40.0, 80, 0.5, none, none).accept);
  testAssert(!decide(39.5, 80, 0.5, none, none).accept);
  testAssert(decide(39.5, 80, 0.5, none, none).reason.find("ABAB") != string::npos);
  testAssert(decide(60.0, 80, 0.5, none, none).accept && !decide(60.0, 80, 0.5, none, none).benchChecked);
  testAssert(!decide(0.0, 0, 0.5, none, none).accept);   // no games, no acceptance
  // Benchmark: 20 openings. A candidate clearly worse than the current net is rejected even with a won ABAB.
  vector<double> cand(20, 0.0), cur(20, 0.0);
  for(int i = 0; i < 20; i++) {
    cur[i] = 0.5 + 0.01 * (i % 3);
    cand[i] = 0.25 + 0.01 * (i % 4);
  }
  Q4Gate::Decision worse = decide(50.0, 80, 0.5, cand, cur);
  testAssert(worse.ababOk && worse.benchChecked && !worse.benchOk && !worse.accept);
  testAssert(worse.benchDiffHi < 0.0);
  // A mean difference of -0.1 with a large spread (12 openings at -0.5, 8 at +0.5) is within the CI: accepted.
  vector<double> noisyCand(20), noisyCur(20, 0.5);
  for(int i = 0; i < 20; i++)
    noisyCand[i] = i < 12 ? 0.0 : 1.0;
  Q4Gate::Decision noisy = decide(50.0, 80, 0.5, noisyCand, noisyCur);
  testAssert(noisy.benchChecked && noisy.benchDiff < 0.0 && noisy.benchOk && noisy.accept);
  // Equal or better on the benchmark: accepted. A lost ABAB rejects regardless.
  testAssert(decide(41.0, 80, 0.5, cur, cand).accept);
  testAssert(!decide(30.0, 80, 0.5, cur, cand).accept);
  // Fewer than 2 openings: the benchmark is not checked.
  testAssert(!decide(50.0, 80, 0.5, {0.1}, {0.9}).benchChecked);
  cout << "M5 passed!" << endl;
}

}  // namespace

void Tests::runQ4MatchTests() {
  cout << "========================================" << endl;
  cout << "Starting Q4 Match Test Suite (M1 - M5)" << endl;
  cout << "========================================" << endl;
  Board::initHash();
  testM1SeatRotationBalance();
  testM2GrudgeTargets();
  testM3PairedOpenings();
  testM4ConfigErrors();
  testM5GatekeeperDecision();
  cout << "All Q4 Match tests PASSED!" << endl;
}
