#ifndef Q4_MATCHENGINE_H_
#define Q4_MATCHENGINE_H_

#include "../../core/config_parser.h"
#include "../../core/logger.h"
#include "../../program/setup.h"
#include "../q4record.h"
#include "../q4rules.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

class NNEvaluator;

// Evaluation matches of Q4 (docs/q4/rounds/R6.md): tables of 4 player slots, played with every distinct seat
// rotation and a shared random opening, by config-driven players (nets at given visits, or q4bots).
namespace Q4Match {

// A config player: "search:<model path>[@visits]" (a net with its own search) or a q4bots name ("random",
// "randomPawn", "greedy", "basher", "grudge").
struct PlayerSpec {
  std::string name;
  std::string rawSpec;
  bool isSearch = false;
  std::string botType;
  std::string modelPath;
  int visits = 200;
};

PlayerSpec parsePlayerSpec(const std::string& name, const std::string& spec);

// One of the 4 slots of a table: an index into the player list, and for a `grudge` player the slot it is out to get
// (spelled "name>k" in the config, k = 0..3), -1 otherwise.
struct Slot {
  int player = 0;
  int grudgeTargetSlot = -1;
};

struct TableSpec {
  std::string name;
  Slot slots[4];
  int numOpenings = 1;
};

// One game: slot `seatSlot[s]` of the table sits in seat s, `grudgeTargetSeat[s]` is the seat a grudge bot in seat s
// targets (-1 for other players).
struct GameSpec {
  int table = 0;
  int opening = 0;
  int rotation = 0;
  int seatSlot[4] = {0, 1, 2, 3};
  int seatPlayer[4] = {0, 0, 0, 0};
  int grudgeTargetSeat[4] = {-1, -1, -1, -1};
  uint64_t seed = 0;
};

// The distinct seat rotations of a table, as rotation numbers 0..3 (rotation r puts slot (s + r) % 4 in seat s). A
// table whose slots repeat with period 2 (ABAB) has two, one with all slots equal has one. Over these rotations
// every slot sits in every seat the same number of times.
std::vector<int> distinctRotations(const TableSpec& table);

// All games of a table: for each opening 0..numOpenings-1, one game per distinct rotation.
std::vector<GameSpec> scheduleTable(const TableSpec& table, int tableIdx, uint64_t masterSeed);

// The opening of index `opening`: `numPlies` random legal plies from the start position (a pawn move, or with
// probability `wallProb` a wall), the same for all tables and rotations. Never ends the game.
std::vector<int> makeOpening(const Q4Rules& rules, uint64_t masterSeed, int opening, int numPlies, double wallProb);

struct MatchConfig {
  std::vector<PlayerSpec> players;
  std::vector<TableSpec> tables;
  int numGameThreads = 1;
  int maxPlies = 400;
  int openingPliesMin = 4;
  int openingPliesMax = 8;
  double openingWallProb = 0.25;
  uint64_t seed = 12345;
};

// Reads players = a,b,c / player_<name> = spec / tables = t1,t2 / table_<t> = slot,slot,slot,slot (or table_<t>_*
// keys), numGameThreads, maxPlies, openingPlies{Min,Max}, openingWallProb. Hard errors on anything unknown.
MatchConfig loadMatchConfig(ConfigParser& cfg);

// Sets the search parameters the config leaves out (the q4search_test.cfg values, Q4's removed features off).
void fillDefaultSearchKeys(ConfigParser& cfg);

struct GameResult {
  GameSpec spec;
  Q4Record record;
  int plies = 0;
  int winnerSeat = -1;        // -1: no winner
  std::string drawReason;     // "", "maxPlies", "repetition", or "unfinished"
};

// Plays the games with `numGameThreads` threads, one NNEvaluator per distinct model path shared by all games. The
// callback is called under a mutex, in completion order. `searchParams` is the base search configuration
// (maxVisits is set per player).
void runGames(
  const MatchConfig& mc,
  const std::vector<GameSpec>& games,
  ConfigParser& cfg,
  const SearchParams& searchParams,
  Logger& logger,
  const std::function<void(const GameResult&)>& onGameDone
);

// The record of a finished game with its match metadata ("match": table, opening, rotation, ...).
std::string resultToJsonLine(const MatchConfig& mc, const GameResult& r);

}  // namespace Q4Match

#endif  // Q4_MATCHENGINE_H_
