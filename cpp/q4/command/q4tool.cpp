#include "../core/global.h"
#include "../core/rand.h"
#include "../core/timer.h"
#include "../external/nlohmann_json/json.hpp"
#include "../main.h"
#include "q4board.h"
#include "q4bots.h"
#include "q4history.h"
#include "q4notation.h"

#include <algorithm>
#include <chrono>
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

}  // namespace

int MainCmds::q4tool(const std::vector<std::string>& args) {
  std::vector<std::string> subArgs = args;
  if(!subArgs.empty() && subArgs[0] == "q4tool")
    subArgs = std::vector<std::string>(subArgs.begin() + 1, subArgs.end());

  if(subArgs.empty()) {
    std::cout << "Usage: katago q4tool <subcommand> [options]\n"
              << "Subcommands:\n"
              << "  legal                 Read JSON positions from stdin, output legal action names\n"
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
        Q4Board board = parseBoardFromJson(j);
        std::vector<int> actions;
        board.getLegalActions(board.toMove, actions);
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
        resp["isFinished"] = board.isFinished();
        resp["winner"] = board.getWinner();
        std::cout << resp.dump() << "\n" << std::flush;
      }
      catch(const std::exception& e) {
        std::cerr << "Error parsing position in q4tool legal: " << e.what() << "\n";
      }
    }
    return 0;
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

  std::cerr << "Unknown q4tool subcommand: " << subcmd << "\n";
  return 1;
}
