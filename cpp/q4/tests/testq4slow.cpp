#include "../../tests/tests.h"
#include "../q4board.h"
#include "../q4history.h"
#include "../q4notation.h"

#include <iostream>
#include <vector>

using namespace std;
using namespace TestCommon;

namespace {

uint64_t perft(const Q4Board& board, int depth) {
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
    total += perft(nextBoard, depth - 1);
  }
  return total;
}

void testWallLegalityFuzz() {
  cout << "Running Q4 Wall Legality Fuzz (T4)..." << endl;
  Rand rand(999);

  // Generate 200 random games, in each testing random candidate wall legality
  for(int g = 0; g < 200; g++) {
    Q4History history;
    int moves = rand.nextUInt(30);
    for(int m = 0; m < moves && !history.isFinished; m++) {
      vector<int> actions;
      history.currentBoard.getLegalActions(history.currentBoard.toMove, actions);
      if(actions.empty()) break;
      history.play(actions[rand.nextUInt((uint32_t)actions.size())]);
    }

    const Q4Board& b = history.currentBoard;
    // Test 100 candidate walls per game
    for(int c = 0; c < 100; c++) {
      int ax = rand.nextUInt(10);
      int ay = rand.nextUInt(10);
      bool isH = rand.nextBool(0.5);

      bool ok = b.isLegalWallBruteForce(ax, ay, isH);
      if(ok) {
        testAssert(!b.wallConflicts(ax, ay, isH));
      }
    }
  }
}

void testDeepPerft() {
  cout << "Running Q4 Deep Perft..." << endl;
  Q4Board startBoard;
  uint64_t d1 = perft(startBoard, 1);
  uint64_t d2 = perft(startBoard, 2);
  cout << "Perft(1) = " << d1 << ", Perft(2) = " << d2 << endl;
  testAssert(d1 == 203);
  testAssert(d2 == 40445);
}

}  // namespace

void Tests::runQ4SlowTests() {
  cout << "========================================" << endl;
  cout << "Starting Q4 Slow Fuzz & Perft Test Suite" << endl;
  cout << "========================================" << endl;

  testWallLegalityFuzz();
  testDeepPerft();

  cout << "All Q4 Slow tests PASSED!" << endl;
}
