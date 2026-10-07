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
  cout << "Running Q4 Wall Legality Fuzz (T4, >= 100,000 pairs)..." << endl;
  Rand rand(999);
  uint64_t testedPairs = 0;

  // Category 1: Random games (200 games, varying plies)
  for(int g = 0; g < 200; g++) {
    Q4History history;
    int moves = rand.nextUInt(40);
    for(int m = 0; m < moves && !history.isFinished; m++) {
      vector<int> actions;
      history.currentBoard.getLegalActions(history.currentBoard.toMove, actions);
      if(actions.empty()) break;
      history.play(actions[rand.nextUInt((uint32_t)actions.size())]);
    }
    const Q4Board& b = history.currentBoard;
    for(int ay = 0; ay < 10; ay++) {
      for(int ax = 0; ax < 10; ax++) {
        testAssert(b.isGeometricallyLegalWall(ax, ay, true) == b.isLegalWallBruteForceBFS(ax, ay, true));
        testAssert(b.isGeometricallyLegalWall(ax, ay, false) == b.isLegalWallBruteForceBFS(ax, ay, false));
        testedPairs += 2;
      }
    }
  }

  // Category 2: Wall-heavy games (bias toward wall moves, 15-25 walls placed)
  for(int g = 0; g < 200; g++) {
    Q4History history;
    int targetWalls = 15 + rand.nextUInt(10);
    int wallsPlaced = 0;
    while(wallsPlaced < targetWalls && !history.isFinished) {
      vector<int> actions;
      history.currentBoard.getLegalActions(history.currentBoard.toMove, actions);
      if(actions.empty()) break;
      vector<int> wallActs;
      for(int act : actions) {
        if(!Q4Board::isPawnAction(act))
          wallActs.push_back(act);
      }
      if(!wallActs.empty() && rand.nextBool(0.85)) {
        history.play(wallActs[rand.nextUInt((uint32_t)wallActs.size())]);
        wallsPlaced++;
      }
      else {
        history.play(actions[rand.nextUInt((uint32_t)actions.size())]);
      }
    }
    const Q4Board& b = history.currentBoard;
    for(int ay = 0; ay < 10; ay++) {
      for(int ax = 0; ax < 10; ax++) {
        testAssert(b.isGeometricallyLegalWall(ax, ay, true) == b.isLegalWallBruteForceBFS(ax, ay, true));
        testAssert(b.isGeometricallyLegalWall(ax, ay, false) == b.isLegalWallBruteForceBFS(ax, ay, false));
        testedPairs += 2;
      }
    }
  }

  // Category 3: Games with eliminated seats
  for(int g = 0; g < 150; g++) {
    Q4History history;
    int movesBeforeElim = rand.nextUInt(20);
    for(int m = 0; m < movesBeforeElim && !history.isFinished; m++) {
      vector<int> actions;
      history.currentBoard.getLegalActions(history.currentBoard.toMove, actions);
      if(actions.empty()) break;
      history.play(actions[rand.nextUInt((uint32_t)actions.size())]);
    }
    // Eliminate 1 or 2 seats
    if(history.currentBoard.getNumAlive() > 2) {
      int elimSeat = rand.nextUInt(4);
      while(!history.currentBoard.isAlive(elimSeat))
        elimSeat = rand.nextUInt(4);
      history.eliminate(elimSeat);
    }
    // Play a few more moves
    for(int m = 0; m < 10 && !history.isFinished; m++) {
      vector<int> actions;
      history.currentBoard.getLegalActions(history.currentBoard.toMove, actions);
      if(actions.empty()) break;
      history.play(actions[rand.nextUInt((uint32_t)actions.size())]);
    }

    const Q4Board& b = history.currentBoard;
    for(int ay = 0; ay < 10; ay++) {
      for(int ax = 0; ax < 10; ax++) {
        testAssert(b.isGeometricallyLegalWall(ax, ay, true) == b.isLegalWallBruteForceBFS(ax, ay, true));
        testAssert(b.isGeometricallyLegalWall(ax, ay, false) == b.isLegalWallBruteForceBFS(ax, ay, false));
        testedPairs += 2;
      }
    }
  }

  cout << "T4 passed! Verified " << testedPairs << " (position, wall) pairs with 100% agreement." << endl;
  testAssert(testedPairs >= 100000);
}

void testDeepPerft() {
  cout << "Running Q4 Deep Perft..." << endl;
  Q4Board startBoard;
  uint64_t d1 = perft(startBoard, 1);
  uint64_t d2 = perft(startBoard, 2);
  uint64_t d3 = perft(startBoard, 3);
  cout << "Start Board: Perft(1) = " << d1 << ", Perft(2) = " << d2 << ", Perft(3) = " << d3 << endl;
  testAssert(d1 == 203);
  testAssert(d2 == 40445);
  testAssert(d3 == 7906929);

  // Position 1 deep perft (d9=45998, d10=145624, d11=553104)
  {
    Q4Board b;
    for(int s = 0; s < 4; s++) { b.wallsLeft[s] = 0; b.occupant[b.pawn[s]] = -1; }
    b.pawn[0] = Q4Board::cellOf(0, 9);  b.occupant[b.pawn[0]] = 0;
    b.pawn[1] = Q4Board::cellOf(0, 10); b.occupant[b.pawn[1]] = 1;
    b.pawn[2] = Q4Board::cellOf(4, 2);  b.occupant[b.pawn[2]] = 2;
    b.pawn[3] = Q4Board::cellOf(4, 3);  b.occupant[b.pawn[3]] = 3;
    b.applyWall(4, 3, true);
    b.toMove = 0;
    b.recomputeDistancesToCenter();
    b.hash = b.getHashFromScratch();

    testAssert(perft(b, 9) == 45998);
    testAssert(perft(b, 10) == 145624);
    testAssert(perft(b, 11) == 553104);
  }

  // Position 2 deep perft (d9=99974, d10=375858, d11=1433908)
  {
    Q4Board b;
    for(int s = 0; s < 4; s++) { b.wallsLeft[s] = 0; b.occupant[b.pawn[s]] = -1; }
    b.pawn[0] = Q4Board::cellOf(2, 2); b.occupant[b.pawn[0]] = 0;
    b.pawn[1] = Q4Board::cellOf(2, 3); b.occupant[b.pawn[1]] = 1;
    b.pawn[2] = Q4Board::cellOf(2, 4); b.occupant[b.pawn[2]] = 2;
    b.pawn[3] = Q4Board::cellOf(9, 9); b.occupant[b.pawn[3]] = 3;
    b.applyWall(2, 2, false);
    b.toMove = 0;
    b.recomputeDistancesToCenter();
    b.hash = b.getHashFromScratch();

    testAssert(perft(b, 9) == 99974);
    testAssert(perft(b, 10) == 375858);
    testAssert(perft(b, 11) == 1433908);
  }

  // Position 3 deep perft (d9=157806, d10=533914, d11=2080330)
  {
    Q4Board b;
    for(int s = 0; s < 4; s++) { b.wallsLeft[s] = 0; b.occupant[b.pawn[s]] = -1; }
    b.pawn[0] = Q4Board::cellOf(2, 5); b.occupant[b.pawn[0]] = 0;
    b.pawn[1] = Q4Board::cellOf(3, 5); b.occupant[b.pawn[1]] = 1;
    b.pawn[2] = Q4Board::cellOf(4, 4); b.occupant[b.pawn[2]] = 2;
    b.pawn[3] = Q4Board::cellOf(9, 9); b.occupant[b.pawn[3]] = 3;
    b.toMove = 2;
    b.recomputeDistancesToCenter();
    b.hash = b.getHashFromScratch();

    testAssert(perft(b, 9) == 157806);
    testAssert(perft(b, 10) == 533914);
    testAssert(perft(b, 11) == 2080330);
  }
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
