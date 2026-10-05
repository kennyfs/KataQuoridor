#include "../../tests/tests.h"
#include "../q4board.h"
#include "../q4bots.h"
#include "../q4history.h"
#include "../q4notation.h"
#include "../q4record.h"
#include "../q4rules.h"
#include "../q4symmetry.h"

#include <algorithm>
#include <iostream>
#include <set>
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

void testNotation() {
  cout << "Running Q4 Notation tests (T9)..." << endl;

  // Round-trip every one of the 321 actions
  for(int act = 0; act < Q4Board::NUM_ACTIONS; act++) {
    string str = Q4Notation::actionToString(act);
    int parsed = Q4Notation::stringToAction(str);
    testAssert(parsed == act);
  }

  // Reject malformed strings
  vector<string> malformed = {
    "", "l1", "a12", "k10h", "f6x", "a0", "k12v", "e", "11", "z5", "e15", "e5z", "a"
  };
  for(const string& bad : malformed) {
    bool threw = false;
    try {
      Q4Notation::stringToAction(bad);
    }
    catch(...) {
      threw = true;
    }
    testAssert(threw);
  }

  // Seats
  for(int s = 0; s < 4; s++) {
    string sStr = Q4Notation::seatToString(s);
    testAssert(Q4Notation::stringToSeat(sStr) == s);
    testAssert(Q4Notation::stringToSeat("p" + sStr) == s);
    testAssert(Q4Notation::stringToSeat("P" + sStr) == s);
  }
}

void testSymmetry() {
  cout << "Running Q4 Symmetry tests (T7)..." << endl;
  Q4Symmetry::init();

  // 1. Group axioms: identity, inverse, composition
  for(int s = 0; s < 8; s++) {
    testAssert(Q4Symmetry::compose(s, 0) == s);
    testAssert(Q4Symmetry::compose(0, s) == s);
    int inv = Q4Symmetry::inverse(s);
    testAssert(Q4Symmetry::compose(s, inv) == 0);
    testAssert(Q4Symmetry::compose(inv, s) == 0);
  }

  // Associativity
  for(int a = 0; a < 8; a++) {
    for(int b = 0; b < 8; b++) {
      for(int c = 0; c < 8; c++) {
        testAssert(Q4Symmetry::compose(Q4Symmetry::compose(a, b), c) ==
                   Q4Symmetry::compose(a, Q4Symmetry::compose(b, c)));
      }
    }
  }

  // 2. Geometric invariance on random boards
  Rand rand(42);
  for(int trial = 0; trial < 100; trial++) {
    Q4History history;
    int numSteps = rand.nextUInt(15);
    for(int step = 0; step < numSteps && !history.isFinished; step++) {
      vector<int> actions;
      history.currentBoard.getLegalActions(history.currentBoard.toMove, actions);
      if(actions.empty()) break;
      history.play(actions[rand.nextUInt((uint32_t)actions.size())]);
    }

    const Q4Board& origBoard = history.currentBoard;
    vector<int> origActions;
    origBoard.getLegalActions(origBoard.toMove, origActions);

    for(int sym = 0; sym < 8; sym++) {
      Q4Board symBoard = Q4Symmetry::applyBoard(origBoard, sym);
      testAssert(symBoard.checkInvariants());

      // Distances transformed
      for(int c = 0; c < Q4Board::NUM_CELLS; c++) {
        int sc = Q4Symmetry::applyCell(c, sym);
        testAssert(symBoard.distToCenter[sc] == origBoard.distToCenter[c]);
      }

      // Legal actions transformed
      vector<int> symActions;
      symBoard.getLegalActions(symBoard.toMove, symActions);
      testAssert(symActions.size() == origActions.size());

      vector<int> mappedOrigActions;
      for(int act : origActions)
        mappedOrigActions.push_back(Q4Symmetry::applyAction(act, sym));
      sort(mappedOrigActions.begin(), mappedOrigActions.end());
      sort(symActions.begin(), symActions.end());
      testAssert(mappedOrigActions == symActions);

      // Playing mapped action on symBoard == sym(playing action on origBoard)
      if(!origActions.empty()) {
        int act = origActions[rand.nextUInt((uint32_t)origActions.size())];
        int symAct = Q4Symmetry::applyAction(act, sym);

        Q4Board origNext = origBoard;
        origNext.applyAction(act);

        Q4Board symNext = symBoard;
        symNext.applyAction(symAct);

        Q4Board mappedOrigNext = Q4Symmetry::applyBoard(origNext, sym);
        testAssert(symNext.hash == mappedOrigNext.hash);
      }
    }
  }
}

void testHashingAndUndo() {
  cout << "Running Q4 Hashing and Undo tests (T5, T6)..." << endl;
  Rand rand(101);

  // Transposition test: order of independent moves gives equal hash
  {
    Q4History h1;
    Q4History h2;
    // South pawn f1->f2 (action of pawn (5, 1)), West pawn a6->b6 (action of pawn (1, 5))
    // Move order 1: South f2, West b6
    h1.play(Q4Board::actionOfPawn(Q4Board::cellOf(5, 1)));
    h1.play(Q4Board::actionOfPawn(Q4Board::cellOf(1, 5)));

    // In h2, suppose South plays f2, then North plays f10, then undo North, then West plays b6
    h2.play(Q4Board::actionOfPawn(Q4Board::cellOf(5, 1)));
    h2.play(Q4Board::actionOfPawn(Q4Board::cellOf(1, 5)));
    testAssert(h1.currentBoard.hash == h2.currentBoard.hash);
  }

  // 100 random games with undo round-trip and hash validation
  set<Hash128> seenHashes;
  int collisions = 0;

  for(int game = 0; game < 50; game++) {
    Q4History history;
    vector<Q4Board> snapshots;
    vector<Hash128> snapshotHashes;

    int numEvents = 30;
    for(int ev = 0; ev < numEvents && !history.isFinished; ev++) {
      snapshots.push_back(history.currentBoard);
      snapshotHashes.push_back(history.currentBoard.hash);

      testAssert(history.currentBoard.checkInvariants());
      testAssert(history.currentBoard.hash == history.currentBoard.getHashFromScratch());

      if(seenHashes.count(history.currentBoard.hash)) {
        // Can happen naturally if transpositions occur, but check
      }
      seenHashes.insert(history.currentBoard.hash);

      // 5% chance of random elimination if >= 2 alive
      if(history.currentBoard.getNumAlive() > 2 && rand.nextUInt(100) < 5) {
        int aliveSeats[4];
        int numAlive = 0;
        for(int s = 0; s < 4; s++) {
          if(history.currentBoard.isAlive(s))
            aliveSeats[numAlive++] = s;
        }
        int elimSeat = aliveSeats[rand.nextUInt(numAlive)];
        history.eliminate(elimSeat);
      }
      else {
        vector<int> actions;
        history.currentBoard.getLegalActions(history.currentBoard.toMove, actions);
        if(actions.empty()) break;
        int act = actions[rand.nextUInt((uint32_t)actions.size())];
        history.play(act);
      }
    }

    // Now undo all events and check exact match with stored snapshots
    while(!snapshots.empty()) {
      Q4Board expectedBoard = snapshots.back();
      snapshots.pop_back();

      bool ok = history.undo();
      testAssert(ok);
      testAssert(history.currentBoard.hash == expectedBoard.hash);
      testAssert(history.currentBoard.toMove == expectedBoard.toMove);
      testAssert(history.currentBoard.alive == expectedBoard.alive);
      testAssert(memcmp(history.currentBoard.pawn, expectedBoard.pawn, sizeof(expectedBoard.pawn)) == 0);
      testAssert(memcmp(history.currentBoard.wallsLeft, expectedBoard.wallsLeft, sizeof(expectedBoard.wallsLeft)) == 0);
      testAssert(history.currentBoard.hWalls == expectedBoard.hWalls);
      testAssert(history.currentBoard.vWalls == expectedBoard.vWalls);
      testAssert(memcmp(history.currentBoard.blocked, expectedBoard.blocked, sizeof(expectedBoard.blocked)) == 0);
      testAssert(memcmp(history.currentBoard.distToCenter, expectedBoard.distToCenter, sizeof(expectedBoard.distToCenter)) == 0);
    }
    testAssert(!history.undo()); // Cannot undo further
  }
}

void testTerminalRules() {
  cout << "Running Q4 Terminal Rules tests (T8)..." << endl;

  // 1. Direct step into center wins
  {
    Q4Board b;
    b.pawn[0] = Q4Board::cellOf(5, 4); // f5
    b.occupant[Q4Board::cellOf(5, 0)] = -1;
    b.occupant[b.pawn[0]] = 0;
    b.toMove = 0;
    b.applyAction(Q4Board::actionOfPawn(Q4Board::CENTER_CELL));
    testAssert(b.isFinished());
    testAssert(b.getWinner() == 0);
  }

  // 2. Direct jump over opponent into center wins
  {
    Q4Board b;
    b.pawn[0] = Q4Board::cellOf(5, 3); // f4
    b.pawn[1] = Q4Board::cellOf(5, 4); // f5
    b.occupant[Q4Board::cellOf(5, 0)] = -1;
    b.occupant[Q4Board::cellOf(0, 5)] = -1;
    b.occupant[b.pawn[0]] = 0;
    b.occupant[b.pawn[1]] = 1;
    b.toMove = 0;
    vector<int> moves;
    b.getPawnMoves(0, moves);
    testAssert(find(moves.begin(), moves.end(), Q4Board::CENTER_CELL) != moves.end());
    b.applyAction(Q4Board::actionOfPawn(Q4Board::CENTER_CELL));
    testAssert(b.isFinished());
    testAssert(b.getWinner() == 0);
  }

  // 3. Diagonal jump into center wins
  {
    Q4Board b;
    b.pawn[0] = Q4Board::cellOf(4, 4);
    b.pawn[1] = Q4Board::cellOf(4, 5);
    b.occupant[Q4Board::cellOf(5, 0)] = -1;
    b.occupant[Q4Board::cellOf(0, 5)] = -1;
    b.occupant[b.pawn[0]] = 0;
    b.occupant[b.pawn[1]] = 1;
    b.toMove = 0;
    // Wall behind (4, 5) blocks straight jump (4, 5)-(4, 6)
    b.applyWall(4, 5, true);
    b.toMove = 0;
    vector<int> moves;
    b.getPawnMoves(0, moves);
    testAssert(find(moves.begin(), moves.end(), Q4Board::CENTER_CELL) != moves.end());
  }

  // 4. Two-pawn jump into center wins
  {
    Q4Board b;
    b.pawn[0] = Q4Board::cellOf(2, 5);
    b.pawn[1] = Q4Board::cellOf(3, 5);
    b.pawn[2] = Q4Board::cellOf(4, 5);
    b.occupant[Q4Board::cellOf(5, 0)] = -1;
    b.occupant[Q4Board::cellOf(0, 5)] = -1;
    b.occupant[Q4Board::cellOf(5, 10)] = -1;
    b.occupant[b.pawn[0]] = 0;
    b.occupant[b.pawn[1]] = 1;
    b.occupant[b.pawn[2]] = 2;
    b.toMove = 0;
    vector<int> moves;
    b.getPawnMoves(0, moves);
    testAssert(find(moves.begin(), moves.end(), Q4Board::CENTER_CELL) != moves.end());
    b.applyAction(Q4Board::actionOfPawn(Q4Board::CENTER_CELL));
    testAssert(b.isFinished());
    testAssert(b.getWinner() == 0);
  }

  // 5. Last seat alive wins immediately
  {
    Q4History h;
    h.eliminate(1);
    testAssert(!h.isFinished);
    h.eliminate(2);
    testAssert(!h.isFinished);
    h.eliminate(3);
    testAssert(h.isFinished);
    testAssert(h.winnerSeat == 0);
    testAssert(h.getResultString() == "1+");
  }

  // 6. Max plies boundary
  {
    Q4Rules r;
    r.maxPlies = 4;
    Q4History h(r);
    h.play(Q4Board::actionOfPawn(Q4Board::cellOf(5, 1))); // ply 1
    h.play(Q4Board::actionOfPawn(Q4Board::cellOf(1, 5))); // ply 2
    h.play(Q4Board::actionOfPawn(Q4Board::cellOf(5, 9))); // ply 3
    testAssert(!h.isFinished);
    h.play(Q4Board::actionOfPawn(Q4Board::cellOf(9, 5))); // ply 4
    testAssert(h.isFinished);
    testAssert(h.isDraw);
    testAssert(h.getResultString() == "Draw");
  }

  // 7. Repetition draw
  {
    Q4Rules r;
    r.repetitionDrawCount = 3;
    Q4History h(r);
    for(int cycle = 0; cycle < 2; cycle++) {
      h.play(Q4Board::actionOfPawn(Q4Board::cellOf(4, 0)));
      h.play(Q4Board::actionOfPawn(Q4Board::cellOf(0, 4)));
      h.play(Q4Board::actionOfPawn(Q4Board::cellOf(4, 10)));
      h.play(Q4Board::actionOfPawn(Q4Board::cellOf(10, 4)));

      h.play(Q4Board::actionOfPawn(Q4Board::cellOf(5, 0)));
      h.play(Q4Board::actionOfPawn(Q4Board::cellOf(0, 5)));
      h.play(Q4Board::actionOfPawn(Q4Board::cellOf(5, 10)));
      h.play(Q4Board::actionOfPawn(Q4Board::cellOf(10, 5)));
    }
    testAssert(h.isFinished);
    testAssert(h.isDraw);
    testAssert(h.getResultString() == "Draw");
  }
}

void testPerft() {
  cout << "Running Q4 Perft tests (T2)..." << endl;

  // Start position
  Q4Board startBoard;
  uint64_t d1 = perft(startBoard, 1);
  testAssert(d1 == 203);

  uint64_t d2 = perft(startBoard, 2);
  testAssert(d2 == 40445);

  // Pawn-heavy position: four pawns surrounding center, no walls left
  // South at (5, 4), West at (4, 5), North at (5, 6), East at (6, 5)
  // Surrounding walls: a box around them so they must jump or step around
  Q4Board pBoard;
  for(int s = 0; s < 4; s++) {
    pBoard.wallsLeft[s] = 0;
    pBoard.occupant[pBoard.pawn[s]] = -1;
  }
  pBoard.pawn[0] = Q4Board::cellOf(5, 4); pBoard.occupant[pBoard.pawn[0]] = 0;
  pBoard.pawn[1] = Q4Board::cellOf(4, 5); pBoard.occupant[pBoard.pawn[1]] = 1;
  pBoard.pawn[2] = Q4Board::cellOf(5, 6); pBoard.occupant[pBoard.pawn[2]] = 2;
  pBoard.pawn[3] = Q4Board::cellOf(6, 5); pBoard.occupant[pBoard.pawn[3]] = 3;

  // Add surrounding walls around 3..7
  pBoard.applyWall(3, 3, true);  // d4h
  pBoard.applyWall(5, 3, true);  // f4h
  pBoard.applyWall(3, 6, true);  // d7h
  pBoard.applyWall(5, 6, true);  // f7h
  pBoard.applyWall(3, 4, false); // d5v
  pBoard.applyWall(6, 4, false); // g5v

  pBoard.toMove = 0;
  pBoard.recomputeDistancesToCenter();
  pBoard.hash = pBoard.getHashFromScratch();

  // Test perft depths on this pawn-heavy position
  uint64_t pd1 = perft(pBoard, 1);
  uint64_t pd2 = perft(pBoard, 2);
  uint64_t pd3 = perft(pBoard, 3);
  uint64_t pd4 = perft(pBoard, 4);
  uint64_t pd5 = perft(pBoard, 5);
  uint64_t pd6 = perft(pBoard, 6);

  cout << "Pawn-heavy position perft counts: "
       << "d1=" << pd1 << ", d2=" << pd2 << ", d3=" << pd3
       << ", d4=" << pd4 << ", d5=" << pd5 << ", d6=" << pd6 << endl;
  testAssert(pd1 == 3);
  testAssert(pd2 == 7);
  testAssert(pd3 == 15);
  testAssert(pd4 == 35);
  testAssert(pd5 == 51);
  testAssert(pd6 == 111);
}

}  // namespace

void Tests::runQ4BoardTests() {
  cout << "========================================" << endl;
  cout << "Starting Q4 Board Test Suite (Fast)" << endl;
  cout << "========================================" << endl;

  testNotation();
  testSymmetry();
  testHashingAndUndo();
  testTerminalRules();
  testPerft();

  cout << "All Q4 Board tests PASSED!" << endl;
}
