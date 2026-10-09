#include "../../tests/tests.h"
#include "../q4board.h"
#include "../q4bots.h"
#include "../q4history.h"
#include "../q4notation.h"
#include "../q4record.h"
#include "../q4rules.h"
#include "../q4style.h"
#include "../q4symmetry.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
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

uint64_t legalActionsHash(const Q4Board& board, int depth, uint64_t h);

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

      // Verify repetition state equals a fresh replay of the remaining events
      Q4History fresh(history.rules);
      for(size_t evIdx = 0; evIdx < history.events.size(); evIdx++) {
        const auto& ev = history.events[evIdx];
        if(ev.isElimination) fresh.eliminate(ev.eliminatedSeat);
        else fresh.play(ev.action);
      }
      testAssert(history.repetitionHashes == fresh.repetitionHashes);
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

  // 8. Repetition right after a wall (N=2 and N=3)
  {
    for(int repCount : {2, 3}) {
      Q4Rules r;
      r.repetitionDrawCount = repCount;
      Q4History h(r);

      // Seat 0 places a wall: a1h (anchor (0, 0)) -> action 221
      h.play(221);
      // Post-wall state P0: toMove = 1 (West).
      // Each cycle: West, North, East, South step out and return, returning to P0.
      for(int cycle = 0; cycle < repCount - 1; cycle++) {
        testAssert(!h.isFinished);
        // Step away
        h.play(Q4Board::actionOfPawn(Q4Board::cellOf(0, 6))); // West
        h.play(Q4Board::actionOfPawn(Q4Board::cellOf(5, 9))); // North
        h.play(Q4Board::actionOfPawn(Q4Board::cellOf(10, 6))); // East
        h.play(Q4Board::actionOfPawn(Q4Board::cellOf(5, 1))); // South

        // Step back
        h.play(Q4Board::actionOfPawn(Q4Board::cellOf(0, 5))); // West returns to a6
        h.play(Q4Board::actionOfPawn(Q4Board::cellOf(5, 10))); // North returns to f11
        h.play(Q4Board::actionOfPawn(Q4Board::cellOf(10, 5))); // East returns to k6
        h.play(Q4Board::actionOfPawn(Q4Board::cellOf(5, 0))); // South returns to f1
      }
      testAssert(h.isFinished);
      testAssert(h.isDraw);
      testAssert(h.getResultString() == "Draw");
    }
  }

  // 9. Repetition right after an elimination (N=2 and N=3)
  {
    for(int repCount : {2, 3}) {
      Q4Rules r;
      r.repetitionDrawCount = repCount;
      Q4History h(r);

      // Eliminate seat 1 (West). Post-elimination position Pe has alive = {0, 2, 3}, toMove = 0.
      h.eliminate(1);
      for(int cycle = 0; cycle < repCount - 1; cycle++) {
        testAssert(!h.isFinished);
        // Step away
        h.play(Q4Board::actionOfPawn(Q4Board::cellOf(5, 1))); // South
        h.play(Q4Board::actionOfPawn(Q4Board::cellOf(5, 9))); // North
        h.play(Q4Board::actionOfPawn(Q4Board::cellOf(10, 6))); // East

        // Step back
        h.play(Q4Board::actionOfPawn(Q4Board::cellOf(5, 0))); // South returns to f1
        h.play(Q4Board::actionOfPawn(Q4Board::cellOf(5, 10))); // North returns to f11
        h.play(Q4Board::actionOfPawn(Q4Board::cellOf(10, 5))); // East returns to k6
      }
      testAssert(h.isFinished);
      testAssert(h.isDraw);
      testAssert(h.getResultString() == "Draw");
    }
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
  uint64_t legalHash = legalActionsHash(startBoard, 2, 0xcbf29ce484222325ULL);

  // 3 non-degenerate positions replacing the degenerate pawn-heavy position (A4)
  // Position 1: Straight jumps, diagonal jump at board edge (a10->b11 over a11), diagonal jump at wall (e3->d4/f4 over e4 with e4h)
  {
    Q4Board b;
    for(int s = 0; s < 4; s++) {
      b.wallsLeft[s] = 0;
      b.occupant[b.pawn[s]] = -1;
    }
    b.pawn[0] = Q4Board::cellOf(0, 9);  b.occupant[b.pawn[0]] = 0; // a10
    b.pawn[1] = Q4Board::cellOf(0, 10); b.occupant[b.pawn[1]] = 1; // a11
    b.pawn[2] = Q4Board::cellOf(4, 2);  b.occupant[b.pawn[2]] = 2; // e3
    b.pawn[3] = Q4Board::cellOf(4, 3);  b.occupant[b.pawn[3]] = 3; // e4
    b.applyWall(4, 3, true); // e4h behind (4, 3)
    b.toMove = 0;
    b.recomputeDistancesToCenter();
    b.hash = b.getHashFromScratch();

    // Verify center is not reachable in one move
    std::vector<int> m0;
    b.getPawnMoves(0, m0);
    testAssert(std::find(m0.begin(), m0.end(), Q4Board::CENTER_CELL) == m0.end());

    testAssert(perft(b, 1) == 3);
    testAssert(perft(b, 2) == 6);
    testAssert(perft(b, 3) == 30);
    testAssert(perft(b, 4) == 90);
    testAssert(perft(b, 5) == 330);
    testAssert(perft(b, 6) == 990);
    testAssert(perft(b, 7) == 3828);
    testAssert(perft(b, 8) == 14388);
    legalHash = legalActionsHash(b, 6, legalHash);
  }

  // Position 2: Two-pawn jump permitted vs denied (c3 at (2, 2) jumping over c4 and c5 with wall c3v)
  {
    Q4Board b;
    for(int s = 0; s < 4; s++) {
      b.wallsLeft[s] = 0;
      b.occupant[b.pawn[s]] = -1;
    }
    b.pawn[0] = Q4Board::cellOf(2, 2); b.occupant[b.pawn[0]] = 0; // c3
    b.pawn[1] = Q4Board::cellOf(2, 3); b.occupant[b.pawn[1]] = 1; // c4
    b.pawn[2] = Q4Board::cellOf(2, 4); b.occupant[b.pawn[2]] = 2; // c5
    b.pawn[3] = Q4Board::cellOf(9, 9); b.occupant[b.pawn[3]] = 3;
    b.applyWall(2, 2, false); // c3v blocks East from (2, 2)
    b.toMove = 0;
    b.recomputeDistancesToCenter();
    b.hash = b.getHashFromScratch();

    // Verify two-pawn jump is permitted to (2, 5)
    std::vector<int> m0;
    b.getPawnMoves(0, m0);
    testAssert(std::find(m0.begin(), m0.end(), Q4Board::cellOf(2, 5)) != m0.end());
    // Verify center is not reachable in one move
    testAssert(std::find(m0.begin(), m0.end(), Q4Board::CENTER_CELL) == m0.end());

    testAssert(perft(b, 1) == 3);
    testAssert(perft(b, 2) == 9);
    testAssert(perft(b, 3) == 35);
    testAssert(perft(b, 4) == 140);
    testAssert(perft(b, 5) == 564);
    testAssert(perft(b, 6) == 2088);
    testAssert(perft(b, 7) == 7776);
    testAssert(perft(b, 8) == 27216);
    legalHash = legalActionsHash(b, 6, legalHash);
  }

  // Position 3: Two-pawn jump into center a few plies deep
  {
    Q4Board b;
    for(int s = 0; s < 4; s++) {
      b.wallsLeft[s] = 0;
      b.occupant[b.pawn[s]] = -1;
    }
    b.pawn[0] = Q4Board::cellOf(2, 5); b.occupant[b.pawn[0]] = 0; // c6
    b.pawn[1] = Q4Board::cellOf(3, 5); b.occupant[b.pawn[1]] = 1; // d6
    b.pawn[2] = Q4Board::cellOf(4, 4); b.occupant[b.pawn[2]] = 2; // e5
    b.pawn[3] = Q4Board::cellOf(9, 9); b.occupant[b.pawn[3]] = 3;
    b.toMove = 2; // Seat 2 to move
    b.recomputeDistancesToCenter();
    b.hash = b.getHashFromScratch();

    // Verify center is not reachable in one move for any seat
    for(int s = 0; s < 4; s++) {
      std::vector<int> ms;
      b.getPawnMoves(s, ms);
      testAssert(std::find(ms.begin(), ms.end(), Q4Board::CENTER_CELL) == ms.end());
    }

    testAssert(perft(b, 1) == 4);
    testAssert(perft(b, 2) == 16);
    testAssert(perft(b, 3) == 64);
    testAssert(perft(b, 4) == 244);
    testAssert(perft(b, 5) == 892);
    testAssert(perft(b, 6) == 2812);
    testAssert(perft(b, 7) == 10792);
    testAssert(perft(b, 8) == 41312);
    legalHash = legalActionsHash(b, 6, legalHash);
  }

  cout << "  getLegalActions hash over the perft trees: " << legalHash << endl;
  testAssert(legalHash == 13831335297760359281ULL);
}

bool canMasksMatchBlocked(const Q4Board& b) {
  for(int c = 0; c < Q4Board::NUM_CELLS; c++) {
    if(b.canN.test(c) != b.canStep(c, Q4Board::DIR_N)) return false;
    if(b.canE.test(c) != b.canStep(c, Q4Board::DIR_E)) return false;
    if(b.canS.test(c) != b.canStep(c, Q4Board::DIR_S)) return false;
    if(b.canW.test(c) != b.canStep(c, Q4Board::DIR_W)) return false;
  }
  return true;
}

// R4b C.3: the flood-fill isLegalWallBruteForce agrees with the old BFS on every anchor and orientation of
// 20,000 positions (random and wall-heavy games, 2-3 alive seats, pawns on the center), and the can-masks
// stay in sync with blocked[] through applyWall, add/removeWallFromBlocked and board copies.
void testWallFloodFillFuzz() {
  cout << "Running Q4 wall flood fill fuzz (new == BFS)..." << endl;
  Rand rand("testWallFloodFillFuzz");
  int numPositions = 0;
  int numIllegalByPath = 0;
  int numCenterPositions = 0;
  int numEliminationPositions = 0;
  while(numPositions < 20000) {
    Q4History history;
    double wallBias = rand.nextDouble() < 0.5 ? 0.0 : 0.85;
    int targetPlies = (int)rand.nextUInt(80);
    for(int m = 0; m < targetPlies && !history.isFinished; m++) {
      if(history.currentBoard.getNumAlive() > 2 && rand.nextUInt(25) == 0) {
        int victim = (int)rand.nextUInt(4);
        if(history.currentBoard.isAlive(victim))
          history.eliminate(victim);
        continue;
      }
      vector<int> actions;
      history.currentBoard.getLegalActions(history.currentBoard.toMove, actions);
      if(actions.empty())
        break;
      vector<int> wallActs;
      for(int act : actions) {
        if(!Q4Board::isPawnAction(act))
          wallActs.push_back(act);
      }
      if(!wallActs.empty() && rand.nextBool(wallBias))
        history.play(wallActs[rand.nextUInt((uint32_t)wallActs.size())]);
      else
        history.play(actions[rand.nextUInt((uint32_t)actions.size())]);
    }

    Q4Board b = history.currentBoard;
    // Move a random alive pawn onto the center (the flood fill must not require it)
    if(rand.nextUInt(10) == 0 && b.occupant[Q4Board::CENTER_CELL] < 0) {
      int s = (int)rand.nextUInt(4);
      if(b.isAlive(s)) {
        b.occupant[b.pawn[s]] = -1;
        b.pawn[s] = Q4Board::CENTER_CELL;
        b.occupant[Q4Board::CENTER_CELL] = (int8_t)s;
        numCenterPositions++;
      }
    }
    if(b.getNumAlive() < 4)
      numEliminationPositions++;
    testAssert(canMasksMatchBlocked(b));

    for(int ay = 0; ay < Q4Board::NUM_ANCHORS; ay++) {
      for(int ax = 0; ax < Q4Board::NUM_ANCHORS; ax++) {
        for(int h = 0; h < 2; h++) {
          bool isH = h == 1;
          bool fast = b.isLegalWallBruteForce(ax, ay, isH);
          bool bfs = b.isLegalWallBruteForceBFS(ax, ay, isH);
          if(fast != bfs)
            cout << Q4Notation::renderBoardAscii(b) << "wall " << Q4Notation::wallToString(ax, ay, isH)
                 << ": flood fill " << fast << ", BFS " << bfs << endl;
          testAssert(fast == bfs);
          testAssert(b.isGeometricallyLegalWall(ax, ay, isH) == bfs);
          if(!bfs && !b.wallConflicts(ax, ay, isH))
            numIllegalByPath++;
        }
      }
    }

    // Add then remove a non-conflicting wall: masks follow blocked[] and come back unchanged
    int ax = (int)rand.nextUInt(Q4Board::NUM_ANCHORS);
    int ay = (int)rand.nextUInt(Q4Board::NUM_ANCHORS);
    bool isH = rand.nextBool(0.5);
    if(!b.wallConflicts(ax, ay, isH)) {
      Q4Board copy = b;
      copy.addWallToBlocked(ax, ay, isH);
      testAssert(canMasksMatchBlocked(copy));
      copy.removeWallFromBlocked(ax, ay, isH);
      testAssert(canMasksMatchBlocked(copy));
      testAssert(copy.canN == b.canN && copy.canE == b.canE && copy.canS == b.canS && copy.canW == b.canW);
    }
    numPositions++;
  }
  cout << "  " << numPositions << " positions, " << numEliminationPositions << " with eliminations, "
       << numCenterPositions << " with a pawn on the center, " << numIllegalByPath
       << " (position, wall) pairs illegal only by path" << endl;
  testAssert(numIllegalByPath > 0);
  testAssert(numCenterPositions > 0);
  testAssert(numEliminationPositions > 0);
}

// R4b C.3: hash of getLegalActions over the perft trees (start position to depth 2, positions 1-3 of
// testPerft to depth 6). The constant was computed with the BFS isLegalWallBruteForce.
uint64_t legalActionsHash(const Q4Board& board, int depth, uint64_t h) {
  std::vector<int> actions;
  board.getLegalActions(board.toMove, actions);
  for(int act : actions)
    h = (h ^ (uint64_t)(act + 1)) * 0x100000001b3ULL;
  h = (h ^ 0xffff) * 0x100000001b3ULL;
  if(depth == 0 || board.isFinished())
    return h;
  for(int act : actions) {
    Q4Board nextBoard = board;
    nextBoard.applyAction(act);
    h = legalActionsHash(nextBoard, depth - 1, h);
  }
  return h;
}

// A5: on 2,000 random games (with eliminations, small maxPlies, repetition rule on with N = 2 and 3),
// Q4PlayState driven by playAssumeLegal and Q4History driven by play agree on board hash, plies,
// isFinished, winner, draw and repetition count after every event.
void testPlayStateConsistency() {
  cout << "Running Q4PlayState consistency tests against Q4History (A5)..." << endl;
  Rand rand("testPlayStateConsistency_seed42");
  for(int g = 0; g < 2000; g++) {
    Q4Rules rules;
    rules.maxPlies = (int)rand.nextInt(10, 40);
    rules.repetitionDrawCount = (rand.nextUInt(2) == 0) ? 2 : 3;

    Q4Board board(rules);
    Q4History hist(board, rules);
    Q4PlayState state(board, rules);

    while(!hist.isFinished) {
      testAssert(state.board.hash == hist.currentBoard.hash);
      testAssert(state.plies == hist.plies);
      testAssert(state.isFinished == hist.isFinished);
      testAssert(state.winnerSeat == hist.winnerSeat);
      testAssert(state.isDraw == hist.isDraw);
      testAssert(state.currentPositionRepetitionCount() == hist.currentPositionRepetitionCount());

      // Occasionally eliminate a random alive seat
      if(state.board.getNumAlive() > 2 && rand.nextUInt(20) == 0) {
        int victim = (int)rand.nextUInt(4);
        if(state.board.isAlive(victim)) {
          hist.eliminate(victim);
          state.eliminate(victim);
          testAssert(state.board.hash == hist.currentBoard.hash);
          testAssert(state.plies == hist.plies);
          testAssert(state.isFinished == hist.isFinished);
          testAssert(state.winnerSeat == hist.winnerSeat);
          testAssert(state.isDraw == hist.isDraw);
          testAssert(state.currentPositionRepetitionCount() == hist.currentPositionRepetitionCount());
          if(hist.isFinished)
            break;
        }
      }

      int toMove = state.board.toMove;
      vector<int> legal;
      state.board.getLegalActions(toMove, legal);
      if(legal.empty())
        break;
      int act = legal[rand.nextUInt(legal.size())];
      hist.play(act);
      state.playAssumeLegal(act);
    }

    testAssert(state.board.hash == hist.currentBoard.hash);
    testAssert(state.plies == hist.plies);
    testAssert(state.isFinished == hist.isFinished);
    testAssert(state.winnerSeat == hist.winnerSeat);
    testAssert(state.isDraw == hist.isDraw);
    testAssert(state.currentPositionRepetitionCount() == hist.currentPositionRepetitionCount());
  }
}

bool sameTracker(const Q4StyleTracker& a, const Q4StyleTracker& b) {
  for(int s = 0; s < 4; s++) {
    if(a.numMoves[s] != b.numMoves[s])
      return false;
    for(int h = 0; h < 2; h++)
      for(int d = 0; d < Q4StyleTracker::NUM_DESCRIPTORS; d++)
        if(a.mean[s][h][d] != b.mean[s][h][d])
          return false;
  }
  return true;
}

// S1 (C++ half): scripted history with hand-derived feature values, and undo restores the tracker exactly.
void testStyleTracker() {
  cout << "Running S1 style tracker (scripted values, undo)..." << endl;
  const float EPS = 1e-6f;
  // Seat 0 steps f1 -> f2 (distance 5 -> 4: progress 0.5, a greedy step). Seat 1 then walls e2h (blocks column e/f
  // between rows 2 and 3), which lengthens seat 0's path from 4 to 6 (its detour is one column to g2 and back), and
  // nobody else's: it hits the race leader (seat 0: arrival estimate 16 vs 18 and 19), victim column 0.25.
  Q4History hist{Q4Rules()};
  hist.play(Q4Notation::stringToAction("f2"));
  {
    float f[Q4StyleTracker::NUM_FEATURES];
    hist.getStyleFeatures(f);   // perspective: seat 1
    // seat 0 is "previous" (k = 3): its block starts at 3 * 19
    const float* b = f + 3 * Q4StyleTracker::PER_SEAT;
    const float exp[9] = {0, 0.5f, 1, 0, 0, 0, 0, 0, 0};
    for(int h = 0; h < 2; h++)
      for(int d = 0; d < 9; d++)
        testAssert(std::fabs(b[h * 9 + d] - exp[d]) < EPS);
    testAssert(std::fabs(b[18] - 1.0f / 64.0f) < EPS);
    for(int i = 0; i < 3 * Q4StyleTracker::PER_SEAT; i++)
      testAssert(f[i] == 0.0f);
  }
  Q4StyleTracker afterFirst = hist.style;
  hist.play(Q4Notation::stringToAction("e2h"));
  testAssert(hist.currentBoard.distToCenter[hist.currentBoard.pawn[0]] == 6);
  {
    float f[Q4StyleTracker::NUM_FEATURES];
    hist.getStyleFeatures(f);   // perspective: seat 2 = (me, next 3, across 0, previous 1)
    const float* b = f + 3 * Q4StyleTracker::PER_SEAT;           // seat 1 (previous)
    // wall, hit the leader, harmless 0; victim = absolute seat 0 = relative column 3 + (0 - 2) mod 4 = 5
    const float exp[9] = {1, 0, 0, 0, 0, 0.25f, 0, 1, 0};
    for(int h = 0; h < 2; h++)
      for(int d = 0; d < 9; d++)
        testAssert(std::fabs(b[h * 9 + d] - exp[d]) < EPS);
    testAssert(std::fabs(b[18] - 1.0f / 64.0f) < EPS);
    const float* a = f + 2 * Q4StyleTracker::PER_SEAT;           // seat 0 (across) unchanged
    const float expA[9] = {0, 0.5f, 1, 0, 0, 0, 0, 0, 0};
    for(int d = 0; d < 9; d++)
      testAssert(std::fabs(a[d] - expA[d]) < EPS);
  }
  // undo restores the tracker
  testAssert(hist.undo());
  testAssert(sameTracker(hist.style, afterFirst));
  testAssert(hist.undo());
  testAssert(sameTracker(hist.style, Q4StyleTracker()));
  testAssert(!hist.undo());

  // random population-like games with eliminations: undo all the way back restores every intermediate tracker
  Rand rand("s1_undo");
  static const char* const bots[5] = {"random", "randomPawn", "greedy", "basher", "grudge"};
  for(int g = 0; g < 40; g++) {
    Q4Rules rules;
    rules.maxPlies = 100;
    Q4History h(rules);
    std::unique_ptr<Q4Bot> players[4];
    for(int s = 0; s < 4; s++)
      players[s] = Q4Bots::makeBot(bots[rand.nextUInt(5)], rand.nextUInt64(), (s + 1) % 4);
    std::vector<Q4StyleTracker> saved = {h.style};
    while(!h.isFinished && h.events.size() < 120) {
      if(rand.nextBool(0.03)) {
        std::vector<int> alive;
        for(int s = 0; s < 4; s++)
          if(h.currentBoard.isAlive(s))
            alive.push_back(s);
        if(alive.size() > 1)
          h.eliminate(alive[rand.nextUInt((uint32_t)alive.size())]);
      }
      else {
        h.play(players[h.currentBoard.toMove]->getMove(h.currentBoard));
      }
      saved.push_back(h.style);
    }
    testAssert(saved.size() == h.events.size() + 1);
    for(int i = (int)saved.size() - 1; i >= 0; i--) {
      testAssert(sameTracker(h.style, saved[i]));
      if(i > 0)
        testAssert(h.undo());
    }
  }
  cout << "S1 (C++) passed!" << endl;
}

void testSgfRecord() {
  // A random game with an elimination, played to a maxPlies draw; every move carries a comment.
  Q4Rules rules;
  rules.maxPlies = 30;
  rules.repetitionDrawCount = 3;
  rules.initialWalls[2] = 5;
  Q4History hist(rules);
  std::unique_ptr<Q4Bot> bot = Q4Bots::makeBot("random", 99, 0);
  Q4Record rec;
  rec.rules = rules;
  rec.players[0].name = "netA"; rec.players[0].type = "selfplay"; rec.players[0].net = "netA"; rec.players[0].visits = 600;
  rec.players[1].type = "weak"; rec.players[1].visits = 20;
  bool eliminated = false;
  while(!hist.isFinished) {
    if(!eliminated && hist.plies >= 5) {
      hist.eliminate(2);
      eliminated = true;
      continue;
    }
    int act = bot->getMove(hist.currentBoard);
    hist.play(act);
  }
  rec.events = hist.events;
  rec.result = hist.getResultString();
  testAssert(rec.result == "Draw");
  testAssert(eliminated);
  rec.gtype = "fork";
  rec.startTurnIdx = 2;
  rec.hasGameHash = true;
  rec.gameHash0 = 0x0123456789abcdefULL;
  rec.gameHash1 = 0xfedcba9876543210ULL;
  for(size_t i = 3; i < rec.events.size(); i += 2) {
    if(rec.events[i].isElimination)
      continue;
    Q4MoveComment mc;
    mc.valid = true;
    float ps[5] = {0.31f, 0.22f, 0.27f, 0.15f, 0.05f};
    for(int c = 0; c < 5; c++) mc.p[c] = ps[c];
    mc.visits = 600 + (int)i;
    mc.weight = 0.5f;
    rec.setMoveComment(i, mc);
  }
  std::string line = rec.toSgfLine();
  testAssert(line.compare(0, 9, "(;FF[4]GM") == 0 && line.find("GM[Q4]SZ[11]") != string::npos);
  testAssert(line.find("RE[0]DR[maxPlies]") != string::npos);
  testAssert(line.find("WN[5]") != string::npos);
  testAssert(line.find(";EL[N]") != string::npos);
  testAssert(line.find("C[0.31 0.22 0.27 0.15 0.05 v=603 weight=0.50]") != string::npos);
  testAssert(line.find('\n') == string::npos);
  Q4Record back = Q4Record::fromSgfLine(line);
  testAssert(back.rules == rules);
  testAssert(back.result == "Draw" && back.drawReason == "maxPlies" && back.gtype == "fork");
  testAssert(back.startTurnIdx == 2 && back.matchOpening < 0);
  testAssert(back.hasGameHash && back.gameHash0 == rec.gameHash0 && back.gameHash1 == rec.gameHash1);
  testAssert(back.players[0].name == "netA" && back.players[0].net == "netA" && back.players[0].visits == 600);
  testAssert(back.players[1].type == "weak" && back.players[1].visits == 20 && back.players[2].net.empty());
  testAssert(back.events.size() == rec.events.size());
  for(size_t i = 0; i < rec.events.size(); i++) {
    testAssert(back.events[i].isElimination == rec.events[i].isElimination);
    testAssert(back.events[i].action == rec.events[i].action);
    testAssert(back.events[i].eliminatedSeat == rec.events[i].eliminatedSeat);
    testAssert(back.moveComments[i].valid == rec.moveComments[i].valid);
    if(rec.moveComments[i].valid) {
      testAssert(back.moveComments[i].visits == rec.moveComments[i].visits);
      testAssert(std::abs(back.moveComments[i].p[1] - 0.22f) < 1e-4f && back.moveComments[i].weight == 0.5f);
    }
  }
  Q4History h2;
  back.replay(h2);
  testAssert(h2.getResultString() == "Draw" && h2.plies == hist.plies);
  testAssert(back.toSgfLine() == line);

  // A decided game with match metadata and no comments; names are escaped
  Q4Rules plain;
  Q4History h3(plain);
  Q4Record win;
  win.rules = plain;
  win.players[3].name = "we]ird\\name";
  while(!h3.isFinished && h3.plies < 300)
    h3.play(bot->getMove(h3.currentBoard));
  win.events = h3.events;
  win.result = h3.isFinished && h3.winnerSeat >= 0 ? h3.getResultString() : "none";
  win.matchTable = "t1"; win.matchOpening = 3; win.matchRotation = 1; win.startTurnIdx = 6;
  std::string wline = win.toSgfLine();
  testAssert(wline.find("table=t1,opening=3,rotation=1") != string::npos);
  Q4Record wback = Q4Record::fromSgfLine(wline);
  testAssert(wback.players[3].name == "we]ird\\name" && wback.result == win.result);
  testAssert(wback.matchTable == "t1" && wback.matchOpening == 3 && wback.matchRotation == 1);
  testAssert(wback.comments.empty() && wback.moveComments.empty());
  if(win.result == "none")
    testAssert(wline.find("RE[?]DR[unfinished]") != string::npos);

  // Malformed lines are rejected
  auto rejects = [](const std::string& text) {
    try { Q4Record::fromSgfLine(text); }
    catch(const StringError&) { return true; }
    return false;
  };
  testAssert(rejects(""));
  testAssert(rejects("{\"rules\": {}}"));
  testAssert(rejects("(;FF[4]GM[1]SZ[17])"));                                    // a Duel game
  testAssert(rejects("(;FF[4]GM[Q4]SZ[11]RE[0];S[f2]"));                         // no closing )
  testAssert(rejects("(;FF[4]GM[Q4]SZ[11]RE[X])"));                              // bad result
  testAssert(rejects("(;FF[4]GM[Q4]SZ[11]RE[S];W[b6])"));                        // not the seat to move
  testAssert(rejects("(;FF[4]GM[Q4]SZ[11]RE[S];S[zz])"));                        // bad action
  testAssert(rejects("(;FF[4]GM[Q4]SZ[11]RE[S];S[f2)"));                         // unterminated value
  Q4MoveComment junk;
  testAssert(!Q4MoveComment::parse("elim", junk) && !junk.valid);
  cout << "SGF record test passed" << endl;
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
  testWallFloodFillFuzz();
  testPlayStateConsistency();
  testStyleTracker();
  testSgfRecord();

  cout << "All Q4 Board tests PASSED!" << endl;
}
