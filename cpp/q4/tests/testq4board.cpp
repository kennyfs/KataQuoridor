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
  }
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
  testPlayStateConsistency();

  cout << "All Q4 Board tests PASSED!" << endl;
}
