/*
 * testrules.cpp
 * Unit tests for Quoridor C++ Core Engine.
 * Step 1: Topology & Coordinate System.
 */

#include "../tests/tests.h"
#include "../game/board.h"
#include "../game/rules.h"
#include "../game/boardhistory.h"
#include "../game/graphhash.h"
#include "../core/global.h"

#include <cassert>
#include <iostream>
#include <set>
#include <vector>

using namespace std;
using namespace TestCommon;

namespace {

// Step 1 - Test 1: Grid constants and color constants
static void testGridConstants() {
  cout << "  [Step 1.1] Grid & Color Constants..." << endl;

  Board b;
  testAssert(b.x_size == 17);
  testAssert(b.y_size == 17);
  testAssert(Board::MAX_LEN == 17);
  testAssert(Board::DEFAULT_LEN == 17);
  testAssert(COMPILE_MAX_BOARD_LEN == 17);
  testAssert(Board::MAX_PLAY_SIZE == 289);
  testAssert(Board::MAX_FENCE_NUM == 10);

  // Colors check
  testAssert(C_EMPTY == 0);
  testAssert(C_BLACK == 1);
  testAssert(C_WHITE == 2);
  testAssert(C_FENCE == 3);
  testAssert(C_WALL == 4);
  testAssert(NUM_BOARD_COLORS == 5);

  // Player check
  testAssert(P_BLACK == 1);
  testAssert(P_WHITE == 2);
  testAssert(getOpp(P_BLACK) == P_WHITE);
  testAssert(getOpp(P_WHITE) == P_BLACK);

  // PlayerIO conversion check
  testAssert(PlayerIO::colorToChar(C_EMPTY) == '.');
  testAssert(PlayerIO::colorToChar(C_BLACK) == 'B');
  testAssert(PlayerIO::colorToChar(C_WHITE) == 'W');
  testAssert(PlayerIO::colorToChar(C_FENCE) == '*');
  testAssert(PlayerIO::colorToChar(C_WALL) == '#');

  testAssert(PlayerIO::playerToStringShort(P_BLACK) == "B");
  testAssert(PlayerIO::playerToStringShort(P_WHITE) == "W");
  testAssert(PlayerIO::playerToString(P_BLACK) == "Black");
  testAssert(PlayerIO::playerToString(P_WHITE) == "White");

  cout << "    -> Passed!" << endl;
}

// Step 1 - Test 2: 81 Pawn Cells geometry (even X, even Y)
static void testPawnCoordinates() {
  cout << "  [Step 1.2] Pawn Cells Geometry (81 locations)..." << endl;

  set<Loc> pawnLocs;
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      Loc loc = Location::pawnLoc(c, r, 17);
      int x = Location::getX(loc, 17);
      int y = Location::getY(loc, 17);

      // Must map to (2c, 2r)
      testAssert(x == 2 * c);
      testAssert(y == 2 * r);
      testAssert(x >= 0 && x <= 16 && x % 2 == 0);
      testAssert(y >= 0 && y <= 16 && y % 2 == 0);

      // Classification checks
      testAssert(Location::isPawnLoc(loc, 17));
      testAssert(!Location::isHWallLoc(loc, 17));
      testAssert(!Location::isVWallLoc(loc, 17));

      // Must be unique
      testAssert(pawnLocs.find(loc) == pawnLocs.end());
      pawnLocs.insert(loc);
    }
  }
  testAssert(pawnLocs.size() == 81);

  cout << "    -> Passed (81/81 distinct pawn cells verified)!" << endl;
}

// Step 1 - Test 3: 64 Horizontal Walls geometry (odd X, odd Y)
static void testHWallCoordinates() {
  cout << "  [Step 1.3] Horizontal Wall Centers & Arms (64 locations)..." << endl;

  set<Loc> hWallLocs;
  for(int r = 0; r < 8; r++) {
    for(int c = 0; c < 8; c++) {
      Loc center = Location::hWallLoc(c, r, 17);
      int x = Location::getX(center, 17);
      int y = Location::getY(center, 17);

      // Center must map to (2c+1, 2r+1)
      testAssert(x == 2 * c + 1);
      testAssert(y == 2 * r + 1);
      testAssert(x >= 1 && x <= 15 && x % 2 == 1);
      testAssert(y >= 1 && y <= 15 && y % 2 == 1);

      // Classification checks
      testAssert(Location::isHWallLoc(center, 17));
      testAssert(!Location::isPawnLoc(center, 17));
      testAssert(!Location::isVWallLoc(center, 17));

      // Occupies 3 cells: (2c, 2r+1), (2c+1, 2r+1), (2c+2, 2r+1)
      Loc leftArm = Location::getLoc(2 * c, 2 * r + 1, 17);
      Loc rightArm = Location::getLoc(2 * c + 2, 2 * r + 1, 17);

      testAssert(Location::getX(leftArm, 17) >= 0 && Location::getX(leftArm, 17) <= 16);
      testAssert(Location::getX(rightArm, 17) >= 0 && Location::getX(rightArm, 17) <= 16);
      testAssert(Location::getY(leftArm, 17) >= 0 && Location::getY(leftArm, 17) <= 16);
      testAssert(Location::getY(rightArm, 17) >= 0 && Location::getY(rightArm, 17) <= 16);

      // Left arm and right arm are distinct from center
      testAssert(leftArm != center);
      testAssert(rightArm != center);
      testAssert(leftArm != rightArm);

      // Centers must be unique
      testAssert(hWallLocs.find(center) == hWallLocs.end());
      hWallLocs.insert(center);
    }
  }
  testAssert(hWallLocs.size() == 64);

  cout << "    -> Passed (64/64 distinct H-wall centers verified)!" << endl;
}

// Step 1 - Test 4: 64 Vertical Walls geometry (odd X, even Y - Upper Arm, Player-Agnostic)
static void testVWallCoordinates() {
  cout << "  [Step 1.4] Vertical Wall Upper Arms & Segments (64 locations, Player-Agnostic)..." << endl;

  set<Loc> vWallLocs;
  for(int r = 0; r < 8; r++) {
    for(int c = 0; c < 8; c++) {
      // Pure geometric function: vWallLoc takes only (c, r), NO Player pla!
      Loc upperArm = Location::vWallLoc(c, r, 17);
      int x = Location::getX(upperArm, 17);
      int y = Location::getY(upperArm, 17);

      // Upper arm must map to (2c+1, 2r)
      testAssert(x == 2 * c + 1);
      testAssert(y == 2 * r);
      testAssert(x >= 1 && x <= 15 && x % 2 == 1);
      testAssert(y >= 0 && y <= 14 && y % 2 == 0);

      // Classification checks
      testAssert(Location::isVWallLoc(upperArm, 17));
      testAssert(!Location::isPawnLoc(upperArm, 17));
      testAssert(!Location::isHWallLoc(upperArm, 17));

      // Occupies 3 cells: upper (2c+1, 2r), center (2c+1, 2r+1), lower (2c+1, 2r+2)
      Loc center = Location::getLoc(2 * c + 1, 2 * r + 1, 17);
      Loc lowerArm = Location::getLoc(2 * c + 1, 2 * r + 2, 17);

      testAssert(Location::getX(center, 17) >= 0 && Location::getX(center, 17) <= 16);
      testAssert(Location::getX(lowerArm, 17) >= 0 && Location::getX(lowerArm, 17) <= 16);
      testAssert(Location::getY(center, 17) >= 0 && Location::getY(center, 17) <= 16);
      testAssert(Location::getY(lowerArm, 17) >= 0 && Location::getY(lowerArm, 17) <= 16);

      // All 3 cells are distinct
      testAssert(upperArm != center);
      testAssert(upperArm != lowerArm);
      testAssert(center != lowerArm);

      // Upper arms must be unique
      testAssert(vWallLocs.find(upperArm) == vWallLocs.end());
      vWallLocs.insert(upperArm);
    }
  }
  testAssert(vWallLocs.size() == 64);

  cout << "    -> Passed (64/64 distinct V-wall upper arms verified, strictly player-agnostic)!" << endl;
}

// Step 1 - Test 5: Disjointness of all 209 legal move representations
static void testDisjointMoveLocations() {
  cout << "  [Step 1.5] Move Location Disjointness (81 + 64 + 64 = 209)..." << endl;

  set<Loc> allMoveLocs;

  // 1. Add 81 Pawn moves
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      Loc loc = Location::pawnLoc(c, r, 17);
      testAssert(allMoveLocs.find(loc) == allMoveLocs.end());
      allMoveLocs.insert(loc);
    }
  }
  testAssert(allMoveLocs.size() == 81);

  // 2. Add 64 H-wall moves (centers)
  for(int r = 0; r < 8; r++) {
    for(int c = 0; c < 8; c++) {
      Loc loc = Location::hWallLoc(c, r, 17);
      testAssert(allMoveLocs.find(loc) == allMoveLocs.end());
      allMoveLocs.insert(loc);
    }
  }
  testAssert(allMoveLocs.size() == 81 + 64);

  // 3. Add 64 V-wall moves (upper arms)
  for(int r = 0; r < 8; r++) {
    for(int c = 0; c < 8; c++) {
      Loc loc = Location::vWallLoc(c, r, 17);
      testAssert(allMoveLocs.find(loc) == allMoveLocs.end());
      allMoveLocs.insert(loc);
    }
  }
  testAssert(allMoveLocs.size() == 209);

  // PASS_LOC (1) and NULL_LOC (0) must not be in allMoveLocs
  testAssert(allMoveLocs.find(Board::PASS_LOC) == allMoveLocs.end());
  testAssert(allMoveLocs.find(Board::NULL_LOC) == allMoveLocs.end());

  // Check that every move location has correct X, Y inside [0, 16]
  for(Loc loc : allMoveLocs) {
    int x = Location::getX(loc, 17);
    int y = Location::getY(loc, 17);
    testAssert(x >= 0 && x < 17);
    testAssert(y >= 0 && y < 17);
    testAssert(Location::getLoc(x, y, 17) == loc);
  }

  cout << "    -> Passed (209/209 mutually disjoint action Locs confirmed)!" << endl;
}

// Step 1 - Test 6: 17x17 Adjacent offsets and steps (stride = 18)
static void testAdjacentOffsets() {
  cout << "  [Step 1.6] Adjacent Offsets for 17x17 Grid..." << endl;

  short offsets[8];
  Location::getAdjacentOffsets(offsets, 17);

  // For 17x17, stride is 17 + 1 = 18
  testAssert(offsets[0] == -18); // North (-Y)
  testAssert(offsets[1] == -1);  // West (-X)
  testAssert(offsets[2] == 1);   // East (+X)
  testAssert(offsets[3] == 18);  // South (+Y)
  testAssert(offsets[4] == -19); // North-West
  testAssert(offsets[5] == -17); // North-East
  testAssert(offsets[6] == 17);  // South-West
  testAssert(offsets[7] == 19);  // South-East

  // Verify across grid interior
  for(int y = 2; y < 15; y++) {
    for(int x = 2; x < 15; x++) {
      Loc loc = Location::getLoc(x, y, 17);
      testAssert(loc + offsets[0] == Location::getLoc(x, y - 1, 17));
      testAssert(loc + offsets[1] == Location::getLoc(x - 1, y, 17));
      testAssert(loc + offsets[2] == Location::getLoc(x + 1, y, 17));
      testAssert(loc + offsets[3] == Location::getLoc(x, y + 1, 17));
      testAssert(loc + offsets[4] == Location::getLoc(x - 1, y - 1, 17));
      testAssert(loc + offsets[5] == Location::getLoc(x + 1, y - 1, 17));
      testAssert(loc + offsets[6] == Location::getLoc(x - 1, y + 1, 17));
      testAssert(loc + offsets[7] == Location::getLoc(x + 1, y + 1, 17));

      // Pawn step = 2 grid units
      testAssert(loc + 2 * offsets[0] == Location::getLoc(x, y - 2, 17));
      testAssert(loc + 2 * offsets[1] == Location::getLoc(x - 2, y, 17));
      testAssert(loc + 2 * offsets[2] == Location::getLoc(x + 2, y, 17));
      testAssert(loc + 2 * offsets[3] == Location::getLoc(x, y + 2, 17));
    }
  }

  cout << "    -> Passed (Offsets match 17x17 stride 18)!" << endl;
}

// Step 1 - Test 7: Initial board setup (Black e9, White e1, 10 fences, colors)
static void testInitialBoardState() {
  cout << "  [Step 1.7] Initial Board Setup & State..." << endl;

  Board b;
  testAssert(b.x_size == 17);
  testAssert(b.y_size == 17);
  testAssert(b.nextPla == P_BLACK);
  testAssert(b.movenum == 0);
  testAssert(b.blackFences == 10);
  testAssert(b.whiteFences == 10);

  // Black starts at e9: column e (4 -> X=8), row 9 (8 -> Y=16)
  Loc expectedBlack = Location::pawnLoc(4, 8, 17);
  testAssert(Location::getX(expectedBlack, 17) == 8);
  testAssert(Location::getY(expectedBlack, 17) == 16);
  testAssert(b.blackPawnLoc == expectedBlack);
  testAssert(b.colors[b.blackPawnLoc] == C_BLACK);

  // White starts at e1: column e (4 -> X=8), row 1 (0 -> Y=0)
  Loc expectedWhite = Location::pawnLoc(4, 0, 17);
  testAssert(Location::getX(expectedWhite, 17) == 8);
  testAssert(Location::getY(expectedWhite, 17) == 0);
  testAssert(b.whitePawnLoc == expectedWhite);
  testAssert(b.colors[b.whitePawnLoc] == C_WHITE);

  // All other on-board positions must be C_EMPTY
  for(int y = 0; y < 17; y++) {
    for(int x = 0; x < 17; x++) {
      Loc loc = Location::getLoc(x, y, 17);
      if(loc == b.blackPawnLoc)
        testAssert(b.colors[loc] == C_BLACK);
      else if(loc == b.whitePawnLoc)
        testAssert(b.colors[loc] == C_WHITE);
      else
        testAssert(b.colors[loc] == C_EMPTY);
    }
  }

  // Off-board locations must be C_WALL
  testAssert(b.colors[Location::getLoc(-1, 0, 17)] == C_WALL);
  testAssert(b.colors[Location::getLoc(0, -1, 17)] == C_WALL);
  testAssert(b.colors[Location::getLoc(17, 0, 17)] == C_WALL);
  testAssert(b.colors[Location::getLoc(0, 17, 17)] == C_WALL);

  // numStonesOnBoard: 2 pawns initially
  testAssert(b.numStonesOnBoard() == 2);
  testAssert(b.numPlaStonesOnBoard(P_BLACK) == 1);
  testAssert(b.numPlaStonesOnBoard(P_WHITE) == 1);
  cout << "    -> Passed (Initial board state 100% verified)!" << endl;
}

// Step 6 - Test 1: Black wins by reaching y=0 (row 0)
static void testBlackWinCondition() {
  cout << "  [Step 6.1] Black Win Condition (reaching y=0)..." << endl;

  Board b;
  BoardHistory hist(b, P_BLACK);
  testAssert(!hist.isGameFinished);
  testAssert(hist.winner == C_EMPTY);

  // Black advances north along column 4: row 8 down to row 0
  // White moves away from c=4,r=0 to c=3,r=0 on move 1, then oscillates between (3,0) and (2,0)
  for(int step = 0; step < 7; step++) {
    // Black moves to (c=4, r = 7 - step)
    Loc bLoc = Location::pawnLoc(4, 7 - step, 17);
    testAssert(hist.isLegal(b, bLoc, P_BLACK));
    hist.makeBoardMoveAssumeLegal(b, bLoc, P_BLACK, NULL);
    testAssert(!hist.isGameFinished);
    testAssert(hist.winner == C_EMPTY);

    // White moves:
    // step 0: (4,0) -> (3,0)
    // step 1: (3,0) -> (2,0)
    // step 2: (2,0) -> (3,0)
    // step 3: (3,0) -> (2,0)...
    int wCol = (step % 2 == 0) ? 3 : 2;
    Loc wLoc = Location::pawnLoc(wCol, 0, 17);
    testAssert(hist.isLegal(b, wLoc, P_WHITE));
    hist.makeBoardMoveAssumeLegal(b, wLoc, P_WHITE, NULL);
    testAssert(!hist.isGameFinished);
    testAssert(hist.winner == C_EMPTY);
  }

  // After 7 pairs of moves (14 moves total):
  // Black is at (c=4, r=1, y=2).
  // White is at (c=3, r=0, y=0).
  // (c=4, r=0, y=0) is empty.
  // Move 15: Black moves to (c=4, r=0, y=0)
  Loc winningLoc = Location::pawnLoc(4, 0, 17);
  testAssert(hist.isLegal(b, winningLoc, P_BLACK));
  hist.makeBoardMoveAssumeLegal(b, winningLoc, P_BLACK, NULL);

  // Terminal assertion
  testAssert(hist.isGameFinished == true);
  testAssert(hist.winner == P_BLACK);
  testAssert(hist.isNoResult == false);
  testAssert(hist.isResignation == false);
  testAssert(hist.moveHistory.size() == 15);

  cout << "    -> Passed (Black reaches y=0 and triggers winner == P_BLACK)!" << endl;
}

// Step 6 - Test 2: White wins by reaching y=16 (row 8)
static void testWhiteWinCondition() {
  cout << "  [Step 6.2] White Win Condition (reaching y=16)..." << endl;

  Board b;
  BoardHistory hist(b, P_BLACK);

  // Black leaves (4,8) immediately to (3,8), then oscillates between (3,8) and (2,8)
  // White advances south along column 4: row 0 up to row 8
  for(int step = 0; step < 7; step++) {
    // Black moves:
    // step 0: (4,8) -> (3,8)
    // step 1: (3,8) -> (2,8)
    // step 2: (2,8) -> (3,8)...
    int bCol = (step % 2 == 0) ? 3 : 2;
    Loc bLoc = Location::pawnLoc(bCol, 8, 17);
    testAssert(hist.isLegal(b, bLoc, P_BLACK));
    hist.makeBoardMoveAssumeLegal(b, bLoc, P_BLACK, NULL);
    testAssert(!hist.isGameFinished);

    // White moves to (c=4, r = step + 1)
    Loc wLoc = Location::pawnLoc(4, step + 1, 17);
    testAssert(hist.isLegal(b, wLoc, P_WHITE));
    hist.makeBoardMoveAssumeLegal(b, wLoc, P_WHITE, NULL);
    testAssert(!hist.isGameFinished);
  }

  // Black makes move 15 between (2,8) and (3,8):
  Loc bLocLast = Location::pawnLoc(2, 8, 17);
  testAssert(hist.isLegal(b, bLocLast, P_BLACK));
  hist.makeBoardMoveAssumeLegal(b, bLocLast, P_BLACK, NULL);
  testAssert(!hist.isGameFinished);

  // White makes move 16 to (c=4, r=8, y=16):
  Loc winningLoc = Location::pawnLoc(4, 8, 17);
  testAssert(hist.isLegal(b, winningLoc, P_WHITE));
  hist.makeBoardMoveAssumeLegal(b, winningLoc, P_WHITE, NULL);

  testAssert(hist.isGameFinished == true);
  testAssert(hist.winner == P_WHITE);
  testAssert(hist.isNoResult == false);
  testAssert(hist.isResignation == false);
  testAssert(hist.moveHistory.size() == 16);

  cout << "    -> Passed (White reaches y=16 and triggers winner == P_WHITE)!" << endl;
}

// Step 6 - Test 3: Pure rules (no draw condition, 300+ moves) & Transposition Merging
static void testPureRulesNoDrawAndTransposition() {
  cout << "  [Step 6.3] Pure Rules (No Draw Condition & Transposition Merging across 300 moves)..." << endl;

  Board bInitial;
  BoardHistory histInitial(bInitial, P_BLACK);

  Board b;
  BoardHistory hist(b, P_BLACK);

  // Black oscillates between e9 (4,8) and e8 (4,7)
  // White oscillates between e1 (4,0) and e2 (4,1)
  Loc bPosA = Location::pawnLoc(4, 8, 17);
  Loc bPosB = Location::pawnLoc(4, 7, 17);
  Loc wPosA = Location::pawnLoc(4, 0, 17);
  Loc wPosB = Location::pawnLoc(4, 1, 17);

  // 75 cycles of 4 moves = 300 moves total
  for(int cycle = 0; cycle < 75; cycle++) {
    // Move 1: Black to B
    hist.makeBoardMoveAssumeLegal(b, bPosB, P_BLACK, NULL);
    testAssert(hist.isGameFinished == false);
    testAssert(hist.winner == C_EMPTY);
    testAssert(hist.isNoResult == false);

    // Move 2: White to B
    hist.makeBoardMoveAssumeLegal(b, wPosB, P_WHITE, NULL);
    testAssert(hist.isGameFinished == false);
    testAssert(hist.winner == C_EMPTY);
    testAssert(hist.isNoResult == false);

    // Move 3: Black to A
    hist.makeBoardMoveAssumeLegal(b, bPosA, P_BLACK, NULL);
    testAssert(hist.isGameFinished == false);
    testAssert(hist.winner == C_EMPTY);
    testAssert(hist.isNoResult == false);

    // Move 4: White to A
    hist.makeBoardMoveAssumeLegal(b, wPosA, P_WHITE, NULL);
    testAssert(hist.isGameFinished == false);
    testAssert(hist.winner == C_EMPTY);
    testAssert(hist.isNoResult == false);
  }

  // Exactly at move 300:
  testAssert((int)hist.moveHistory.size() == 300);
  testAssert(hist.isGameFinished == false); // NO DRAW in pure Quoridor rules!
  testAssert(hist.isNoResult == false);
  testAssert(hist.winner == C_EMPTY);

  // Pure Markov Transposition Merging Verification:
  // At move 300, pawns are at initial locations and 0 walls placed.
  // Board pos_hash, sitHash, situationRulesAndKoHash, and GraphHash::getStateHash MUST match initial!
  testAssert(b.pos_hash == bInitial.pos_hash);
  testAssert(b.getSitHash(P_BLACK) == bInitial.getSitHash(P_BLACK));
  testAssert(BoardHistory::getSituationRulesAndKoHash(b, hist, P_BLACK, 0.5) ==
             BoardHistory::getSituationRulesAndKoHash(bInitial, histInitial, P_BLACK, 0.5));
  testAssert(GraphHash::getStateHash(hist, P_BLACK, 0.5) ==
             GraphHash::getStateHash(histInitial, P_BLACK, 0.5));

  cout << "    -> Passed (Pure rules: 300 moves without draw, Markov state & transposition hash verified)!" << endl;
}

// Step 6 - Test 4: Resignation & RecentBoards Ring Buffer
static void testResignationAndRecentBoards() {
  cout << "  [Step 6.4] Resignation and RecentBoards ring buffer..." << endl;

  Board b;
  BoardHistory hist(b, P_BLACK);

  // Test recentBoards tracking
  Loc bLoc1 = Location::pawnLoc(4, 7, 17);
  hist.makeBoardMoveAssumeLegal(b, bLoc1, P_BLACK, NULL);
  testAssert(hist.getRecentBoard(0).blackPawnLoc == bLoc1);
  testAssert(hist.getRecentBoard(1).blackPawnLoc == Location::pawnLoc(4, 8, 17));

  // Test resignation
  hist.setWinnerByResignation(P_WHITE);
  testAssert(hist.isGameFinished == true);
  testAssert(hist.winner == P_WHITE);
  testAssert(hist.isResignation == true);
  testAssert(hist.isNoResult == false);

  // Test clear resets state
  Board bFresh;
  hist.clear(bFresh, P_BLACK);
  testAssert(hist.isGameFinished == false);
  testAssert(hist.winner == C_EMPTY);
  testAssert(hist.isResignation == false);
  testAssert(hist.moveHistory.empty());

  cout << "    -> Passed (Resignation, ring buffer and clear verified)!" << endl;
}

// Step 6 - Test 5: Strict vs Tolerant isLegal & Terminal State Reset
static void testBoardHistoryIsLegalStrictAndTolerant() {
  cout << "  [Step 6.5] Strict vs Tolerant isLegal & Terminal State Reset..." << endl;
  Board b;
  BoardHistory hist(b, P_BLACK);

  Loc bMove = Location::pawnLoc(4, 7, 17);
  Loc wMove = Location::pawnLoc(4, 1, 17);

  // Black's turn initially
  testAssert(hist.isLegal(b, bMove, P_BLACK));
  testAssert(!hist.isLegal(b, wMove, P_WHITE));           // Strict: not White's turn
  testAssert(hist.isLegalTolerant(b, wMove, P_WHITE));     // Tolerant: allows White
  testAssert(!hist.isLegalTolerant(b, wMove, C_EMPTY));    // Tolerant: invalid player

  // Resignation triggers terminal state
  hist.setWinnerByResignation(P_WHITE);
  testAssert(hist.isGameFinished);
  testAssert(!hist.isLegal(b, bMove, P_BLACK));            // Strict: game finished
  testAssert(hist.isLegalTolerant(b, bMove, P_BLACK));     // Tolerant: allows exploration

  // Playing a move in tolerant mode resets terminal state
  bool suc = hist.makeBoardMoveTolerant(b, bMove, P_BLACK);
  testAssert(suc);
  testAssert(!hist.isGameFinished);
  testAssert(hist.winner == C_EMPTY);
  testAssert(!hist.isResignation);
  testAssert(hist.presumedNextMovePla == P_WHITE);

  cout << "    -> Passed (Strict/Tolerant distinction and terminal reset verified)!" << endl;
}

// Step 6 - Test 6: Rules parsing and validation
static void testRulesParsingValidation() {
  cout << "  [Step 6.6] Rules Parsing Validation..." << endl;
  Rules r;

  // Valid named rules
  testAssert(Rules::tryParseRules("quoridor", r));
  testAssert(Rules::tryParseRules("DEFAULT", r));
  testAssert(Rules::tryParseRules("tromptaylor", r));

  // Valid JSON rules
  testAssert(Rules::tryParseRules("{}", r));
  testAssert(Rules::tryParseRules("{\"rules\": \"quoridor\"}", r));

  // Invalid inputs
  testAssert(!Rules::tryParseRules("invalid_rules", r));
  testAssert(!Rules::tryParseRules("[1, 2, 3]", r));
  testAssert(!Rules::tryParseRules("123", r));
  testAssert(!Rules::tryParseRules("", r));

  // tryParseRulesWithoutKomi
  testAssert(Rules::tryParseRulesWithoutKomi("quoridor", r, 5.5f));
  testAssert(r.komi == 5.5f);
  testAssert(!Rules::tryParseRulesWithoutKomi("garbage", r, 5.5f));

  // parseRules throws on invalid
  bool threw = false;
  try {
    Rules::parseRules("nonsense");
  }
  catch(const StringError&) {
    threw = true;
  }
  testAssert(threw);

  cout << "    -> Passed (Rules validation and exception handling verified)!" << endl;
}

} // namespace

void Tests::runRulesTests() {
  cout << "=== Running Quoridor Rules, Topology & BoardHistory Tests ===" << endl;
  testGridConstants();
  testPawnCoordinates();
  testHWallCoordinates();
  testVWallCoordinates();
  testDisjointMoveLocations();
  testAdjacentOffsets();
  testInitialBoardState();
  testBlackWinCondition();
  testWhiteWinCondition();
  testPureRulesNoDrawAndTransposition();
  testResignationAndRecentBoards();
  testBoardHistoryIsLegalStrictAndTolerant();
  testRulesParsingValidation();
  cout << "=== All Quoridor Step 1 & Step 6 Tests Passed Successfully! ===" << endl;
}

