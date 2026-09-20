/*
 * testrules.cpp
 * Unit tests for Quoridor C++ Core Engine.
 * Step 1: Topology & Coordinate System.
 */

#include "../tests/tests.h"
#include "../game/board.h"
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

} // namespace

void Tests::runRulesTests() {
  cout << "=== Running Quoridor Step 1 Rules & Topology Tests ===" << endl;
  testGridConstants();
  testPawnCoordinates();
  testHWallCoordinates();
  testVWallCoordinates();
  testDisjointMoveLocations();
  testAdjacentOffsets();
  testInitialBoardState();
  cout << "=== All Quoridor Step 1 Tests Passed Successfully! ===" << endl;
}
