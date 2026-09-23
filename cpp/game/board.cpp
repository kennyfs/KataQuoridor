/*
 * board.cpp
 * Quoridor Board implementation on 17x17 unified grid for KataQuoridor.
 */

#include "../game/board.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "../core/rand.h"

using namespace std;

// STATIC VARS-----------------------------------------------------------------------------
bool Board::IS_ZOBRIST_INITALIZED = false;
Hash128 Board::ZOBRIST_SIZE_X_HASH[MAX_LEN + 1];
Hash128 Board::ZOBRIST_SIZE_Y_HASH[MAX_LEN + 1];
Hash128 Board::ZOBRIST_BOARD_HASH[MAX_ARR_SIZE][NUM_BOARD_COLORS];
Hash128 Board::ZOBRIST_BOARD_HASH2[MAX_ARR_SIZE][NUM_BOARD_COLORS];
Hash128 Board::ZOBRIST_PLAYER_HASH[4];
Hash128 Board::ZOBRIST_FENCENUM_HASH[MAX_FENCE_NUM + 1][2];
const Hash128 Board::ZOBRIST_GAME_IS_OVER = //Based on sha256 hash of Board::ZOBRIST_GAME_IS_OVER
  Hash128(0xb6f9e465597a77eeULL, 0xf1d583d960a4ce7fULL);

// PLAYER IO-------------------------------------------------------------------------------
namespace PlayerIO {
  char colorToChar(Color c) {
    switch(c) {
      case C_EMPTY: return '.';
      case C_BLACK: return 'B';
      case C_WHITE: return 'W';
      case C_FENCE: return '*';
      case C_WALL: return '#';
      default: return '?';
    }
  }

  string playerToStringShort(Player p) {
    switch(p) {
      case P_BLACK: return "B";
      case P_WHITE: return "W";
      case C_EMPTY: return "E";
      default: return "?";
    }
  }

  string playerToString(Player p) {
    switch(p) {
      case P_BLACK: return "Black";
      case P_WHITE: return "White";
      case C_EMPTY: return "Empty";
      default: return "Unknown";
    }
  }

  bool tryParsePlayer(const string& s, Player& pla) {
    string str = Global::toLower(Global::trim(s));
    if(str == "b" || str == "black" || str == "p1" || str == "1") {
      pla = P_BLACK;
      return true;
    }
    if(str == "w" || str == "white" || str == "p2" || str == "2") {
      pla = P_WHITE;
      return true;
    }
    return false;
  }

  Player parsePlayer(const string& s) {
    Player pla = C_EMPTY;
    if(tryParsePlayer(s, pla))
      return pla;
    throw StringError("Could not parse player: " + s);
  }
}

// LOCATION--------------------------------------------------------------------------------
void Location::getAdjacentOffsets(short adj_offsets[8], int x_size) {
  adj_offsets[0] = -(x_size + 1); // North / -Y
  adj_offsets[1] = -1;            // West / -X
  adj_offsets[2] = 1;             // East / +X
  adj_offsets[3] = (x_size + 1);  // South / +Y
  adj_offsets[4] = -(x_size + 1) - 1; // North-West
  adj_offsets[5] = -(x_size + 1) + 1; // North-East
  adj_offsets[6] = (x_size + 1) - 1;  // South-West
  adj_offsets[7] = (x_size + 1) + 1;  // South-East
}

bool Location::isAdjacent(Loc loc0, Loc loc1, int x_size) {
  return loc0 == loc1 - (x_size + 1) || loc0 == loc1 - 1 || loc0 == loc1 + 1 || loc0 == loc1 + (x_size + 1);
}
// [delete in future] for Anti-Mirror Go.
Loc Location::getMirrorLoc(Loc loc, int x_size, int y_size) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  (void)y_size;
  int x = getX(loc, x_size);
  int y = getY(loc, x_size);
  return getLoc(x_size - 1 - x, y, x_size);
}

Loc Location::getCenterLoc(int x_size, int y_size) {
  return getLoc(x_size / 2, y_size / 2, x_size);
}

Loc Location::getCenterLoc(const Board& b) {
  return getCenterLoc(b.x_size, b.y_size);
}

bool Location::isCentral(Loc loc, int x_size, int y_size) {
  int x = getX(loc, x_size);
  int y = getY(loc, x_size);
  return x >= (x_size - 1) / 2 && x <= x_size / 2 && y >= (y_size - 1) / 2 && y <= y_size / 2;
}

bool Location::isNearCentral(Loc loc, int x_size, int y_size) {
  int x = getX(loc, x_size);
  int y = getY(loc, x_size);
  return x >= (x_size - 1) / 2 - 1 && x <= x_size / 2 + 1 && y >= (y_size - 1) / 2 - 1 && y <= y_size / 2 + 1;
}

int Location::distance(Loc loc0, Loc loc1, int x_size) {
  int dx = getX(loc1, x_size) - getX(loc0, x_size);
  int dy = getY(loc1, x_size) - getY(loc0, x_size);
  return (dx >= 0 ? dx : -dx) + (dy >= 0 ? dy : -dy);
}

int Location::euclideanDistanceSquared(Loc loc0, Loc loc1, int x_size) {
  int dx = getX(loc1, x_size) - getX(loc0, x_size);
  int dy = getY(loc1, x_size) - getY(loc0, x_size);
  return dx * dx + dy * dy;
}

string Location::toStringMach(Loc loc, int x_size) {
  if(loc == Board::PASS_LOC)
    return string("pass");
  if(loc == Board::NULL_LOC)
    return string("null");
  char buf[128];
  sprintf(buf, "(%d,%d)", getX(loc, x_size), getY(loc, x_size));
  return string(buf);
}

string Location::toStringMach(Loc loc, const Board& b) {
  return toStringMach(loc, b.x_size);
}

string Location::toString(Loc loc, int x_size, int y_size) {
  if(loc == Board::PASS_LOC)
    return string("pass");
  if(loc == Board::NULL_LOC)
    return string("null");

  int x = getX(loc, x_size);
  int y = getY(loc, x_size);
  if(x < 0 || x >= x_size || y < 0 || y >= y_size)
    return toStringMach(loc, x_size);

  // If even x, even y -> Pawn destination in algebraic notation (e.g. e8, e1)
  if(x % 2 == 0 && y % 2 == 0) {
    char col = (char)('a' + (x / 2));
    char row = (char)('1' + (y / 2));
    string s;
    s += col;
    s += row;
    return s;
  }
  // If odd x, odd y -> Horizontal wall (e.g. e2h)
  else if(x % 2 == 1 && y % 2 == 1) {
    char col = (char)('a' + (x / 2));
    char row = (char)('1' + (y / 2));
    string s;
    s += col;
    s += row;
    s += 'h';
    return s;
  }
  // If odd x, even y -> Vertical wall (e.g. e2v), upper arm is (2c+1, 2r)
  else if(x % 2 == 1 && y % 2 == 0) {
    char col = (char)('a' + (x / 2));
    char row = (char)('1' + (y / 2));
    string s;
    s += col;
    s += row;
    s += 'v';
    return s;
  }

  return toStringMach(loc, x_size);
}

string Location::toString(Loc loc, const Board& b) {
  return toString(loc, b.x_size, b.y_size);
}

static bool tryParseLetterCoordinate(char c, int& x) {
  if(c >= 'A' && c <= 'H')
    x = c - 'A';
  else if(c >= 'a' && c <= 'h')
    x = c - 'a';
  else if(c >= 'J' && c <= 'Z')
    x = c - 'A' - 1;
  else if(c >= 'j' && c <= 'z')
    x = c - 'a' - 1;
  else
    return false;
  return true;
}

bool Location::tryOfString(const string& str, int x_size, int y_size, Loc& result) {
  string s = Global::trim(str);
  if(s.length() < 2)
    return false;
  if(Global::isEqualCaseInsensitive(s, string("pass")) || Global::isEqualCaseInsensitive(s, string("pss"))) {
    result = Board::PASS_LOC;
    return true;
  }

  bool hasMovePrefix = false;
  bool hasWallPrefix = false;

  if(Global::isPrefix(s, "move ") || Global::isPrefix(s, "MOVE ") || Global::isPrefix(s, "Move ")) {
    hasMovePrefix = true;
    s = Global::trim(s.substr(5));
  }
  else if(Global::isPrefix(s, "wall ") || Global::isPrefix(s, "WALL ") || Global::isPrefix(s, "Wall ")) {
    hasWallPrefix = true;
    string rest = Global::trim(s.substr(5));
    vector<string> parts = Global::split(rest, ' ');
    if(parts.size() == 2)
      s = parts[0] + parts[1];
    else
      s = rest;
  }

  if(s.length() < 2)
    return false;

  if(s[0] == '(') {
    if(s[s.length() - 1] != ')')
      return false;
    s = s.substr(1, s.length() - 2);
    vector<string> pieces = Global::split(s, ',');
    if(pieces.size() != 2)
      return false;
    int x;
    int y;
    bool sucX = Global::tryStringToInt(pieces[0], x);
    bool sucY = Global::tryStringToInt(pieces[1], y);
    if(!sucX || !sucY)
      return false;
    result = Location::getLoc(x, y, x_size);
    return true;
  }

  // Quoridor algebraic notation checks:
  // e.g. "e8" or "E8" -> pawn move (len 2, col in a..i, row in 1..9)
  char colChar = (char)tolower(s[0]);
  if(s.length() == 2 && isalpha(s[0]) && isdigit(s[1])) {
    if(hasWallPrefix) // Mutually exclusive: "wall e8" is illegal
      return false;
    if(colChar >= 'a' && colChar <= 'i' && s[1] >= '1' && s[1] <= '9') {
      int c = colChar - 'a';
      int r = s[1] - '1';
      result = Location::pawnLoc(c, r, x_size);
      return true;
    }
    return false;
  }

  // e.g. "e2h" or "e2v" -> wall placement (len 3, col a..h, row 1..8, 'h' or 'v')
  if(s.length() == 3 && isalpha(s[0]) && isdigit(s[1])) {
    if(hasMovePrefix) // Mutually exclusive: "move e2v" is illegal
      return false;
    char ori = (char)tolower(s[2]);
    if(ori == 'h' || ori == 'v') {
      if(colChar >= 'a' && colChar <= 'h' && s[1] >= '1' && s[1] <= '8') {
        int c = colChar - 'a';
        int r = s[1] - '1';
        if(ori == 'h') {
          result = Location::hWallLoc(c, r, x_size);
          return true;
        }
        else if(ori == 'v') {
          // Pure geometric upper arm Loc (2*c+1, 2*r) - identical for both players
          result = Location::vWallLoc(c, r, x_size);
          return true;
        }
      }
      return false;
    }
  }

  if(hasMovePrefix || hasWallPrefix)
    return false;

  // Standard KataGo letter coordinate (A1, E9, Q17)
  int x;
  if(!tryParseLetterCoordinate(s[0], x))
    return false;

  int y;
  bool sucY = Global::tryStringToInt(s.substr(1), y);
  if(!sucY)
    return false;
  y = y_size - y;

  if(x < 0 || x >= x_size || y < 0 || y >= y_size)
    return false;

  result = Location::getLoc(x, y, x_size);
  return true;
}

bool Location::tryOfString(const string& str, const Board& b, Loc& result) {
  return tryOfString(str, b.x_size, b.y_size, result);
}

bool Location::tryOfString(const string& str, const Board& b, Player pla, Loc& result) {
  (void)pla;
  return tryOfString(str, b.x_size, b.y_size, result);
}

bool Location::tryOfStringAllowNull(const string& str, int x_size, int y_size, Loc& result) {
  if(Global::isEqualCaseInsensitive(str, string("null"))) {
    result = Board::NULL_LOC;
    return true;
  }
  return tryOfString(str, x_size, y_size, result);
}

bool Location::tryOfStringAllowNull(const string& str, const Board& b, Loc& result) {
  return tryOfStringAllowNull(str, b.x_size, b.y_size, result);
}

Loc Location::ofString(const string& str, int x_size, int y_size) {
  Loc result;
  if(tryOfString(str, x_size, y_size, result))
    return result;
  throw StringError("Could not parse board location: " + str);
}

Loc Location::ofString(const string& str, const Board& b) {
  Loc result;
  if(tryOfString(str, b, result))
    return result;
  throw StringError("Could not parse board location: " + str);
}

Loc Location::ofStringAllowNull(const string& str, int x_size, int y_size) {
  Loc result;
  if(tryOfStringAllowNull(str, x_size, y_size, result))
    return result;
  throw StringError("Could not parse board location: " + str);
}

Loc Location::ofStringAllowNull(const string& str, const Board& b) {
  Loc result;
  if(tryOfStringAllowNull(str, b, result))
    return result;
  throw StringError("Could not parse board location: " + str);
}

vector<Loc> Location::parseSequence(const string& str, const Board& b) {
  vector<string> pieces = Global::split(Global::trim(str), ' ');
  vector<Loc> locs;
  for(size_t i = 0; i < pieces.size(); i++) {
    string p = Global::trim(pieces[i]);
    if(p.length() > 0)
      locs.push_back(Location::ofString(p, b));
  }
  return locs;
}

// CONSTRUCTORS AND INITIALIZATION----------------------------------------------------------
Board::Board() {
  init(DEFAULT_LEN, DEFAULT_LEN);
}

Board::Board(int x, int y) {
  init(x, y);
}

Board::Board(const Board& other) {
  x_size = other.x_size;
  y_size = other.y_size;
  memcpy(colors, other.colors, sizeof(Color) * MAX_ARR_SIZE);
  blackFences = other.blackFences;
  whiteFences = other.whiteFences;
  blackPawnLoc = other.blackPawnLoc;
  whitePawnLoc = other.whitePawnLoc;
  movenum = other.movenum;
  pos_hash = other.pos_hash;
  memcpy(adj_offsets, other.adj_offsets, sizeof(short) * 8);
  nextPla = other.nextPla;
  ko_loc = other.ko_loc;
  numBlackCaptures = other.numBlackCaptures;
  numWhiteCaptures = other.numWhiteCaptures;
  cachedPathP1 = other.cachedPathP1;
  cachedPathP2 = other.cachedPathP2;
}

void Board::init(int xS, int yS) {
  assert(IS_ZOBRIST_INITALIZED);
  if(xS < 0 || yS < 0 || xS > MAX_LEN || yS > MAX_LEN)
    throw StringError("Board::init - invalid board size");

  x_size = xS;
  y_size = yS;

  for(int i = 0; i < MAX_ARR_SIZE; i++) {
    colors[i] = C_WALL;
    chain_head[i] = i;
    next_in_chain[i] = i;
  }

  movenum = 0;

  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      Loc loc = Location::getLoc(x, y, x_size);
      colors[loc] = C_EMPTY;
    }
  }

  Location::getAdjacentOffsets(adj_offsets, x_size);

  nextPla = P_BLACK;
  blackFences = MAX_FENCE_NUM;
  whiteFences = MAX_FENCE_NUM;
  ko_loc = NULL_LOC;
  numBlackCaptures = 0;
  numWhiteCaptures = 0;

  // Pawn initial positions:
  // Black (P1) starts at e9: x = 4 -> 2*4 = 8, y = 8 -> 2*8 = 16
  // White (P2) starts at e1: x = 4 -> 2*4 = 8, y = 0 -> 2*0 = 0
  int midX = (x_size - 1) / 2; // 8
  whitePawnLoc = Location::getLoc(midX, 0, x_size);
  colors[whitePawnLoc] = C_WHITE;

  blackPawnLoc = Location::getLoc(midX, y_size - 1, x_size);
  colors[blackPawnLoc] = C_BLACK;

  // Compute initial hash
  pos_hash = ZOBRIST_SIZE_X_HASH[x_size] ^ ZOBRIST_SIZE_Y_HASH[y_size];
  pos_hash ^= ZOBRIST_BOARD_HASH[whitePawnLoc][C_WHITE];
  pos_hash ^= ZOBRIST_BOARD_HASH[blackPawnLoc][C_BLACK];
  pos_hash ^= ZOBRIST_FENCENUM_HASH[blackFences][0];
  pos_hash ^= ZOBRIST_FENCENUM_HASH[whiteFences][1];

  // Initial cached paths
  cachedPathP1.clear();
  bfsReachable(blackPawnLoc, 0, &cachedPathP1);

  cachedPathP2.clear();
  bfsReachable(whitePawnLoc, y_size - 1, &cachedPathP2);
}

void Board::initHash() {
  if(IS_ZOBRIST_INITALIZED)
    return;
  Rand rand("Board::initHash() for KataQuoridor");

  auto nextHash = [&rand]() {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    return Hash128(h0, h1);
  };

  for(int i = 0; i < 4; i++)
    ZOBRIST_PLAYER_HASH[i] = nextHash();

  for(int i = 0; i < MAX_ARR_SIZE; i++) {
    for(Color j = 0; j < NUM_BOARD_COLORS; j++) {
      if(j == C_EMPTY || j == C_WALL) {
        ZOBRIST_BOARD_HASH[i][j] = Hash128();
      }
      else {
        ZOBRIST_BOARD_HASH[i][j] = nextHash();
      }
    }
  }

  rand.init("Board::initHash() for ZOBRIST_FENCENUM_HASH hashes");
  for(int i = 0; i < MAX_FENCE_NUM + 1; i++) {
    ZOBRIST_FENCENUM_HASH[i][0] = nextHash();
    ZOBRIST_FENCENUM_HASH[i][1] = nextHash();
  }

  rand.init("Board::initHash() for ZOBRIST_SIZE hashes");
  for(int i = 0; i < MAX_LEN + 1; i++) {
    ZOBRIST_SIZE_X_HASH[i] = nextHash();
    ZOBRIST_SIZE_Y_HASH[i] = nextHash();
  }

  //Reseed and compute one more set of zobrist hashes, mixed a bit differently
  rand.init("Board::initHash() for second set of ZOBRIST hashes");
  for(int i = 0; i<MAX_ARR_SIZE; i++) {
    for(Color j = 0; j< NUM_BOARD_COLORS; j++) {
      ZOBRIST_BOARD_HASH2[i][j] = nextHash();
      ZOBRIST_BOARD_HASH2[i][j].hash0 = Hash::murmurMix(ZOBRIST_BOARD_HASH2[i][j].hash0);
      ZOBRIST_BOARD_HASH2[i][j].hash1 = Hash::splitMix64(ZOBRIST_BOARD_HASH2[i][j].hash1);
    }
  }

  IS_ZOBRIST_INITALIZED = true;
}

bool Board::isOnBoard(Loc loc) const {
  return loc >= 0 && loc < MAX_ARR_SIZE && colors[loc] != C_WALL;
}

bool Board::isOnBoardPawn(Loc loc) const {
  if (!(loc >= 0 && loc < MAX_ARR_SIZE && colors[loc] != C_WALL)) return false;
  int x = Location::getX(loc, x_size);
  int y = Location::getY(loc, x_size);
  return (x % 2 == 0 && y % 2 == 0);
}

bool Board::isOnBoardFence(Loc loc) const {
  if (!(loc >= 0 && loc < MAX_ARR_SIZE && colors[loc] != C_WALL)) return false;
  int x = Location::getX(loc, x_size);
  int y = Location::getY(loc, x_size);
  return (x % 2 == 1 && y % 2 == 1) || (x % 2 == 1 && y % 2 == 0);
}

bool Board::isEmpty() const {
  return numStonesOnBoard() == 0;
}

int Board::numStonesOnBoard() const {
  int count = 0;
  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      Loc loc = Location::getLoc(x, y, x_size);
      if(colors[loc] == C_BLACK || colors[loc] == C_WHITE)
        count++;
    }
  }
  return count;
}

int Board::numPlaStonesOnBoard(Player pla) const {
  return (pla == P_BLACK || pla == P_WHITE) ? 1 : 0;
}

bool Board::setStone(Loc loc, Color color) {
  if(!isOnBoard(loc))
    return false;
  Color oldColor = colors[loc];
  if(oldColor == color)
    return true;
  colors[loc] = color;
  pos_hash ^= ZOBRIST_BOARD_HASH[loc][oldColor];
  pos_hash ^= ZOBRIST_BOARD_HASH[loc][color];
  return true;
}

bool Board::setStones(const vector<Move>& placements) {
  for(const Move& m : placements) {
    if(!setStone(m.loc, m.pla))
      return false;
  }
  return true;
}

void Board::placeFence(Loc center, bool isVertical) {
  auto placeOne = [this](Loc loc) {
    colors[loc] = C_FENCE;
    // hash for empty is 0 so no need to XOR it
    pos_hash ^= ZOBRIST_BOARD_HASH[loc][C_FENCE];
  };
  placeOne(center);
  if(isVertical) {
    placeOne(center + adj_offsets[0]);
    placeOne(center + adj_offsets[3]);
  }
  else {
    placeOne(center + adj_offsets[1]);
    placeOne(center + adj_offsets[2]);
  }
}

void Board::removeFence(Loc center, bool isVertical) {
  auto removeOne = [this](Loc loc) {
    colors[loc] = C_EMPTY;
    pos_hash ^= ZOBRIST_BOARD_HASH[loc][C_FENCE];
  };
  removeOne(center);
  if(isVertical) {
    removeOne(center + adj_offsets[0]);
    removeOne(center + adj_offsets[3]);
  }
  else {
    removeOne(center + adj_offsets[1]);
    removeOne(center + adj_offsets[2]);
  }
}

// Check if a pawn can step directly from pawn cell `from` to adjacent pawn cell `to`
bool Board::canPawnStep(Loc from, Loc to) const {
  if(!isOnBoardPawn(from) || !isOnBoardPawn(to))
    return false;
  int x0 = Location::getX(from, x_size);
  int y0 = Location::getY(from, x_size);
  int x1 = Location::getX(to, x_size);
  int y1 = Location::getY(to, x_size);
  if(x0 % 2 != 0 || y0 % 2 != 0 || x1 % 2 != 0 || y1 % 2 != 0)
    return false;

  int dx = x1 - x0;
  int dy = y1 - y0;
  if(!((abs(dx) == 2 && dy == 0) || (dx == 0 && abs(dy) == 2)))
    return false;

  Loc mid = Location::getLoc(x0 + dx / 2, y0 + dy / 2, x_size);
  return colors[mid] != C_FENCE;
}

bool Board::isLegalPawnMove(Loc loc, Player pla) const {
  if(!isOnBoardPawn(loc) || colors[loc] != C_EMPTY)
    return false;
  Loc myLoc = (pla == P_BLACK) ? blackPawnLoc : whitePawnLoc;
  Loc oppLoc = (pla == P_BLACK) ? whitePawnLoc : blackPawnLoc;

  int myX = Location::getX(myLoc, x_size);
  int myY = Location::getY(myLoc, x_size);
  int targetX = Location::getX(loc, x_size);
  int targetY = Location::getY(loc, x_size);

  int dx = targetX - myX;
  int dy = targetY - myY;
  int dist = abs(dx) + abs(dy);
  if(dist != 2 && dist != 4)
    return false;

  // Direct step of 1 cell (distance 2 on 17x17 grid)
  if(dist == 2) {
    if(loc == oppLoc)
      return false;
    return canPawnStep(myLoc, loc);
  }

  // dist == 4: Jump over opponent (either straight or diagonal)
  const int dirs[4][2] = {{0, -2}, {-2, 0}, {2, 0}, {0, 2}};
  for(int d = 0; d < 4; d++) {
    int nx = myX + dirs[d][0];
    int ny = myY + dirs[d][1];
    if(nx < 0 || nx >= x_size || ny < 0 || ny >= y_size)
      continue;
    Loc step1 = Location::getLoc(nx, ny, x_size);
    if(step1 != oppLoc || !canPawnStep(myLoc, step1))
      continue;

    // Opponent pawn is adjacent! Check straight jump
    int jx = nx + dirs[d][0];
    int jy = ny + dirs[d][1];
    bool canStraight = false;
    if(jx >= 0 && jx < x_size && jy >= 0 && jy < y_size) {
      Loc step2 = Location::getLoc(jx, jy, x_size);
      if(canPawnStep(oppLoc, step2) && colors[step2] == C_EMPTY) {
        if(step2 == loc)
          return true;
        canStraight = true;
      }
    }
    // If straight jump blocked by wall or board edge, diagonal jumps allowed
    if(!canStraight) {
      int d1 = (d == 0 || d == 3) ? 1 : 0;
      int d2 = (d == 0 || d == 3) ? 2 : 3;
      int perpDirs[2] = {d1, d2};
      for(int i = 0; i < 2; i++) {
        int pd = perpDirs[i];
        int diagX = nx + dirs[pd][0];
        int diagY = ny + dirs[pd][1];
        if(diagX >= 0 && diagX < x_size && diagY >= 0 && diagY < y_size) {
          Loc diagLoc = Location::getLoc(diagX, diagY, x_size);
          if(diagLoc == loc && canPawnStep(oppLoc, diagLoc) && colors[diagLoc] == C_EMPTY)
            return true;
        }
      }
    }
  }

  return false;
}

vector<Loc> Board::getLegalPawnDestinations(Player pla) const {
  vector<Loc> moves;
  moves.reserve(2);
  Loc myLoc = (pla == P_BLACK) ? blackPawnLoc : whitePawnLoc;
  Loc oppLoc = (pla == P_BLACK) ? whitePawnLoc : blackPawnLoc;

  int myX = Location::getX(myLoc, x_size);
  int myY = Location::getY(myLoc, x_size);

  const int dirs[4][2] = {{0, -2}, {-2, 0}, {2, 0}, {0, 2}};

  for(int d = 0; d < 4; d++) {
    int nx = myX + dirs[d][0];
    int ny = myY + dirs[d][1];
    if(nx < 0 || nx >= x_size || ny < 0 || ny >= y_size)
      continue;
    Loc step1 = Location::getLoc(nx, ny, x_size);
    if(!canPawnStep(myLoc, step1))
      continue;

    if(step1 != oppLoc) {
      if(colors[step1] == C_EMPTY)
        moves.push_back(step1);
    }
    else {
      // Opponent pawn is adjacent! Check straight jump
      int jx = nx + dirs[d][0];
      int jy = ny + dirs[d][1];
      bool canStraight = false;
      if(jx >= 0 && jx < x_size && jy >= 0 && jy < y_size) {
        Loc step2 = Location::getLoc(jx, jy, x_size);
        if(canPawnStep(oppLoc, step2) && colors[step2] == C_EMPTY) {
          moves.push_back(step2);
          canStraight = true;
        }
      }
      // If straight jump blocked by wall or board edge, diagonal jumps allowed
      if(!canStraight) {
        int d1 = (d == 0 || d == 3) ? 1 : 0;
        int d2 = (d == 0 || d == 3) ? 2 : 3;
        int perpDirs[2] = {d1, d2};
        for(int i = 0; i < 2; i++) {
          int pd = perpDirs[i];
          int dx = nx + dirs[pd][0];
          int dy = ny + dirs[pd][1];
          if(dx >= 0 && dx < x_size && dy >= 0 && dy < y_size) {
            Loc diagLoc = Location::getLoc(dx, dy, x_size);
            if(canPawnStep(oppLoc, diagLoc) && colors[diagLoc] == C_EMPTY)
              moves.push_back(diagLoc);
          }
        }
      }
    }
  }

  return moves;
}

bool Board::bfsReachable(Loc start, int targetY, vector<Loc>* outPath) const {
  if(!isOnBoardPawn(start))
    return false;

  bool visited[MAX_ARR_SIZE];
  memset(visited, 0, sizeof(visited));
  Loc parent[MAX_ARR_SIZE];
  if(outPath != nullptr)
    memset(parent, 0, sizeof(parent));

  Loc q[81];
  int qHead = 0;
  int qTail = 0;

  q[qTail++] = start;
  visited[start] = true;

  Loc goalFound = NULL_LOC;
  const int dirs[4][2] = {{0, -2}, {-2, 0}, {2, 0}, {0, 2}};

  while(qHead < qTail) {
    Loc curr = q[qHead++];

    int cy = Location::getY(curr, x_size);
    if(cy == targetY) {
      goalFound = curr;
      break;
    }

    int cx = Location::getX(curr, x_size);
    for(int d = 0; d < 4; d++) {
      int nx = cx + dirs[d][0];
      int ny = cy + dirs[d][1];
      if(nx < 0 || nx >= x_size || ny < 0 || ny >= y_size)
        continue;
      Loc next = Location::getLoc(nx, ny, x_size);
      if(!visited[next] && canPawnStep(curr, next)) {
        visited[next] = true;
        if(outPath != nullptr)
          parent[next] = curr;
        q[qTail++] = next;
      }
    }
  }

  if(goalFound == NULL_LOC)
    return false;

  if(outPath != nullptr) {
    outPath->clear();
    outPath->reserve(81);
    Loc curr = goalFound;
    while(curr != NULL_LOC) {
      outPath->push_back(curr);
      curr = parent[curr];
    }
    reverse(outPath->begin(), outPath->end());
  }

  return true;
}

static bool pathCrossesWall(const vector<Loc>& path, Loc arm1, Loc arm2) {
  if(path.size() < 2) return false;
  for(size_t i = 0; i + 1 < path.size(); i++) {
    Loc edgeMid = (path[i] + path[i+1]) >> 1;
    if(edgeMid == arm1 || edgeMid == arm2)
      return true;
  }
  return false;
}

bool Board::checkNoFullBlockLazy(int c, int r, bool isVertical) const {
  Loc center = Location::getLoc(2 * c + 1, 2 * r + 1, x_size);
  Loc arm1 = isVertical ? (center + adj_offsets[0]) : (center + adj_offsets[1]);
  Loc arm2 = isVertical ? (center + adj_offsets[3]) : (center + adj_offsets[2]);

  bool cutP1 = pathCrossesWall(cachedPathP1, arm1, arm2);
  bool cutP2 = pathCrossesWall(cachedPathP2, arm1, arm2);

  if(!cutP1 && !cutP2)
    return true;

  Board* mutableThis = const_cast<Board*>(this);
  mutableThis->placeFence(center, isVertical);

  vector<Loc> newPathP1;
  vector<Loc> newPathP2;
  bool legal = bfsReachable(blackPawnLoc, 0, cutP1 ? &newPathP1 : nullptr) &&
               bfsReachable(whitePawnLoc, y_size - 1, cutP2 ? &newPathP2 : nullptr);

  mutableThis->removeFence(center, isVertical);

  if(legal) {
    if(cutP1) cachedPathP1 = newPathP1;
    if(cutP2) cachedPathP2 = newPathP2;
  }

  return legal;
}

bool Board::isLegalWallPlacement(int c, int r, bool isVertical, Player pla) const {
  int fencesLeft = (pla == P_BLACK) ? blackFences : whiteFences;
  if(fencesLeft <= 0)
    return false;
  if(c < 0 || c >= 8 || r < 0 || r >= 8)
    return false;

  Loc center = Location::getLoc(2 * c + 1, 2 * r + 1, x_size);
  if(colors[center] != C_EMPTY)
    return false;

  if(isVertical) {
    Loc top = center + adj_offsets[0];
    Loc bot = center + adj_offsets[3];
    if(colors[top] != C_EMPTY || colors[bot] != C_EMPTY)
      return false;
  }
  else {
    Loc left = center + adj_offsets[1];
    Loc right = center + adj_offsets[2];
    if(colors[left] != C_EMPTY || colors[right] != C_EMPTY)
      return false;
  }

  return checkNoFullBlockLazy(c, r, isVertical);
}

bool Board::isLegal(Loc loc, Player pla, bool isMultiStoneSuicideLegal) const {
  (void)isMultiStoneSuicideLegal;
  if(loc == PASS_LOC || loc == NULL_LOC)
    return false;
  if(!isOnBoard(loc))
    return false;
  if(colors[loc] != C_EMPTY)
    return false;

  int x = Location::getX(loc, x_size);
  int y = Location::getY(loc, x_size);

  // Case 1: Pawn move at even x, even y
  if(x % 2 == 0 && y % 2 == 0) {
    return isLegalPawnMove(loc, pla);
  }
  // Case 2: Horizontal fence at odd x, odd y
  if(x % 2 == 1 && y % 2 == 1) {
    int c = x / 2;
    int r = y / 2;
    return isLegalWallPlacement(c, r, false, pla);
  }
  // Case 3: Vertical fence at odd x, even y (upper arm is 2c+1, 2r)
  if(x % 2 == 1 && y % 2 == 0) {
    int c = x / 2;
    int r = y / 2;
    return isLegalWallPlacement(c, r, true, pla);
  }

  return false;
}

void Board::playMoveAssumeLegal(Loc loc, Player pla) {
  movenum++;

  int x = Location::getX(loc, x_size);
  int y = Location::getY(loc, x_size);

  if(x % 2 == 0 && y % 2 == 0) {
    // Pawn move
    if(pla == P_BLACK) {
      setStone(blackPawnLoc, C_EMPTY);
      setStone(loc, C_BLACK);
      blackPawnLoc = loc;
      cachedPathP1.clear();
      bfsReachable(blackPawnLoc, 0, &cachedPathP1);
    }
    else {
      setStone(whitePawnLoc, C_EMPTY);
      setStone(loc, C_WHITE);
      whitePawnLoc = loc;
      cachedPathP2.clear();
      bfsReachable(whitePawnLoc, y_size - 1, &cachedPathP2);
    }
  }
  else if(x % 2 == 1 && y % 2 == 1) {
    // Horizontal fence
    placeFence(loc, false);
    if(pla == P_BLACK) {
      pos_hash ^= ZOBRIST_FENCENUM_HASH[blackFences][0];
      blackFences--;
      pos_hash ^= ZOBRIST_FENCENUM_HASH[blackFences][0];
    }
    else {
      pos_hash ^= ZOBRIST_FENCENUM_HASH[whiteFences][1];
      whiteFences--;
      pos_hash ^= ZOBRIST_FENCENUM_HASH[whiteFences][1];
    }
    // Update cached paths if affected
    Loc arm1 = loc + adj_offsets[1];
    Loc arm2 = loc + adj_offsets[2];
    if(pathCrossesWall(cachedPathP1, arm1, arm2)) {
      cachedPathP1.clear();
      bfsReachable(blackPawnLoc, 0, &cachedPathP1);
    }
    if(pathCrossesWall(cachedPathP2, arm1, arm2)) {
      cachedPathP2.clear();
      bfsReachable(whitePawnLoc, y_size - 1, &cachedPathP2);
    }
  }
  else if(x % 2 == 1 && y % 2 == 0) {
    // Vertical fence: upper arm is loc, center is loc + adj_offsets[3] (down 1)
    Loc center = loc + adj_offsets[3];
    placeFence(center, true);
    if(pla == P_BLACK) {
      pos_hash ^= ZOBRIST_FENCENUM_HASH[blackFences][0];
      blackFences--;
      pos_hash ^= ZOBRIST_FENCENUM_HASH[blackFences][0];
    }
    else {
      pos_hash ^= ZOBRIST_FENCENUM_HASH[whiteFences][1];
      whiteFences--;
      pos_hash ^= ZOBRIST_FENCENUM_HASH[whiteFences][1];
    }
    // Update cached paths if affected
    Loc top = center + adj_offsets[0];
    Loc bot = center + adj_offsets[3];
    if(pathCrossesWall(cachedPathP1, top, bot)) {
      cachedPathP1.clear();
      bfsReachable(blackPawnLoc, 0, &cachedPathP1);
    }
    if(pathCrossesWall(cachedPathP2, top, bot)) {
      cachedPathP2.clear();
      bfsReachable(whitePawnLoc, y_size - 1, &cachedPathP2);
    }
  }

  // Switch player
  nextPla = getOpp(nextPla);
}

bool Board::playMove(Loc loc, Player pla, bool isMultiStoneSuicideLegal) {
  if(!isLegal(loc, pla, isMultiStoneSuicideLegal))
    return false;
  playMoveAssumeLegal(loc, pla);
  return true;
}

Board::MoveRecord Board::playMoveRecorded(Loc loc, Player pla) {
  MoveRecord record;
  record.pla = pla;
  record.loc = loc;
  record.oldBlackPawnLoc = blackPawnLoc;
  record.oldWhitePawnLoc = whitePawnLoc;
  record.oldBlackFences = blackFences;
  record.oldWhiteFences = whiteFences;
  record.oldMovenum = movenum;
  record.oldNextPla = nextPla;
  record.oldPosHash = pos_hash;
  record.numModifiedCells = 0;

  int x = Location::getX(loc, x_size);
  int y = Location::getY(loc, x_size);

  if(x % 2 == 1 && y % 2 == 1) {
    // Horizontal fence: center is loc
    record.modifiedCells[0] = loc;
    record.modifiedCells[1] = loc + adj_offsets[1];
    record.modifiedCells[2] = loc + adj_offsets[2];
    record.numModifiedCells = 3;
  }
  else if(x % 2 == 1 && y % 2 == 0) {
    // Vertical fence: upper arm is loc, center is loc + adj_offsets[3]
    Loc center = loc + adj_offsets[3];
    record.modifiedCells[0] = center;
    record.modifiedCells[1] = center + adj_offsets[0];
    record.modifiedCells[2] = center + adj_offsets[3];
    record.numModifiedCells = 3;
  }

  playMoveAssumeLegal(loc, pla);
  return record;
}

void Board::undo(MoveRecord record) {
  movenum = record.oldMovenum;
  nextPla = record.oldNextPla;
  pos_hash = record.oldPosHash;

  int x = Location::getX(record.loc, x_size);
  int y = Location::getY(record.loc, x_size);

  if(x % 2 == 0 && y % 2 == 0) {
    // Revert pawn move
    // Note: Opponent's cached path remains valid because BFS pathfinding only
    // checks for fence blocking (via canPawnStep) and is unaffected by pawn movements.
    if(record.pla == P_BLACK) {
      colors[blackPawnLoc] = C_EMPTY;
      blackPawnLoc = record.oldBlackPawnLoc;
      colors[blackPawnLoc] = C_BLACK;
      cachedPathP1.clear();
      bfsReachable(blackPawnLoc, 0, &cachedPathP1);
    }
    else {
      colors[whitePawnLoc] = C_EMPTY;
      whitePawnLoc = record.oldWhitePawnLoc;
      colors[whitePawnLoc] = C_WHITE;
      cachedPathP2.clear();
      bfsReachable(whitePawnLoc, y_size - 1, &cachedPathP2);
    }
  }
  else {
    // Revert fence placement
    for(int i = 0; i < record.numModifiedCells; i++)
      colors[record.modifiedCells[i]] = C_EMPTY;
    blackFences = record.oldBlackFences;
    whiteFences = record.oldWhiteFences;
    cachedPathP1.clear();
    bfsReachable(blackPawnLoc, 0, &cachedPathP1);
    cachedPathP2.clear();
    bfsReachable(whitePawnLoc, y_size - 1, &cachedPathP2);
  }
}

int Board::getShortestPathDistance(Player pla) const {
  Loc start = (pla == P_BLACK) ? blackPawnLoc : whitePawnLoc;
  int targetY = (pla == P_BLACK) ? 0 : (y_size - 1);
  vector<Loc> path;
  if(bfsReachable(start, targetY, &path))
    return (int)path.size() - 1;
  return -1;
}

vector<Loc> Board::findShortestPath(Player pla) const {
  Loc start = (pla == P_BLACK) ? blackPawnLoc : whitePawnLoc;
  int targetY = (pla == P_BLACK) ? 0 : (y_size - 1);
  vector<Loc> path;
  bfsReachable(start, targetY, &path);
  return path;
}

void Board::calDistMap(Player pla, int32_t* res) const {
  for(int i = 0; i < 81; i++)
    res[i] = -1;

  int targetY = (pla == P_BLACK) ? 0 : (y_size - 1);

  Loc q[81];
  int qHead = 0;
  int qTail = 0;
  bool visited[MAX_ARR_SIZE];
  memset(visited, 0, sizeof(visited));

  for(int c = 0; c < 9; c++) {
    Loc goal = Location::getLoc(2 * c, targetY, x_size);
    q[qTail++] = goal;
    visited[goal] = true;
    res[c * 9 + targetY / 2] = 0;
  }

  const int dirs[4][2] = {{0, -2}, {-2, 0}, {2, 0}, {0, 2}};

  while(qHead < qTail) {
    Loc curr = q[qHead++];

    int cx = Location::getX(curr, x_size);
    int cy = Location::getY(curr, x_size);
    int curDist = res[(cx / 2) * 9 + (cy / 2)];

    for(int d = 0; d < 4; d++) {
      int nx = cx + dirs[d][0];
      int ny = cy + dirs[d][1];
      if(nx < 0 || nx >= x_size || ny < 0 || ny >= y_size)
        continue;
      Loc next = Location::getLoc(nx, ny, x_size);
      if(!visited[next] && canPawnStep(curr, next)) {
        visited[next] = true;
        res[(nx / 2) * 9 + (ny / 2)] = curDist + 1;
        q[qTail++] = next;
      }
    }
  }
}

bool Board::isBoardNotConnected() const {
  return !bfsReachable(blackPawnLoc, 0) || !bfsReachable(whitePawnLoc, y_size - 1);
}

Player Board::nextnextPla() const {
  return nextPla;
}

Player Board::prevPla() const {
  return getOpp(nextPla);
}

Hash128 Board::getSitHash(Player pla) const {
  return pos_hash ^ ZOBRIST_PLAYER_HASH[pla];
}

Hash128 Board::getPosHashAfterMove(Loc loc, Player pla) const {
  Board copy = *this;
  copy.playMove(loc, pla);
  return copy.pos_hash;
}

Board Board::getMirroredX() const {
  Board b(x_size, y_size);
  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      Loc src = Location::getLoc(x, y, x_size);
      Loc dst = Location::getLoc(x_size - 1 - x, y, x_size);
      b.colors[dst] = colors[src];
    }
  }
  b.blackPawnLoc = Location::getMirrorLoc(blackPawnLoc, x_size, y_size);
  b.whitePawnLoc = Location::getMirrorLoc(whitePawnLoc, x_size, y_size);
  b.blackFences = blackFences;
  b.whiteFences = whiteFences;
  b.movenum = movenum;
  b.nextPla = nextPla;
  b.cachedPathP1.clear();
  b.bfsReachable(b.blackPawnLoc, 0, &b.cachedPathP1);
  b.cachedPathP2.clear();
  b.bfsReachable(b.whitePawnLoc, y_size - 1, &b.cachedPathP2);

  b.pos_hash = ZOBRIST_SIZE_X_HASH[b.x_size] ^ ZOBRIST_SIZE_Y_HASH[b.y_size];
  b.pos_hash ^= ZOBRIST_BOARD_HASH[b.whitePawnLoc][C_WHITE];
  b.pos_hash ^= ZOBRIST_BOARD_HASH[b.blackPawnLoc][C_BLACK];
  b.pos_hash ^= ZOBRIST_FENCENUM_HASH[b.blackFences][0];
  b.pos_hash ^= ZOBRIST_FENCENUM_HASH[b.whiteFences][1];
  for(int y = 0; y < b.y_size; y++) {
    for(int x = 0; x < b.x_size; x++) {
      Loc loc = Location::getLoc(x, y, b.x_size);
      if(b.colors[loc] == C_FENCE) {
        b.pos_hash ^= ZOBRIST_BOARD_HASH[loc][C_FENCE];
      }
    }
  }
  return b;
}

void Board::calculateArea(Color* result, bool, bool, bool, bool) const {
  for(int i = 0; i < MAX_ARR_SIZE; i++)
    result[i] = C_EMPTY;
}

void Board::calculateIndependentLifeArea(Color* result, int& whiteMinusBlack, bool, bool, bool, bool) const {
  for(int i = 0; i < MAX_ARR_SIZE; i++)
    result[i] = C_EMPTY;
  whiteMinusBlack = 0;
}

void Board::checkConsistency() const {
  assert(blackFences >= 0 && blackFences <= MAX_FENCE_NUM);
  assert(whiteFences >= 0 && whiteFences <= MAX_FENCE_NUM);
  assert(colors[blackPawnLoc] == C_BLACK);
  assert(colors[whitePawnLoc] == C_WHITE);
}

bool Board::isEqualForTesting(const Board& other, bool checkNumCaptures, bool checkSimpleKo) const {
  (void)checkNumCaptures;
  (void)checkSimpleKo;
  if(x_size != other.x_size || y_size != other.y_size)
    return false;
  if(blackPawnLoc != other.blackPawnLoc || whitePawnLoc != other.whitePawnLoc)
    return false;
  if(blackFences != other.blackFences || whiteFences != other.whiteFences)
    return false;
  if(nextPla != other.nextPla || movenum != other.movenum)
    return false;
  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      Loc loc = Location::getLoc(x, y, x_size);
      if(colors[loc] != other.colors[loc])
        return false;
    }
  }
  return true;
}

Board Board::parseBoard(int xSize, int ySize, const string& s, char lineDelimiter) {
  Board b(xSize, ySize);
  (void)s;
  (void)lineDelimiter;
  return b;
}

string Board::toStringSimple(const Board& board, char lineDelimiter) {
  ostringstream out;
  printBoard(out, board, NULL_LOC, nullptr);
  (void)lineDelimiter;
  return out.str();
}

void Board::printBoard(ostream& out, const Board& board, Loc markLoc, const vector<Move>* hist) {
  if(hist != nullptr)
    out << "MoveNum: " << hist->size() << " ";
  out << "HASH: " << board.pos_hash << "\n";
  out << "Black (P1) Fences Left: " << board.blackFences << "\n";
  out << "White (P2) Fences Left: " << board.whiteFences << "\n";
  out << "Next Player: " << PlayerIO::playerToString(board.nextPla) << "\n";

  for(int y = board.y_size - 1; y >= 0; y--) {
    out << setw(2) << y << " ";
    for(int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x, y, board.x_size);
      Color c = board.colors[loc];
      char ch = '.';
      if(c == C_BLACK) ch = 'B';
      else if(c == C_WHITE) ch = 'W';
      else if(c == C_FENCE) {
        if(x % 2 == 1 && y % 2 == 1) ch = '+';
        else if(y % 2 == 1) ch = '-';
        else ch = '|';
      }
      else {
        if(x % 2 == 0 && y % 2 == 0) ch = '.';
        else ch = ' ';
      }
      if(loc == markLoc)
        out << '@';
      else
        out << ch;
      out << ' ';
    }
    out << "\n";
  }
  out << "   ";
  for(int x = 0; x < board.x_size; x++) {
    out << (x % 10) << ' ';
  }
  out << "\n";
}

ostream& operator<<(ostream& out, const Board& board) {
  Board::printBoard(out, board, Board::NULL_LOC, nullptr);
  return out;
}

nlohmann::json Board::toJson(const Board& board) {
  nlohmann::json j;
  j["x_size"] = board.x_size;
  j["y_size"] = board.y_size;
  j["blackPawnLoc"] = board.blackPawnLoc;
  j["whitePawnLoc"] = board.whitePawnLoc;
  j["blackFences"] = board.blackFences;
  j["whiteFences"] = board.whiteFences;
  j["nextPla"] = board.nextPla;
  j["movenum"] = board.movenum;
  j["pos_hash"] = board.pos_hash.toString();

  vector<int> colArr;
  for(int i = 0; i < Board::MAX_ARR_SIZE; i++)
    colArr.push_back((int)board.colors[i]);
  j["colors"] = colArr;

  return j;
}

Board Board::ofJson(const nlohmann::json& j) {
  int xs = j["x_size"].get<int>();
  int ys = j["y_size"].get<int>();
  Board b(xs, ys);
  b.blackPawnLoc = j["blackPawnLoc"].get<Loc>();
  b.whitePawnLoc = j["whitePawnLoc"].get<Loc>();
  b.blackFences = j["blackFences"].get<int>();
  b.whiteFences = j["whiteFences"].get<int>();
  b.nextPla = j["nextPla"].get<Color>();
  b.movenum = j["movenum"].get<int>();

  vector<int> colArr = j["colors"].get<vector<int>>();
  for(int i = 0; i < Board::MAX_ARR_SIZE && i < (int)colArr.size(); i++)
    b.colors[i] = (Color)colArr[i];

  b.pos_hash = ZOBRIST_SIZE_X_HASH[b.x_size] ^ ZOBRIST_SIZE_Y_HASH[b.y_size];
  b.pos_hash ^= ZOBRIST_BOARD_HASH[b.whitePawnLoc][C_WHITE];
  b.pos_hash ^= ZOBRIST_BOARD_HASH[b.blackPawnLoc][C_BLACK];
  b.pos_hash ^= ZOBRIST_FENCENUM_HASH[b.blackFences][0];
  b.pos_hash ^= ZOBRIST_FENCENUM_HASH[b.whiteFences][1];
  for(int y = 0; y < b.y_size; y++) {
    for(int x = 0; x < b.x_size; x++) {
      Loc loc = Location::getLoc(x, y, b.x_size);
      if(b.colors[loc] == C_FENCE) {
        b.pos_hash ^= ZOBRIST_BOARD_HASH[loc][C_FENCE];
      }
    }
  }

  b.cachedPathP1.clear();
  b.bfsReachable(b.blackPawnLoc, 0, &b.cachedPathP1);
  b.cachedPathP2.clear();
  b.bfsReachable(b.whitePawnLoc, b.y_size - 1, &b.cachedPathP2);

  b.checkConsistency();
  return b;
}
