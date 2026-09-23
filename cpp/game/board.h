/*
 * board.h
 * Quoridor Board for KataQuoridor on 17x17 unified grid representation.
 * Adapted from KataGo 1.18.2.
 */

#ifndef GAME_BOARD_H_
#define GAME_BOARD_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../external/nlohmann_json/json.hpp"

#ifdef COMPILE_MAX_BOARD_LEN
#undef COMPILE_MAX_BOARD_LEN
#endif
#define COMPILE_MAX_BOARD_LEN 17 // 17x17 grid for 9x9 Quoridor

/*
 * 17x17 Board representation:
 * A 9x9 Quoridor board is mapped to a 17x17 grid:
 * - Even X, Even Y: Pawn cell (X/2, Y/2). Holds C_EMPTY, C_BLACK, or C_WHITE.
 * - Even X, Odd Y: Horizontal fence arm between rows Y/2 and Y/2+1 at column X/2.
 * - Odd X, Even Y: Vertical fence arm between columns X/2 and X/2+1 at row Y/2.
 * - Odd X, Odd Y: Fence center at anchor (X/2, Y/2).
 *
 * Placed fence segments occupy three 17x17 cells marked C_FENCE:
 * - Horizontal fence at anchor (c, r) (0 <= c, r <= 7):
 *     (2*c, 2*r+1), (2*c+1, 2*r+1), (2*c+2, 2*r+1)
 * - Vertical fence at anchor (c, r) (0 <= c, r <= 7):
 *     (2*c+1, 2*r), (2*c+1, 2*r+1), (2*c+1, 2*r+2)
 *
 * Action encoding (policy index in 17x17 grid):
 * - Pawn destination (c, r): Loc at (2*c, 2*r) (even x, even y) -> 81 moves
 * - Horizontal fence at (c, r): Loc at (2*c+1, 2*r+1) (odd x, odd y, center) -> 64 moves
 * - Vertical fence at (c, r): Loc at (2*c+1, 2*r) (odd x, even y, upper arm) -> 64 moves
 *   (Player-agnostic geometric Loc: identical coordinate for both Black and White)
 * - Total legal moves = 81 + 64 + 64 = 209
 * - Total policy slots = 17*17 + 1 = 290 (+1 is PASS_LOC, always illegal)
 */

// TYPES AND CONSTANTS-----------------------------------------------------------------

struct Board;

// Player
typedef int8_t Player;
static constexpr Player P_BLACK = 1; // Player 1, starts at e9 (8, 16), moves toward Y = 0
static constexpr Player P_WHITE = 2; // Player 2, starts at e1 (8, 0), moves toward Y = 16

// Color of a point on the board
typedef int8_t Color;
static constexpr Color C_EMPTY = 0;
static constexpr Color C_BLACK = 1;
static constexpr Color C_WHITE = 2;
static constexpr Color C_FENCE = 3; // Wall / fence segment
static constexpr Color C_WALL = 4;  // Board boundary / off-board
static constexpr int NUM_BOARD_COLORS = 5;

static inline Color getOpp(Color c) {
  return c ^ 3;
}

namespace PlayerIO {
  char colorToChar(Color c);
  std::string playerToStringShort(Player p);
  std::string playerToString(Player p);
  bool tryParsePlayer(const std::string& s, Player& pla);
  Player parsePlayer(const std::string& s);
}

// Location of a point on the board
// (x, y) is represented as (x+1) + (y+1)*(x_size+1)
typedef short Loc;
namespace Location {
  inline Loc getLoc(int x, int y, int x_size) { return (Loc)((x + 1) + (y + 1) * (x_size + 1)); }
  inline int getX(Loc loc, int x_size) { return (loc % (x_size + 1)) - 1; }
  inline int getY(Loc loc, int x_size) { return (loc / (x_size + 1)) - 1; }

  // Pure geometric Loc definitions for Quoridor
  inline Loc pawnLoc(int c, int r, int x_size = 17) { return getLoc(2 * c, 2 * r, x_size); }
  inline Loc hWallLoc(int c, int r, int x_size = 17) { return getLoc(2 * c + 1, 2 * r + 1, x_size); }
  inline Loc vWallLoc(int c, int r, int x_size = 17) { return getLoc(2 * c + 1, 2 * r, x_size); }

  inline bool isPawnLoc(Loc loc, int x_size = 17) {
    int x = getX(loc, x_size);
    int y = getY(loc, x_size);
    return x >= 0 && x < x_size && y >= 0 && y < x_size && (x % 2 == 0) && (y % 2 == 0);
  }
  inline bool isHWallLoc(Loc loc, int x_size = 17) {
    int x = getX(loc, x_size);
    int y = getY(loc, x_size);
    return x >= 1 && x <= x_size - 2 && y >= 1 && y <= x_size - 2 && (x % 2 == 1) && (y % 2 == 1);
  }
  inline bool isVWallLoc(Loc loc, int x_size = 17) {
    int x = getX(loc, x_size);
    int y = getY(loc, x_size);
    return x >= 1 && x <= x_size - 2 && y >= 0 && y <= x_size - 3 && (x % 2 == 1) && (y % 2 == 0);
  }

  void getAdjacentOffsets(short adj_offsets[8], int x_size);
  bool isAdjacent(Loc loc0, Loc loc1, int x_size);
  Loc getMirrorLoc(Loc loc, int x_size, int y_size);
  Loc getCenterLoc(int x_size, int y_size);
  Loc getCenterLoc(const Board& b);
  bool isCentral(Loc loc, int x_size, int y_size);
  bool isNearCentral(Loc loc, int x_size, int y_size);
  int distance(Loc loc0, Loc loc1, int x_size);
  int euclideanDistanceSquared(Loc loc0, Loc loc1, int x_size);

  std::string toString(Loc loc, int x_size, int y_size);
  std::string toString(Loc loc, const Board& b);
  std::string toStringMach(Loc loc, int x_size);
  std::string toStringMach(Loc loc, const Board& b);

  bool tryOfString(const std::string& str, int x_size, int y_size, Loc& result);
  bool tryOfString(const std::string& str, const Board& b, Loc& result);
  bool tryOfString(const std::string& str, const Board& b, Player pla, Loc& result);
  Loc ofString(const std::string& str, int x_size, int y_size);
  Loc ofString(const std::string& str, const Board& b);

  bool tryOfStringAllowNull(const std::string& str, int x_size, int y_size, Loc& result);
  bool tryOfStringAllowNull(const std::string& str, const Board& b, Loc& result);
  Loc ofStringAllowNull(const std::string& str, int x_size, int y_size);
  Loc ofStringAllowNull(const std::string& str, const Board& b);

  std::vector<Loc> parseSequence(const std::string& str, const Board& b);
}

STRUCT_NAMED_PAIR(Loc, loc, Player, pla, Move);

struct Board {
  static void initHash();

  static constexpr int MAX_LEN = COMPILE_MAX_BOARD_LEN;
  static constexpr int DEFAULT_LEN = 17;
  static constexpr int MAX_PLAY_SIZE = MAX_LEN * MAX_LEN;
  static constexpr int MAX_ARR_SIZE = (MAX_LEN + 1) * (MAX_LEN + 2) + 1;
  static constexpr int MAX_FENCE_NUM = 10;

  static constexpr Loc NULL_LOC = 0;
  static constexpr Loc PASS_LOC = 1;

  // Zobrist Hashing
  static bool IS_ZOBRIST_INITALIZED;
  static Hash128 ZOBRIST_SIZE_X_HASH[MAX_LEN + 1];
  static Hash128 ZOBRIST_SIZE_Y_HASH[MAX_LEN + 1];
  static Hash128 ZOBRIST_BOARD_HASH[MAX_ARR_SIZE][NUM_BOARD_COLORS];
  static Hash128 ZOBRIST_BOARD_HASH2[MAX_ARR_SIZE][NUM_BOARD_COLORS];
  static Hash128 ZOBRIST_PLAYER_HASH[4];
  static Hash128 ZOBRIST_FENCENUM_HASH[MAX_FENCE_NUM + 1][2];
  static const Hash128 ZOBRIST_GAME_IS_OVER;

  // Compatibility stubs for unchanged modules until respective steps
  static const Hash128 ZOBRIST_PASS_ENDS_PHASE;
  static Hash128 ZOBRIST_KO_LOC_HASH[MAX_ARR_SIZE];
  Loc chain_head[MAX_ARR_SIZE];
  Loc next_in_chain[MAX_ARR_SIZE];

  struct MoveRecord {
    Player pla;
    Loc loc;
    Loc oldBlackPawnLoc;
    Loc oldWhitePawnLoc;
    int oldBlackFences;
    int oldWhiteFences;
    int oldMovenum;
    Player oldNextPla;
    Hash128 oldPosHash;
    // For walls: cells modified
    Loc modifiedCells[3];
    int numModifiedCells;
  };

  Board();
  Board(int x, int y);
  Board(const Board& other);
  Board& operator=(const Board&) = default;

  // Primary Quoridor gameplay functions
  bool isLegal(Loc loc, Player pla, bool isMultiStoneSuicideLegal = false) const;
  bool isLegalIgnoringKo(Loc loc, Player pla, bool isMultiStoneSuicideLegal = false) const { (void)isMultiStoneSuicideLegal; return isLegal(loc, pla); }
  bool isOnBoard(Loc loc) const;
  bool isOnBoardPawn(Loc loc) const;
  bool isOnBoardFence(Loc loc) const;
  bool isEmpty() const;
  int numStonesOnBoard() const;
  int numPlaStonesOnBoard(Player pla) const;

  bool setStone(Loc loc, Color color);
  bool setStones(const std::vector<Move>& placements);

  bool playMove(Loc loc, Player pla, bool isMultiStoneSuicideLegal = false);
  void playMoveAssumeLegal(Loc loc, Player pla);
  MoveRecord playMoveRecorded(Loc loc, Player pla);
  void undo(MoveRecord record);

  Player nextnextPla() const;
  Player prevPla() const;
  Hash128 getSitHash(Player pla) const;
  Hash128 getSitHashWithSimpleKo(Player pla) const { return getSitHash(pla); }
  Hash128 getPosHashAfterMove(Loc loc, Player pla) const;

  void clearSimpleKoLoc() {}
  void setSimpleKoLoc(Loc loc) { (void)loc; }

  // Quoridor Movement & Wall queries
  bool canPawnStep(Loc from, Loc to) const;
  bool isLegalPawnMove(Loc loc, Player pla) const;
  std::vector<Loc> getLegalPawnDestinations(Player pla) const;
  bool isLegalWallPlacement(int c, int r, bool isVertical, Player pla) const;
  bool checkNoFullBlockLazy(int c, int r, bool isVertical) const;
  bool bfsReachable(Loc start, int targetY, std::vector<Loc>* outPath = nullptr) const;
  int getShortestPathDistance(Player pla) const;
  std::vector<Loc> findShortestPath(Player pla) const;
  void calDistMap(Player pla, int32_t* res) const;
  bool isBoardNotConnected() const;

  Board getMirroredX() const;

  // Compatibility stubs for KataGo search / helpers
  double sqrtBoardArea() const { return 17.0; }
  int getChainSize(Loc loc) const { (void)loc; return 1; }
  int getNumLiberties(Loc loc) const { (void)loc; return 4; }
  int getNumLibertiesAfterPlay(Loc loc, Player pla, int max) const { (void)loc; (void)pla; (void)max; return 4; }
  void getBoundNumLibertiesAfterPlay(Loc loc, Player pla, int& lowerBound, int& upperBound) const { (void)loc; (void)pla; lowerBound = 4; upperBound = 4; }
  int getNumImmediateLiberties(Loc loc) const { (void)loc; return 4; }
  bool isSuicide(Loc loc, Player pla) const { (void)loc; (void)pla; return false; }
  bool isIllegalSuicide(Loc loc, Player pla, bool isMultiStoneSuicideLegal) const { (void)loc; (void)pla; (void)isMultiStoneSuicideLegal; return false; }
  bool isKoBanned(Loc loc) const { (void)loc; return false; }
  bool isSimpleEye(Loc loc, Player pla) const { (void)loc; (void)pla; return false; }
  bool pocketIsSingleColor(Loc loc, Color color, int maxDepth) const { (void)loc; (void)color; (void)maxDepth; return false; }
  bool wouldBeCapture(Loc loc, Player pla) const { (void)loc; (void)pla; return false; }
  bool wouldBeKoCapture(Loc loc, Player pla) const { (void)loc; (void)pla; return false; }
  Loc getKoCaptureLoc(Loc loc, Player pla) const { (void)loc; (void)pla; return NULL_LOC; }
  bool isAdjacentToPla(Loc loc, Player pla) const { (void)loc; (void)pla; return false; }
  bool isAdjacentOrDiagonalToPla(Loc loc, Player pla) const { (void)loc; (void)pla; return false; }
  bool isAdjacentToChain(Loc loc, Loc chain) const { (void)loc; (void)chain; return false; }
  bool isNonPassAliveSelfConnection(Loc loc, Player pla, const Color* passAliveArea) const { (void)loc; (void)pla; (void)passAliveArea; return false; }
  bool simpleRepetitionBoundGt(Loc loc, int bound) const { (void)loc; (void)bound; return false; }
  bool searchIsLadderCaptured(Loc loc, bool defenderFirst, std::vector<Loc>& buf) { (void)loc; (void)defenderFirst; (void)buf; return false; }
  bool searchIsLadderCapturedAttackerFirst2Libs(Loc loc, std::vector<Loc>& buf, std::vector<Loc>& workingMoves) { (void)loc; (void)buf; (void)workingMoves; return false; }

  void calculateArea(Color* result, bool nonPassAliveStones, bool safeBigTerritories, bool unsafeBigTerritories, bool isMultiStoneSuicideLegal) const;
  void calculateIndependentLifeArea(Color* result, int& whiteMinusBlackIndependentLifeRegionCount, bool keepTerritories, bool keepStones, bool excludeTerritoryAdjacentToAtari, bool isMultiStoneSuicideLegal) const;

  bool setStoneFailIfNoLibs(Loc loc, Color color) { return setStone(loc, color); }
  bool setStonesFailIfNoLibs(const std::vector<Move>& placements) { return setStones(placements); }
  int setStonesTolerant(const std::vector<Move>& placements) { setStones(placements); return 0; }
  void regenChainsFromColors() {}

  void checkConsistency() const;
  bool isEqualForTesting(const Board& other, bool checkNumCaptures = true, bool checkSimpleKo = true) const;

  static Board parseBoard(int xSize, int ySize, const std::string& s, char lineDelimiter = '\n');
  static void printBoard(std::ostream& out, const Board& board, Loc markLoc, const std::vector<Move>* hist);
  static std::string toStringSimple(const Board& board, char lineDelimiter = '\n');
  static nlohmann::json toJson(const Board& board);
  static Board ofJson(const nlohmann::json& data);

  // Data fields - pure Quoridor state (ZERO Go chains/liberties/stonePool residue)
  int x_size;
  int y_size;
  Color colors[MAX_ARR_SIZE];
  int blackFences;
  int whiteFences;
  Loc blackPawnLoc;
  Loc whitePawnLoc;
  int movenum;
  Color nextPla;
  Hash128 pos_hash;
  short adj_offsets[8];

  Loc ko_loc;
  int numBlackCaptures;
  int numWhiteCaptures;

  // Cached paths for Grant's Lazy BFS
  mutable std::vector<Loc> cachedPathP1;
  mutable std::vector<Loc> cachedPathP2;

private:
  void init(int xS, int yS);
  void placeFence(Loc center, bool isVertical);
  void removeFence(Loc center, bool isVertical);

  friend std::ostream& operator<<(std::ostream& out, const Board& board);
};

#endif // GAME_BOARD_H_
