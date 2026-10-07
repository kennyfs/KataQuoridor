#ifndef Q4_BOARD_H_
#define Q4_BOARD_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "q4rules.h"

#include <bitset>
#include <cstdint>
#include <vector>

// A set of the 121 cells, bit c = cell c (lo: cells 0..63, hi: cells 64..120; bits above 120 stay zero).
// Used by the flood fill in isLegalWallBruteForce.
struct Q4Bits121 {
  static constexpr uint64_t HI_MASK = ((uint64_t)1 << 57) - 1;
  uint64_t lo = 0;
  uint64_t hi = 0;

  inline bool operator==(const Q4Bits121& o) const { return lo == o.lo && hi == o.hi; }
  inline bool operator!=(const Q4Bits121& o) const { return !(*this == o); }
  inline Q4Bits121 operator|(const Q4Bits121& o) const { return {lo | o.lo, hi | o.hi}; }
  inline Q4Bits121 operator&(const Q4Bits121& o) const { return {lo & o.lo, hi & o.hi}; }
  inline bool test(int c) const { return c < 64 ? ((lo >> c) & 1) : ((hi >> (c - 64)) & 1); }
  inline void set(int c) { if(c < 64) lo |= (uint64_t)1 << c; else hi |= (uint64_t)1 << (c - 64); }
  inline void clear(int c) { if(c < 64) lo &= ~((uint64_t)1 << c); else hi &= ~((uint64_t)1 << (c - 64)); }
  inline void assign(int c, bool v) { if(v) set(c); else clear(c); }

  // Shift toward higher cell indices by n (1 = east, 11 = north), dropping bits above 120.
  template<int n> inline Q4Bits121 shl() const { return {lo << n, ((hi << n) | (lo >> (64 - n))) & HI_MASK}; }
  // Shift toward lower cell indices by n (1 = west, 11 = south).
  template<int n> inline Q4Bits121 shr() const { return {(lo >> n) | (hi << (64 - n)), hi >> n}; }
};

struct Q4Board {
  static constexpr int BOARD_SIZE = 11;
  static constexpr int NUM_CELLS = 121;
  static constexpr int CENTER_CELL = 60;  // (5, 5): 5 * 11 + 5
  static constexpr int NUM_SEATS = 4;
  static constexpr int NUM_ANCHORS = 10;
  static constexpr int NUM_WALL_ANCHORS = 100;
  static constexpr int NUM_ACTIONS = 321;
  static constexpr int NULL_ACTION = -1;

  // Directions: 0: North (+y), 1: East (+x), 2: South (-y), 3: West (-x)
  static constexpr int DIR_N = 0;
  static constexpr int DIR_E = 1;
  static constexpr int DIR_S = 2;
  static constexpr int DIR_W = 3;

  static const int DIR_OFFSET[4];       // {+11, +1, -11, -1}
  static const int DIR_OPPOSITE[4];     // {2, 3, 0, 1}
  static const int START_CELLS[4];      // {5, 55, 115, 65} -> f1, a6, f11, k6

  // Zobrist tables
  static bool IS_ZOBRIST_INITIALIZED;
  static Hash128 ZOBRIST_PAWN[NUM_SEATS][NUM_CELLS];
  static Hash128 ZOBRIST_HWALL[NUM_WALL_ANCHORS];
  static Hash128 ZOBRIST_VWALL[NUM_WALL_ANCHORS];
  static Hash128 ZOBRIST_WALLS_LEFT[NUM_SEATS][11];
  static Hash128 ZOBRIST_TO_MOVE[NUM_SEATS];
  static Hash128 ZOBRIST_ALIVE[NUM_SEATS];

  static void initHash();

  // State members
  int8_t pawn[NUM_SEATS];               // Cell index (0..120) or -1 if eliminated
  int8_t occupant[NUM_CELLS];           // Seat index (0..3) or -1 if unoccupied
  int8_t wallsLeft[NUM_SEATS];          // Walls remaining for each seat
  uint8_t alive;                        // Bitmask: (1 << seat) if seat is alive
  int8_t toMove;                        // Seat index (0..3)

  std::bitset<NUM_WALL_ANCHORS> hWalls; // Placed horizontal walls
  std::bitset<NUM_WALL_ANCHORS> vWalls; // Placed vertical walls

  // Edge blocking: bit (1 << dir) set if movement in that direction is blocked
  uint8_t blocked[NUM_CELLS];

  // Open-edge masks, kept in sync with blocked[]: bit c set iff the step from c in that direction is open.
  // Board-edge bits in blocked[] keep canE clear on x == 10 and canW clear on x == 0, which stops row wrap.
  Q4Bits121 canN;
  Q4Bits121 canE;
  Q4Bits121 canS;
  Q4Bits121 canW;

  // Distances to center (walls-only, 255 if unreachable)
  uint8_t distToCenter[NUM_CELLS];

  // Cached shortest paths to center for alive seats (Plan §5.3)
  std::bitset<110> hPathEdges;
  std::bitset<110> vPathEdges;

  Hash128 hash;

  // Constructors
  Q4Board();
  explicit Q4Board(const Q4Rules& rules);

  // Helper getters
  static inline int cellOf(int x, int y) { return y * BOARD_SIZE + x; }
  static inline int cellX(int c) { return c % BOARD_SIZE; }
  static inline int cellY(int c) { return c / BOARD_SIZE; }

  static inline int anchorOf(int ax, int ay) { return ay * NUM_ANCHORS + ax; }
  static inline int anchorX(int a) { return a % NUM_ANCHORS; }
  static inline int anchorY(int a) { return a / NUM_ANCHORS; }

  static inline int actionOfPawn(int c) { return c; }
  static inline int actionOfVWall(int a) { return 121 + a; }
  static inline int actionOfHWall(int a) { return 221 + a; }

  static inline bool isPawnAction(int action) { return action >= 0 && action < 121; }
  static inline bool isVWallAction(int action) { return action >= 121 && action < 221; }
  static inline bool isHWallAction(int action) { return action >= 221 && action < 321; }

  inline bool isAlive(int seat) const { return (alive & (1 << seat)) != 0; }
  int getNumAlive() const;
  int getNextAlive(int currSeat) const;

  inline bool canStep(int c, int dir) const {
    return !(blocked[c] & (1 << dir));
  }

  // Wall conflict and legality
  bool wallConflicts(int ax, int ay, bool isHorizontal) const;
  bool isGeometricallyLegalWall(int ax, int ay, bool isHorizontal) const;
  // Full reachability check by bitboard flood fill from the center.
  bool isLegalWallBruteForce(int ax, int ay, bool isHorizontal) const;
  // Previous BFS version of isLegalWallBruteForce, kept as a reference for tests.
  bool isLegalWallBruteForceBFS(int ax, int ay, bool isHorizontal) const;
  bool isLegalWall(int ax, int ay, bool isHorizontal) const;

  // Legal moves generation
  int getPawnMoves(int seat, int* outMoves) const;
  void getPawnMoves(int seat, std::vector<int>& outMoves) const;
  int getLegalActions(int seat, int* outActions) const;
  void getLegalActions(int seat, std::vector<int>& outActions) const;
  bool isLegalAction(int action, int seat) const;

  // Apply moves and events
  Hash128 getHashAfterPawnMove(int destCell) const;
  void applyPawnMove(int destCell);
  void applyWall(int ax, int ay, bool isHorizontal);
  void applyAction(int action);
  void eliminateSeat(int seat);

  // Terminal state checks
  bool isFinished() const;
  int getWinner() const;  // 0..3 if won, -1 if no winner

  void recomputeDistancesToCenter();
  void recomputeCachedPaths();
  Hash128 getHashFromScratch() const;
  bool checkInvariants() const;

  void addWallToBlocked(int ax, int ay, bool isHorizontal);
  void removeWallFromBlocked(int ax, int ay, bool isHorizontal);

private:
  void initGridBoundaries();
  // Refresh the can-mask bits of cell c from blocked[c].
  void syncCanMasks(int c);
};

#endif  // Q4_BOARD_H_
