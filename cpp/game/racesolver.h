#ifndef GAME_RACESOLVER_H_
#define GAME_RACESOLVER_H_

// Exact solver for the pawn race that remains when the walls are fixed (docs/QuoridorRaceSolver.md).
//
// With the walls fixed, a game of pawn moves only is a game on the states (Black's square, White's square, side to
// move): 2 * 81 * 81 states. It is solved by retrograde analysis (attractors with move counters, one per lead
// threshold). The value of a state is the lead s = t + komi (White's view, see docs/QuoridorIOv2.md section 2) that
// the optimal play ends with, 0 if neither side can force the end (infinite play is a draw).

#include "../core/global.h"
#include "../game/board.h"
#include "../game/boardhistory.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

struct RaceSolution {
  static constexpr int NUM_SQ = 81;
  static constexpr int NUM_STATES = 2 * NUM_SQ * NUM_SQ;
  static constexpr int16_t INVALID = 32767; // val2 of states that are not part of the race (same square, game over)
  static constexpr int MAX_MOVES = 6;

  int komi2; // 2 * komi, odd

  // Per state (see stateIndex):
  //  - val2: 2 * s (an odd number for a win / loss, 0 for a draw, INVALID);
  //  - plies: the number of plies the game lasts when both play optimally, i.e. keep the value (the lead) and the winner
  //    finishes as fast as possible and the loser delays as long as possible (0 for a draw);
  //  - forced: the number of plies within which the winner can force a result at least as good as val2 whatever the
  //    loser does, even if the loser gives up lead to stall (>= plies; 0 for a draw). Use this to check that the
  //    solved result fits in the plies left before the maxPlies draw.
  int16_t val2[NUM_STATES];
  int16_t plies[NUM_STATES];
  int16_t forced[NUM_STATES];

  // Walls only distances to the goal rows, per square. distBlack: to row 0 (Black's goal); distWhite: to row 8.
  int8_t distBlack[NUM_SQ];
  int8_t distWhite[NUM_SQ];
  // Neighbor square by direction (0 north = row - 1, 1 west, 2 east, 3 south = row + 1), -1 if a wall or the edge blocks.
  int8_t nbr[NUM_SQ][4];

  static int sqOf(int c, int r) { return r * 9 + c; }
  static int sqOfLoc(Loc loc, int xSize) { return (Location::getY(loc, xSize) / 2) * 9 + (Location::getX(loc, xSize) / 2); }
  static Loc locOfSq(int sq) { return Location::pawnLoc(sq % 9, sq / 9); }
  static int stateIndex(Player nextPla, int blackSq, int whiteSq) {
    return ((nextPla == P_BLACK ? 0 : 1) * NUM_SQ + blackSq) * NUM_SQ + whiteSq;
  }

  // The pawn squares `me` can move to when the opponent stands on `opp` (straight and diagonal jumps included).
  // Returns the number of moves, at most MAX_MOVES.
  int genMoves(int me, int opp, int8_t out[MAX_MOVES]) const;

  // The 2*s of the game that ends when `mover`'s pawn arrives on its goal row while the loser stands on `loserSq`.
  int terminalVal2(Player mover, int loserSq) const;

  // Number of non-draw, valid states (for tests / statistics).
  int numSolvedStates() const;
};

struct RaceResult {
  bool solved = false;
  // True if both players have no walls (so the result is the exact value of the game); false for a proven win of the
  // only player that still has walls (the race lead is a lower bound for that player).
  bool exact = false;
  Player winner = C_EMPTY; // C_EMPTY for a draw
  int lead2 = 0;           // 2 * s (White's view), 0 for a draw
  int plies = 0;           // plies until the game ends under optimal play (0 for a draw)
};

namespace RaceSolver {
  // Solves the race for the walls on `board` and komi (not cached).
  std::shared_ptr<RaceSolution> solve(const Board& board, float komi);

  // The cached solution (thread safe; solves on a miss).
  std::shared_ptr<const RaceSolution> getCached(const Board& board, float komi);

  // Cache limits and statistics (global, shared by all threads).
  void setMaxCacheEntries(size_t n);
  size_t getMaxCacheEntries();
  void clearCache();
  struct CacheStats {
    uint64_t hits;
    uint64_t misses;
    uint64_t solveMicros; // total time spent solving
    size_t entries;
  };
  CacheStats getCacheStats();
  void resetCacheStats();

  // Whether the solver applies to this board at all (a 9x9 board).
  bool isSupportedBoard(const Board& board);

  // The solved result of the position `board` with `nextPla` to move, played under `hist`'s rules, or unsolved:
  //  - both players have no walls: the exact value;
  //  - exactly one player has walls and wins the pure race (restricted to pawn moves): that player's win, with the
  //    race lead as a lower bound;
  //  - otherwise unsolved.
  // The result is only used when the plies it needs fit in hist.pliesUntilDraw() and, with the repetition rule on,
  // when no earlier position of the history could turn the solved line into a draw (else unsolved).
  RaceResult evaluate(const Board& board, const BoardHistory& hist, Player nextPla);

  // The optimal pawn moves for the exact race (both players without walls): the moves that keep the value, among them
  // the ones that finish fastest if the mover is winning, slowest if losing, all of them for a draw. Empty if the
  // position is not exactly solvable.
  std::vector<Loc> optimalMoves(const Board& board, const BoardHistory& hist, Player pla);
}

#endif // GAME_RACESOLVER_H_
