#ifndef Q4_PLAY_STATE_H_
#define Q4_PLAY_STATE_H_

#include "q4board.h"
#include "q4rules.h"
#include "../core/hash.h"

#include <string>
#include <vector>

// Lightweight game state representation for search playouts (Plan §8.4).
// Contains board, plies, repetition hashes, and terminal state logic without undo history.
class Q4PlayState {
public:
  Q4Rules rules;
  Q4Board board;
  int plies;
  std::vector<Hash128> repetitionHashes; // Hashes since last wall or elimination
  bool isFinished;
  int winnerSeat; // 0..3, or -1 if no winner
  bool isDraw;

  Q4PlayState();
  explicit Q4PlayState(const Q4Rules& rules);
  explicit Q4PlayState(const Q4Board& board, const Q4Rules& rules);

  void clear(const Q4Board& board, const Q4Rules& rules);
  void playAssumeLegal(int action);
  void eliminate(int seat);

  int currentPositionRepetitionCount() const;
  std::string getResultString() const;

private:
  void recordRepetition();
  void checkTerminal();
};

#endif  // Q4_PLAY_STATE_H_
