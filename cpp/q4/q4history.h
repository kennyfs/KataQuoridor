#ifndef Q4_HISTORY_H_
#define Q4_HISTORY_H_

#include "q4board.h"
#include "q4rules.h"
#include "q4playstate.h"
#include "q4style.h"

#include <string>
#include <vector>

struct Q4Event {
  bool isElimination;
  int action;          // 0..320 if !isElimination
  int eliminatedSeat;  // 0..3 if isElimination
};

class Q4History {
public:
  Q4Rules rules;
  Q4Board initialBoard;
  Q4Board currentBoard;

  // Search playout state representing current position
  Q4PlayState state;

  // Style statistics of the real history (docs/q4/Q4IO.md §9); a copy before each event is kept, as for boards.
  Q4StyleTracker style;
  std::vector<Q4StyleTracker> styleHistory;

  std::vector<Q4Board> boardHistory; // Stored state BEFORE each event
  std::vector<Q4Event> events;

  int plies;
  bool isFinished;
  int winnerSeat;      // 0..3, or -1 if no winner
  bool isDraw;

  // Hashes of positions since last irreversible change (wall placement or elimination)
  std::vector<Hash128> repetitionHashes;
  std::vector<std::vector<Hash128>> repetitionHistory; // repetitionHashes before each event

  Q4History();
  explicit Q4History(const Q4Rules& rules);
  explicit Q4History(const Q4Board& board, const Q4Rules& rules);

  void clear(const Q4Board& board, const Q4Rules& rules);
  void play(int action);
  void eliminate(int seat);
  bool undo();

  const Q4PlayState& getState() const { return state; }
  // The 76 style features for the seat to move (metadata slots 0..75 of Q4 I/O v2).
  void getStyleFeatures(float out[Q4StyleTracker::NUM_FEATURES]) const { style.encode(currentBoard.toMove, out); }
  int currentPositionRepetitionCount() const;
  std::string getResultString() const;

private:
  void syncFromState();
};

#endif  // Q4_HISTORY_H_
