#ifndef Q4_STYLE_H_
#define Q4_STYLE_H_

#include "q4board.h"

// Per-seat style statistics of the real game history (docs/q4/Q4IO.md §9). Game logic only, no net.
struct Q4StyleTracker {
  static constexpr int NUM_DESCRIPTORS = 9;
  static constexpr int NUM_HALFLIVES = 2;
  static constexpr int PER_SEAT = NUM_HALFLIVES * NUM_DESCRIPTORS + 1;  // 19
  static constexpr int NUM_FEATURES = 4 * PER_SEAT;                      // 76, metadata slots 0..75
  static constexpr double HALFLIVES[NUM_HALFLIVES] = {4.0, 16.0};

  double mean[4][NUM_HALFLIVES][NUM_DESCRIPTORS];
  int numMoves[4];

  Q4StyleTracker();
  void reset();

  // The 9 descriptors of the move `action` played by before.toMove on `before`, giving `after`.
  static void descriptors(const Q4Board& before, const Q4Board& after, int action, double x[NUM_DESCRIPTORS]);

  void observeMove(const Q4Board& before, const Q4Board& after, int action);
  void observeElimination(int seat);

  // The 76 features for the seat `perspective` to move: seats in relative order, victim columns rotated.
  void encode(int perspective, float out[NUM_FEATURES]) const;
};

#endif  // Q4_STYLE_H_
