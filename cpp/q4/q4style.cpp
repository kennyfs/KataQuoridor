#include "q4style.h"

#include "q4bots.h"

#include <algorithm>
#include <cmath>
#include <vector>

constexpr double Q4StyleTracker::HALFLIVES[Q4StyleTracker::NUM_HALFLIVES];

Q4StyleTracker::Q4StyleTracker() {
  reset();
}

void Q4StyleTracker::reset() {
  for(int s = 0; s < 4; s++) {
    numMoves[s] = 0;
    for(int h = 0; h < NUM_HALFLIVES; h++)
      for(int d = 0; d < NUM_DESCRIPTORS; d++)
        mean[s][h][d] = 0.0;
  }
}

static int seatDistance(const Q4Board& board, int seat) {
  if(!board.isAlive(seat) || board.pawn[seat] < 0)
    return -1;
  return board.distToCenter[board.pawn[seat]];
}

void Q4StyleTracker::descriptors(const Q4Board& before, const Q4Board& after, int action, double x[NUM_DESCRIPTORS]) {
  for(int i = 0; i < NUM_DESCRIPTORS; i++)
    x[i] = 0.0;
  const int m = before.toMove;

  if(Q4Board::isPawnAction(action)) {
    const int dest = action;  // pawn actions are numbered by destination cell
    const int dBefore = seatDistance(before, m);
    const int dAfter = seatDistance(after, m);
    x[1] = std::max(-1.0, std::min(1.0, (double)(dBefore - dAfter) / 2.0));
    std::vector<int> dests;
    before.getPawnMoves(m, dests);
    int minDist = 255;
    for(int c : dests)
      minDist = std::min(minDist, (int)before.distToCenter[c]);
    x[2] = (before.distToCenter[dest] == minDist) ? 1.0 : 0.0;
    return;
  }

  x[0] = 1.0;
  // The race leader among the other alive seats, by the arrival estimates on the board before the move.
  int lead = -1;
  int bestEstimate = 0;
  for(int s = 0; s < Q4Board::NUM_SEATS; s++) {
    if(s == m || !before.isAlive(s))
      continue;
    int est = Q4Bots::arrivalEstimate(before, s);
    if(lead < 0 || est < bestEstimate) {
      lead = s;
      bestEstimate = est;
    }
  }
  int delta[4] = {0, 0, 0, 0};
  int maxOpponentDelta = 0;
  for(int s = 0; s < Q4Board::NUM_SEATS; s++) {
    int d0 = seatDistance(before, s);
    int d1 = seatDistance(after, s);
    if(d0 < 0 || d1 < 0 || d0 == 255 || d1 == 255)
      continue;
    delta[s] = d1 - d0;
    x[3 + s] = std::min((double)delta[s] / 8.0, 1.0);
    if(s != m)
      maxOpponentDelta = std::max(maxOpponentDelta, delta[s]);
  }
  x[7] = (lead >= 0 && maxOpponentDelta > 0 && delta[lead] == maxOpponentDelta) ? 1.0 : 0.0;
  x[8] = (maxOpponentDelta == 0) ? 1.0 : 0.0;
}

void Q4StyleTracker::observeMove(const Q4Board& before, const Q4Board& after, int action) {
  const int m = before.toMove;
  double x[NUM_DESCRIPTORS];
  descriptors(before, after, action, x);
  numMoves[m]++;
  const int n = numMoves[m];
  for(int h = 0; h < NUM_HALFLIVES; h++) {
    const double rate = std::max(1.0 / n, 1.0 - std::pow(2.0, -1.0 / HALFLIVES[h]));
    for(int d = 0; d < NUM_DESCRIPTORS; d++)
      mean[m][h][d] += (x[d] - mean[m][h][d]) * rate;
  }
}

void Q4StyleTracker::observeElimination(int seat) {
  numMoves[seat] = 0;
  for(int h = 0; h < NUM_HALFLIVES; h++)
    for(int d = 0; d < NUM_DESCRIPTORS; d++)
      mean[seat][h][d] = 0.0;
}

void Q4StyleTracker::encode(int perspective, float out[NUM_FEATURES]) const {
  for(int k = 0; k < 4; k++) {
    const int s = (perspective + k) % 4;
    float* o = out + k * PER_SEAT;
    for(int h = 0; h < NUM_HALFLIVES; h++) {
      for(int d = 0; d < NUM_DESCRIPTORS; d++) {
        int col = d;
        if(d >= 3 && d <= 6)
          col = 3 + ((d - 3) - perspective + 4) % 4;  // absolute victim -> relative column
        o[h * NUM_DESCRIPTORS + col] = (float)mean[s][h][d];
      }
    }
    o[NUM_HALFLIVES * NUM_DESCRIPTORS] = (float)std::min((double)numMoves[s] / 64.0, 1.0);
  }
}
