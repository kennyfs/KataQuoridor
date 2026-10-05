#ifndef Q4_BOTS_H_
#define Q4_BOTS_H_

#include "../core/rand.h"
#include "q4board.h"

#include <memory>
#include <string>

class Q4Bot {
public:
  virtual ~Q4Bot() = default;
  virtual int getMove(const Q4Board& board) = 0;
  virtual std::string getName() const = 0;
};

namespace Q4Bots {
  int arrivalEstimate(const Q4Board& board, int seat);
  int findRaceLeader(const Q4Board& board);

  std::unique_ptr<Q4Bot> makeBot(const std::string& type, uint64_t seed, int targetSeat = -1);
}

#endif  // Q4_BOTS_H_
