#ifndef Q4_NOTATION_H_
#define Q4_NOTATION_H_

#include "q4board.h"

#include <string>

namespace Q4Notation {
  std::string cellToString(int cell);
  int stringToCell(const std::string& str);

  std::string wallToString(int ax, int ay, bool isHorizontal);

  std::string actionToString(int action);
  int stringToAction(const std::string& str);

  std::string seatToString(int seat);
  int stringToSeat(const std::string& str);

  std::string renderBoardAscii(const Q4Board& board, int plies = 0);
}

#endif  // Q4_NOTATION_H_
