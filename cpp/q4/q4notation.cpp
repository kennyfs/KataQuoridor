#include "q4notation.h"

#include "../core/global.h"

#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace Q4Notation {

std::string cellToString(int cell) {
  if(cell < 0 || cell >= Q4Board::NUM_CELLS)
    throw StringError("cellToString: cell index out of range: " + Global::intToString(cell));
  int x = Q4Board::cellX(cell);
  int y = Q4Board::cellY(cell);
  std::string s;
  s += (char)('a' + x);
  s += Global::intToString(y + 1);
  return s;
}

int stringToCell(const std::string& str) {
  std::string s = Global::trim(str);
  if(s.size() < 2 || s.size() > 3)
    throw StringError("stringToCell: invalid length for cell string: " + str);

  char colChar = std::tolower(s[0]);
  if(colChar < 'a' || colChar > 'k')
    throw StringError("stringToCell: column letter out of range (a-k): " + str);
  int x = colChar - 'a';

  int y1Based;
  try {
    y1Based = Global::stringToInt(s.substr(1));
  }
  catch(...) {
    throw StringError("stringToCell: invalid row number: " + str);
  }

  if(y1Based < 1 || y1Based > 11)
    throw StringError("stringToCell: row number out of range (1-11): " + str);
  int y = y1Based - 1;

  return Q4Board::cellOf(x, y);
}

std::string wallToString(int ax, int ay, bool isHorizontal) {
  if(ax < 0 || ax >= Q4Board::NUM_ANCHORS || ay < 0 || ay >= Q4Board::NUM_ANCHORS)
    throw StringError("wallToString: anchor out of range (" + Global::intToString(ax) + "," + Global::intToString(ay) + ")");
  std::string s;
  s += (char)('a' + ax);
  s += Global::intToString(ay + 1);
  s += (isHorizontal ? 'h' : 'v');
  return s;
}

std::string actionToString(int action) {
  if(action < 0 || action >= Q4Board::NUM_ACTIONS)
    throw StringError("actionToString: action index out of range: " + Global::intToString(action));
  if(Q4Board::isPawnAction(action)) {
    return cellToString(action);
  }
  if(Q4Board::isVWallAction(action)) {
    int a = action - 121;
    return wallToString(Q4Board::anchorX(a), Q4Board::anchorY(a), false);
  }
  if(Q4Board::isHWallAction(action)) {
    int a = action - 221;
    return wallToString(Q4Board::anchorX(a), Q4Board::anchorY(a), true);
  }
  throw StringError("actionToString: unrecognized action: " + Global::intToString(action));
}

int stringToAction(const std::string& str) {
  std::string s = Global::trim(str);
  if(s.empty())
    throw StringError("stringToAction: empty string");

  char lastChar = std::tolower(s.back());
  if(lastChar == 'h' || lastChar == 'v') {
    // Wall action
    if(s.size() < 3 || s.size() > 4)
      throw StringError("stringToAction: invalid wall string length: " + str);

    char colChar = std::tolower(s[0]);
    if(colChar < 'a' || colChar > 'j')
      throw StringError("stringToAction: wall anchor column out of range (a-j): " + str);
    int ax = colChar - 'a';

    std::string rowStr = s.substr(1, s.size() - 2);
    int ay1Based;
    try {
      ay1Based = Global::stringToInt(rowStr);
    }
    catch(...) {
      throw StringError("stringToAction: invalid anchor row number: " + str);
    }
    if(ay1Based < 1 || ay1Based > 10)
      throw StringError("stringToAction: wall anchor row out of range (1-10): " + str);
    int ay = ay1Based - 1;

    int a = Q4Board::anchorOf(ax, ay);
    return (lastChar == 'v') ? Q4Board::actionOfVWall(a) : Q4Board::actionOfHWall(a);
  }
  else {
    // Pawn move
    int cell = stringToCell(s);
    return Q4Board::actionOfPawn(cell);
  }
}

std::string seatToString(int seat) {
  if(seat < 0 || seat >= Q4Board::NUM_SEATS)
    throw StringError("seatToString: invalid seat: " + Global::intToString(seat));
  return Global::intToString(seat + 1);
}

int stringToSeat(const std::string& str) {
  std::string s = Global::trim(str);
  if(s.empty())
    throw StringError("stringToSeat: empty string");

  if(s[0] == 'p' || s[0] == 'P')
    s = s.substr(1);

  int num;
  try {
    num = Global::stringToInt(s);
  }
  catch(...) {
    throw StringError("stringToSeat: invalid seat string: " + str);
  }

  if(num >= 1 && num <= 4)
    return num - 1;
  throw StringError("stringToSeat: seat number out of range (1-4): " + str);
}

std::string renderBoardAscii(const Q4Board& board, int plies) {
  std::ostringstream out;
  out << "=== KataQuoridor Q4 (Four at a Table) ===\n";
  out << "Ply: " << plies << " | To move: Player " << (board.toMove + 1)
      << " | Alive: [";
  for(int s = 0; s < Q4Board::NUM_SEATS; s++) {
    if(s > 0) out << " ";
    out << (board.isAlive(s) ? Global::intToString(s + 1) : "X");
  }
  out << "] | Walls left: [";
  for(int s = 0; s < Q4Board::NUM_SEATS; s++) {
    if(s > 0) out << ", ";
    out << "P" << (s + 1) << ":" << (int)board.wallsLeft[s];
  }
  out << "]\n\n";

  // Rows from 10 down to 0
  for(int y = Q4Board::BOARD_SIZE - 1; y >= 0; y--) {
    // Print row of cells and vertical edges
    out << std::setw(2) << (y + 1) << " ";
    for(int x = 0; x < Q4Board::BOARD_SIZE; x++) {
      int c = Q4Board::cellOf(x, y);
      if(board.occupant[c] >= 0) {
        out << " " << (board.occupant[c] + 1) << " ";
      }
      else if(c == Q4Board::CENTER_CELL) {
        out << " * ";
      }
      else {
        out << " . ";
      }

      if(x < Q4Board::BOARD_SIZE - 1) {
        // Vertical boundary between (x, y) and (x+1, y)
        if(board.blocked[c] & (1 << Q4Board::DIR_E))
          out << "|";
        else
          out << " ";
      }
    }
    out << "\n";

    // Print horizontal boundary between row y and row y-1
    if(y > 0) {
      out << "   ";
      for(int x = 0; x < Q4Board::BOARD_SIZE; x++) {
        int c = Q4Board::cellOf(x, y);
        if(board.blocked[c] & (1 << Q4Board::DIR_S))
          out << "---";
        else
          out << "   ";

        if(x < Q4Board::BOARD_SIZE - 1) {
          out << "+";
        }
      }
      out << "\n";
    }
  }

  // Column letters along bottom
  out << "    ";
  for(int x = 0; x < Q4Board::BOARD_SIZE; x++) {
    out << " " << (char)('a' + x) << " ";
    if(x < Q4Board::BOARD_SIZE - 1) out << " ";
  }
  out << "\n";

  // Distances summary
  out << "Distances to center: ";
  for(int s = 0; s < Q4Board::NUM_SEATS; s++) {
    if(s > 0) out << ", ";
    out << "P" << (s + 1) << ": ";
    if(board.isAlive(s) && board.pawn[s] >= 0)
      out << (int)board.distToCenter[board.pawn[s]];
    else
      out << "elim";
  }
  out << "\n";

  return out.str();
}

}  // namespace Q4Notation
