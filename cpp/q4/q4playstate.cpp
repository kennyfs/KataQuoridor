#include "q4playstate.h"

#include <algorithm>
#include <stdexcept>

Q4PlayState::Q4PlayState() : Q4PlayState(Q4Rules()) {}

Q4PlayState::Q4PlayState(const Q4Rules& r) : Q4PlayState(Q4Board(r), r) {}

Q4PlayState::Q4PlayState(const Q4Board& b, const Q4Rules& r) {
  clear(b, r);
}

void Q4PlayState::clear(const Q4Board& b, const Q4Rules& r) {
  rules = r;
  board = b;
  plies = 0;
  isFinished = false;
  winnerSeat = -1;
  isDraw = false;
  repetitionHashes.clear();
  recordRepetition();
  checkTerminal();
}

void Q4PlayState::recordRepetition() {
  if(rules.repetitionDrawCount >= 2) {
    Hash128 h = board.hash;
    repetitionHashes.push_back(h);
    int count = 0;
    for(const auto& prevHash : repetitionHashes) {
      if(prevHash == h)
        count++;
    }
    if(count >= rules.repetitionDrawCount) {
      isDraw = true;
      isFinished = true;
    }
  }
}

void Q4PlayState::checkTerminal() {
  if(isFinished)
    return;

  int w = board.getWinner();
  if(w >= 0) {
    winnerSeat = w;
    isFinished = true;
    return;
  }

  if(plies >= rules.maxPlies) {
    isDraw = true;
    isFinished = true;
    return;
  }
}

void Q4PlayState::playAssumeLegal(int action) {
  if(isFinished)
    throw StringError("Cannot play in finished game");

  bool wasWall = !Q4Board::isPawnAction(action);
  board.applyAction(action);
  plies++;

  if(wasWall) {
    repetitionHashes.clear();
  }
  recordRepetition();
  checkTerminal();
}

void Q4PlayState::eliminate(int seat) {
  if(isFinished)
    throw StringError("Cannot eliminate in finished game");
  if(seat < 0 || seat >= Q4Board::NUM_SEATS || !board.isAlive(seat))
    return;

  board.eliminateSeat(seat);
  repetitionHashes.clear();
  recordRepetition();
  checkTerminal();
}

int Q4PlayState::currentPositionRepetitionCount() const {
  int count = 0;
  for(const auto& h : repetitionHashes) {
    if(h == board.hash)
      count++;
  }
  return count;
}

std::string Q4PlayState::getResultString() const {
  if(winnerSeat >= 0)
    return Global::intToString(winnerSeat + 1) + "+";
  if(isDraw)
    return "Draw";
  return "none";
}
