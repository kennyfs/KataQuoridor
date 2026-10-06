#include "q4history.h"

#include <algorithm>
#include <stdexcept>

Q4History::Q4History() : Q4History(Q4Rules()) {}

Q4History::Q4History(const Q4Rules& r) : Q4History(Q4Board(r), r) {}

Q4History::Q4History(const Q4Board& b, const Q4Rules& r) {
  clear(b, r);
}

void Q4History::clear(const Q4Board& b, const Q4Rules& r) {
  rules = r;
  initialBoard = b;
  currentBoard = b;
  boardHistory.clear();
  events.clear();
  repetitionHistory.clear();
  plies = 0;
  isFinished = false;
  winnerSeat = -1;
  isDraw = false;
  repetitionHashes.clear();
  recordRepetition();
  checkTerminal();
}

void Q4History::recordRepetition() {
  if(rules.repetitionDrawCount >= 2) {
    Hash128 h = currentBoard.hash;
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

void Q4History::checkTerminal() {
  if(isFinished)
    return;

  int w = currentBoard.getWinner();
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

void Q4History::play(int action) {
  if(isFinished)
    throw StringError("Cannot play in finished game");

  if(!currentBoard.isLegalAction(action, currentBoard.toMove))
    throw StringError("Illegal action in Q4History::play: " + Global::intToString(action));

  boardHistory.push_back(currentBoard);
  repetitionHistory.push_back(repetitionHashes);
  Q4Event ev;
  ev.isElimination = false;
  ev.action = action;
  ev.eliminatedSeat = -1;
  events.push_back(ev);

  bool wasWall = !Q4Board::isPawnAction(action);
  currentBoard.applyAction(action);
  plies++;

  if(wasWall) {
    repetitionHashes.clear();
  }
  recordRepetition();
  checkTerminal();
}

void Q4History::eliminate(int seat) {
  if(isFinished)
    throw StringError("Cannot eliminate in finished game");
  if(seat < 0 || seat >= Q4Board::NUM_SEATS || !currentBoard.isAlive(seat))
    return;

  boardHistory.push_back(currentBoard);
  repetitionHistory.push_back(repetitionHashes);
  Q4Event ev;
  ev.isElimination = true;
  ev.action = Q4Board::NULL_ACTION;
  ev.eliminatedSeat = seat;
  events.push_back(ev);

  currentBoard.eliminateSeat(seat);
  repetitionHashes.clear();
  recordRepetition();
  checkTerminal();
}

bool Q4History::undo() {
  if(boardHistory.empty() || events.empty() || repetitionHistory.empty())
    return false;

  currentBoard = boardHistory.back();
  boardHistory.pop_back();

  repetitionHashes = repetitionHistory.back();
  repetitionHistory.pop_back();

  Q4Event lastEv = events.back();
  events.pop_back();

  if(!lastEv.isElimination) {
    plies--;
  }

  isFinished = false;
  winnerSeat = -1;
  isDraw = false;
  return true;
}

int Q4History::currentPositionRepetitionCount() const {
  int count = 0;
  for(const auto& h : repetitionHashes) {
    if(h == currentBoard.hash)
      count++;
  }
  return count;
}

std::string Q4History::getResultString() const {
  if(winnerSeat >= 0)
    return Global::intToString(winnerSeat + 1) + "+";
  if(isDraw)
    return "Draw";
  return "none";
}
