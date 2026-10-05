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
  else {
    recordRepetition();
  }

  checkTerminal();
}

void Q4History::eliminate(int seat) {
  if(isFinished)
    throw StringError("Cannot eliminate in finished game");
  if(seat < 0 || seat >= Q4Board::NUM_SEATS || !currentBoard.isAlive(seat))
    return;

  boardHistory.push_back(currentBoard);
  Q4Event ev;
  ev.isElimination = true;
  ev.action = Q4Board::NULL_ACTION;
  ev.eliminatedSeat = seat;
  events.push_back(ev);

  currentBoard.eliminateSeat(seat);
  repetitionHashes.clear();

  checkTerminal();
}

bool Q4History::undo() {
  if(boardHistory.empty() || events.empty())
    return false;

  currentBoard = boardHistory.back();
  boardHistory.pop_back();

  Q4Event lastEv = events.back();
  events.pop_back();

  if(!lastEv.isElimination) {
    plies--;
  }

  // Rebuild repetition history from scratch since last wall or elimination
  repetitionHashes.clear();
  if(rules.repetitionDrawCount >= 2) {
    // Find index of most recent wall or elimination in history
    int startIdx = 0;
    for(int i = (int)events.size() - 1; i >= 0; i--) {
      if(events[i].isElimination || !Q4Board::isPawnAction(events[i].action)) {
        startIdx = i + 1;
        break;
      }
    }
    // Replay hashes from startIdx to current
    if(startIdx == 0) {
      repetitionHashes.push_back(initialBoard.hash);
    }
    for(size_t i = startIdx; i < boardHistory.size(); i++) {
      // The state after event i is boardHistory[i+1] or currentBoard
      if(i + 1 < boardHistory.size())
        repetitionHashes.push_back(boardHistory[i + 1].hash);
      else
        repetitionHashes.push_back(currentBoard.hash);
    }
    if(startIdx > 0 && startIdx == (int)events.size()) {
      repetitionHashes.push_back(currentBoard.hash);
    }
  }

  isFinished = false;
  winnerSeat = -1;
  isDraw = false;

  // Re-check repetition draw on the restored state
  if(rules.repetitionDrawCount >= 2 && !repetitionHashes.empty()) {
    Hash128 h = currentBoard.hash;
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

  checkTerminal();
  return true;
}

std::string Q4History::getResultString() const {
  if(winnerSeat >= 0)
    return Global::intToString(winnerSeat + 1) + "+";
  if(isDraw)
    return "Draw";
  return "none";
}
