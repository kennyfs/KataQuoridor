#include "q4history.h"

#include <algorithm>
#include <stdexcept>

Q4History::Q4History() : Q4History(Q4Rules()) {}

Q4History::Q4History(const Q4Rules& r) : Q4History(Q4Board(r), r) {}

Q4History::Q4History(const Q4Board& b, const Q4Rules& r) {
  clear(b, r);
}

void Q4History::syncFromState() {
  rules = state.rules;
  currentBoard = state.board;
  plies = state.plies;
  isFinished = state.isFinished;
  winnerSeat = state.winnerSeat;
  isDraw = state.isDraw;
  repetitionHashes = state.repetitionHashes;
}

void Q4History::clear(const Q4Board& b, const Q4Rules& r) {
  initialBoard = b;
  boardHistory.clear();
  events.clear();
  repetitionHistory.clear();
  state.clear(b, r);
  syncFromState();
}

void Q4History::play(int action) {
  if(state.isFinished)
    throw StringError("Cannot play in finished game");

  if(!state.board.isLegalAction(action, state.board.toMove))
    throw StringError("Illegal action in Q4History::play: " + Global::intToString(action));

  boardHistory.push_back(state.board);
  repetitionHistory.push_back(state.repetitionHashes);
  Q4Event ev;
  ev.isElimination = false;
  ev.action = action;
  ev.eliminatedSeat = -1;
  events.push_back(ev);

  state.playAssumeLegal(action);
  syncFromState();
}

void Q4History::eliminate(int seat) {
  if(state.isFinished)
    throw StringError("Cannot eliminate in finished game");
  if(seat < 0 || seat >= Q4Board::NUM_SEATS || !state.board.isAlive(seat))
    return;

  boardHistory.push_back(state.board);
  repetitionHistory.push_back(state.repetitionHashes);
  Q4Event ev;
  ev.isElimination = true;
  ev.action = Q4Board::NULL_ACTION;
  ev.eliminatedSeat = seat;
  events.push_back(ev);

  state.eliminate(seat);
  syncFromState();
}

bool Q4History::undo() {
  if(boardHistory.empty() || events.empty() || repetitionHistory.empty())
    return false;

  state.board = boardHistory.back();
  boardHistory.pop_back();

  state.repetitionHashes = repetitionHistory.back();
  repetitionHistory.pop_back();

  Q4Event lastEv = events.back();
  events.pop_back();

  if(!lastEv.isElimination) {
    state.plies--;
  }

  state.isFinished = false;
  state.winnerSeat = -1;
  state.isDraw = false;
  syncFromState();
  return true;
}

int Q4History::currentPositionRepetitionCount() const {
  return state.currentPositionRepetitionCount();
}

std::string Q4History::getResultString() const {
  return state.getResultString();
}
