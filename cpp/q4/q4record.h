#ifndef Q4_RECORD_H_
#define Q4_RECORD_H_

#include "q4history.h"
#include "q4rules.h"

#include <cstdint>
#include <string>
#include <vector>

struct Q4PlayerInfo {
  std::string name;
  std::string type;
  std::string net;
  int visits = 0;
};

// The search values at the position before a move, from the absolute seats' point of view (docs/q4/Q4Sgf.md §4):
// "0.31 0.22 0.27 0.15 0.05 v=600 weight=1.00". valid is false for a move without a search.
struct Q4MoveComment {
  bool valid = false;
  float p[5] = {0, 0, 0, 0, 0};   // win probabilities of seats 0..3, then the draw
  int64_t visits = 0;             // the unreduced visits of the search
  float weight = 0.0f;            // the training target weight of the turn

  std::string toString() const;
  // Parses the text of a move comment; returns false (and leaves valid = false) if it is not one.
  static bool parse(const std::string& text, Q4MoveComment& out);
};

struct Q4Record {
  Q4Rules rules;
  std::vector<Q4PlayerInfo> players; // size 4
  std::string result;                // "1+", "2+", "3+", "4+", "Draw", or "none"
  std::vector<Q4Event> events;
  // Optional per-event comments: the raw text (written as is) and the parsed move comment. Both are empty or have
  // one entry per event.
  std::vector<std::string> comments;
  std::vector<Q4MoveComment> moveComments;

  std::string gtype;                 // "normal", "fork", "mixed" (root C: gtype)
  int startTurnIdx = 0;              // opening plies that were not played by the engines (root C: startTurnIdx)
  std::string drawReason;            // "", "maxPlies", "repetition" or "unfinished"

  // Optional match metadata (written by q4match and the gatekeeper); matchOpening < 0 means none.
  std::string matchTable;
  int matchOpening = -1;
  int matchRotation = 0;

  // Optional: the game hash of the training rows (globalTargetsNC C44-49), written by q4selfplay.
  // Root C: gameHash = 32 hex digits, hash0 then hash1.
  bool hasGameHash;
  uint64_t gameHash0;
  uint64_t gameHash1;

  Q4Record();

  // Sets the comment of event i (resizing the comment vectors to the number of events).
  void setMoveComment(size_t i, const Q4MoveComment& mc);

  // One game as one Duel-style SGF line, without the final newline (docs/q4/Q4Sgf.md).
  std::string toSgfLine() const;
  static Q4Record fromSgfLine(const std::string& line);

  // Replays the record from start to finish into history
  void replay(Q4History& history) const;
};

#endif  // Q4_RECORD_H_
