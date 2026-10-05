#ifndef Q4_RECORD_H_
#define Q4_RECORD_H_

#include "q4history.h"
#include "q4rules.h"

#include <string>
#include <vector>

struct Q4PlayerInfo {
  std::string name;
  std::string type;
  std::string net;
  int visits = 0;
};

struct Q4Record {
  Q4Rules rules;
  std::vector<Q4PlayerInfo> players; // size 4
  std::string result;                // "1+", "2+", "3+", "4+", "Draw", or "none"
  std::vector<Q4Event> events;
  std::vector<std::string> comments; // optional per-event comments

  Q4Record();

  std::string toJsonLine() const;
  static Q4Record fromJsonLine(const std::string& line);

  // Replays the record from start to finish into history
  void replay(Q4History& history) const;
};

#endif  // Q4_RECORD_H_
