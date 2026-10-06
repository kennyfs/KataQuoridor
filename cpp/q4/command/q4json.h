#ifndef Q4_JSON_H_
#define Q4_JSON_H_

#include "../../external/nlohmann_json/json.hpp"
#include "../q4board.h"
#include "../q4history.h"

// JSON position format shared by q4tool's subcommands (defined in q4tool.cpp): a history is
// {"rules": {...}, "initialBoard": {...} (optional), "events": [{"action": "e5h"} | {"elim": 2}, ...]}, or just a
// board {"pawns": ..., ...}.
Q4Board parseBoardFromJson(const nlohmann::json& j);
nlohmann::json boardToJson(const Q4Board& board);
Q4History parseHistoryFromJson(const nlohmann::json& j);

#endif  // Q4_JSON_H_
