#include "q4rules.h"

#include "../core/global.h"
#include "../external/nlohmann_json/json.hpp"

#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

Q4Rules::Q4Rules()
  : maxPlies(400),
    repetitionDrawCount(0) {
  for(int i = 0; i < 4; i++)
    initialWalls[i] = 7;
}

Q4Rules::Q4Rules(int plies, int repDraw, int walls)
  : maxPlies(plies),
    repetitionDrawCount(repDraw) {
  for(int i = 0; i < 4; i++)
    initialWalls[i] = walls;
  validate();
}

Q4Rules::Q4Rules(int plies, int repDraw, const int walls[4])
  : maxPlies(plies),
    repetitionDrawCount(repDraw) {
  for(int i = 0; i < 4; i++)
    initialWalls[i] = walls[i];
  validate();
}

void Q4Rules::validate() const {
  if(maxPlies < 1)
    throw StringError("Q4Rules: maxPlies must be >= 1");
  if(repetitionDrawCount < 0)
    throw StringError("Q4Rules: repetitionDrawCount must be >= 0");
  for(int i = 0; i < 4; i++) {
    if(initialWalls[i] < 0 || initialWalls[i] > 10)
      throw StringError("Q4Rules: initialWalls per seat must be between 0 and 10");
  }
}

bool Q4Rules::operator==(const Q4Rules& other) const {
  if(maxPlies != other.maxPlies || repetitionDrawCount != other.repetitionDrawCount)
    return false;
  for(int i = 0; i < 4; i++) {
    if(initialWalls[i] != other.initialWalls[i])
      return false;
  }
  return true;
}

bool Q4Rules::operator!=(const Q4Rules& other) const {
  return !(*this == other);
}

std::string Q4Rules::toString() const {
  std::ostringstream out;
  out << "Q4:maxPlies=" << maxPlies
      << ",repetitionDrawCount=" << repetitionDrawCount
      << ",initialWalls=" << initialWalls[0] << "," << initialWalls[1]
      << "," << initialWalls[2] << "," << initialWalls[3];
  return out.str();
}

std::string Q4Rules::toJson() const {
  json j;
  j["maxPlies"] = maxPlies;
  j["repetitionDrawCount"] = repetitionDrawCount;
  j["initialWalls"] = {initialWalls[0], initialWalls[1], initialWalls[2], initialWalls[3]};
  return j.dump();
}

Q4Rules Q4Rules::fromJson(const std::string& jsonStr) {
  json j = json::parse(jsonStr);
  Q4Rules rules;
  if(j.contains("maxPlies"))
    rules.maxPlies = j["maxPlies"].get<int>();
  if(j.contains("repetitionDrawCount"))
    rules.repetitionDrawCount = j["repetitionDrawCount"].get<int>();
  if(j.contains("initialWalls")) {
    if(j["initialWalls"].is_array()) {
      auto arr = j["initialWalls"];
      if(arr.size() == 4) {
        for(int i = 0; i < 4; i++)
          rules.initialWalls[i] = arr[i].get<int>();
      }
      else if(arr.size() == 1) {
        int w = arr[0].get<int>();
        for(int i = 0; i < 4; i++)
          rules.initialWalls[i] = w;
      }
    }
    else if(j["initialWalls"].is_number_integer()) {
      int w = j["initialWalls"].get<int>();
      for(int i = 0; i < 4; i++)
        rules.initialWalls[i] = w;
    }
  }
  rules.validate();
  return rules;
}

Q4Rules Q4Rules::parse(const std::string& str) {
  std::string s = Global::trim(str);
  if(s.rfind("Q4:", 0) == 0)
    s = s.substr(3);

  Q4Rules rules;
  std::vector<std::string> parts = Global::split(s, ',');
  for(size_t i = 0; i < parts.size(); i++) {
    std::string part = Global::trim(parts[i]);
    if(part.empty())
      continue;
    size_t eq = part.find('=');
    if(eq == std::string::npos)
      throw StringError("Invalid Q4 rules key=value: " + part);
    std::string key = Global::trim(part.substr(0, eq));
    std::string val = Global::trim(part.substr(eq + 1));

    if(key == "maxPlies") {
      rules.maxPlies = Global::stringToInt(val);
    }
    else if(key == "repetitionDrawCount") {
      rules.repetitionDrawCount = Global::stringToInt(val);
    }
    else if(key == "initialWalls") {
      // Could be single number or 4 comma-separated numbers; but notice comma separates parts.
      // If there were comma-separated numbers in initialWalls=7,7,7,7 then parts[i] was initialWalls=7
      // and subsequent parts were 7, 7, 7.
      // Let's handle both!
      std::vector<int> wallVals;
      wallVals.push_back(Global::stringToInt(val));
      while(wallVals.size() < 4 && i + 1 < parts.size() && parts[i + 1].find('=') == std::string::npos) {
        i++;
        wallVals.push_back(Global::stringToInt(Global::trim(parts[i])));
      }
      if(wallVals.size() == 1) {
        for(int sIdx = 0; sIdx < 4; sIdx++)
          rules.initialWalls[sIdx] = wallVals[0];
      }
      else if(wallVals.size() == 4) {
        for(int sIdx = 0; sIdx < 4; sIdx++)
          rules.initialWalls[sIdx] = wallVals[sIdx];
      }
      else {
        throw StringError("Invalid initialWalls in Q4 rules: expected 1 or 4 values");
      }
    }
    else {
      throw StringError("Unknown Q4 rules key: " + key);
    }
  }

  rules.validate();
  return rules;
}

Q4Rules Q4Rules::parseRulesOrJson(const std::string& str) {
  std::string s = Global::trim(str);
  if(s.rfind("{", 0) == 0)
    return fromJson(s);
  return parse(s);
}
