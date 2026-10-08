#include "q4record.h"
#include "q4notation.h"

#include "../core/global.h"
#include "../external/nlohmann_json/json.hpp"

using json = nlohmann::json;

Q4Record::Q4Record()
  : hasGameHash(false), gameHash0(0), gameHash1(0) {
  players.resize(4);
  for(int i = 0; i < 4; i++) {
    players[i].name = "P" + Global::intToString(i + 1);
    players[i].type = "bot";
  }
  result = "none";
}

std::string Q4Record::toJsonLine() const {
  json j;

  // Rules
  json rj;
  rj["maxPlies"] = rules.maxPlies;
  rj["repetitionDrawCount"] = rules.repetitionDrawCount;
  rj["initialWalls"] = {rules.initialWalls[0], rules.initialWalls[1],
                        rules.initialWalls[2], rules.initialWalls[3]};
  j["rules"] = rj;

  // Players
  json pj = json::array();
  for(const auto& p : players) {
    json item;
    item["name"] = p.name;
    item["type"] = p.type;
    if(!p.net.empty()) item["net"] = p.net;
    if(p.visits > 0) item["visits"] = p.visits;
    pj.push_back(item);
  }
  j["players"] = pj;

  j["result"] = result;

  // Events
  json ej = json::array();
  for(const auto& ev : events) {
    json item;
    if(ev.isElimination) {
      item["elim"] = ev.eliminatedSeat + 1; // 1-based seat notation in JSON
    }
    else {
      item["a"] = Q4Notation::actionToString(ev.action);
    }
    ej.push_back(item);
  }
  j["events"] = ej;

  if(!comments.empty()) {
    j["comments"] = comments;
  }

  if(matchOpening >= 0) {
    json mj;
    mj["table"] = matchTable;
    mj["opening"] = matchOpening;
    mj["rotation"] = matchRotation;
    mj["openingPlies"] = matchOpeningPlies;
    mj["drawReason"] = drawReason;
    j["match"] = mj;
  }

  if(hasGameHash)
    j["gameHash"] = Global::strprintf("%016llx%016llx", (unsigned long long)gameHash0, (unsigned long long)gameHash1);

  return j.dump();
}

Q4Record Q4Record::fromJsonLine(const std::string& line) {
  json j = json::parse(line);
  Q4Record rec;

  if(j.contains("rules")) {
    rec.rules = Q4Rules::fromJson(j["rules"].dump());
  }

  if(j.contains("players") && j["players"].is_array()) {
    rec.players.clear();
    for(const auto& item : j["players"]) {
      Q4PlayerInfo p;
      if(item.contains("name")) p.name = item["name"].get<std::string>();
      if(item.contains("type")) p.type = item["type"].get<std::string>();
      if(item.contains("net")) p.net = item["net"].get<std::string>();
      if(item.contains("visits")) p.visits = item["visits"].get<int>();
      rec.players.push_back(p);
    }
    while(rec.players.size() < 4) {
      Q4PlayerInfo p;
      p.name = "P" + Global::intToString((int)rec.players.size() + 1);
      rec.players.push_back(p);
    }
  }

  if(j.contains("result")) {
    rec.result = j["result"].get<std::string>();
  }

  if(j.contains("events") && j["events"].is_array()) {
    rec.events.clear();
    for(const auto& item : j["events"]) {
      Q4Event ev;
      if(item.contains("elim")) {
        ev.isElimination = true;
        ev.action = Q4Board::NULL_ACTION;
        ev.eliminatedSeat = item["elim"].get<int>() - 1; // Convert back to 0-based
      }
      else if(item.contains("a")) {
        ev.isElimination = false;
        ev.action = Q4Notation::stringToAction(item["a"].get<std::string>());
        ev.eliminatedSeat = -1;
      }
      else {
        throw StringError("Malformed event in Q4Record: " + item.dump());
      }
      rec.events.push_back(ev);
    }
  }

  if(j.contains("comments") && j["comments"].is_array()) {
    rec.comments = j["comments"].get<std::vector<std::string>>();
  }

  if(j.contains("match")) {
    const auto& mj = j["match"];
    rec.matchTable = mj.value("table", std::string());
    rec.matchOpening = mj.value("opening", 0);
    rec.matchRotation = mj.value("rotation", 0);
    rec.matchOpeningPlies = mj.value("openingPlies", 0);
    rec.drawReason = mj.value("drawReason", std::string());
  }

  if(j.contains("gameHash")) {
    std::string h = j["gameHash"].get<std::string>();
    if(h.size() != 32)
      throw StringError("Malformed gameHash in Q4Record: " + h);
    rec.hasGameHash = true;
    rec.gameHash0 = std::stoull(h.substr(0, 16), nullptr, 16);
    rec.gameHash1 = std::stoull(h.substr(16, 16), nullptr, 16);
  }

  return rec;
}

void Q4Record::replay(Q4History& history) const {
  history.clear(Q4Board(rules), rules);
  for(const auto& ev : events) {
    if(ev.isElimination) {
      history.eliminate(ev.eliminatedSeat);
    }
    else {
      history.play(ev.action);
    }
  }
}
