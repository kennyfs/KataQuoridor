#include "q4record.h"
#include "q4notation.h"

#include "../core/global.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>

namespace {

const char SEAT_LETTERS[4] = {'S', 'W', 'N', 'E'};

int letterToSeat(const std::string& s) {
  if(s.size() == 1)
    for(int i = 0; i < 4; i++)
      if(s[0] == SEAT_LETTERS[i])
        return i;
  throw StringError("Q4 SGF: bad seat letter '" + s + "'");
}

std::string escapeText(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for(char c : s) {
    if(c == ']' || c == '\\')
      out += '\\';
    out += (c == '\n' || c == '\r') ? ' ' : c;
  }
  return out;
}

struct Prop {
  std::string id;
  std::string value;
};
typedef std::vector<Prop> Node;

// Splits "(;ID[v]ID[v];ID[v])" into its nodes.
std::vector<Node> parseSgfNodes(const std::string& line) {
  size_t n = line.size();
  size_t i = 0;
  while(i < n && (line[i] == ' ' || line[i] == '\t'))
    i++;
  if(i >= n || line[i] != '(')
    throw StringError("Q4 SGF: line does not start with '('");
  i++;
  std::vector<Node> nodes;
  bool closed = false;
  while(i < n) {
    char c = line[i];
    if(c == ';') {
      nodes.emplace_back();
      i++;
    }
    else if(c == ')') {
      closed = true;
      i++;
      break;
    }
    else if(c >= 'A' && c <= 'Z') {
      if(nodes.empty())
        throw StringError("Q4 SGF: property before the first node");
      size_t j = i;
      while(j < n && line[j] >= 'A' && line[j] <= 'Z')
        j++;
      Prop p;
      p.id = line.substr(i, j - i);
      if(j >= n || line[j] != '[')
        throw StringError("Q4 SGF: property " + p.id + " has no value");
      j++;
      bool end = false;
      while(j < n) {
        if(line[j] == '\\' && j + 1 < n) {
          p.value += line[j + 1];
          j += 2;
        }
        else if(line[j] == ']') {
          end = true;
          j++;
          break;
        }
        else {
          p.value += line[j++];
        }
      }
      if(!end)
        throw StringError("Q4 SGF: unterminated value of " + p.id);
      nodes.back().push_back(p);
      i = j;
    }
    else if(c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      i++;
    }
    else {
      throw StringError(std::string("Q4 SGF: unexpected character '") + c + "'");
    }
  }
  if(!closed)
    throw StringError("Q4 SGF: missing closing ')'");
  if(nodes.empty())
    throw StringError("Q4 SGF: no root node");
  return nodes;
}

const std::string* findProp(const Node& node, const std::string& id) {
  for(const Prop& p : node)
    if(p.id == id)
      return &p.value;
  return nullptr;
}

std::map<std::string, std::string> parseKV(const std::string& text) {
  std::map<std::string, std::string> kv;
  for(const std::string& part : Global::split(text, ',')) {
    size_t eq = part.find('=');
    if(eq == std::string::npos)
      continue;
    kv[Global::trim(part.substr(0, eq))] = Global::trim(part.substr(eq + 1));
  }
  return kv;
}

std::string resultToLetter(const std::string& result) {
  if(result == "Draw")
    return "0";
  if(result == "none")
    return "?";
  if(result.size() == 2 && result[1] == '+' && result[0] >= '1' && result[0] <= '4')
    return std::string(1, SEAT_LETTERS[result[0] - '1']);
  throw StringError("Q4 SGF: unknown result '" + result + "'");
}

std::string letterToResult(const std::string& s) {
  if(s == "0")
    return "Draw";
  if(s == "?")
    return "none";
  return Global::intToString(letterToSeat(s) + 1) + "+";
}

int toInt(const std::string& s, const char* what) {
  int v;
  if(!Global::tryStringToInt(s, v))
    throw StringError(std::string("Q4 SGF: bad integer for ") + what + ": '" + s + "'");
  return v;
}

}  // namespace

std::string Q4MoveComment::toString() const {
  char buf[160];
  std::snprintf(
    buf, sizeof(buf), "%.2f %.2f %.2f %.2f %.2f v=%lld weight=%.2f", p[0], p[1], p[2], p[3], p[4],
    (long long)visits, weight
  );
  return buf;
}

bool Q4MoveComment::parse(const std::string& text, Q4MoveComment& out) {
  out = Q4MoveComment();
  std::vector<std::string> tok = Global::split(Global::trim(text), ' ');
  std::vector<std::string> parts;
  for(const std::string& t : tok)
    if(!t.empty())
      parts.push_back(t);
  if(parts.size() < 5)
    return false;
  Q4MoveComment mc;
  for(int k = 0; k < 5; k++) {
    char* end = nullptr;
    mc.p[k] = std::strtof(parts[k].c_str(), &end);
    if(end == parts[k].c_str() || *end != 0)
      return false;
  }
  bool hasV = false;
  for(size_t k = 5; k < parts.size(); k++) {
    size_t eq = parts[k].find('=');
    if(eq == std::string::npos)
      continue;
    std::string key = parts[k].substr(0, eq);
    std::string val = parts[k].substr(eq + 1);
    if(key == "v") {
      mc.visits = std::strtoll(val.c_str(), nullptr, 10);
      hasV = true;
    }
    else if(key == "weight") {
      mc.weight = std::strtof(val.c_str(), nullptr);
    }
  }
  if(!hasV)
    return false;
  mc.valid = true;
  out = mc;
  return true;
}

Q4Record::Q4Record()
  : gtype("normal"), hasGameHash(false), gameHash0(0), gameHash1(0) {
  players.resize(4);
  for(int i = 0; i < 4; i++) {
    players[i].name = "P" + Global::intToString(i + 1);
    players[i].type = "bot";
  }
  result = "none";
}

void Q4Record::setMoveComment(size_t i, const Q4MoveComment& mc) {
  comments.resize(events.size());
  moveComments.resize(events.size());
  comments[i] = mc.valid ? mc.toString() : std::string();
  moveComments[i] = mc;
}

std::string Q4Record::toSgfLine() const {
  if(players.size() != 4)
    throw StringError("Q4 SGF: a record needs 4 players");
  std::ostringstream out;
  out << "(;FF[4]GM[Q4]SZ[11]";
  static const char* const PLAYER_IDS[4] = {"PS", "PW", "PN", "PE"};
  static const char* const WALL_IDS[4] = {"WS", "WW", "WN", "WE"};
  for(int s = 0; s < 4; s++)
    out << PLAYER_IDS[s] << "[" << escapeText(players[s].name) << "]";
  for(int s = 0; s < 4; s++)
    out << WALL_IDS[s] << "[" << rules.initialWalls[s] << "]";
  out << "RU[Q4:repetitionDrawCount=" << rules.repetitionDrawCount << ",maxPlies=" << rules.maxPlies << "]";
  out << "RE[" << resultToLetter(result) << "]";

  std::string dr = drawReason;
  if(result == "none")
    dr = "unfinished";
  else if(result == "Draw" && dr.empty()) {
    int plies = 0;
    for(const Q4Event& ev : events)
      if(!ev.isElimination)
        plies++;
    dr = plies >= rules.maxPlies ? "maxPlies" : "repetition";
  }
  else if(result != "Draw")
    dr = "";
  if(!dr.empty())
    out << "DR[" << dr << "]";

  std::ostringstream c;
  c << "gtype=" << (gtype.empty() ? "normal" : gtype);
  if(hasGameHash)
    c << ",gameHash=" << Global::strprintf("%016llx%016llx", (unsigned long long)gameHash0, (unsigned long long)gameHash1);
  c << ",startTurnIdx=" << startTurnIdx;
  for(int s = 0; s < 4; s++) {
    c << ",type" << s << "=" << players[s].type;
    if(!players[s].net.empty())
      c << ",net" << s << "=" << players[s].net;
    if(players[s].visits > 0)
      c << ",v" << s << "=" << players[s].visits;
  }
  if(matchOpening >= 0)
    c << ",table=" << matchTable << ",opening=" << matchOpening << ",rotation=" << matchRotation;
  out << "C[" << escapeText(c.str()) << "]";

  Q4History hist(rules);
  for(size_t i = 0; i < events.size(); i++) {
    const Q4Event& ev = events[i];
    out << ";";
    if(ev.isElimination) {
      out << "EL[" << SEAT_LETTERS[ev.eliminatedSeat] << "]";
      hist.eliminate(ev.eliminatedSeat);
    }
    else {
      out << SEAT_LETTERS[hist.currentBoard.toMove] << "[" << Q4Notation::actionToString(ev.action) << "]";
      hist.play(ev.action);
    }
    if(i < comments.size() && !comments[i].empty())
      out << "C[" << escapeText(comments[i]) << "]";
  }
  out << ")";
  return out.str();
}

Q4Record Q4Record::fromSgfLine(const std::string& line) {
  std::vector<Node> nodes = parseSgfNodes(line);
  const Node& root = nodes[0];
  Q4Record rec;

  const std::string* gm = findProp(root, "GM");
  if(gm == nullptr || *gm != "Q4")
    throw StringError("Q4 SGF: not a Q4 game (GM[" + (gm ? *gm : std::string()) + "])");
  const std::string* sz = findProp(root, "SZ");
  if(sz == nullptr || *sz != "11")
    throw StringError("Q4 SGF: SZ must be 11");

  static const char* const PLAYER_IDS[4] = {"PS", "PW", "PN", "PE"};
  static const char* const WALL_IDS[4] = {"WS", "WW", "WN", "WE"};
  for(int s = 0; s < 4; s++) {
    if(const std::string* v = findProp(root, PLAYER_IDS[s]))
      rec.players[s].name = *v;
    if(const std::string* v = findProp(root, WALL_IDS[s]))
      rec.rules.initialWalls[s] = toInt(*v, WALL_IDS[s]);
  }
  if(const std::string* ru = findProp(root, "RU")) {
    std::string body = *ru;
    if(body.compare(0, 3, "Q4:") == 0)
      body = body.substr(3);
    auto kv = parseKV(body);
    if(kv.count("maxPlies"))
      rec.rules.maxPlies = toInt(kv["maxPlies"], "maxPlies");
    if(kv.count("repetitionDrawCount"))
      rec.rules.repetitionDrawCount = toInt(kv["repetitionDrawCount"], "repetitionDrawCount");
  }
  rec.rules.validate();
  if(const std::string* re = findProp(root, "RE"))
    rec.result = letterToResult(*re);
  if(const std::string* dr = findProp(root, "DR"))
    rec.drawReason = *dr;
  if(const std::string* c = findProp(root, "C")) {
    auto kv = parseKV(*c);
    if(kv.count("gtype"))
      rec.gtype = kv["gtype"];
    if(kv.count("startTurnIdx"))
      rec.startTurnIdx = toInt(kv["startTurnIdx"], "startTurnIdx");
    if(kv.count("gameHash")) {
      const std::string& h = kv["gameHash"];
      if(h.size() != 32)
        throw StringError("Malformed gameHash in Q4 SGF: " + h);
      rec.hasGameHash = true;
      rec.gameHash0 = std::stoull(h.substr(0, 16), nullptr, 16);
      rec.gameHash1 = std::stoull(h.substr(16, 16), nullptr, 16);
    }
    for(int s = 0; s < 4; s++) {
      std::string si = Global::intToString(s);
      if(kv.count("type" + si))
        rec.players[s].type = kv["type" + si];
      if(kv.count("net" + si))
        rec.players[s].net = kv["net" + si];
      if(kv.count("v" + si))
        rec.players[s].visits = toInt(kv["v" + si], "v");
    }
    if(kv.count("table")) {
      rec.matchTable = kv["table"];
      rec.matchOpening = kv.count("opening") ? toInt(kv["opening"], "opening") : 0;
      rec.matchRotation = kv.count("rotation") ? toInt(kv["rotation"], "rotation") : 0;
    }
  }

  Q4History hist(rec.rules);
  bool anyComment = false;
  for(size_t k = 1; k < nodes.size(); k++) {
    const Node& node = nodes[k];
    Q4Event ev;
    ev.isElimination = false;
    ev.action = Q4Board::NULL_ACTION;
    ev.eliminatedSeat = -1;
    bool haveEvent = false;
    for(const Prop& p : node) {
      if(p.id == "EL") {
        ev.isElimination = true;
        ev.eliminatedSeat = letterToSeat(p.value);
        haveEvent = true;
      }
      else if(p.id == "S" || p.id == "W" || p.id == "N" || p.id == "E") {
        ev.action = Q4Notation::stringToAction(p.value);
        if(letterToSeat(p.id) != hist.currentBoard.toMove)
          throw StringError("Q4 SGF: " + p.id + "[" + p.value + "] is not the seat to move");
        haveEvent = true;
      }
    }
    if(!haveEvent)
      throw StringError("Q4 SGF: node without a move");
    if(ev.isElimination)
      hist.eliminate(ev.eliminatedSeat);
    else
      hist.play(ev.action);
    rec.events.push_back(ev);
    std::string text;
    if(const std::string* c = findProp(node, "C"))
      text = *c;
    rec.comments.push_back(text);
    Q4MoveComment mc;
    Q4MoveComment::parse(text, mc);
    rec.moveComments.push_back(mc);
    anyComment = anyComment || !text.empty();
  }
  if(!anyComment) {
    rec.comments.clear();
    rec.moveComments.clear();
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
