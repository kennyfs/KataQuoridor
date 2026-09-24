#include "../game/rules.h"

#include "../external/nlohmann_json/json.hpp"

#include <sstream>

using namespace std;
using json = nlohmann::json;

Rules::Rules()
  : koRule(KO_POSITIONAL),
    scoringRule(SCORING_AREA),
    taxRule(TAX_NONE),
    multiStoneSuicideLegal(false),
    hasButton(false),
    whiteHandicapBonusRule(WHB_ZERO),
    friendlyPassOk(false),
    komi(0.0f)
{}

Rules::Rules(
  int kRule,
  int sRule,
  int tRule,
  bool suic,
  bool button,
  int whbRule,
  bool pOk,
  float km
)
  : koRule(kRule),
    scoringRule(sRule),
    taxRule(tRule),
    multiStoneSuicideLegal(suic),
    hasButton(button),
    whiteHandicapBonusRule(whbRule),
    friendlyPassOk(pOk),
    komi(km)
{}

Rules::~Rules() {}

bool Rules::operator==(const Rules& other) const {
  (void)other;
  return true;
}

bool Rules::operator!=(const Rules& other) const {
  return !(*this == other);
}

bool Rules::equalsIgnoringKomi(const Rules& other) const {
  (void)other;
  return true;
}

bool Rules::gameResultWillBeInteger() const {
  return true;
}

Rules Rules::getQuoridorRules() {
  return Rules();
}

Rules Rules::getTrompTaylorish() {
  return getQuoridorRules();
}

Rules Rules::getSimpleTerritory() {
  return getQuoridorRules();
}

set<string> Rules::koRuleStrings() { return {"POSITIONAL"}; }
set<string> Rules::scoringRuleStrings() { return {"AREA"}; }
set<string> Rules::taxRuleStrings() { return {"NONE"}; }
set<string> Rules::whiteHandicapBonusRuleStrings() { return {"0"}; }

int Rules::parseKoRule(const string&) { return KO_POSITIONAL; }
int Rules::parseScoringRule(const string&) { return SCORING_AREA; }
int Rules::parseTaxRule(const string&) { return TAX_NONE; }
int Rules::parseWhiteHandicapBonusRule(const string&) { return WHB_ZERO; }

string Rules::writeKoRule(int) { return "POSITIONAL"; }
string Rules::writeScoringRule(int) { return "AREA"; }
string Rules::writeTaxRule(int) { return "NONE"; }
string Rules::writeWhiteHandicapBonusRule(int) { return "0"; }

bool Rules::komiIsIntOrHalfInt(float) { return true; }

Rules Rules::parseRules(const string& str) {
  Rules rules;
  if(!tryParseRules(str, rules))
    throw StringError("Unknown rules: " + str);
  return rules;
}

Rules Rules::parseRulesWithoutKomi(const string& str, float komi) {
  Rules rules;
  if(!tryParseRulesWithoutKomi(str, rules, komi))
    throw StringError("Unknown rules: " + str);
  return rules;
}

bool Rules::tryParseRules(const string& str, Rules& buf) {
  string s = Global::trim(str);
  if(Global::isEqualCaseInsensitive(s, "quoridor") ||
     Global::isEqualCaseInsensitive(s, "default") ||
     Global::isEqualCaseInsensitive(s, "tromptaylor")) {
    buf = getQuoridorRules();
    return true;
  }
  try {
    json input = json::parse(s);
    if(input.is_object()) {
      buf = getQuoridorRules();
      return true;
    }
  }
  catch(...) {
    return false;
  }
  return false;
}

bool Rules::tryParseRulesWithoutKomi(const string& str, Rules& buf, float komi) {
  if(!tryParseRules(str, buf))
    return false;
  buf.komi = komi;
  return true;
}

Rules Rules::updateRules(const string&, const string&, const Rules& priorRules) {
  return priorRules;
}

string Rules::toString() const {
  return "Quoridor";
}

string Rules::toStringNoKomi() const {
  return toString();
}

string Rules::toStringNoKomiMaybeNice() const {
  return toString();
}

json Rules::toJson() const {
  json ret;
  // Minimal fields to satisfy any older consumers
  ret["ko"] = "POSITIONAL";
  ret["scoring"] = "AREA";
  ret["tax"] = "NONE";
  ret["suicide"] = false;
  ret["hasButton"] = false;
  ret["whiteHandicapBonus"] = "0";
  ret["friendlyPassOk"] = false;
  ret["komi"] = komi;
  return ret;
}

json Rules::toJsonNoKomi() const {
  return toJson();
}

json Rules::toJsonNoKomiMaybeOmitStuff() const {
  return toJson();
}

string Rules::toJsonString() const {
  return toJson().dump();
}

string Rules::toJsonStringNoKomi() const {
  return toJsonString();
}

string Rules::toJsonStringNoKomiMaybeOmitStuff() const {
  return toJsonString();
}

ostream& operator<<(ostream& out, const Rules& rules) {
  out << rules.toJsonString();
  return out;
}

const Hash128 Rules::ZOBRIST_KO_RULE_HASH[4] = {
  Hash128(0ULL, 0ULL),
  Hash128(0ULL, 0ULL),
  Hash128(0ULL, 0ULL),
  Hash128(0ULL, 0ULL)
};

const Hash128 Rules::ZOBRIST_SCORING_RULE_HASH[2] = {
  Hash128(0ULL, 0ULL),
  Hash128(0ULL, 0ULL)
};

const Hash128 Rules::ZOBRIST_TAX_RULE_HASH[3] = {
  Hash128(0ULL, 0ULL),
  Hash128(0ULL, 0ULL),
  Hash128(0ULL, 0ULL)
};

const Hash128 Rules::ZOBRIST_MULTI_STONE_SUICIDE_HASH = Hash128(0ULL, 0ULL);
const Hash128 Rules::ZOBRIST_BUTTON_HASH = Hash128(0ULL, 0ULL);
const Hash128 Rules::ZOBRIST_FRIENDLY_PASS_OK_HASH = Hash128(0ULL, 0ULL);
const Hash128 Rules::ZOBRIST_PASS_ALIVE_UNDER_SUICIDE_HASH = Hash128(0ULL, 0ULL);
const Hash128 Rules::ZOBRIST_EXCLUDE_TERRITORY_ADJ_ATARI_HASH = Hash128(0ULL, 0ULL);
