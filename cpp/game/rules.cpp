#include "../game/rules.h"

#include "../game/board.h"

#include "../external/nlohmann_json/json.hpp"

#include <cmath>
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
    komi(DEFAULT_KOMI),
    maxPlies(DEFAULT_MAX_PLIES),
    timeBonusPerPly(0.0f),
    blackInitialFences(DEFAULT_INITIAL_FENCES),
    whiteInitialFences(DEFAULT_INITIAL_FENCES)
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
    komi(km),
    maxPlies(DEFAULT_MAX_PLIES),
    timeBonusPerPly(0.0f),
    blackInitialFences(DEFAULT_INITIAL_FENCES),
    whiteInitialFences(DEFAULT_INITIAL_FENCES)
{}

Rules::~Rules() {}

bool Rules::operator==(const Rules& other) const {
  return komi == other.komi && equalsIgnoringKomi(other);
}

bool Rules::operator!=(const Rules& other) const {
  return !(*this == other);
}

//The Go rule fields are inert stubs, so they are not compared.
bool Rules::equalsIgnoringKomi(const Rules& other) const {
  return
    maxPlies == other.maxPlies &&
    timeBonusPerPly == other.timeBonusPerPly &&
    blackInitialFences == other.blackInitialFences &&
    whiteInitialFences == other.whiteInitialFences;
}

bool Rules::gameResultWillBeInteger() const {
  return false;
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

bool Rules::komiIsIntOrHalfInt(float komi) {
  return std::isfinite(komi) && komi * 2 == (int)(komi * 2);
}

bool Rules::isValidKomi(float komi) {
  if(!std::isfinite(komi) || std::fabs(komi) > MAX_KOMI)
    return false;
  double k = (double)komi - 0.5;
  return k == std::floor(k);
}

float Rules::roundKomi(double komi) {
  if(!std::isfinite(komi))
    throw StringError("Komi is not finite");
  double rounded = std::floor(komi) + 0.5;
  if(rounded < -MAX_KOMI) rounded = -MAX_KOMI;
  if(rounded > MAX_KOMI) rounded = MAX_KOMI;
  return (float)rounded;
}

void Rules::validateOrThrow(const string& what) const {
  if(!isValidKomi(komi))
    throw StringError(what + ": komi must be a half-integer (n + 0.5) with |komi| <= " + Global::floatToString(MAX_KOMI) + ", got " + Global::floatToString(komi));
  if(maxPlies < 1 || maxPlies > MAX_MAX_PLIES)
    throw StringError(what + ": maxPlies must be in [1, " + Global::intToString(MAX_MAX_PLIES) + "], got " + Global::intToString(maxPlies));
  if(!std::isfinite(timeBonusPerPly) || timeBonusPerPly < 0.0f || timeBonusPerPly > MAX_TIME_BONUS_PER_PLY)
    throw StringError(what + ": timeBonusPerPly must be in [0, " + Global::floatToString(MAX_TIME_BONUS_PER_PLY) + "], got " + Global::floatToString(timeBonusPerPly));
  if(blackInitialFences < 0 || blackInitialFences > Board::MAX_FENCE_NUM)
    throw StringError(what + ": blackInitialWalls must be in [0, " + Global::intToString(Board::MAX_FENCE_NUM) + "], got " + Global::intToString(blackInitialFences));
  if(whiteInitialFences < 0 || whiteInitialFences > Board::MAX_FENCE_NUM)
    throw StringError(what + ": whiteInitialWalls must be in [0, " + Global::intToString(Board::MAX_FENCE_NUM) + "], got " + Global::intToString(whiteInitialFences));
}

//--------------------------------------------------------------------------------------------------------
//Serialization. Keys: komi (only where komi is included), maxPlies, timeBonusPerPly, blackInitialWalls,
//whiteInitialWalls. The string form is "Quoridor" for the standard game, else "Quoridor:key=value,..." listing the
//non-default keys; it is what SGF RU holds.

static const char* KEY_KOMI = "komi";
static const char* KEY_MAX_PLIES = "maxPlies";
static const char* KEY_TIME_BONUS = "timeBonusPerPly";
static const char* KEY_BLACK_WALLS = "blackInitialWalls";
static const char* KEY_WHITE_WALLS = "whiteInitialWalls";

//Go rule keys that older consumers may still send. Accepted and ignored.
static const set<string> LEGACY_GO_KEYS = {
  "ko", "scoring", "tax", "suicide", "hasButton", "whiteHandicapBonus", "friendlyPassOk"
};

set<string> Rules::quoridorRuleKeys() {
  return {KEY_MAX_PLIES, KEY_TIME_BONUS, KEY_BLACK_WALLS, KEY_WHITE_WALLS};
}

static string canonicalKey(const string& key) {
  for(const string& k : {string(KEY_KOMI), string(KEY_MAX_PLIES), string(KEY_TIME_BONUS), string(KEY_BLACK_WALLS), string(KEY_WHITE_WALLS)}) {
    if(Global::isEqualCaseInsensitive(key, k))
      return k;
  }
  return "";
}

//Sets one key from a string value, throwing StringError if the key is unknown or the value doesn't parse.
//Range checks are left to validateOrThrow.
static void setRuleFromString(Rules& rules, const string& keyOrig, const string& valueOrig, bool allowKomi) {
  string key = canonicalKey(Global::trim(keyOrig));
  string value = Global::trim(valueOrig);
  if(key == "")
    throw StringError("Unknown Quoridor rule: '" + keyOrig + "' (known: maxPlies, timeBonusPerPly, blackInitialWalls, whiteInitialWalls)");
  if(key == KEY_KOMI) {
    if(!allowKomi)
      throw StringError("komi is not a rule here, use the komi command");
    if(!Global::tryStringToFloat(value, rules.komi))
      throw StringError("Could not parse komi: '" + value + "'");
  }
  else if(key == KEY_TIME_BONUS) {
    if(!Global::tryStringToFloat(value, rules.timeBonusPerPly))
      throw StringError("Could not parse timeBonusPerPly: '" + value + "'");
  }
  else {
    int x;
    if(!Global::tryStringToInt(value, x))
      throw StringError("Could not parse " + key + " as an integer: '" + value + "'");
    if(key == KEY_MAX_PLIES) rules.maxPlies = x;
    else if(key == KEY_BLACK_WALLS) rules.blackInitialFences = x;
    else if(key == KEY_WHITE_WALLS) rules.whiteInitialFences = x;
    else ASSERT_UNREACHABLE;
  }
}

static Rules parseRulesOrThrow(const string& str) {
  string s = Global::trim(str);
  Rules rules = Rules::getQuoridorRules();
  if(Global::isEqualCaseInsensitive(s, "quoridor") ||
     Global::isEqualCaseInsensitive(s, "default") ||
     Global::isEqualCaseInsensitive(s, "tromptaylor")) {
    return rules;
  }
  if(s.size() > 9 && Global::isEqualCaseInsensitive(s.substr(0,9), "quoridor:")) {
    for(const string& item : Global::split(s.substr(9), ',')) {
      vector<string> kv = Global::split(item, '=');
      if(kv.size() != 2)
        throw StringError("Could not parse rules: '" + str + "'");
      setRuleFromString(rules, kv[0], kv[1], true);
    }
    rules.validateOrThrow("Rules '" + str + "'");
    return rules;
  }

  json input;
  try {
    input = json::parse(s);
  }
  catch(nlohmann::detail::exception&) {
    throw StringError("Could not parse rules: '" + str + "'");
  }
  if(!input.is_object())
    throw StringError("Could not parse rules: '" + str + "'");
  for(auto it = input.begin(); it != input.end(); ++it) {
    const string& key = it.key();
    if(LEGACY_GO_KEYS.count(key) > 0)
      continue;
    const json& v = it.value();
    string value;
    if(v.is_string())
      value = v.get<string>();
    else if(v.is_number_integer())
      value = Global::int64ToString(v.get<int64_t>());
    else if(v.is_number())
      value = Global::doubleToStringHighPrecision(v.get<double>());
    else
      throw StringError("Could not parse rules: '" + str + "', bad value for " + key);
    setRuleFromString(rules, key, value, true);
  }
  rules.validateOrThrow("Rules '" + str + "'");
  return rules;
}

Rules Rules::parseRules(const string& str) {
  return parseRulesOrThrow(str);
}

Rules Rules::parseRulesWithoutKomi(const string& str, float komi) {
  Rules rules = parseRulesOrThrow(str);
  rules.komi = komi;
  rules.validateOrThrow("Rules '" + str + "'");
  return rules;
}

bool Rules::tryParseRules(const string& str, Rules& buf) {
  try {
    buf = parseRules(str);
    return true;
  }
  catch(const StringError&) {
    return false;
  }
}

bool Rules::tryParseRulesWithoutKomi(const string& str, Rules& buf, float komi) {
  try {
    buf = parseRulesWithoutKomi(str, komi);
    return true;
  }
  catch(const StringError&) {
    return false;
  }
}

Rules Rules::updateRules(const string& key, const string& value, const Rules& priorRules) {
  Rules rules = priorRules;
  setRuleFromString(rules, key, value, false);
  rules.validateOrThrow("Rule " + key);
  return rules;
}

static string toStringHelper(const Rules& rules, bool includeKomi) {
  const Rules d = Rules::getQuoridorRules();
  vector<string> items;
  if(includeKomi && rules.komi != d.komi)
    items.push_back(string(KEY_KOMI) + "=" + Global::floatToString(rules.komi));
  if(rules.maxPlies != d.maxPlies)
    items.push_back(string(KEY_MAX_PLIES) + "=" + Global::intToString(rules.maxPlies));
  if(rules.timeBonusPerPly != d.timeBonusPerPly)
    items.push_back(string(KEY_TIME_BONUS) + "=" + Global::floatToString(rules.timeBonusPerPly));
  if(rules.blackInitialFences != d.blackInitialFences)
    items.push_back(string(KEY_BLACK_WALLS) + "=" + Global::intToString(rules.blackInitialFences));
  if(rules.whiteInitialFences != d.whiteInitialFences)
    items.push_back(string(KEY_WHITE_WALLS) + "=" + Global::intToString(rules.whiteInitialFences));
  if(items.size() <= 0)
    return "Quoridor";
  return "Quoridor:" + Global::concat(items, ",");
}

string Rules::toString() const {
  return toStringHelper(*this, true);
}

string Rules::toStringNoKomi() const {
  return toStringHelper(*this, false);
}

string Rules::toStringNoKomiMaybeNice() const {
  return toStringNoKomi();
}

//Floats are written with their shortest decimal form (0.05, not 0.05000000074505806).
static double niceDouble(float x) {
  return std::stod(Global::floatToString(x));
}

json Rules::toJson() const {
  json ret = toJsonNoKomi();
  ret[KEY_KOMI] = niceDouble(komi);
  return ret;
}

json Rules::toJsonNoKomi() const {
  json ret;
  ret[KEY_MAX_PLIES] = maxPlies;
  ret[KEY_TIME_BONUS] = niceDouble(timeBonusPerPly);
  ret[KEY_BLACK_WALLS] = blackInitialFences;
  ret[KEY_WHITE_WALLS] = whiteInitialFences;
  return ret;
}

json Rules::toJsonNoKomiMaybeOmitStuff() const {
  return toJsonNoKomi();
}

string Rules::toJsonString() const {
  return toJson().dump();
}

string Rules::toJsonStringNoKomi() const {
  return toJsonNoKomi().dump();
}

string Rules::toJsonStringNoKomiMaybeOmitStuff() const {
  return toJsonNoKomiMaybeOmitStuff().dump();
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
