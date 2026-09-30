#ifndef GAME_RULES_H_
#define GAME_RULES_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../external/nlohmann_json/json.hpp"

#include <set>
#include <string>
#include <iostream>

struct Rules {
  // Compatibility stubs for unchanged modules until Step 7-10
  static const int KO_SIMPLE = 0;
  static const int KO_POSITIONAL = 1;
  static const int KO_SITUATIONAL = 2;
  static const int KO_SPIGHT = 3;
  int koRule;

  static const int SCORING_AREA = 0;
  static const int SCORING_TERRITORY = 1;
  int scoringRule;

  static const int TAX_NONE = 0;
  static const int TAX_SEKI = 1;
  static const int TAX_ALL = 2;
  int taxRule;

  bool multiStoneSuicideLegal;
  bool hasButton;

  static const int WHB_ZERO = 0;
  static const int WHB_N = 1;
  static const int WHB_N_MINUS_ONE = 2;
  int whiteHandicapBonusRule;

  bool friendlyPassOk;

  // ---------------------------------------------------------------------------------------------
  // Quoridor rules and scoring (Quoridor I/O v2, see docs/QuoridorIOv2.md). All from White's perspective.
  //
  //   whiteMargin  = +max(1, dB) if White's pawn reached its goal, -max(1, dW) if Black's did
  //                  (Board::whiteMarginWhenWonBy; d = the loser's shortest-path distance).
  //   tempo t      = whiteMargin if White's pawn reached its goal, whiteMargin + 1 if Black's did
  //                  (W+1 -> 1, B+1 -> 0, B+2 -> -1: one step of t is one tempo).
  //   lead s       = t + komi (half-integer). White wins iff s > 0, even if Black's pawn reached the goal.
  //   utility u    = s + sign(s) * timeBonusPerPly * (maxPlies - T), T = plies at game end.
  //   Reaching maxPlies plies without a pawn on its goal is a draw, s = u = 0.
  // BoardHistory computes these (finalWhiteLead = s, finalWhiteMinusBlackScore = u).
  // ---------------------------------------------------------------------------------------------

  // Half-integer (n + 0.5), |komi| <= MAX_KOMI. The standard game has komi -0.5.
  float komi;
  static constexpr float DEFAULT_KOMI = -0.5f;
  static constexpr float MAX_KOMI = 20.5f;
  // Upstream's (Go) bounds on user-provided komi; any Quoridor komi is within them.
  static constexpr float MIN_USER_KOMI = -400.0f;
  static constexpr float MAX_USER_KOMI = 400.0f;

  // Draw once the game reaches this many plies (counted from the game start) without a pawn on its goal.
  int maxPlies;
  static constexpr int DEFAULT_MAX_PLIES = 300;
  static constexpr int MAX_MAX_PLIES = 100000;

  // The time bonus lambda of the utility score. A property of the training run: search must use the lambda the
  // net was trained with.
  float timeBonusPerPly;
  static constexpr float MAX_TIME_BONUS_PER_PLY = 1.0f;

  // Walls each player starts the game with (fence handicap), 0..Board::MAX_FENCE_NUM.
  int blackInitialFences;
  int whiteInitialFences;
  static constexpr int DEFAULT_INITIAL_FENCES = 10;

  Rules();
  Rules(
    int koRule,
    int scoringRule,
    int taxRule,
    bool multiStoneSuicideLegal,
    bool hasButton,
    int whiteHandicapBonusRule,
    bool friendlyPassOk,
    float komi
  );
  ~Rules();

  bool operator==(const Rules& other) const;
  bool operator!=(const Rules& other) const;

  // Factory
  static Rules getQuoridorRules();

  // Stub serialization kept for GTP/SGF compatibility
  std::string toString() const;
  std::string toStringNoKomi() const;
  std::string toStringNoKomiMaybeNice() const;
  std::string toJsonString() const;
  std::string toJsonStringNoKomi() const;
  std::string toJsonStringNoKomiMaybeOmitStuff() const;
  nlohmann::json toJson() const;
  nlohmann::json toJsonNoKomi() const;
  nlohmann::json toJsonNoKomiMaybeOmitStuff() const;

  static Rules parseRules(const std::string& str);
  static Rules parseRulesWithoutKomi(const std::string& str, float komi);
  static bool tryParseRules(const std::string& str, Rules& buf);
  static bool tryParseRulesWithoutKomi(const std::string& str, Rules& buf, float komi);

  // Sets one Quoridor rule by its key (maxPlies, timeBonusPerPly, blackInitialWalls, whiteInitialWalls), as in
  // kata-set-rule. Komi is not a rule key here (use the komi command). Throws StringError on bad input.
  static Rules updateRules(const std::string& key, const std::string& value, const Rules& priorRules);
  static std::set<std::string> quoridorRuleKeys();

  bool equalsIgnoringKomi(const Rules& other) const;
  // Whether the final score can be a draw by the score alone (Go: integer komi). Never for Quoridor: komi is a
  // half-integer and draws come only from maxPlies.
  bool gameResultWillBeInteger() const;
  // Upstream: komi is on the 0.5 grid. Use isValidKomi to validate a Quoridor komi.
  static bool komiIsIntOrHalfInt(float komi);

  // A legal Quoridor komi: finite, n + 0.5 for an integer n, and |komi| <= MAX_KOMI.
  static bool isValidKomi(float komi);
  // The nearest legal Quoridor komi (ties round up), clipped to [-MAX_KOMI, MAX_KOMI].
  static float roundKomi(double komi);
  // Throws StringError unless every Quoridor field is in range. `what` names the source for the message.
  void validateOrThrow(const std::string& what) const;

  static Rules getTrompTaylorish();
  static Rules getSimpleTerritory();

  static std::set<std::string> koRuleStrings();
  static std::set<std::string> scoringRuleStrings();
  static std::set<std::string> taxRuleStrings();
  static std::set<std::string> whiteHandicapBonusRuleStrings();
  static int parseKoRule(const std::string& s);
  static int parseScoringRule(const std::string& s);
  static int parseTaxRule(const std::string& s);
  static int parseWhiteHandicapBonusRule(const std::string& s);
  static std::string writeKoRule(int koRule);
  static std::string writeScoringRule(int scoringRule);
  static std::string writeTaxRule(int taxRule);
  static std::string writeWhiteHandicapBonusRule(int whiteHandicapBonusRule);

  friend std::ostream& operator<<(std::ostream& out, const Rules& rules);

  static const Hash128 ZOBRIST_KO_RULE_HASH[4];
  static const Hash128 ZOBRIST_SCORING_RULE_HASH[2];
  static const Hash128 ZOBRIST_TAX_RULE_HASH[3];
  static const Hash128 ZOBRIST_MULTI_STONE_SUICIDE_HASH;
  static const Hash128 ZOBRIST_BUTTON_HASH;
  static const Hash128 ZOBRIST_FRIENDLY_PASS_OK_HASH;
  static const Hash128 ZOBRIST_PASS_ALIVE_UNDER_SUICIDE_HASH;
  static const Hash128 ZOBRIST_EXCLUDE_TERRITORY_ADJ_ATARI_HASH;
};

#endif  // GAME_RULES_H_
