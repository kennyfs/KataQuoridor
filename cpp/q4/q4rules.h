#ifndef Q4_RULES_H_
#define Q4_RULES_H_

#include <string>
#include <vector>

struct Q4Rules {
  int maxPlies = 400;
  int repetitionDrawCount = 0;
  int initialWalls[4] = {7, 7, 7, 7};
  static constexpr int MAX_REPETITION_DRAW_COUNT = 1000;

  Q4Rules();
  Q4Rules(int maxPlies, int repetitionDrawCount, int walls = 7);
  Q4Rules(int maxPlies, int repetitionDrawCount, const int walls[4]);

  bool operator==(const Q4Rules& other) const;
  bool operator!=(const Q4Rules& other) const;

  std::string toString() const;
  std::string toJson() const;

  static Q4Rules parse(const std::string& str);
  static Q4Rules fromJson(const std::string& json);
  static Q4Rules parseRulesOrJson(const std::string& str);

  void validate() const;
};

#endif  // Q4_RULES_H_
