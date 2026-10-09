#ifndef Q4_PLAYSETTINGS_H_
#define Q4_PLAYSETTINGS_H_

#include "../../core/config_parser.h"
#include "../../search/searchprint.h"
#include "../search/q4search.h"
#include "q4population.h"

namespace Q4Play {

struct Q4PlaySettings {
  // Play a bunch of mostly policy-distributed moves at the start to initialize a game.
  bool initGamesWithPolicy;
  double policyInitAreaProp;       // Avg number of moves is this * board area (121)
  double policyInitGammaShape;     // Controls the shape of policy init
  double policyInitAreaTemperature;

  // Occasionally try some alternative moves and search the responses to them.
  double sidePositionProb;

  // Occasionally fork an entire new game to try out an experimental move in the opening
  double earlyForkGameProb;             // Expected number of early forked games per game
  double earlyForkGameExpectedMoveProp; // Fork on average within the first board area * this prop moves
  double forkGameProb;                  // Expected number of forked games per game
  int forkGameMinChoices;               // Fork between the favorite of this many random legal moves, at minimum
  int earlyForkGameMaxChoices;          // Fork between the favorite of this many random legal moves, at maximum
  int forkGameMaxChoices;               // Fork between the favorite of this many random legal moves, at maximum

  // With this probability, use only this many visits for a move, and record it with only this weight
  double cheapSearchProb;
  int cheapSearchVisits;
  float cheapSearchTargetWeight;

  // Attenuate the number of visits used in positions where one seat is winning
  bool reduceVisits;
  double reduceVisitsThreshold;          // When max winrate exceeds this
  int reduceVisitsThresholdLookback;     // Value must be more extreme over the last this many turns
  int reducedVisitsMin;                  // Number of visits at the most extreme winrate
  float reducedVisitsWeight;             // Amount of weight to put on the training sample at minimum visits winrate.

  // Probabilistically favor samples that had high policy surprise (kl divergence).
  double policySurpriseDataWeight;
  // Probabilistically favor samples that had high value surprise (kl divergence).
  double valueSurpriseDataWeight;
  bool useSearchValueSurprise;
  // Scale frequency weights for writing data by this
  double scaleDataWeight;

  // Don't stochastically integerify target weights
  bool noResolveTargetWeights;

  // Q4 elimination probability per game
  double q4EliminationProb;

  // Population self-play: learners and other players at one table (docs/q4/rounds/R7.md)
  Q4PopulationSettings population;

  // q4OpponentMode (Plan §12.3) of every search of a self-play game
  Q4S::OpponentMode opponentMode;

  // Enable full data recording and minor tweaks applying only for self-play training.
  bool forSelfPlay;

  // Record time taken per move
  bool recordTimePerMove;

  // Search tree printing options
  PrintTreeOptions logSearchTreeOptions;

  Q4PlaySettings();
  ~Q4PlaySettings();

  static Q4PlaySettings loadForSelfplay(ConfigParser& cfg);
};

}  // namespace Q4Play

#endif  // Q4_PLAYSETTINGS_H_
