#ifndef Q4_POPULATION_H_
#define Q4_POPULATION_H_

#include <string>
#include <vector>

#include "../../core/config_parser.h"
#include "../../core/rand.h"

// Population self-play (Plan §11, docs/q4/rounds/R7.md): a game seats 1-4 learners (the current net, one shared
// search) and fills the other seats from a pool.
namespace Q4Play {

// The kind of the player in a seat; the value is written to globalTargetsNC C61.
enum Q4SeatKind {
  SEAT_LEARNER = 0,
  SEAT_WEAK = 1,
  SEAT_SNAPSHOT = 2,
  SEAT_GREEDY = 3,
  SEAT_RANDOM_PAWN = 4,
  SEAT_BASHER = 5,
  SEAT_GRUDGE = 6,
  NUM_SEAT_KINDS = 7
};

const char* seatKindName(int kind);  // "learner", "weak", "snapshot", "greedy", "randomPawn", "basher", "grudge"
int seatKindOfName(const std::string& name);  // -1 if unknown

struct Q4PopulationSettings {
  double mixedProb = 0.0;           // q4PopulationMixedProb: probability that a game is not all-learner
  double kindWeights[NUM_SEAT_KINDS] = {0, 0, 0, 0, 0, 0, 0};  // q4PopulationWeights (learner weight is 0)
  // q4PopulationSnapshotAges: the snapshots are the models this many exports back from the current net (models dir
  // mtime order; 1 = the previous export). An age beyond the oldest clamps to the oldest; duplicates collapse.
  std::vector<int> snapshotAges;
  int snapshotVisits = 0;           // q4PopulationSnapshotVisits
  double snapshotTemperatureEarly = 0.0;  // q4PopulationSnapshotChosenMoveTemperatureEarly / ...Temperature
  double snapshotTemperature = 0.0;
  int snapshotCacheSizePowerOfTwo = 0;    // NN cache of the snapshot evaluators
  std::vector<int> weakVisits;      // q4PopulationWeakVisits
  // q4PopulationObserverRowWeight: the target weight of an observer row (before KataGo's surprise weighting and
  // stochastic integerization), so that a non-learner ply yields a row about as often as a learner ply does
  double observerRowWeight = 0.0;
  double weakTemperatureMin = 0.0;  // q4PopulationWeakTemperatureMin / Max
  double weakTemperatureMax = 0.0;

  // All keys are required (no silent defaults).
  static Q4PopulationSettings load(ConfigParser& cfg);
  bool usesSnapshots() const { return mixedProb > 0.0 && !snapshotAges.empty() && kindWeights[SEAT_SNAPSHOT] > 0.0; }
};

struct Q4SeatSpec {
  int kind = SEAT_LEARNER;
  int weakVisits = 0;           // SEAT_WEAK
  double weakTemperature = 0;   // SEAT_WEAK
  int snapshotIdx = -1;         // SEAT_SNAPSHOT: index into the snapshot list the sampler was given
  int grudgeTarget = -1;        // SEAT_GRUDGE: a learner seat
};

struct Q4Composition {
  Q4SeatSpec seats[4];
  int numLearners() const;
};

// Draws a composition. With probability 1 - mixedProb all four seats are learners (and nothing else is drawn).
// Otherwise the number of learners is uniform in {1,2,3}, their seats uniform, and every other seat draws its kind
// with the weights (snapshot is excluded if numSnapshotsAvailable == 0, i.e. the kind is redrawn).
Q4Composition sampleComposition(Rand& rand, const Q4PopulationSettings& settings, int numSnapshotsAvailable);

}  // namespace Q4Play

#endif  // Q4_POPULATION_H_
