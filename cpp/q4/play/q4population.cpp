#include "q4population.h"

#include "../../core/global.h"

namespace Q4Play {

static const char* const KIND_NAMES[NUM_SEAT_KINDS] = {
  "learner", "weak", "snapshot", "greedy", "randomPawn", "basher", "grudge"
};

const char* seatKindName(int kind) {
  if(kind < 0 || kind >= NUM_SEAT_KINDS)
    return "unknown";
  return KIND_NAMES[kind];
}

int seatKindOfName(const std::string& name) {
  for(int k = 0; k < NUM_SEAT_KINDS; k++)
    if(name == KIND_NAMES[k])
      return k;
  return -1;
}

int Q4Composition::numLearners() const {
  int n = 0;
  for(int s = 0; s < 4; s++)
    n += (seats[s].kind == SEAT_LEARNER) ? 1 : 0;
  return n;
}

Q4PopulationSettings Q4PopulationSettings::load(ConfigParser& cfg) {
  Q4PopulationSettings p;
  p.mixedProb = cfg.getDouble("q4PopulationMixedProb", 0.0, 1.0);
  p.numSnapshots = cfg.getInt("q4PopulationNumSnapshots", 0, 1000);
  p.snapshotVisits = cfg.getInt("q4PopulationSnapshotVisits", 1, 10000000);
  p.snapshotTemperatureEarly = cfg.getDouble("q4PopulationSnapshotChosenMoveTemperatureEarly", 0.0, 5.0);
  p.snapshotTemperature = cfg.getDouble("q4PopulationSnapshotChosenMoveTemperature", 0.0, 5.0);
  p.snapshotCacheSizePowerOfTwo = cfg.getInt("q4PopulationSnapshotCacheSizePowerOfTwo", 10, 30);
  p.weakTemperatureMin = cfg.getDouble("q4PopulationWeakTemperatureMin", 0.0, 5.0);
  p.weakTemperatureMax = cfg.getDouble("q4PopulationWeakTemperatureMax", 0.0, 5.0);
  if(p.weakTemperatureMax < p.weakTemperatureMin)
    throw StringError("q4PopulationWeakTemperatureMax < q4PopulationWeakTemperatureMin");
  for(const std::string& v : Global::split(cfg.getString("q4PopulationWeakVisits"), ',')) {
    if(Global::trim(v).empty())
      continue;
    int n = Global::stringToInt(Global::trim(v));
    if(n < 1)
      throw StringError("q4PopulationWeakVisits: expected positive visit counts, got " + v);
    p.weakVisits.push_back(n);
  }
  if(p.weakVisits.empty())
    throw StringError("q4PopulationWeakVisits is empty");

  // "kind:weight, kind:weight, ...": every non-learner kind must be listed exactly once.
  bool seen[NUM_SEAT_KINDS] = {false, false, false, false, false, false, false};
  for(const std::string& item : Global::split(cfg.getString("q4PopulationWeights"), ',')) {
    std::string t = Global::trim(item);
    if(t.empty())
      continue;
    size_t colon = t.find(':');
    if(colon == std::string::npos)
      throw StringError("q4PopulationWeights: expected kind:weight, got '" + t + "'");
    std::string name = Global::trim(t.substr(0, colon));
    int kind = seatKindOfName(name);
    if(kind <= SEAT_LEARNER)
      throw StringError("q4PopulationWeights: unknown kind '" + name + "' (expected weak, snapshot, greedy, randomPawn, basher or grudge)");
    if(seen[kind])
      throw StringError("q4PopulationWeights: kind " + name + " listed twice");
    seen[kind] = true;
    double w;
    if(!Global::tryStringToDouble(Global::trim(t.substr(colon + 1)), w) || w < 0.0)
      throw StringError("q4PopulationWeights: bad weight in '" + t + "'");
    p.kindWeights[kind] = w;
  }
  for(int k = SEAT_WEAK; k < NUM_SEAT_KINDS; k++)
    if(!seen[k])
      throw StringError(std::string("q4PopulationWeights: kind ") + KIND_NAMES[k] + " is missing (use weight 0 to turn it off)");
  return p;
}

Q4Composition sampleComposition(Rand& rand, const Q4PopulationSettings& settings, int numSnapshotsAvailable) {
  Q4Composition comp;
  if(!rand.nextBool(settings.mixedProb))
    return comp;

  const int numLearners = 1 + (int)rand.nextUInt(3);
  // Learner seats: a uniform subset of the given size (partial Fisher-Yates).
  int perm[4] = {0, 1, 2, 3};
  for(int i = 0; i < numLearners; i++) {
    int j = i + (int)rand.nextUInt((uint32_t)(4 - i));
    std::swap(perm[i], perm[j]);
  }
  bool isLearner[4] = {false, false, false, false};
  int learnerSeats[3];
  for(int i = 0; i < numLearners; i++) {
    isLearner[perm[i]] = true;
    learnerSeats[i] = perm[i];
  }
  // Every other seat draws a kind (seats in order, so the stream is reproducible).
  double weights[NUM_SEAT_KINDS];
  for(int k = 0; k < NUM_SEAT_KINDS; k++)
    weights[k] = settings.kindWeights[k];
  if(numSnapshotsAvailable <= 0)
    weights[SEAT_SNAPSHOT] = 0.0;
  double total = 0.0;
  for(int k = SEAT_WEAK; k < NUM_SEAT_KINDS; k++)
    total += weights[k];
  if(total <= 0.0)
    throw StringError("Q4 population: no kind with positive weight is available for a non-learner seat");
  for(int s = 0; s < 4; s++) {
    if(isLearner[s])
      continue;
    double r = rand.nextDouble() * total;
    int kind = NUM_SEAT_KINDS - 1;
    for(int k = SEAT_WEAK; k < NUM_SEAT_KINDS; k++) {
      if(weights[k] <= 0.0)
        continue;
      if(r < weights[k]) {
        kind = k;
        break;
      }
      r -= weights[k];
    }
    while(weights[kind] <= 0.0)
      kind--;
    Q4SeatSpec& spec = comp.seats[s];
    spec.kind = kind;
    if(kind == SEAT_WEAK) {
      spec.weakVisits = settings.weakVisits[rand.nextUInt((uint32_t)settings.weakVisits.size())];
      spec.weakTemperature = rand.nextDouble(settings.weakTemperatureMin, settings.weakTemperatureMax);
    }
    else if(kind == SEAT_SNAPSHOT) {
      spec.snapshotIdx = (int)rand.nextUInt((uint32_t)numSnapshotsAvailable);
    }
    else if(kind == SEAT_GRUDGE) {
      spec.grudgeTarget = learnerSeats[rand.nextUInt((uint32_t)numLearners)];
    }
  }
  return comp;
}

}  // namespace Q4Play
