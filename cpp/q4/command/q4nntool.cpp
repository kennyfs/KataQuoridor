#include "command/q4nntool.h"

#include "../core/config_parser.h"
#include "../core/global.h"
#include "../core/rand.h"
#include "../core/timer.h"
#include "../external/nlohmann_json/json.hpp"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"
#include "command/q4json.h"
#include "nn/q4nn.h"
#include "q4board.h"
#include "q4notation.h"
#include "q4symmetry.h"
#include "../core/test.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <unistd.h>

using json = nlohmann::json;

namespace {

// Game kinds of sampleHistories: 0 pawn shuffles, 1 near maxPlies, 2 wall-heavy, 3 eliminations, 4 walls run out, 5 random
// (the repetition games come twice: they give the fewest positions with repeating and drawing moves)
constexpr int NUM_GAME_SLOTS = 7;
constexpr int KIND_OF_SLOT[NUM_GAME_SLOTS] = {0, 1, 2, 3, 4, 5, 0};

void playGameKind(int kind, Rand& rand, std::vector<Q4History>& out, size_t maxPositions) {
  Q4Rules rules;
  int maxPliesPlayed = 100;
  switch(kind) {
    case 0: rules.repetitionDrawCount = 2 + (int)rand.nextUInt(2); maxPliesPlayed = 40; break;  // pawn shuffles
    case 1: rules.maxPlies = 30 + (int)rand.nextUInt(20); break;                                // near maxPlies
    case 2: maxPliesPlayed = 80; break;                                                         // wall-heavy
    case 3: maxPliesPlayed = 70; break;                                                         // eliminations
    case 4:                                                                                     // walls run out
      for(int s = 0; s < 4; s++) rules.initialWalls[s] = (int)rand.nextUInt(4);
      rules.repetitionDrawCount = (rand.nextUInt(2) == 0) ? 0 : 3;
      break;
    default: rules.repetitionDrawCount = (rand.nextUInt(2) == 0) ? 0 : 2; break;
  }
  Q4History history(rules);
  int startPawns[4];
  for(int s = 0; s < 4; s++) startPawns[s] = history.currentBoard.pawn[s];
  while(!history.isFinished && history.plies < maxPliesPlayed && out.size() < maxPositions) {
    if((kind != 1 || history.plies >= rules.maxPlies - 12) && (kind != 0 || history.plies >= 8))
      out.push_back(history);
    int seat = history.currentBoard.toMove;
    if(kind == 3 && history.currentBoard.getNumAlive() > 2 && rand.nextUInt(8) == 0) {
      int victim = (seat + 1 + (int)rand.nextUInt(3)) % 4;
      if(history.currentBoard.isAlive(victim)) {
        history.eliminate(victim);
        continue;
      }
    }
    std::vector<int> actions;
    history.currentBoard.getLegalActions(seat, actions);
    if(actions.empty())
      break;
    int act = actions[rand.nextUInt((uint32_t)actions.size())];
    if(kind == 0) {
      // Step away from the start cell and back, so that positions repeat.
      int cycle = history.plies / 4;
      for(int a : actions) {
        if(cycle % 2 == 0 ? (a < 121 && a != startPawns[seat]) : a == startPawns[seat]) {
          act = a;
          break;
        }
      }
    }
    else if(kind == 2 || kind == 3 || kind == 4) {
      std::vector<int> walls;
      for(int a : actions) if(a >= 121) walls.push_back(a);
      if(!walls.empty() && rand.nextUInt(4) != 0)
        act = walls[rand.nextUInt((uint32_t)walls.size())];
    }
    history.play(act);
  }
}

std::vector<std::string> readStdinLines() {
  std::vector<std::string> lines;
  std::string line;
  while(std::getline(std::cin, line)) {
    line = Global::trim(line);
    if(!line.empty())
      lines.push_back(line);
  }
  return lines;
}

struct ModelOptions {
  std::string modelFile;
  std::string configFile;
  int symmetry = 0;
  bool jsonOutput = false;
  int numPositions = 100;
  uint64_t seed = 42;
  int numThreads = 1;
  int batchSize = 64;
  int seconds = 5;
  bool randomPositions = false;
  bool stdinPositions = false;
};

ModelOptions parseOptions(const std::vector<std::string>& args) {
  ModelOptions o;
  for(size_t i = 0; i < args.size(); i++) {
    const std::string& a = args[i];
    auto next = [&]() -> const std::string& {
      if(i + 1 >= args.size())
        throw StringError("Missing value after " + a);
      return args[++i];
    };
    if(a == "-model") o.modelFile = next();
    else if(a == "-config") o.configFile = next();
    else if(a == "-symmetry") o.symmetry = Global::stringToInt(next());
    else if(a == "-json") o.jsonOutput = true;
    else if(a == "-n" || a == "-num-positions") o.numPositions = Global::stringToInt(next());
    else if(a == "-seed") o.seed = (uint64_t)Global::stringToInt64(next());
    else if(a == "-threads") o.numThreads = Global::stringToInt(next());
    else if(a == "-batch-size") o.batchSize = Global::stringToInt(next());
    else if(a == "-seconds") o.seconds = Global::stringToInt(next());
    else if(a == "-random") o.randomPositions = true;
    else if(a == "-stdin") o.stdinPositions = true;
    else throw StringError("Unknown option: " + a);
  }
  if(o.symmetry < 0 || o.symmetry >= Q4Symmetry::NUM_SYMMETRIES)
    throw StringError("-symmetry must be in 0..7");
  return o;
}

// The evaluator keeps a pointer to its logger for as long as it lives, so the logger must outlive it.
// The config may switch it on (logToStderr = true).
Logger& evaluatorLogger(ConfigParser& cfg) {
  static Logger logger(&cfg, false, false, false, false);
  return logger;
}

NNEvaluator* makeEvaluator(const ModelOptions& o, int maxBatchSize, int numThreads) {
  if(o.modelFile.empty())
    throw StringError("-model <file> is required");
  ConfigParser cfg;
  if(!o.configFile.empty()) cfg.initialize(o.configFile);
  if(!cfg.contains("nnCacheSizePowerOfTwo")) cfg.overrideKey("nnCacheSizePowerOfTwo", "16");
  if(!cfg.contains("nnMutexPoolSizePowerOfTwo")) cfg.overrideKey("nnMutexPoolSizePowerOfTwo", "12");
  // NN server threads: numNNServerThreadsPerModel (GPU backends) / numEigenThreadsPerModel (Eigen)
  if(!cfg.contains("numNNServerThreadsPerModel")) cfg.overrideKey("numNNServerThreadsPerModel", Global::intToString(numThreads));
  if(!cfg.contains("numEigenThreadsPerModel")) cfg.overrideKey("numEigenThreadsPerModel", Global::intToString(numThreads));
  Rand seedRand(o.seed);
  return Setup::initializeNNEvaluator(
    o.modelFile, o.modelFile, "", cfg, evaluatorLogger(cfg), seedRand, maxBatchSize, Q4NNConst::POS_LEN, Q4NNConst::POS_LEN,
    Setup::MaxBatchSizeRequest::explicitSize(maxBatchSize), true, false, Setup::SETUP_FOR_GTP
  );
}

std::vector<float> toVec(const float* p, int n) {
  return std::vector<float>(p, p + n);
}

// positions: from stdin (one JSON history per line) or sampled
std::vector<Q4History> loadPositions(const ModelOptions& o, std::vector<json>* extra = nullptr) {
  std::vector<Q4History> out;
  if(o.randomPositions || (!o.stdinPositions && isatty(fileno(stdin))))
    return Q4NNTool::sampleHistories(o.seed, o.numPositions);
  for(const std::string& line : readStdinLines()) {
    json j = json::parse(line);
    out.push_back(parseHistoryFromJson(j));
    if(extra) extra->push_back(j);
  }
  return out;
}

void dumpInputs(const ModelOptions& o) {
  std::vector<Q4History> positions = loadPositions(o);
  for(const Q4History& history : positions) {
    const Q4Board& board = history.currentBoard;
    std::vector<float> sym0Spatial(Q4NN::NUM_SPATIAL_CHANNELS * Q4NN::POS_AREA), global(Q4NN::NUM_GLOBAL_FEATURES);
    Q4NN::RawDistances rawDist;
    Q4NN::fillRow(board, history, false, sym0Spatial.data(), global.data(), &rawDist);

    json j;
    j["board"] = boardToJson(board);
    j["toMove"] = board.toMove;
    j["rules"] = {
      {"maxPlies", history.rules.maxPlies},
      {"repetitionDrawCount", history.rules.repetitionDrawCount},
      {"initialWalls", {history.rules.initialWalls[0], history.rules.initialWalls[1],
                        history.rules.initialWalls[2], history.rules.initialWalls[3]}}
    };
    j["plies"] = history.plies;
    json events = json::array();
    for(const Q4Event& ev : history.events) {
      json e = json::object();
      if(ev.isElimination) e["elim"] = ev.eliminatedSeat;
      else e["action"] = Q4Notation::actionToString(ev.action);
      events.push_back(e);
    }
    j["events"] = events;
    std::vector<int> rawDistFlat;
    for(int k = 0; k < Q4NN::NUM_RAW_DIST_CHANNELS; k++)
      for(int c = 0; c < Q4NN::POS_AREA; c++)
        rawDistFlat.push_back(rawDist.d[k][c]);
    j["rawDist"] = rawDistFlat;
    j["global"] = global;
    json syms = json::array();
    for(int sym = 0; sym < Q4Symmetry::NUM_SYMMETRIES; sym++) {
      std::vector<float> symSpatial(sym0Spatial.size());
      Q4NN::applyInputSymmetry(sym0Spatial.data(), symSpatial.data(), sym, false);
      syms.push_back({{"sym", sym}, {"spatial", symSpatial}});
    }
    j["symmetries"] = syms;
    std::cout << j.dump() << "\n";
  }
}

void evalNN(const ModelOptions& o) {
  NNEvaluator* nnEval = makeEvaluator(o, 16, 1);
  std::vector<json> raws;
  std::vector<Q4History> positions;
  if(isatty(fileno(stdin)) && !o.stdinPositions) {
    positions.push_back(Q4History());  // the start position
    raws.push_back(json::object());
  }
  else {
    positions = loadPositions(o, &raws);
  }
  for(size_t i = 0; i < positions.size(); i++) {
    int sym = o.symmetry;
    if(raws[i].contains("sym")) sym = raws[i]["sym"].get<int>();
    else if(raws[i].contains("symmetry")) sym = raws[i]["symmetry"].get<int>();
    const Q4History& history = positions[i];
    NNResultBuf buf;
    Q4NN::Eval eval;
    bool skipCache = true;
    if(raws[i].contains("skipCache")) skipCache = raws[i]["skipCache"].get<bool>();
    Q4NN::evaluate(*nnEval, buf, history, sym, skipCache, eval);
    const Q4RawNNOutput* raw = buf.result->q4Raw.get();
    std::vector<int> legalActions;
    history.currentBoard.getLegalActions(history.currentBoard.toMove, legalActions);

    if(o.jsonOutput || !isatty(fileno(stdout))) {
      // "raw" = what the net returned for the transformed input (symmetry sym), undecoded: T17 compares it with
      // PyTorch channel by channel. "decoded" = the same outputs in game space.
      json j;
      j["sym"] = sym;
      j["toMove"] = history.currentBoard.toMove;
      j["raw"] = {
        {"policy", toVec(raw->policyLogits, Q4NN::POLICY_SLOTS)},
        {"value", toVec(raw->valueLogits, Q4NN::NUM_VALUE_LOGITS)},
        {"misc", toVec(raw->miscValues, Q4NN::NUM_MISC)},
        {"trajectory", toVec(raw->trajectoryLogits, Q4NN::TRAJECTORY_SLOTS)}
      };
      std::vector<float> probs(Q4NN::NUM_ACTIONS);
      Q4NN::softmaxLegal(eval.policyLogits[0], legalActions, probs.data());
      j["decoded"] = {
        {"policyLogits", {toVec(eval.policyLogits[0], Q4NN::NUM_ACTIONS), toVec(eval.policyLogits[1], Q4NN::NUM_ACTIONS)}},
        {"legalActions", legalActions},
        {"searchPolicyProbs", probs},
        {"valueRel", toVec(eval.valueRel, Q4NN::NUM_VALUE_LOGITS)},
        {"valueAbs", toVec(eval.valueAbs, Q4NN::NUM_VALUE_LOGITS)},
        {"valueAbsMasked", toVec(eval.valueAbsMasked, Q4NN::NUM_VALUE_LOGITS)},
        {"shorttermWinlossError", eval.shorttermWinlossError},
        {"trajectory", toVec(eval.trajectory, Q4NN::POS_AREA)}
      };
      std::cout << j.dump() << "\n" << std::flush;
    }
    else {
      std::cout << Q4NN::formatEval(eval, legalActions, 10) << std::flush;
    }
  }
  delete nnEval;
}

// T17: the average over the 8 symmetries of the decoded outputs is the same for a position and for any symmetric image
// of it. Reported per position, with the spread of the single-symmetry outputs (the nets are not equivariant, so the
// spread shows that the invariance is not trivial).
void symAvg(const ModelOptions& o) {
  NNEvaluator* nnEval = makeEvaluator(o, 16, 1);
  std::vector<Q4History> positions = loadPositions(o);
  const int numSyms = Q4Symmetry::NUM_SYMMETRIES;
  for(const Q4History& history : positions) {
    std::vector<float> avgPolicy[numSyms];   // [T] -> averaged search-policy probs, indexed by the actions of the original position
    std::vector<float> avgValue[numSyms];
    std::vector<float> avgTraj[numSyms];
    double spread = 0.0;
    for(int t = 0; t < numSyms; t++) {
      Q4History image = Q4Symmetry::applyHistory(history, t);
      std::vector<int> legal;
      image.currentBoard.getLegalActions(image.currentBoard.toMove, legal);
      std::vector<float> policy(Q4NN::NUM_ACTIONS, 0.0f), value(5, 0.0f), traj(Q4NN::POS_AREA, 0.0f);
      std::vector<float> first(Q4NN::NUM_ACTIONS, 0.0f);
      for(int s = 0; s < numSyms; s++) {
        NNResultBuf buf;
        Q4NN::Eval eval;
        Q4NN::evaluate(*nnEval, buf, image, s, true, eval);
        std::vector<float> probs(Q4NN::NUM_ACTIONS);
        Q4NN::softmaxLegal(eval.policyLogits[0], legal, probs.data());
        for(int a : legal) policy[a] += probs[a] / numSyms;
        for(int i = 0; i < 5; i++) value[i] += eval.valueAbs[i] / numSyms;
        for(int c = 0; c < Q4NN::POS_AREA; c++) traj[c] += eval.trajectory[c] / numSyms;
        if(t == 0) {
          if(s == 0) first = probs;
          else for(int a : legal) spread = std::max(spread, (double)std::abs(probs[a] - first[a]));
        }
      }
      // Back to the actions of the original position: the action a of the original is applyAction(a, t) in the image.
      std::vector<float> mapped(Q4NN::NUM_ACTIONS, 0.0f), mappedTraj(Q4NN::POS_AREA, 0.0f);
      for(int a = 0; a < Q4NN::NUM_ACTIONS; a++)
        mapped[a] = policy[Q4Symmetry::applyAction(a, t)];
      for(int c = 0; c < Q4NN::POS_AREA; c++)
        mappedTraj[c] = traj[Q4Symmetry::applyCell(c, t)];
      avgPolicy[t] = mapped;
      avgValue[t] = value;
      avgTraj[t] = mappedTraj;
    }
    double dPolicy = 0.0, dValue = 0.0, dTraj = 0.0;
    for(int t = 1; t < numSyms; t++) {
      for(int a = 0; a < Q4NN::NUM_ACTIONS; a++) dPolicy = std::max(dPolicy, (double)std::abs(avgPolicy[t][a] - avgPolicy[0][a]));
      for(int i = 0; i < 5; i++) dValue = std::max(dValue, (double)std::abs(avgValue[t][i] - avgValue[0][i]));
      for(int c = 0; c < Q4NN::POS_AREA; c++) dTraj = std::max(dTraj, (double)std::abs(avgTraj[t][c] - avgTraj[0][c]));
    }
    json j = {{"maxPolicyDiff", dPolicy}, {"maxValueDiff", dValue}, {"maxTrajectoryDiff", dTraj}, {"singleSymPolicySpread", spread}};
    std::cout << j.dump() << "\n" << std::flush;
  }
  delete nnEval;
}

// T18: scenarios of the NN cache with a real evaluator. One JSON line per scenario: whether the evaluation was
// served from the cache, and whether it should be.
void nnCache(const ModelOptions& o) {
  NNEvaluator* nnEval = makeEvaluator(o, 16, 1);
  auto evalHit = [&](const Q4History& h, int sym) {
    uint64_t before = nnEval->numCacheHits();
    NNResultBuf buf;
    Q4NN::Eval eval;
    Q4NN::evaluate(*nnEval, buf, h, sym, false, eval);
    return nnEval->numCacheHits() > before;
  };
  auto report = [&](const std::string& name, bool hit, bool expectedHit) {
    std::cout << json({{"scenario", name}, {"hit", hit}, {"expectedHit", expectedHit}}).dump() << "\n";
  };
  auto play = [](Q4History& h, const std::string& a) { h.play(Q4Notation::stringToAction(a)); };

  Q4Rules rules;
  rules.repetitionDrawCount = 3;
  Q4History start(rules);
  report("first evaluation misses", evalHit(start, 0), false);
  report("same position, same symmetry hits", evalHit(start, 0), true);
  report("same position, another symmetry hits", evalHit(start, 3), true);

  // The same position reached by two move orders (walls placed by different seats): identical inputs, hits.
  Q4History a(rules), b(rules);
  play(a, "e5h"); play(a, "b7v");
  play(b, "b7v"); play(b, "e5h");
  report("transposition: first order misses", evalHit(a, 0), false);
  report("transposition: other order hits", evalHit(b, 0), true);

  // Same board, other ply count: out and back with every pawn returns to the start position after 8 plies.
  Q4History shuffled(rules);
  for(const char* m : {"f2", "b6", "f10", "j6", "f1", "a6", "f11", "k6"}) play(shuffled, m);
  report("same board, ply count 8 instead of 0, misses", evalHit(shuffled, 0), false);

  // Other rules the inputs read.
  { Q4Rules r = rules; r.maxPlies = 120; Q4History h(r); report("other maxPlies misses", evalHit(h, 0), false); }
  { Q4Rules r = rules; r.repetitionDrawCount = 2; Q4History h(r); report("other repetitionDrawCount misses", evalHit(h, 0), false); }
  { Q4Rules r = rules; r.repetitionDrawCount = 0; Q4History h(r); report("repetition rule off misses", evalHit(h, 0), false); }
  { Q4Rules r = rules; r.initialWalls[1] = 4; Q4History h(r); report("other wall supply misses", evalHit(h, 0), false); }

  // Repetition state alone: the position after 8 plies of shuffling is the start position for the second time. Its ply
  // counter is reset to 0 here (hand-made history), so that board, rules and ply count equal those of `start`; what
  // differs is the number of earlier occurrences (progress input, repeating-move planes).
  Q4History repeated = shuffled;
  repeated.plies = 0;
  testAssert(repeated.currentBoard.hash == start.currentBoard.hash);
  testAssert(repeated.currentPositionRepetitionCount() == 2 && start.currentPositionRepetitionCount() == 1);
  report("same board and ply count, other repetition state, misses", evalHit(repeated, 0), false);
  report("... and the start position is still cached", evalHit(start, 0), true);
  delete nnEval;
}

// B8: input-fill time (CPU) and NN evaluations per second through NNEvaluator::evaluateQ4Raw.
void nnBench(const ModelOptions& o) {
  std::vector<Q4History> positions = Q4NNTool::sampleHistories(o.seed, 2000);
  {
    std::vector<float> spatial(Q4NN::NUM_SPATIAL_CHANNELS * Q4NN::POS_AREA), global(Q4NN::NUM_GLOBAL_FEATURES);
    for(const Q4History& h : positions) Q4NN::fillRow(h.currentBoard, h, true, spatial.data(), global.data());  // warm up
    ClockTimer t;
    const int reps = 5;
    for(int r = 0; r < reps; r++)
      for(const Q4History& h : positions)
        Q4NN::fillRow(h.currentBoard, h, true, spatial.data(), global.data());
    double secs = t.getSeconds();
    double perPosUs = secs / (reps * positions.size()) * 1e6;
    std::cout << json({{"what", "fillRow"}, {"positions", reps * positions.size()}, {"microsecondsPerPosition", perPosUs}}).dump() << "\n";
    // The other CPU work around one evaluation: the symmetry of the input row and the decoding of the policy.
    std::vector<float> symSpatial(spatial.size()), policy(Q4NN::POLICY_SLOTS, 0.5f), decoded(Q4NN::NUM_POLICY_VARIANTS * Q4NN::NUM_ACTIONS);
    const int symReps = 20000;
    t.reset();
    for(int r = 0; r < symReps; r++)
      Q4NN::applyInputSymmetry(spatial.data(), symSpatial.data(), 1 + r % 7, true);
    double symUs = t.getSeconds() / symReps * 1e6;
    t.reset();
    for(int r = 0; r < symReps; r++)
      Q4NN::mapPolicyToGame(policy.data(), 1 + r % 7, decoded.data());
    double mapUs = t.getSeconds() / symReps * 1e6;
    std::cout << json({{"what", "applyInputSymmetry"}, {"microsecondsPerRow", symUs}}).dump() << "\n";
    std::cout << json({{"what", "mapPolicyToGame"}, {"microsecondsPerRow", mapUs}}).dump() << "\n";
  }
  if(o.modelFile.empty())
    return;

  // Rows are prepared up front, then client threads evaluate them through the evaluator (all cache misses).
  const int numClientThreads = o.numThreads;
  NNEvaluator* nnEval = makeEvaluator(o, o.batchSize, 1);
  const bool nhwc = nnEval->getInputsUseNHWC();
  std::vector<std::vector<float>> spatials, globals;
  std::vector<Hash128> keys;
  for(const Q4History& h : positions) {
    std::vector<float> sp(Q4NN::NUM_SPATIAL_CHANNELS * Q4NN::POS_AREA), gl(Q4NN::NUM_GLOBAL_FEATURES);
    Q4NN::fillRow(h.currentBoard, h, nhwc, sp.data(), gl.data());
    spatials.push_back(sp);
    globals.push_back(gl);
    keys.push_back(Q4NN::getCacheHash(h.currentBoard, h, 0));
  }
  std::atomic<bool> stop(false);
  std::atomic<uint64_t> total(0);
  auto client = [&](int idx) {
    NNResultBuf buf;
    size_t i = (size_t)idx * 37;
    while(!stop.load()) {
      i = (i + 1) % positions.size();
      nnEval->evaluateQ4Raw(spatials[i].data(), globals[i].data(), keys[i], buf, true);
      total.fetch_add(1);
    }
  };
  std::vector<std::thread> threads;
  for(int t = 0; t < numClientThreads; t++) threads.emplace_back(client, t);
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));  // warm up
  uint64_t start = total.load();
  ClockTimer timer;
  std::this_thread::sleep_for(std::chrono::seconds(o.seconds));
  uint64_t done = total.load() - start;
  double secs = timer.getSeconds();
  stop.store(true);
  for(std::thread& t : threads) t.join();
  std::cout << json({{"what", "evaluateQ4Raw"}, {"batchSize", o.batchSize}, {"clientThreads", numClientThreads},
                     {"evalsPerSecond", done / secs}, {"seconds", secs}}).dump() << "\n";
  delete nnEval;
}

}  // namespace

namespace Q4NNTool {

std::vector<Q4History> sampleHistories(uint64_t seed, int n) {
  Rand rand(seed);
  std::vector<Q4History> pool;
  // Enough games of every kind, then n positions spread evenly over the pool (deterministic).
  int games = 0;
  while((int)pool.size() < 4 * n || games < 2 * NUM_GAME_SLOTS) {
    std::vector<Q4History> g;
    int kind = KIND_OF_SLOT[games % NUM_GAME_SLOTS];
    playGameKind(kind, rand, g, (size_t)4 * n);
    // Take a few positions per game so that the pool mixes many games (all of the short repetition games).
    for(size_t i = 0; i < g.size(); i += (kind == 0 ? 1 : 3))
      pool.push_back(g[i]);
    games++;
  }
  std::vector<Q4History> out;
  for(int i = 0; i < n; i++)
    out.push_back(pool[(size_t)i * pool.size() / n]);
  return out;
}

int run(const std::string& subcmd, const std::vector<std::string>& args) {
  try {
    ModelOptions o = parseOptions(std::vector<std::string>(args.begin() + 1, args.end()));
    if(subcmd == "dumpinputs") dumpInputs(o);
    else if(subcmd == "evalnn") evalNN(o);
    else if(subcmd == "symavg") symAvg(o);
    else if(subcmd == "nncache") nnCache(o);
    else if(subcmd == "nnbench") nnBench(o);
    else throw StringError("Unknown NN subcommand " + subcmd);
  }
  catch(const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

}  // namespace Q4NNTool
