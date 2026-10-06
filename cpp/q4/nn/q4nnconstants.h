#ifndef Q4_NN_CONSTANTS_H_
#define Q4_NN_CONSTANTS_H_

// Dependency-free constants of the Q4 neural network I/O (docs/q4/Q4IO.md). This header is included by the shared
// NN code (neuralnet/*), so it must not include anything: shared code asks isQ4IOVersion() and uses these numbers
// instead of spelling them out.

namespace Q4NNConst {
  // Model option D: 100 + Q4 I/O version.
  static constexpr int Q4_IO_VERSION_BASE = 100;
  static constexpr int Q4_IO_VERSION_1 = 101;
  static constexpr int Q4_IO_VERSION_LATEST = Q4_IO_VERSION_1;
  inline bool isQ4IOVersion(int ioVersion) { return ioVersion >= Q4_IO_VERSION_BASE; }

  static constexpr int POS_LEN = 11;
  static constexpr int POS_AREA = POS_LEN * POS_LEN;  // 121
  static constexpr int NUM_SPATIAL_CHANNELS = 27;
  static constexpr int NUM_GLOBAL_FEATURES = 28;

  // Policy: 2 variants (0 = search policy, 1 = style policy) x 3 planes (pawn, V wall, H wall), variant-major.
  static constexpr int NUM_POLICY_VARIANTS = 2;
  static constexpr int NUM_POLICY_PLANES = 3;
  static constexpr int POLICY_SLOTS_PER_VARIANT = NUM_POLICY_PLANES * POS_AREA;  // 363
  static constexpr int POLICY_SLOTS = NUM_POLICY_VARIANTS * POLICY_SLOTS_PER_VARIANT;  // 726

  static constexpr int NUM_VALUE_LOGITS = 5;  // me, next, across, previous, draw
  static constexpr int NUM_MISC = 6;
  static constexpr int NUM_OWNERSHIP_CHANNELS = 1;  // my trajectory
  static constexpr int TRAJECTORY_SLOTS = NUM_OWNERSHIP_CHANNELS * POS_AREA;  // 121

  static constexpr int NUM_ACTIONS = 321;  // 121 pawn moves, 100 vertical walls, 100 horizontal walls
}

#endif  // Q4_NN_CONSTANTS_H_
