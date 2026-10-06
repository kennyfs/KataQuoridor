#ifndef Q4SEARCH_SEARCHNODETABLE_H_
#define Q4SEARCH_SEARCHNODETABLE_H_

#include <cstdint>
#include <map>
#include <vector>

#include "../../core/global.h"
#include "../../core/hash.h"
#include "../../search/mutexpool.h"

namespace Q4S {

struct SearchNode;

struct SearchNodeTable {
  std::vector<std::map<Hash128, SearchNode*>> entries;
  MutexPool* mutexPool;
  uint32_t numShards;

  SearchNodeTable(int numShardsPowerOfTwo);
  ~SearchNodeTable();

  uint32_t getIndex(uint64_t hash) const;
};

}  // namespace Q4S

#endif  // Q4SEARCH_SEARCHNODETABLE_H_
