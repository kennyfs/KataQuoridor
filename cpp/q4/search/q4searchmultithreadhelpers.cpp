#include "q4search.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "../../core/test.h"
#include "q4searchnode.h"

namespace Q4S {

static void threadTaskLoop(Search* search, int threadIdx) {
  while(true) {
    std::function<void(int)>* task;
    bool suc = search->threadTasks[threadIdx-1].waitPop(task);
    if(!suc)
      return;

    try {
      (*task)(threadIdx);
    }
    catch(const std::exception& e) {
      if(search->logger != nullptr)
        search->logger->write(std::string("ERROR: Search thread failed: ") + e.what());
      search->threadTasksRemaining->add(-1);
      throw;
    }
    catch(const std::string& e) {
      if(search->logger != nullptr)
        search->logger->write("ERROR: Search thread failed: " + e);
      search->threadTasksRemaining->add(-1);
      throw;
    }
    catch(...) {
      if(search->logger != nullptr)
        search->logger->write("ERROR: Search thread failed with unexpected throw");
      search->threadTasksRemaining->add(-1);
      throw;
    }
    search->threadTasksRemaining->add(-1);
  }
}

int Search::numAdditionalThreadsToUseForTasks() const {
  return searchParams.numThreads - 1;
}

void Search::spawnThreadsIfNeeded() {
  int desiredNumAdditionalThreads = numAdditionalThreadsToUseForTasks();
  if(numThreadsSpawned >= desiredNumAdditionalThreads)
    return;
  killThreads();
  threadTasks = new ThreadSafeQueue<std::function<void(int)>*>[desiredNumAdditionalThreads];
  threadTasksRemaining = new ThreadSafeCounter();
  threads = new std::thread[desiredNumAdditionalThreads];
  for(int i = 0; i < desiredNumAdditionalThreads; i++)
    threads[i] = std::thread(threadTaskLoop, this, i + 1);
  numThreadsSpawned = desiredNumAdditionalThreads;
}

void Search::killThreads() {
  if(numThreadsSpawned <= 0)
    return;
  for(int i = 0; i < numThreadsSpawned; i++)
    threadTasks[i].close();
  for(int i = 0; i < numThreadsSpawned; i++)
    threads[i].join();
  delete[] threadTasks;
  delete threadTasksRemaining;
  delete[] threads;
  threadTasks = NULL;
  threadTasksRemaining = NULL;
  threads = NULL;
  numThreadsSpawned = 0;
}

void Search::respawnThreads() {
  killThreads();
  spawnThreadsIfNeeded();
}

void Search::performTaskWithThreads(std::function<void(int)>* task, int capThreads) {
  spawnThreadsIfNeeded();
  int numAdditionalThreadsToUse = std::min(capThreads - 1, numAdditionalThreadsToUseForTasks());
  if(numAdditionalThreadsToUse <= 0) {
    (*task)(0);
  }
  else {
    testAssert(numAdditionalThreadsToUse <= numThreadsSpawned);
    threadTasksRemaining->add(numAdditionalThreadsToUse);
    for(int i = 0; i < numAdditionalThreadsToUse; i++)
      threadTasks[i].forcePush(task);
    (*task)(0);
    threadTasksRemaining->waitUntilZero();
  }
}

static void maybeAppendShuffledIntRange(int cap, Rand* rand, std::vector<int>& randBuf) {
  if(rand != NULL) {
    size_t randBufStart = randBuf.size();
    for(int i = 0; i < cap; i++)
      randBuf.push_back(i);
    for(int i = 1; i < cap; i++) {
      int r = (int)(rand->nextUInt() % (uint32_t)(i + 1));
      int tmp = randBuf[randBufStart + i];
      randBuf[randBufStart + i] = randBuf[randBufStart + r];
      randBuf[randBufStart + r] = tmp;
    }
  }
}

void Search::applyRecursivelyPostOrderMulithreaded(const std::vector<SearchNode*>& nodes, std::function<void(SearchNode*,int)>* f) {
  searchNodeAge += 1;

  int numAdditionalThreads = numAdditionalThreadsToUseForTasks();
  std::vector<Rand*> rands(numAdditionalThreads + 1, NULL);
  for(int threadIdx = 1; threadIdx < numAdditionalThreads + 1; threadIdx++)
    rands[threadIdx] = new Rand(nonSearchRand.nextUInt64());

  int numChildren = (int)nodes.size();
  std::function<void(int)> g = [&](int threadIdx) {
    assert(threadIdx >= 0 && threadIdx < (int)rands.size());
    Rand* rand = rands[threadIdx];
    std::unordered_set<SearchNode*> nodeBuf;
    std::vector<int> randBuf;

    size_t randBufStart = randBuf.size();
    maybeAppendShuffledIntRange(numChildren, rand, randBuf);
    for(int i = 0; i < numChildren; i++) {
      int childIdx = rand != NULL ? randBuf[randBufStart + i] : i;
      applyRecursivelyPostOrderMulithreadedHelper(nodes[childIdx], threadIdx, (PCG32*)rand, nodeBuf, randBuf, f);
    }
    randBuf.resize(randBufStart);
  };
  performTaskWithThreads(&g, 0x3fffFFFF);
  for(int threadIdx = 1; threadIdx < numAdditionalThreads + 1; threadIdx++)
    delete rands[threadIdx];
}

void Search::applyRecursivelyPostOrderMulithreadedHelper(
  SearchNode* node, int threadIdx, PCG32* rand, std::unordered_set<SearchNode*>& nodeBuf, std::vector<int>& randBuf, std::function<void(SearchNode*,int)>* f
) {
  if(node->nodeAge.load(std::memory_order_acquire) == searchNodeAge)
    return;
  if(nodeBuf.find(node) != nodeBuf.end())
    return;

  SearchNodeChildrenReference children = node->getChildren();
  int numChildren = children.iterateAndCountChildren();

  if(numChildren > 0) {
    size_t randBufStart = randBuf.size();
    maybeAppendShuffledIntRange(numChildren, (Rand*)rand, randBuf);

    nodeBuf.insert(node);
    for(int i = 0; i < numChildren; i++) {
      int childIdx = rand != NULL ? randBuf[randBufStart + i] : i;
      applyRecursivelyPostOrderMulithreadedHelper(children[childIdx].getIfAllocated(), threadIdx, rand, nodeBuf, randBuf, f);
    }
    randBuf.resize(randBufStart);
    nodeBuf.erase(node);
  }

  std::lock_guard<std::mutex> lock(mutexPool->getMutex(node->mutexIdx));
  if(node->nodeAge.load(std::memory_order_acquire) == searchNodeAge)
    return;
  if(f != NULL)
    (*f)(node, threadIdx);
  node->nodeAge.store(searchNodeAge, std::memory_order_release);
}

void Search::applyRecursivelyAnyOrderMulithreaded(const std::vector<SearchNode*>& nodes, std::function<void(SearchNode*,int)>* f) {
  searchNodeAge += 1;

  int numAdditionalThreads = numAdditionalThreadsToUseForTasks();
  std::vector<Rand*> rands(numAdditionalThreads + 1, NULL);
  for(int threadIdx = 1; threadIdx < numAdditionalThreads + 1; threadIdx++)
    rands[threadIdx] = new Rand(nonSearchRand.nextUInt64());

  int numChildren = (int)nodes.size();
  std::function<void(int)> g = [&](int threadIdx) {
    assert(threadIdx >= 0 && threadIdx < (int)rands.size());
    Rand* rand = rands[threadIdx];
    std::unordered_set<SearchNode*> nodeBuf;
    std::vector<int> randBuf;

    size_t randBufStart = randBuf.size();
    maybeAppendShuffledIntRange(numChildren, rand, randBuf);
    for(int i = 0; i < numChildren; i++) {
      int childIdx = rand != NULL ? randBuf[randBufStart + i] : i;
      applyRecursivelyAnyOrderMulithreadedHelper(nodes[childIdx], threadIdx, (PCG32*)rand, nodeBuf, randBuf, f);
    }
    randBuf.resize(randBufStart);
  };
  performTaskWithThreads(&g, 0x3fffFFFF);
  for(int threadIdx = 1; threadIdx < numAdditionalThreads + 1; threadIdx++)
    delete rands[threadIdx];
}

void Search::applyRecursivelyAnyOrderMulithreadedHelper(
  SearchNode* node, int threadIdx, PCG32* rand, std::unordered_set<SearchNode*>& nodeBuf, std::vector<int>& randBuf, std::function<void(SearchNode*,int)>* f
) {
  if(node->nodeAge.load(std::memory_order_acquire) == searchNodeAge)
    return;
  if(nodeBuf.find(node) != nodeBuf.end())
    return;

  SearchNodeChildrenReference children = node->getChildren();
  int numChildren = children.iterateAndCountChildren();

  if(numChildren > 0) {
    size_t randBufStart = randBuf.size();
    maybeAppendShuffledIntRange(numChildren, (Rand*)rand, randBuf);

    nodeBuf.insert(node);
    for(int i = 0; i < numChildren; i++) {
      int childIdx = rand != NULL ? randBuf[randBufStart + i] : i;
      applyRecursivelyAnyOrderMulithreadedHelper(children[childIdx].getIfAllocated(), threadIdx, rand, nodeBuf, randBuf, f);
    }
    randBuf.resize(randBufStart);
    nodeBuf.erase(node);
  }

  uint32_t oldAge = node->nodeAge.exchange(searchNodeAge, std::memory_order_acq_rel);
  if(oldAge == searchNodeAge)
    return;
  if(f != NULL)
    (*f)(node, threadIdx);
}

std::vector<SearchNode*> Search::enumerateTreePostOrder() {
  std::atomic<int64_t> sizeCounter(0);
  std::function<void(SearchNode*,int)> f = [&](SearchNode* node, int threadIdx) noexcept {
    (void)node;
    (void)threadIdx;
    sizeCounter.fetch_add(1, std::memory_order_relaxed);
  };
  applyRecursivelyPostOrderMulithreaded({rootNode}, &f);

  int64_t size = sizeCounter.load(std::memory_order_relaxed);
  std::vector<SearchNode*> nodes(size, NULL);
  std::atomic<int64_t> indexCounter(0);
  std::function<void(SearchNode*,int)> g = [&](SearchNode* node, int threadIdx) noexcept {
    (void)threadIdx;
    int64_t index = indexCounter.fetch_add(1, std::memory_order_relaxed);
    assert(index >= 0 && index < size);
    nodes[index] = node;
  };
  applyRecursivelyPostOrderMulithreaded({rootNode}, &g);
  assert(indexCounter.load(std::memory_order_relaxed) == size);
  return nodes;
}

}  // namespace Q4S
