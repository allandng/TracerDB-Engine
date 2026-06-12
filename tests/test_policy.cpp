// Eviction policy behavior, including the property TracerDB relies on:
// LRU-K keeps frequently re-referenced (hot) pages resident across large
// one-off sequential scans, where plain LRU evicts them.
#include <cassert>
#include <cstdio>

#include "tracerdb/cache.h"

using namespace tracerdb;

// Workload: 2 hot pages touched before every scan chunk (like centroid pages
// touched by every query), interleaved with one-off scans of *fresh* cold
// pages each round (like posting lists — each page touched once and never
// again). Returns hits on the hot pages after warmup.
static uint64_t hot_page_hits(const std::string& pol) {
  const std::string path = "/tmp/tracerdb_test_policy_" + pol + ".bin";
  std::remove(path.c_str());
  const uint64_t hot[] = {0, 1};
  const uint64_t ncold = 64;  // per round, all distinct
  const int warmup = 3, rounds = 20;
  const size_t nframes = 8;

  Pager pager(path, true);
  pager.ensure_pages(2 + ncold * (warmup + rounds));
  PageCache cache(pager, nframes, make_policy(pol, nframes));

  uint64_t next_cold = 2;
  auto touch = [&](uint64_t pg) {
    PageGuard g(cache, pg);
    (void)g;
  };
  auto scan_fresh = [&] {
    for (uint64_t c = 0; c < ncold; ++c) touch(next_cold++);
  };

  // Warmup: hot pages accumulate >= K references spaced wider than LRU-K's
  // correlated reference period (the scans in between space them out).
  for (int r = 0; r < warmup; ++r) {
    for (uint64_t h : hot) touch(h);
    scan_fresh();
  }

  cache.reset_stats();
  uint64_t hot_hits = 0;
  for (int round = 0; round < rounds; ++round) {
    for (uint64_t h : hot) {
      uint64_t before = cache.stats().hits;
      touch(h);
      if (cache.stats().hits > before) ++hot_hits;
    }
    // One-off scan: every page referenced exactly once, ever (infinite
    // backward-K-distance under LRU-K).
    scan_fresh();
  }
  std::remove(path.c_str());
  return hot_hits;
}

int main() {
  uint64_t lru = hot_page_hits("lru");
  uint64_t lruk = hot_page_hits("lru2");
  uint64_t clock = hot_page_hits("clock");
  std::printf("hot-page hits under scans (max 40): lru=%llu clock=%llu lru-k=%llu\n",
              (unsigned long long)lru, (unsigned long long)clock, (unsigned long long)lruk);

  // Plain LRU loses the hot pages on every scan; only round 0 (still
  // resident from warmup) can hit.
  assert(lru <= 2);
  // LRU-K must keep them resident essentially always.
  assert(lruk >= 38);
  // And strictly beat LRU.
  assert(lruk > lru);

  std::printf("test_policy OK\n");
  return 0;
}
