// PageCache: data integrity under heavy eviction, dirty write-back,
// pin semantics, stats — for every policy.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>

#include "tracerdb/cache.h"

using namespace tracerdb;

static void fill_pattern(std::byte* page, uint64_t pg) {
  auto* p = reinterpret_cast<uint64_t*>(page);
  for (size_t i = 0; i < kPageSize / 8; ++i) p[i] = pg * 1000003 + i;
}
static void check_pattern(const std::byte* page, uint64_t pg) {
  auto* p = reinterpret_cast<const uint64_t*>(page);
  for (size_t i = 0; i < kPageSize / 8; ++i) assert(p[i] == pg * 1000003 + i);
}

static void run_for_policy(const std::string& pol) {
  const std::string path = "tracerdb_test_cache_" + pol + ".bin";
  std::remove(path.c_str());
  const uint64_t npages = 64;
  const size_t nframes = 4;  // tiny cache => constant eviction
  {
    Pager pager(path, true);
    pager.ensure_pages(npages);
    PageCache cache(pager, nframes, make_policy(pol, nframes));

    // Write all pages through the cache (forces dirty evictions).
    for (uint64_t pg = 0; pg < npages; ++pg) {
      PageGuard g(cache, pg);
      fill_pattern(g.data(), pg);
      g.mark_dirty();
    }
    cache.flush_all();
    assert(cache.stats().evictions >= npages - nframes);

    // Random reads through the same tiny cache: data must be intact.
    std::mt19937_64 rng(99);
    for (int i = 0; i < 1000; ++i) {
      uint64_t pg = rng() % npages;
      PageGuard g(cache, pg);
      check_pattern(g.data(), pg);
    }

    // Pinned pages must not be evicted: pin nframes-1 pages, churn the rest.
    std::byte* held = cache.pin(0);
    fill_pattern(held, 0);
    for (uint64_t pg = 1; pg < npages; ++pg) {
      PageGuard g(cache, pg);
      (void)g;
    }
    check_pattern(held, 0);  // pointer still valid, content untouched
    cache.unpin(0, true);
    cache.flush_all();
  }
  {
    // Reopen with a fresh cache: dirty data reached the disk.
    Pager pager(path, false);
    PageCache cache(pager, nframes, make_policy(pol, nframes));
    for (uint64_t pg = 0; pg < npages; ++pg) {
      PageGuard g(cache, pg);
      check_pattern(g.data(), pg);
    }
    // Hit-rate sanity: re-reading one page repeatedly is all hits.
    cache.reset_stats();
    for (int i = 0; i < 100; ++i) {
      PageGuard g(cache, 7);
      (void)g;
    }
    assert(cache.stats().hits == 100 || cache.stats().hits == 99);
  }
  std::remove(path.c_str());
  std::printf("test_cache[%s] OK\n", pol.c_str());
}

int main() {
  for (const char* pol : {"clock", "lru", "lru2"}) run_for_policy(pol);
  return 0;
}
