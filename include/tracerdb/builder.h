// TracerDB — index builder.
//
// Memory-bounded build, no matter the dataset size:
//   1. train centroids on a capped sample (kmeans.h)
//   2. pass 1: stream the dataset, assign each vector to its nearest list;
//      spill assignments to a temp file, keep only per-list counts in RAM
//   3. compute the exact file layout from the counts, pre-extend the file
//   4. pass 2: stream dataset + assignments again and scatter each vector to
//      its final (page, slot) — *through the page cache*, which absorbs the
//      random page writes within its fixed frame budget
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "tracerdb/cache.h"
#include "tracerdb/dataset.h"
#include "tracerdb/format.h"
#include "tracerdb/kmeans.h"
#include "tracerdb/pager.h"
#include "tracerdb/policy.h"

namespace tracerdb {

struct BuildOptions {
  uint32_t nlist = 0;  // 0 = auto (~sqrt(n))
  // k-means sample cap; 0 = auto. Centroid quality needs a few dozen
  // training points per list — too few and recall collapses.
  size_t sample = 0;
  int iters = 10;              // k-means iterations
  size_t cache_frames = 4096;  // build-time page cache budget (16 MiB default)
  // CLOCK, not LRU-K: the scatter pass re-touches *recently started* posting
  // pages (pure recency), and LRU-K's evict-pages-with-short-history rule is
  // pathological for that write pattern. LRU-K earns its keep on the read
  // path (search), where centroid pages are re-referenced by every query.
  std::string policy = "clock";
  uint64_t seed = 42;
  bool verbose = true;
};

inline void build_index(const std::string& raw_path, const std::string& index_path,
                        const BuildOptions& opt) {
  DatasetReader reader(raw_path);
  const uint32_t dim = reader.dim();
  const uint64_t n = reader.nvec();
  Layout::validate_dim(dim);

  uint32_t nlist = opt.nlist;
  if (nlist == 0)
    nlist = static_cast<uint32_t>(
        std::clamp<uint64_t>(static_cast<uint64_t>(std::sqrt(static_cast<double>(n))), 16, 65536));
  size_t sample = opt.sample ? opt.sample : std::max<size_t>(50000, 48 * nlist);
  Timer total;

  // ---- 1. train ----
  std::vector<float> centroids =
      train_centroids(reader, nlist, sample, opt.iters, opt.seed, opt.verbose);

  // ---- 2. pass 1: assign, count, spill assignments ----
  const std::string tmp_path = index_path + ".assign.tmp";
  std::vector<uint64_t> counts(nlist, 0);
  {
    FILE* tmp = std::fopen(tmp_path.c_str(), "wb");
    if (!tmp) die("fopen " + tmp_path);
    reader.reset();
    std::vector<float> buf(static_cast<size_t>(1024) * dim);
    std::vector<uint32_t> abuf(1024);
    size_t got;
    uint64_t done = 0;
    Timer t;
    while ((got = reader.read_batch(buf.data(), 1024)) > 0) {
      for (size_t i = 0; i < got; ++i) {
        uint32_t c = nearest_centroid(buf.data() + i * dim, centroids.data(), nlist, dim);
        abuf[i] = c;
        ++counts[c];
      }
      if (std::fwrite(abuf.data(), sizeof(uint32_t), got, tmp) != got) die("write assignments");
      done += got;
      if (opt.verbose && (done % (1u << 20) < 1024))
        std::fprintf(stderr, "[assign] %llu/%llu vectors\r", (unsigned long long)done,
                     (unsigned long long)n);
    }
    std::fclose(tmp);
    if (opt.verbose)
      std::fprintf(stderr, "[assign] %llu vectors in %.1fs            \n",
                   (unsigned long long)done, t.seconds());
  }

  // ---- 3. layout ----
  Layout lay{dim, nlist};
  std::vector<ListExtent> extents(nlist);
  uint64_t next_page = lay.lists_start();
  for (uint32_t c = 0; c < nlist; ++c) {
    extents[c].start_page = next_page;
    extents[c].count = counts[c];
    next_page += lay.list_pages(counts[c]);
  }
  const uint64_t total_pages = next_page;

  // ---- 4. pass 2: scatter through the page cache ----
  Pager pager(index_path, /*create=*/true);
  pager.ensure_pages(total_pages);
  PageCache cache(pager, opt.cache_frames, make_policy(opt.policy, opt.cache_frames));

  // Header.
  {
    PageGuard g(cache, 0);
    auto* h = reinterpret_cast<IndexHeader*>(g.data());
    *h = IndexHeader{kMagic,
                     kVersion,
                     dim,
                     nlist,
                     0,
                     n,
                     lay.centroid_start(),
                     lay.centroid_pages(),
                     lay.extent_start(),
                     lay.extent_pages()};
    g.mark_dirty();
  }
  // Centroids.
  for (uint32_t c = 0; c < nlist; ++c) {
    uint64_t pg = lay.centroid_start() + c / lay.cent_per_page();
    PageGuard g(cache, pg);
    std::memcpy(g.data() + (c % lay.cent_per_page()) * lay.vec_bytes(),
                centroids.data() + static_cast<size_t>(c) * dim, lay.vec_bytes());
    g.mark_dirty();
  }
  // Extent table.
  for (uint32_t c = 0; c < nlist; ++c) {
    uint64_t pg = lay.extent_start() + c / lay.extents_per_page();
    PageGuard g(cache, pg);
    std::memcpy(g.data() + (c % lay.extents_per_page()) * sizeof(ListExtent), &extents[c],
                sizeof(ListExtent));
    g.mark_dirty();
  }
  // Posting entries.
  {
    FILE* tmp = std::fopen(tmp_path.c_str(), "rb");
    if (!tmp) die("fopen " + tmp_path);
    reader.reset();
    std::vector<float> buf(static_cast<size_t>(1024) * dim);
    std::vector<uint32_t> abuf(1024);
    std::vector<uint64_t> cursor(nlist, 0);
    uint64_t id = 0;
    size_t got;
    Timer t;
    while ((got = reader.read_batch(buf.data(), 1024)) > 0) {
      if (std::fread(abuf.data(), sizeof(uint32_t), got, tmp) != got) die("read assignments");
      for (size_t i = 0; i < got; ++i, ++id) {
        uint32_t c = abuf[i];
        uint64_t slot = cursor[c]++;
        uint64_t pg = extents[c].start_page + slot / lay.entries_per_page();
        size_t off = (slot % lay.entries_per_page()) * lay.entry_stride();
        PageGuard g(cache, pg);
        std::memcpy(g.data() + off, &id, sizeof(uint64_t));
        std::memcpy(g.data() + off + 8, buf.data() + i * dim, lay.vec_bytes());
        g.mark_dirty();
      }
      if (opt.verbose && (id % (1u << 20) < 1024))
        std::fprintf(stderr, "[scatter] %llu/%llu vectors\r", (unsigned long long)id,
                     (unsigned long long)n);
    }
    std::fclose(tmp);
    if (opt.verbose)
      std::fprintf(stderr, "[scatter] %llu vectors in %.1fs            \n",
                   (unsigned long long)id, t.seconds());
  }
  cache.flush_all();
  std::remove(tmp_path.c_str());

  if (opt.verbose) {
    const CacheStats& s = cache.stats();
    std::fprintf(stderr,
                 "[build] done in %.1fs: %llu vectors, %u lists, %s index file\n"
                 "[build] cache(%s, %s): %.1f%% hit rate, %llu evictions, "
                 "%llu page reads, %llu page writes\n",
                 total.seconds(), (unsigned long long)n, nlist,
                 human_bytes(static_cast<double>(total_pages) * kPageSize).c_str(),
                 cache.policy_name(), human_bytes(cache.budget_bytes()).c_str(),
                 100.0 * s.hit_rate(), (unsigned long long)s.evictions,
                 (unsigned long long)pager.stats().reads, (unsigned long long)pager.stats().writes);
  }
}

}  // namespace tracerdb
