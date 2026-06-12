// End-to-end: generate a small dataset, build an index with a tiny cache,
// search it, and verify against in-RAM brute force.
//  - nprobe = nlist must return *exactly* the brute-force top-k (IVF scans
//    every list, so the result is exact).
//  - nprobe = 4 must achieve decent recall on clustered data.
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <random>
#include <vector>

#include "tracerdb/builder.h"
#include "tracerdb/dataset.h"
#include "tracerdb/index.h"

using namespace tracerdb;

int main() {
  const char* raw_path = "/tmp/tracerdb_test_index.vec";
  const char* idx_path = "/tmp/tracerdb_test_index.tdb";
  std::remove(raw_path);
  std::remove(idx_path);

  const uint64_t n = 6000;
  const uint32_t dim = 24;
  const uint32_t nclusters = 32;

  // Clustered data, kept in RAM for brute force.
  std::mt19937_64 rng(5);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  std::vector<float> centers(static_cast<size_t>(nclusters) * dim);
  for (auto& x : centers) x = gauss(rng);
  std::vector<float> data(n * dim);
  for (uint64_t i = 0; i < n; ++i) {
    const float* c = centers.data() + (rng() % nclusters) * dim;
    for (uint32_t d = 0; d < dim; ++d) data[i * dim + d] = c[d] + 0.1f * gauss(rng);
  }
  {
    DatasetWriter w(raw_path, dim);
    w.append(data.data(), n);
    w.finish();
  }

  BuildOptions opt;
  opt.nlist = 32;
  opt.sample = 4000;
  opt.iters = 8;
  opt.cache_frames = 16;  // deliberately tiny: build must still be correct
  opt.verbose = false;
  build_index(raw_path, idx_path, opt);

  TracerIndex index(idx_path, 16, "lru2");
  assert(index.size() == n);
  assert(index.dim() == dim);
  assert(index.nlist() == 32);

  const size_t k = 10;
  size_t exact_ok = 0, recall_hits = 0, recall_total = 0;
  for (int q = 0; q < 25; ++q) {
    std::vector<float> query(dim);
    uint64_t row = rng() % n;
    for (uint32_t d = 0; d < dim; ++d) query[d] = data[row * dim + d] + 0.05f * gauss(rng);

    // Brute force.
    std::vector<std::pair<float, uint64_t>> bf(n);
    for (uint64_t i = 0; i < n; ++i)
      bf[i] = {l2sq(query.data(), data.data() + i * dim, dim), i};
    std::partial_sort(bf.begin(), bf.begin() + k, bf.end());

    // nprobe = nlist: must match brute force exactly (as an id set).
    auto res = index.search(query.data(), k, 32);
    assert(res.size() == k);
    std::vector<uint64_t> got, want;
    for (auto& r : res) got.push_back(r.id);
    for (size_t i = 0; i < k; ++i) want.push_back(bf[i].second);
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    if (got == want) ++exact_ok;
    // Distances must be sorted ascending.
    for (size_t i = 1; i < res.size(); ++i) assert(res[i - 1].dist <= res[i].dist);

    // nprobe = 4: count recall.
    auto res4 = index.search(query.data(), k, 4);
    for (auto& r : res4)
      if (std::find(want.begin(), want.end(), r.id) != want.end()) ++recall_hits;
    recall_total += k;
  }
  std::printf("exact (nprobe=nlist) matched brute force: %zu/25\n", exact_ok);
  std::printf("recall@10 (nprobe=4/32): %.3f\n", double(recall_hits) / recall_total);
  assert(exact_ok == 25);
  assert(double(recall_hits) / recall_total > 0.8);

  std::remove(raw_path);
  std::remove(idx_path);
  std::printf("test_index OK\n");
  return 0;
}
