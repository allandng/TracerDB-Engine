// tracer_bench — the headline benchmark.
//
// Runs a query workload against the index under a fixed page-cache budget,
// once per eviction policy, and reports QPS, latency, cache hit rate, disk
// traffic, recall@k (vs. exact ground truth computed by streaming the raw
// file), and peak RSS — demonstrating semantic search over a dataset far
// larger than RAM, with memory bounded by the cache budget.
//
//   tracer_bench index.tdb --raw data.vec --queries 200 --k 10 --nprobe 8 \
//                --cache-mb 8 --policies clock,lru,lru2 --gt 32
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "tracerdb/dataset.h"
#include "tracerdb/index.h"

#include "cli.h"

using namespace tracerdb;

namespace {

// Exact top-k for the first G queries, computed in one bounded-memory
// streaming pass over the raw dataset (G small heaps, batched reads).
std::vector<std::vector<uint64_t>> exact_topk(DatasetReader& raw,
                                              const std::vector<std::vector<float>>& queries,
                                              size_t G, size_t k) {
  const uint32_t dim = raw.dim();
  struct Cand {
    float d;
    uint64_t id;
    bool operator<(const Cand& o) const { return d < o.d; }  // max-heap
  };
  std::vector<std::vector<Cand>> heaps(G);
  raw.reset();
  std::vector<float> buf(static_cast<size_t>(1024) * dim);
  uint64_t id = 0;
  size_t got;
  while ((got = raw.read_batch(buf.data(), 1024)) > 0) {
    for (size_t i = 0; i < got; ++i, ++id) {
      const float* v = buf.data() + i * dim;
      for (size_t q = 0; q < G; ++q) {
        float d = l2sq(queries[q].data(), v, dim);
        auto& h = heaps[q];
        if (h.size() < k) {
          h.push_back({d, id});
          std::push_heap(h.begin(), h.end());
        } else if (d < h.front().d) {
          std::pop_heap(h.begin(), h.end());
          h.back() = {d, id};
          std::push_heap(h.begin(), h.end());
        }
      }
    }
  }
  std::vector<std::vector<uint64_t>> gt(G);
  for (size_t q = 0; q < G; ++q)
    for (const auto& c : heaps[q]) gt[q].push_back(c.id);
  return gt;
}

struct RunResult {
  std::string policy;
  double qps = 0, mean_ms = 0, p95_ms = 0, hit_rate = 0, recall = 0, read_mb = 0;
  uint64_t evictions = 0, disk_reads = 0;
};

}  // namespace

int main(int argc, char** argv) {
  auto args = cli::Args::parse(argc, argv);
  if (args.positional.size() != 1 || !args.has("raw")) {
    std::fprintf(stderr,
                 "usage: tracer_bench <index.tdb> --raw <data.vec> [--queries Q] [--k K]\n"
                 "       [--nprobe P] [--cache-mb M] [--policies clock,lru,lru2]\n"
                 "       [--gt G] [--noise S] [--seed X]\n");
    return 2;
  }
  const std::string index_path = args.positional[0];
  const size_t Q = args.u64("queries", 200);
  const size_t k = args.u64("k", 10);
  const size_t nprobe = args.u64("nprobe", 8);
  const double cache_mb = args.f64("cache-mb", 8.0);
  const size_t frames = cli::mb_to_frames(cache_mb);
  const size_t G = std::min<size_t>(args.u64("gt", 32), Q);
  const float noise = static_cast<float>(args.f64("noise", 0.05));
  const uint64_t seed = args.u64("seed", 7);

  try {
    DatasetReader raw(args.str("raw", ""));
    const uint32_t dim = raw.dim();

    // Query workload: random dataset rows, lightly perturbed so the answer
    // isn't a trivial self-match.
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> row(0, raw.nvec() - 1);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<std::vector<float>> queries(Q, std::vector<float>(dim));
    for (size_t q = 0; q < Q; ++q) {
      raw.read_row(row(rng), queries[q].data());
      for (uint32_t d = 0; d < dim; ++d) queries[q][d] += noise * gauss(rng);
    }

    std::printf("computing exact ground truth for %zu queries (streaming scan)...\n", G);
    auto gt = exact_topk(raw, queries, G, k);

    std::vector<std::string> policies;
    {
      std::stringstream ss(args.str("policies", "clock,lru,lru2"));
      std::string p;
      while (std::getline(ss, p, ',')) policies.push_back(p);
    }

    std::vector<RunResult> results;
    for (const auto& pol : policies) {
      TracerIndex index(index_path, frames, pol);
      index.cache().reset_stats();
      const uint64_t reads0 = index.pager().stats().reads;

      std::vector<double> lat;
      lat.reserve(Q);
      size_t inter = 0, total_gt = 0;
      Timer wall;
      for (size_t q = 0; q < Q; ++q) {
        Timer t;
        auto res = index.search(queries[q].data(), k, nprobe);
        lat.push_back(t.seconds() * 1e3);
        if (q < G) {
          for (const auto& r : res)
            if (std::find(gt[q].begin(), gt[q].end(), r.id) != gt[q].end()) ++inter;
          total_gt += gt[q].size();
        }
      }
      double wall_s = wall.seconds();

      std::sort(lat.begin(), lat.end());
      RunResult rr;
      rr.policy = pol;
      rr.qps = Q / wall_s;
      rr.mean_ms = std::accumulate(lat.begin(), lat.end(), 0.0) / Q;
      rr.p95_ms = lat[static_cast<size_t>(0.95 * (Q - 1))];
      rr.hit_rate = index.cache().stats().hit_rate();
      rr.evictions = index.cache().stats().evictions;
      rr.disk_reads = index.pager().stats().reads - reads0;
      rr.read_mb = static_cast<double>(rr.disk_reads) * kPageSize / (1024.0 * 1024.0);
      rr.recall = total_gt ? static_cast<double>(inter) / total_gt : 0.0;
      results.push_back(rr);
    }

    std::printf("\n=== TracerDB bench: %llu vectors x %u dims, k=%zu, nprobe=%zu ===\n",
                (unsigned long long)raw.nvec(), dim, k, nprobe);
    std::printf("page cache budget: %s (%zu frames of %zu B)\n",
                human_bytes(static_cast<double>(frames) * kPageSize).c_str(), frames, kPageSize);
    std::printf("%-8s %9s %9s %9s %9s %10s %11s %9s\n", "policy", "QPS", "mean ms", "p95 ms",
                "hit rate", "evictions", "disk reads", "recall");
    for (const auto& r : results)
      std::printf("%-8s %9.1f %9.3f %9.3f %8.1f%% %10llu %7llu (%s) %8.3f\n", r.policy.c_str(),
                  r.qps, r.mean_ms, r.p95_ms, 100.0 * r.hit_rate,
                  (unsigned long long)r.evictions, (unsigned long long)r.disk_reads,
                  human_bytes(r.read_mb * 1024 * 1024).c_str(), r.recall);
    std::printf("peak RSS: %s\n", human_bytes(static_cast<double>(peak_rss_bytes())).c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
