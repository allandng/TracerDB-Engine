// tracer_query — run a single query against an index, using row R of the raw
// dataset as the query vector.
//
//   tracer_query index.tdb --raw data.vec --row 123 --k 10 --nprobe 8
#include <cstdio>
#include <vector>

#include "tracerdb/dataset.h"
#include "tracerdb/index.h"

#include "cli.h"

using namespace tracerdb;

int main(int argc, char** argv) {
  auto args = cli::Args::parse(argc, argv);
  if (args.positional.size() != 1 || !args.has("raw")) {
    std::fprintf(stderr,
                 "usage: tracer_query <index.tdb> --raw <data.vec> [--row R] [--k K]\n"
                 "       [--nprobe P] [--cache-mb M] [--policy clock|lru|lru2]\n");
    return 2;
  }
  try {
    TracerIndex index(args.positional[0], cli::mb_to_frames(args.f64("cache-mb", 8.0)),
                      args.str("policy", "lru2"));
    DatasetReader raw(args.str("raw", ""));
    std::vector<float> q(index.dim());
    raw.read_row(args.u64("row", 0), q.data());

    Timer t;
    auto res = index.search(q.data(), args.u64("k", 10), args.u64("nprobe", 8));
    double ms = t.seconds() * 1e3;

    std::printf("top-%zu for row %llu (%.2f ms):\n", res.size(),
                (unsigned long long)args.u64("row", 0), ms);
    for (const auto& r : res)
      std::printf("  id=%-10llu dist=%.4f\n", (unsigned long long)r.id, r.dist);
    const auto& s = index.cache().stats();
    std::printf("cache: %.1f%% hits (%llu hits / %llu misses)\n", 100.0 * s.hit_rate(),
                (unsigned long long)s.hits, (unsigned long long)s.misses);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
