// tracer_build — build a TracerDB index from a raw dataset.
//
//   tracer_build data.vec index.tdb --nlist 1024 --cache-mb 16 --policy lru2
#include <cstdio>

#include "tracerdb/builder.h"

#include "cli.h"

using namespace tracerdb;

int main(int argc, char** argv) {
  auto args = cli::Args::parse(argc, argv);
  if (args.positional.size() != 2) {
    std::fprintf(stderr,
                 "usage: tracer_build <data.vec> <index.tdb> [--nlist N] [--sample S]\n"
                 "       [--iters I] [--cache-mb M] [--policy clock|lru|lru2] [--seed X]\n");
    return 2;
  }
  BuildOptions opt;
  opt.nlist = static_cast<uint32_t>(args.u64("nlist", 0));
  opt.sample = args.u64("sample", 0);
  opt.iters = static_cast<int>(args.u64("iters", 10));
  opt.cache_frames = cli::mb_to_frames(args.f64("cache-mb", 16.0));
  opt.policy = args.str("policy", "clock");
  opt.seed = args.u64("seed", 42);
  try {
    build_index(args.positional[0], args.positional[1], opt);
    std::printf("peak RSS: %s\n", human_bytes(static_cast<double>(peak_rss_bytes())).c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
