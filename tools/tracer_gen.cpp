// tracer_gen — generate a synthetic clustered embedding dataset, streamed to
// disk with bounded memory (cluster centers only).
//
//   tracer_gen out.vec --n 500000 --dim 128 --clusters 1000 --noise 0.15
#include <cstdio>
#include <random>
#include <vector>

#include "tracerdb/common.h"
#include "tracerdb/dataset.h"

#include "cli.h"

using namespace tracerdb;

int main(int argc, char** argv) {
  auto args = cli::Args::parse(argc, argv);
  if (args.positional.size() != 1) {
    std::fprintf(stderr,
                 "usage: tracer_gen <out.vec> [--n N] [--dim D] [--clusters C] "
                 "[--noise S] [--seed X]\n");
    return 2;
  }
  const uint64_t n = args.u64("n", 500000);
  const uint32_t dim = static_cast<uint32_t>(args.u64("dim", 128));
  const uint32_t clusters = static_cast<uint32_t>(args.u64("clusters", 1000));
  const float noise = static_cast<float>(args.f64("noise", 0.15));
  const uint64_t seed = args.u64("seed", 1234);

  std::mt19937_64 rng(seed);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  std::uniform_int_distribution<uint32_t> pick(0, clusters - 1);

  // Cluster centers on the unit sphere (typical of normalized embeddings).
  std::vector<float> centers(static_cast<size_t>(clusters) * dim);
  for (uint32_t c = 0; c < clusters; ++c) {
    float* v = centers.data() + static_cast<size_t>(c) * dim;
    float norm = 0;
    for (uint32_t d = 0; d < dim; ++d) {
      v[d] = gauss(rng);
      norm += v[d] * v[d];
    }
    norm = std::sqrt(norm);
    for (uint32_t d = 0; d < dim; ++d) v[d] /= norm;
  }

  DatasetWriter w(args.positional[0], dim);
  std::vector<float> batch(static_cast<size_t>(1024) * dim);
  Timer t;
  uint64_t written = 0;
  while (written < n) {
    size_t b = static_cast<size_t>(std::min<uint64_t>(1024, n - written));
    for (size_t i = 0; i < b; ++i) {
      const float* c = centers.data() + static_cast<size_t>(pick(rng)) * dim;
      float* v = batch.data() + i * dim;
      for (uint32_t d = 0; d < dim; ++d) v[d] = c[d] + noise * gauss(rng);
    }
    w.append(batch.data(), b);
    written += b;
  }
  w.finish();
  double bytes = 16.0 + static_cast<double>(n) * dim * 4;
  std::printf("wrote %llu vectors (dim %u, %u clusters) -> %s (%s) in %.1fs\n",
              (unsigned long long)n, dim, clusters, args.positional[0].c_str(),
              human_bytes(bytes).c_str(), t.seconds());
  return 0;
}
