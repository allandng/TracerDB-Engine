// TracerDB — k-means training for IVF coarse quantization.
//
// Trains on a bounded sample (stride-sampled from the dataset stream) so the
// training footprint is capped no matter how large the dataset is. Random
// init + Lloyd iterations; empty clusters are reseeded from random samples.
#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "tracerdb/common.h"
#include "tracerdb/dataset.h"

namespace tracerdb {

inline uint32_t nearest_centroid(const float* v, const float* centroids, uint32_t nlist,
                                 uint32_t dim) {
  uint32_t best = 0;
  float bestd = l2sq(v, centroids, dim);
  for (uint32_t c = 1; c < nlist; ++c) {
    float d = l2sq(v, centroids + static_cast<size_t>(c) * dim, dim);
    if (d < bestd) {
      bestd = d;
      best = c;
    }
  }
  return best;
}

inline std::vector<float> train_centroids(DatasetReader& r, uint32_t nlist, size_t sample_cap,
                                          int iters, uint64_t seed, bool verbose) {
  const uint32_t dim = r.dim();
  const uint64_t n = r.nvec();
  if (n < nlist) die("dataset smaller than nlist");

  // Stride-sample up to sample_cap vectors into RAM.
  const uint64_t stride = std::max<uint64_t>(1, n / std::min<uint64_t>(n, sample_cap));
  std::vector<float> sample;
  sample.reserve(std::min<uint64_t>(n, sample_cap) * dim);
  {
    r.reset();
    std::vector<float> buf(static_cast<size_t>(1024) * dim);
    uint64_t idx = 0;
    size_t got;
    while ((got = r.read_batch(buf.data(), 1024)) > 0) {
      for (size_t i = 0; i < got; ++i, ++idx) {
        if (idx % stride == 0 && sample.size() / dim < sample_cap)
          sample.insert(sample.end(), buf.begin() + i * dim, buf.begin() + (i + 1) * dim);
      }
    }
  }
  const size_t S = sample.size() / dim;
  if (S < nlist) die("sample smaller than nlist; raise --sample");
  if (verbose)
    std::fprintf(stderr, "[kmeans] %zu samples (%s), nlist=%u, %d iterations\n", S,
                 human_bytes(static_cast<double>(sample.size()) * 4).c_str(), nlist, iters);

  std::mt19937_64 rng(seed);
  std::vector<float> cent(static_cast<size_t>(nlist) * dim);
  {
    // Init from nlist distinct random samples.
    std::vector<uint64_t> perm(S);
    for (size_t i = 0; i < S; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (uint32_t c = 0; c < nlist; ++c)
      std::copy_n(sample.data() + perm[c] * dim, dim, cent.data() + static_cast<size_t>(c) * dim);
  }

  std::vector<uint32_t> assign(S);
  std::vector<uint64_t> count(nlist);
  std::vector<double> acc(static_cast<size_t>(nlist) * dim);
  for (int it = 0; it < iters; ++it) {
    std::fill(count.begin(), count.end(), 0);
    std::fill(acc.begin(), acc.end(), 0.0);
    double sse = 0;
    for (size_t i = 0; i < S; ++i) {
      const float* v = sample.data() + i * dim;
      uint32_t c = nearest_centroid(v, cent.data(), nlist, dim);
      assign[i] = c;
      ++count[c];
      double* a = acc.data() + static_cast<size_t>(c) * dim;
      for (uint32_t d = 0; d < dim; ++d) a[d] += v[d];
      sse += l2sq(v, cent.data() + static_cast<size_t>(c) * dim, dim);
    }
    for (uint32_t c = 0; c < nlist; ++c) {
      float* ce = cent.data() + static_cast<size_t>(c) * dim;
      if (count[c] == 0) {
        // Reseed empty cluster from a random sample.
        size_t i = rng() % S;
        std::copy_n(sample.data() + i * dim, dim, ce);
        continue;
      }
      const double* a = acc.data() + static_cast<size_t>(c) * dim;
      for (uint32_t d = 0; d < dim; ++d) ce[d] = static_cast<float>(a[d] / count[c]);
    }
    if (verbose) std::fprintf(stderr, "[kmeans] iter %d/%d  mse=%.4f\n", it + 1, iters, sse / S);
  }
  return cent;
}

}  // namespace tracerdb
