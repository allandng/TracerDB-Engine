// TracerDB — common utilities: aligned allocation, timing, RSS, errors.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <sys/resource.h>

namespace tracerdb {

// One page = one direct-I/O unit. 4096 matches the logical block size of
// virtually every disk and the alignment O_DIRECT requires on Linux.
constexpr size_t kPageSize = 4096;
constexpr size_t kIoAlign = 4096;

[[noreturn]] inline void die(const std::string& msg) {
  throw std::runtime_error("tracerdb: " + msg);
}

struct AlignedDeleter {
  void operator()(void* p) const { ::free(p); }
};
using AlignedBuf = std::unique_ptr<std::byte[], AlignedDeleter>;

// O_DIRECT rejects buffers that are not block-aligned, so every buffer that
// touches the pager comes from here.
inline AlignedBuf alloc_aligned(size_t bytes) {
  void* p = nullptr;
  if (posix_memalign(&p, kIoAlign, bytes) != 0) throw std::bad_alloc();
  std::memset(p, 0, bytes);
  return AlignedBuf(static_cast<std::byte*>(p));
}

class Timer {
 public:
  Timer() : t0_(std::chrono::steady_clock::now()) {}
  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  }
  void reset() { t0_ = std::chrono::steady_clock::now(); }

 private:
  std::chrono::steady_clock::time_point t0_;
};

inline size_t peak_rss_bytes() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
  return static_cast<size_t>(ru.ru_maxrss);  // bytes on macOS
#else
  return static_cast<size_t>(ru.ru_maxrss) * 1024;  // kilobytes on Linux
#endif
}

inline std::string human_bytes(double b) {
  char buf[64];
  const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  int u = 0;
  while (b >= 1024.0 && u < 4) { b /= 1024.0; ++u; }
  snprintf(buf, sizeof(buf), "%.1f %s", b, units[u]);
  return buf;
}

// Squared L2 distance. Written so -O3 -ffast-math auto-vectorizes it
// (NEON on the Pi/Apple Silicon, AVX on x86).
inline float l2sq(const float* __restrict a, const float* __restrict b, size_t d) {
  float s = 0.0f;
  for (size_t i = 0; i < d; ++i) {
    float t = a[i] - b[i];
    s += t * t;
  }
  return s;
}

}  // namespace tracerdb
