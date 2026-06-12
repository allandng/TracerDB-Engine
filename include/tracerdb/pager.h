// TracerDB — Pager: raw page I/O that deliberately bypasses the kernel page
// cache. On Linux the file is opened with O_DIRECT; on macOS (no O_DIRECT)
// we get the same effect with fcntl(F_NOCACHE). Either way, the OS is out of
// the caching business and every cached byte is accounted for by *our*
// user-space page cache.
#pragma once

#include <cerrno>
#include <cstdint>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "tracerdb/common.h"

namespace tracerdb {

class Pager {
 public:
  struct Stats {
    uint64_t reads = 0;   // pages read from disk
    uint64_t writes = 0;  // pages written to disk
  };

  Pager(const std::string& path, bool create) : path_(path) {
    int flags = create ? (O_RDWR | O_CREAT | O_TRUNC) : O_RDWR;
#ifdef O_DIRECT
    flags |= O_DIRECT;  // Linux: bypass the kernel page cache entirely
#endif
    fd_ = ::open(path.c_str(), flags, 0644);
    if (fd_ < 0) die("open(" + path + "): " + std::strerror(errno));
#ifdef __APPLE__
    // macOS has no O_DIRECT; F_NOCACHE tells the unified buffer cache not to
    // retain pages for this descriptor.
    if (fcntl(fd_, F_NOCACHE, 1) < 0) die("fcntl(F_NOCACHE): " + std::string(std::strerror(errno)));
#endif
    struct stat st;
    if (fstat(fd_, &st) != 0) die("fstat: " + std::string(std::strerror(errno)));
    npages_ = static_cast<uint64_t>(st.st_size) / kPageSize;
  }

  ~Pager() {
    if (fd_ >= 0) ::close(fd_);
  }
  Pager(const Pager&) = delete;
  Pager& operator=(const Pager&) = delete;

  // buf must be kIoAlign-aligned (use alloc_aligned).
  void read_page(uint64_t page_id, void* buf) {
    ssize_t n = ::pread(fd_, buf, kPageSize, off(page_id));
    if (n != static_cast<ssize_t>(kPageSize))
      die("pread page " + std::to_string(page_id) + " of " + path_ + ": " +
          (n < 0 ? std::strerror(errno) : "short read"));
    ++stats_.reads;
  }

  // Read n contiguous pages with a single I/O (extent readahead).
  void read_pages(uint64_t page_id, size_t n, void* buf) {
    ssize_t want = static_cast<ssize_t>(n * kPageSize);
    ssize_t got = ::pread(fd_, buf, want, off(page_id));
    if (got != want)
      die("pread run at page " + std::to_string(page_id) + ": " +
          (got < 0 ? std::strerror(errno) : "short read"));
    stats_.reads += n;
  }

  void write_page(uint64_t page_id, const void* buf) {
    ssize_t n = ::pwrite(fd_, buf, kPageSize, off(page_id));
    if (n != static_cast<ssize_t>(kPageSize))
      die("pwrite page " + std::to_string(page_id) + ": " +
          (n < 0 ? std::strerror(errno) : "short write"));
    ++stats_.writes;
    if (page_id >= npages_) npages_ = page_id + 1;
  }

  // Pre-extend the file (sparse where supported). Direct reads of holes
  // return zeros, so a freshly extended page is a valid all-zero page.
  void ensure_pages(uint64_t n) {
    if (n <= npages_) return;
    if (ftruncate(fd_, static_cast<off_t>(n) * kPageSize) != 0)
      die("ftruncate: " + std::string(std::strerror(errno)));
    npages_ = n;
  }

  void sync() {
    if (fsync(fd_) != 0) die("fsync: " + std::string(std::strerror(errno)));
  }

  uint64_t page_count() const { return npages_; }
  const Stats& stats() const { return stats_; }

 private:
  static off_t off(uint64_t page_id) { return static_cast<off_t>(page_id) * kPageSize; }

  std::string path_;
  int fd_ = -1;
  uint64_t npages_ = 0;
  Stats stats_;
};

}  // namespace tracerdb
