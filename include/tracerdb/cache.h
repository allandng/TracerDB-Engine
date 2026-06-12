// TracerDB — user-space page cache (buffer pool).
//
// A fixed budget of page frames backed by one aligned allocation. All file
// access goes through pin()/unpin(); since the file itself is opened with
// O_DIRECT / F_NOCACHE, this pool is the *only* cache between the index and
// the disk, so its frame budget is a hard, honest bound on memory used for
// data — exactly what a RAM-constrained edge device needs.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "tracerdb/common.h"
#include "tracerdb/pager.h"
#include "tracerdb/policy.h"

namespace tracerdb {

struct CacheStats {
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t evictions = 0;
  uint64_t dirty_writebacks = 0;
  uint64_t prefetched = 0;  // pages pulled in by extent readahead

  double hit_rate() const {
    uint64_t t = hits + misses;
    return t ? static_cast<double>(hits) / static_cast<double>(t) : 0.0;
  }
};

class PageCache {
 public:
  // Largest single readahead I/O: 16 pages = 64 KiB.
  static constexpr size_t kMaxReadahead = 16;

  PageCache(Pager& pager, size_t nframes, std::unique_ptr<EvictionPolicy> policy)
      : pager_(pager),
        nframes_(nframes),
        pool_(alloc_aligned(nframes * kPageSize)),
        scratch_(alloc_aligned(kMaxReadahead * kPageSize)),
        frames_(nframes),
        policy_(std::move(policy)) {
    if (nframes_ == 0) die("page cache needs at least 1 frame");
    map_.reserve(nframes_ * 2);
    free_.reserve(nframes_);
    for (size_t f = nframes_; f-- > 0;) free_.push_back(static_cast<frame_id_t>(f));
  }

  // Pin a page and return its 4 KiB of data. The pointer stays valid (and
  // the page stays resident) until the matching unpin().
  //
  // readahead > 0 is a hint that the caller will scan forward: on a miss,
  // up to that many following pages are pulled in with the same single
  // disk read (capped at kMaxReadahead, stopping at already-resident
  // pages). Prefetched pages enter unpinned with one reference, so under
  // LRU-K they stay "cold" and can never displace the hot set.
  std::byte* pin(uint64_t page_id, size_t readahead = 0) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = map_.find(page_id);
    if (it != map_.end()) {
      Frame& fr = frames_[it->second];
      ++fr.pins;
      ++stats_.hits;
      policy_->on_access(it->second);
      return data(it->second);
    }
    ++stats_.misses;

    // Extend the run while pages are non-resident and on disk.
    size_t want = 1 + std::min(readahead, kMaxReadahead - 1);
    want = static_cast<size_t>(
        std::min<uint64_t>(want, pager_.page_count() > page_id ? pager_.page_count() - page_id : 1));
    size_t run = 1;
    while (run < want && map_.find(page_id + run) == map_.end()) ++run;

    if (run == 1) {
      frame_id_t f = acquire_frame();
      pager_.read_page(page_id, data(f));
      install(f, page_id, /*pins=*/1);
      return data(f);
    }
    pager_.read_pages(page_id, run, scratch_.get());
    frame_id_t first = kInvalidFrame;
    for (size_t i = 0; i < run; ++i) {
      frame_id_t f = i == 0 ? acquire_frame() : try_acquire_frame();
      if (f == kInvalidFrame) break;  // pool too contended; keep what we have
      std::memcpy(data(f), scratch_.get() + i * kPageSize, kPageSize);
      install(f, page_id + i, i == 0 ? 1u : 0u);
      if (i == 0) first = f;
      else ++stats_.prefetched;
    }
    return data(first);
  }

  void unpin(uint64_t page_id, bool dirty) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = map_.find(page_id);
    if (it == map_.end()) die("unpin of non-resident page " + std::to_string(page_id));
    Frame& fr = frames_[it->second];
    if (fr.pins == 0) die("unpin of unpinned page " + std::to_string(page_id));
    --fr.pins;
    fr.dirty |= dirty;
  }

  void flush_all() {
    std::lock_guard<std::mutex> lk(mu_);
    for (frame_id_t f = 0; f < nframes_; ++f) {
      Frame& fr = frames_[f];
      if (fr.valid && fr.dirty) {
        pager_.write_page(fr.page, data(f));
        ++stats_.dirty_writebacks;
        fr.dirty = false;
      }
    }
    pager_.sync();
  }

  size_t frame_count() const { return nframes_; }
  size_t budget_bytes() const { return nframes_ * kPageSize; }
  const CacheStats& stats() const { return stats_; }
  const char* policy_name() const { return policy_->name(); }
  void reset_stats() {
    std::lock_guard<std::mutex> lk(mu_);
    stats_ = CacheStats{};
  }

 private:
  struct Frame {
    uint64_t page = UINT64_MAX;
    uint32_t pins = 0;
    bool dirty = false;
    bool valid = false;
  };

  std::byte* data(frame_id_t f) { return pool_.get() + static_cast<size_t>(f) * kPageSize; }

  void install(frame_id_t f, uint64_t page_id, uint32_t pins) {
    Frame& fr = frames_[f];
    fr.page = page_id;
    fr.pins = pins;
    fr.dirty = false;
    fr.valid = true;
    map_.emplace(page_id, f);
    policy_->on_fill(f, page_id);
  }

  frame_id_t acquire_frame() {
    frame_id_t f = try_acquire_frame();
    if (f == kInvalidFrame) die("page cache exhausted: all frames pinned");
    return f;
  }

  frame_id_t try_acquire_frame() {
    if (!free_.empty()) {
      frame_id_t f = free_.back();
      free_.pop_back();
      return f;
    }
    frame_id_t f = policy_->pick_victim(
        [this](frame_id_t v) { return frames_[v].valid && frames_[v].pins == 0; });
    if (f == kInvalidFrame) return kInvalidFrame;
    Frame& fr = frames_[f];
    if (fr.dirty) {
      pager_.write_page(fr.page, data(f));
      ++stats_.dirty_writebacks;
    }
    map_.erase(fr.page);
    policy_->on_clear(f);
    fr = Frame{};
    ++stats_.evictions;
    return f;
  }

  Pager& pager_;
  size_t nframes_;
  AlignedBuf pool_;
  AlignedBuf scratch_;  // staging buffer for multi-page readahead I/O
  std::vector<Frame> frames_;
  std::unordered_map<uint64_t, frame_id_t> map_;
  std::vector<frame_id_t> free_;
  std::unique_ptr<EvictionPolicy> policy_;
  std::mutex mu_;
  CacheStats stats_;
};

// RAII pin: unpins on destruction.
class PageGuard {
 public:
  PageGuard(PageCache& cache, uint64_t page_id, size_t readahead = 0)
      : cache_(&cache), page_id_(page_id), data_(cache.pin(page_id, readahead)) {}
  ~PageGuard() {
    if (cache_) cache_->unpin(page_id_, dirty_);
  }
  PageGuard(const PageGuard&) = delete;
  PageGuard& operator=(const PageGuard&) = delete;
  PageGuard(PageGuard&& o) noexcept
      : cache_(o.cache_), page_id_(o.page_id_), data_(o.data_), dirty_(o.dirty_) {
    o.cache_ = nullptr;
  }

  std::byte* data() { return data_; }
  const std::byte* data() const { return data_; }
  void mark_dirty() { dirty_ = true; }

 private:
  PageCache* cache_;
  uint64_t page_id_;
  std::byte* data_;
  bool dirty_ = false;
};

}  // namespace tracerdb
