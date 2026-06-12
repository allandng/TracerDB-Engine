// TracerDB — pluggable page-replacement policies.
//
// The cache asks the policy for a victim frame on every miss once all frames
// are occupied. Three policies are provided:
//
//   CLOCK  — classic second-chance: one reference bit per frame, a sweeping
//            hand. O(1) amortized, near-LRU behavior, no timestamps.
//   LRU    — exact least-recently-used. Simple, but a single sequential scan
//            of a large posting list flushes every hot page out.
//   LRU-K  — O'Neil et al.'s LRU-K (default K=2). A frame's priority is its
//            K-th most recent reference. Pages referenced only once (a scan)
//            have backward-K-distance = infinity and are evicted first, so
//            frequently re-referenced pages — the IVF centroid pages that
//            every single query touches — survive arbitrarily large one-off
//            posting-list scans. This is the scan resistance that makes a
//            tiny cache work for vector search.
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <list>
#include <queue>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tracerdb/common.h"

namespace tracerdb {

using frame_id_t = uint32_t;
constexpr frame_id_t kInvalidFrame = UINT32_MAX;

class EvictionPolicy {
 public:
  virtual ~EvictionPolicy() = default;
  // Page page_id was loaded into frame f (counts as a reference). Policies
  // that retain per-page history across evictions (LRU-K) use page_id.
  virtual void on_fill(frame_id_t f, uint64_t page_id) = 0;
  // The page in frame f was referenced (one pin = one reference).
  virtual void on_access(frame_id_t f) = 0;
  // Frame f was evicted / invalidated; forget its history.
  virtual void on_clear(frame_id_t f) = 0;
  // Choose a victim among frames where evictable(f) is true (resident and
  // unpinned). Return kInvalidFrame if none qualifies.
  virtual frame_id_t pick_victim(const std::function<bool(frame_id_t)>& evictable) = 0;
  virtual const char* name() const = 0;
};

class ClockPolicy final : public EvictionPolicy {
 public:
  explicit ClockPolicy(size_t nframes) : ref_(nframes, 0) {}

  void on_fill(frame_id_t f, uint64_t) override { ref_[f] = 1; }
  void on_access(frame_id_t f) override { ref_[f] = 1; }
  void on_clear(frame_id_t f) override { ref_[f] = 0; }

  frame_id_t pick_victim(const std::function<bool(frame_id_t)>& evictable) override {
    const size_t n = ref_.size();
    // Two full sweeps suffice: the first clears reference bits, the second
    // must find a victim unless everything is pinned.
    for (size_t step = 0; step < 2 * n + 1; ++step) {
      frame_id_t f = static_cast<frame_id_t>(hand_);
      hand_ = (hand_ + 1) % n;
      if (!evictable(f)) continue;
      if (ref_[f]) {
        ref_[f] = 0;  // second chance
      } else {
        return f;
      }
    }
    return kInvalidFrame;
  }

  const char* name() const override { return "clock"; }

 private:
  std::vector<uint8_t> ref_;
  size_t hand_ = 0;
};

class LruPolicy final : public EvictionPolicy {
 public:
  explicit LruPolicy(size_t nframes) : last_used_(nframes, 0) {}

  void on_fill(frame_id_t f, uint64_t) override { last_used_[f] = ++clock_; }
  void on_access(frame_id_t f) override { last_used_[f] = ++clock_; }
  void on_clear(frame_id_t f) override { last_used_[f] = 0; }

  frame_id_t pick_victim(const std::function<bool(frame_id_t)>& evictable) override {
    frame_id_t best = kInvalidFrame;
    uint64_t best_t = UINT64_MAX;
    for (frame_id_t f = 0; f < last_used_.size(); ++f) {
      if (!evictable(f)) continue;
      if (last_used_[f] < best_t) {
        best_t = last_used_[f];
        best = f;
      }
    }
    return best;
  }

  const char* name() const override { return "lru"; }

 private:
  std::vector<uint64_t> last_used_;
  uint64_t clock_ = 0;
};

class LruKPolicy final : public EvictionPolicy {
 public:
  // Exact LRU-K with O(log n) amortized eviction via a lazy-invalidation
  // min-heap: every reference pushes the frame's new (class, priority) state
  // tagged with a per-frame stamp; pick_victim pops entries whose stamp no
  // longer matches. Frames with fewer than K references ("cold", class 0,
  // priority = most recent reference) are evicted before frames with a full
  // history ("warm", class 1, priority = K-th most recent reference, i.e.
  // maximum backward K-distance first).
  //
  // retain: the Retained Information Period from the LRU-K paper — a frame
  // referenced within the last `retain` ticks is exempt from eviction while
  // any other candidate exists. Without this, a freshly prefetched run of
  // scan pages (each holding a single-reference history, i.e. the most
  // evictable class) gets cannibalized by its own batch before the scan can
  // consume it. The period is capped at half the pool: in a tiny cache a
  // long window would cover every cold frame and push eviction onto the hot
  // set.
  explicit LruKPolicy(size_t nframes, size_t k = 2, uint64_t retain = 64)
      : k_(k),
        retain_(std::min<uint64_t>(retain, nframes / 2)),
        hist_(nframes),
        nref_(nframes, 0),
        stamp_(nframes, 0),
        tracked_(nframes, false),
        page_of_(nframes, 0) {
    for (auto& h : hist_) h.assign(k_, 0);
  }

  void on_fill(frame_id_t f, uint64_t page_id) override {
    nref_[f] = 0;
    std::fill(hist_[f].begin(), hist_[f].end(), 0);
    page_of_[f] = page_id;
    // Retained Information (LRU-K paper): if this page was evicted recently,
    // restore its reference history. Without this a hot page in a small
    // cache could never accumulate K spaced-out references — it would be
    // evicted between them and restart from scratch every time.
    auto it = retained_.find(page_id);
    if (it != retained_.end()) {
      hist_[f] = it->second.times;
      nref_[f] = it->second.nref;
      retained_.erase(it);
    }
    tracked_[f] = true;
    record(f);
  }
  void on_access(frame_id_t f) override { record(f); }
  void on_clear(frame_id_t f) override {
    if (nref_[f] > 0) {
      retained_[page_of_[f]] = {hist_[f], nref_[f]};
      if (retained_.size() > 8 * hist_.size()) sweep_retained();
    }
    nref_[f] = 0;
    std::fill(hist_[f].begin(), hist_[f].end(), 0);
    tracked_[f] = false;
    ++stamp_[f];  // invalidate any heap entries for this frame
  }

  frame_id_t pick_victim(const std::function<bool(frame_id_t)>& evictable) override {
    side_.clear();
    frame_id_t found = kInvalidFrame;
    while (!heap_.empty()) {
      Entry e = heap_.top();
      if (e.stamp != stamp_[e.f]) {  // stale state, discard
        heap_.pop();
        continue;
      }
      if (!evictable(e.f) || most_recent(e.f) + retain_ > clock_) {
        heap_.pop();
        side_.push_back(e);  // pinned or retained: reconsider later
        continue;
      }
      heap_.pop();
      found = e.f;
      break;
    }
    if (found == kInvalidFrame) {
      // Everything is pinned or retained; take the best evictable candidate
      // ignoring retention.
      const Entry* best = nullptr;
      for (const Entry& e : side_)
        if (evictable(e.f) && (!best || e.key() < best->key())) best = &e;
      if (best) found = best->f;
    }
    for (const Entry& e : side_)
      if (e.f != found) heap_.push(e);
    return found;
  }

  const char* name() const override { return "lru-k"; }

 private:
  struct Entry {
    uint64_t cls;   // 0 = cold (< K refs), 1 = warm
    uint64_t time;  // cold: most recent ref; warm: K-th most recent ref
    frame_id_t f;
    uint64_t stamp;
    std::pair<uint64_t, uint64_t> key() const { return {cls, time}; }
  };
  struct Cmp {  // min-heap on (cls, time)
    bool operator()(const Entry& a, const Entry& b) const { return a.key() > b.key(); }
  };

  void record(frame_id_t f) {
    auto& h = hist_[f];
    uint64_t now = ++clock_;
    if (nref_[f] > 0 && most_recent(f) + retain_ > now) {
      // Correlated Reference Period (same window as retention): references
      // in quick succession — e.g. a prefetch fill followed by the scan that
      // consumes the page — count as a single reference, so a one-pass scan
      // can never promote its pages to "warm". Only re-references spaced
      // further apart (a page hot across queries) build a real K-history.
      h[(nref_[f] - 1) % k_] = now;
    } else {
      // h is a ring of the K most recent reference times; slot = nref mod K.
      h[nref_[f] % k_] = now;
      ++nref_[f];
    }
    push_state(f);
    // Lazy entries accumulate one per reference; periodically rebuild from
    // live state so the heap stays O(nframes).
    if (heap_.size() > 16 * hist_.size()) rebuild();
  }

  void push_state(frame_id_t f) {
    Entry e;
    e.f = f;
    e.stamp = ++stamp_[f];
    if (nref_[f] < k_) {
      e.cls = 0;
      e.time = most_recent(f);
    } else {
      e.cls = 1;
      e.time = kth_recent(f);
    }
    heap_.push(e);
  }

  void rebuild() {
    heap_ = {};
    for (frame_id_t f = 0; f < hist_.size(); ++f)
      if (tracked_[f]) push_state(f);
  }

  uint64_t most_recent(frame_id_t f) const {
    return nref_[f] == 0 ? 0 : hist_[f][(nref_[f] - 1) % k_];
  }
  uint64_t kth_recent(frame_id_t f) const {
    // With >= K references, the oldest entry in the ring is the K-th most
    // recent: it sits at slot nref mod K (the next slot to be overwritten).
    return hist_[f][nref_[f] % k_];
  }

  // Drop retained histories whose last reference is outside the history
  // horizon; called rarely (amortized O(1) per eviction).
  void sweep_retained() {
    const uint64_t horizon = std::max<uint64_t>(16 * hist_.size(), 256);
    for (auto it = retained_.begin(); it != retained_.end();) {
      uint64_t last = it->second.times[(it->second.nref - 1) % k_];
      if (last + horizon < clock_)
        it = retained_.erase(it);
      else
        ++it;
    }
  }

  struct Retained {
    std::vector<uint64_t> times;
    uint64_t nref;
  };

  size_t k_;
  uint64_t retain_;
  uint64_t clock_ = 0;
  std::vector<std::vector<uint64_t>> hist_;
  std::vector<uint64_t> nref_;
  std::vector<uint64_t> stamp_;
  std::vector<bool> tracked_;
  std::vector<uint64_t> page_of_;
  std::unordered_map<uint64_t, Retained> retained_;
  std::priority_queue<Entry, std::vector<Entry>, Cmp> heap_;
  std::vector<Entry> side_;
};

inline std::unique_ptr<EvictionPolicy> make_policy(const std::string& name, size_t nframes,
                                                   size_t k = 2) {
  if (name == "clock") return std::make_unique<ClockPolicy>(nframes);
  if (name == "lru") return std::make_unique<LruPolicy>(nframes);
  if (name == "lru2" || name == "lruk" || name == "lru-k")
    return std::make_unique<LruKPolicy>(nframes, k);
  die("unknown eviction policy '" + name + "' (expected clock|lru|lru2)");
}

}  // namespace tracerdb
