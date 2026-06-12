// TracerDB — disk-backed IVF search.
//
// Every byte of vector data is read through the page cache; nothing about
// the index is memory-mapped or slurped into RAM. Per query:
//   1. scan the centroid pages (a handful of pages the cache quickly learns
//      to keep resident — they are re-referenced by every query, so LRU-K
//      gives them maximal priority) and pick the nprobe nearest lists
//   2. stream those lists' posting pages through the cache, maintaining a
//      top-k heap
// RAM use is bounded by the cache budget + O(nlist) metadata.
#pragma once

#include <algorithm>
#include <cstdint>
#include <queue>
#include <string>
#include <vector>

#include "tracerdb/cache.h"
#include "tracerdb/format.h"
#include "tracerdb/pager.h"
#include "tracerdb/policy.h"

namespace tracerdb {

struct SearchResult {
  uint64_t id;
  float dist;
};

class TracerIndex {
 public:
  TracerIndex(const std::string& path, size_t cache_frames, const std::string& policy,
              size_t lruk_k = 2)
      : pager_(path, /*create=*/false),
        cache_(pager_, cache_frames, make_policy(policy, cache_frames, lruk_k)) {
    {
      PageGuard g(cache_, 0);
      std::memcpy(&hdr_, g.data(), sizeof(hdr_));
    }
    if (hdr_.magic != kMagic) die("not a TracerDB index: " + path);
    if (hdr_.version != kVersion) die("index version mismatch");
    lay_ = Layout{hdr_.dim, hdr_.nlist};

    // The extent table is tiny fixed metadata (16 B per list); load it once.
    extents_.resize(hdr_.nlist);
    for (uint32_t c = 0; c < hdr_.nlist; ++c) {
      uint64_t pg = hdr_.extent_start + c / lay_.extents_per_page();
      PageGuard g(cache_, pg);
      std::memcpy(&extents_[c], g.data() + (c % lay_.extents_per_page()) * sizeof(ListExtent),
                  sizeof(ListExtent));
    }
  }

  std::vector<SearchResult> search(const float* query, size_t k, size_t nprobe) {
    nprobe = std::min<size_t>(nprobe, hdr_.nlist);

    // ---- coarse step: distance to every centroid, via the cache ----
    std::vector<std::pair<float, uint32_t>> cd;
    cd.reserve(hdr_.nlist);
    for (uint64_t pg = 0; pg < hdr_.centroid_pages; ++pg) {
      PageGuard g(cache_, hdr_.centroid_start + pg, hdr_.centroid_pages - pg - 1);
      uint32_t first = static_cast<uint32_t>(pg * lay_.cent_per_page());
      uint32_t last = std::min<uint32_t>(hdr_.nlist, first + lay_.cent_per_page());
      for (uint32_t c = first; c < last; ++c) {
        const float* cent =
            reinterpret_cast<const float*>(g.data() + (c - first) * lay_.vec_bytes());
        cd.emplace_back(l2sq(query, cent, hdr_.dim), c);
      }
    }
    std::partial_sort(cd.begin(), cd.begin() + nprobe, cd.end());

    // ---- fine step: scan the nprobe posting lists ----
    // Max-heap of size k on distance: top() is the worst kept result.
    std::priority_queue<SearchResult, std::vector<SearchResult>,
                        decltype(&TracerIndex::heap_less)>
        heap(&TracerIndex::heap_less);
    for (size_t p = 0; p < nprobe; ++p) {
      const ListExtent& ext = extents_[cd[p].second];
      uint64_t remaining = ext.count;
      const uint64_t npages = lay_.list_pages(ext.count);
      for (uint64_t pg = 0; remaining > 0; ++pg) {
        // Posting lists are contiguous extents: hint the cache to pull the
        // rest of the list in with one large read.
        PageGuard g(cache_, ext.start_page + pg, npages - pg - 1);
        size_t in_page = std::min<uint64_t>(remaining, lay_.entries_per_page());
        for (size_t e = 0; e < in_page; ++e) {
          const std::byte* entry = g.data() + e * lay_.entry_stride();
          const float* vec = reinterpret_cast<const float*>(entry + 8);
          float d = l2sq(query, vec, hdr_.dim);
          if (heap.size() < k) {
            uint64_t id;
            std::memcpy(&id, entry, sizeof(id));
            heap.push({id, d});
          } else if (d < heap.top().dist) {
            uint64_t id;
            std::memcpy(&id, entry, sizeof(id));
            heap.pop();
            heap.push({id, d});
          }
        }
        remaining -= in_page;
      }
    }

    std::vector<SearchResult> out(heap.size());
    for (size_t i = heap.size(); i-- > 0;) {
      out[i] = heap.top();
      heap.pop();
    }
    return out;
  }

  uint32_t dim() const { return hdr_.dim; }
  uint32_t nlist() const { return hdr_.nlist; }
  uint64_t size() const { return hdr_.nvec; }
  PageCache& cache() { return cache_; }
  const Pager& pager() const { return pager_; }

 private:
  static bool heap_less(const SearchResult& a, const SearchResult& b) { return a.dist < b.dist; }

  Pager pager_;
  PageCache cache_;
  IndexHeader hdr_{};
  Layout lay_{};
  std::vector<ListExtent> extents_;
};

}  // namespace tracerdb
