// TracerDB — on-disk index format.
//
// One file of 4 KiB pages:
//
//   page 0                       IndexHeader
//   [centroid_start, +cpages)   centroids, cent_per_page() per page, no
//                               vector ever straddles a page boundary
//   [extent_start, +epages)     ListExtent[nlist]: where each IVF list lives
//   [lists...]                  posting pages: fixed-stride entries of
//                               { uint64 id, float vec[dim] }
//
// Keeping every record inside a single page is what lets the page cache be
// the unit of both I/O and eviction — no record ever needs two frames.
#pragma once

#include <cstdint>

#include "tracerdb/common.h"

namespace tracerdb {

constexpr uint64_t kMagic = 0x5452414345524442ULL;  // "TRACERDB"
constexpr uint32_t kVersion = 1;

struct IndexHeader {
  uint64_t magic;
  uint32_t version;
  uint32_t dim;
  uint32_t nlist;
  uint32_t pad_;
  uint64_t nvec;
  uint64_t centroid_start;
  uint64_t centroid_pages;
  uint64_t extent_start;
  uint64_t extent_pages;
};
static_assert(sizeof(IndexHeader) <= kPageSize, "header must fit one page");

struct ListExtent {
  uint64_t start_page;  // first posting page of this list
  uint64_t count;       // number of vectors in this list
};
static_assert(sizeof(ListExtent) == 16, "extent layout");

struct Layout {
  uint32_t dim;
  uint32_t nlist;

  size_t vec_bytes() const { return static_cast<size_t>(dim) * sizeof(float); }
  size_t cent_per_page() const { return kPageSize / vec_bytes(); }
  size_t centroid_pages() const { return (nlist + cent_per_page() - 1) / cent_per_page(); }

  size_t extents_per_page() const { return kPageSize / sizeof(ListExtent); }
  size_t extent_pages() const { return (nlist + extents_per_page() - 1) / extents_per_page(); }

  // Posting entry: 8-byte id + vector, fixed stride.
  size_t entry_stride() const { return 8 + vec_bytes(); }
  size_t entries_per_page() const { return kPageSize / entry_stride(); }
  size_t list_pages(uint64_t count) const {
    return (count + entries_per_page() - 1) / entries_per_page();
  }

  uint64_t centroid_start() const { return 1; }
  uint64_t extent_start() const { return centroid_start() + centroid_pages(); }
  uint64_t lists_start() const { return extent_start() + extent_pages(); }

  static void validate_dim(uint32_t dim) {
    if (dim == 0) die("dim must be > 0");
    Layout l{dim, 1};
    if (l.entry_stride() > kPageSize)
      die("dim " + std::to_string(dim) + " too large: an entry must fit in one " +
          std::to_string(kPageSize) + "-byte page (max dim " +
          std::to_string((kPageSize - 8) / sizeof(float)) + ")");
  }
};

}  // namespace tracerdb
