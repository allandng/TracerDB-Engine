// Pager: direct-I/O page roundtrip, file growth, hole reads.
#include <cassert>
#include <cstdio>
#include <cstring>

#include "tracerdb/pager.h"

using namespace tracerdb;

int main() {
  const char* path = "tracerdb_test_pager.bin";
  std::remove(path);
  {
    Pager p(path, true);
    auto buf = alloc_aligned(kPageSize);
    auto buf2 = alloc_aligned(kPageSize);

    // Write distinct patterns to 8 pages, read back.
    for (uint64_t pg = 0; pg < 8; ++pg) {
      std::memset(buf.get(), static_cast<int>(0x10 + pg), kPageSize);
      p.write_page(pg, buf.get());
    }
    assert(p.page_count() == 8);
    for (uint64_t pg = 0; pg < 8; ++pg) {
      p.read_page(pg, buf2.get());
      for (size_t i = 0; i < kPageSize; ++i)
        assert(static_cast<uint8_t>(buf2.get()[i]) == 0x10 + pg);
    }
    assert(p.stats().reads == 8 && p.stats().writes == 8);

    // ensure_pages + reading a hole returns zeros.
    p.ensure_pages(16);
    assert(p.page_count() == 16);
    p.read_page(12, buf2.get());
    for (size_t i = 0; i < kPageSize; ++i) assert(buf2.get()[i] == std::byte{0});
    p.sync();
  }
  {
    // Reopen without create: size persists.
    Pager p(path, false);
    assert(p.page_count() == 16);
    auto buf = alloc_aligned(kPageSize);
    p.read_page(3, buf.get());
    for (size_t i = 0; i < kPageSize; ++i) assert(static_cast<uint8_t>(buf.get()[i]) == 0x13);
  }
  std::remove(path);
  std::printf("test_pager OK\n");
  return 0;
}
