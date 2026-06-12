// TracerDB — raw vector dataset file (build input / ground-truth source).
//
// Plain binary: u64 nvec, u32 dim, u32 reserved, then nvec*dim floats.
// This file is only ever read/written as a sequential stream with a small
// bounded buffer — buffered stdio is fine here; the *index* is what lives
// behind direct I/O and the page cache.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "tracerdb/common.h"

namespace tracerdb {

struct DatasetHeader {
  uint64_t nvec;
  uint32_t dim;
  uint32_t reserved;
};

class DatasetWriter {
 public:
  DatasetWriter(const std::string& path, uint32_t dim) : dim_(dim) {
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) die("fopen(w) " + path);
    DatasetHeader h{0, dim, 0};
    if (std::fwrite(&h, sizeof(h), 1, f_) != 1) die("write header");
  }
  ~DatasetWriter() {
    if (f_) finish();
  }

  void append(const float* v, size_t n) {
    if (std::fwrite(v, sizeof(float) * dim_, n, f_) != n) die("write vectors");
    nvec_ += n;
  }

  void finish() {
    DatasetHeader h{nvec_, dim_, 0};
    std::fseek(f_, 0, SEEK_SET);
    if (std::fwrite(&h, sizeof(h), 1, f_) != 1) die("rewrite header");
    std::fclose(f_);
    f_ = nullptr;
  }

 private:
  FILE* f_;
  uint32_t dim_;
  uint64_t nvec_ = 0;
};

class DatasetReader {
 public:
  explicit DatasetReader(const std::string& path) : path_(path) {
    f_ = std::fopen(path.c_str(), "rb");
    if (!f_) die("fopen(r) " + path);
    if (std::fread(&h_, sizeof(h_), 1, f_) != 1) die("read header " + path);
  }
  ~DatasetReader() {
    if (f_) std::fclose(f_);
  }

  uint64_t nvec() const { return h_.nvec; }
  uint32_t dim() const { return h_.dim; }

  // Sequential batch read; returns vectors read (0 at EOF).
  size_t read_batch(float* out, size_t maxn) {
    return std::fread(out, sizeof(float) * h_.dim, maxn, f_);
  }

  void reset() { std::fseek(f_, sizeof(DatasetHeader), SEEK_SET); }

  void read_row(uint64_t row, float* out) {
    long off = static_cast<long>(sizeof(DatasetHeader) +
                                 row * static_cast<uint64_t>(h_.dim) * sizeof(float));
    if (std::fseek(f_, off, SEEK_SET) != 0) die("seek row");
    if (std::fread(out, sizeof(float) * h_.dim, 1, f_) != 1) die("read row");
  }

 private:
  std::string path_;
  FILE* f_;
  DatasetHeader h_{};
};

}  // namespace tracerdb
