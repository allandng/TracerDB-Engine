# TracerDB

A disk-backed vector search engine for edge devices, written from scratch in
C++17. TracerDB performs semantic (nearest-neighbor) search over embedding
datasets far larger than RAM by bypassing the operating system's page cache
entirely and managing every cached byte itself, with a user-space page cache
and an LRU-K replacement policy tuned for vector-index access patterns.

On a 280 MiB index of 500,000 × 128-dim embeddings, searched through a
**2 MiB** page cache (peak process RSS ≈ 5 MiB):

| policy | QPS | mean ms | hit rate | disk read / 200 queries | recall@10 |
|--------|----:|--------:|---------:|------------------------:|----------:|
| CLOCK  | 158 | 6.3     | 93.1%    | 495 MiB                 | 0.794     |
| LRU    | 152 | 6.6     | 93.2%    | 495 MiB                 | 0.794     |
| LRU-2  | **201** | **5.0** | **95.5%** | **305 MiB**        | 0.794     |

Same data, same cache budget — the only difference is the eviction policy.
LRU-K does 38% less disk I/O because it refuses to evict the index pages that
every query touches.

It scales: a 1.1 GiB index (2,000,000 × 128-dim vectors) searched through a
16 MiB cache runs at 21.2 QPS / 47 ms mean latency with recall@10 = 0.854
(nprobe 24/2048) and **21.9 MiB peak RSS** — memory does not grow with the
dataset, only with the budget you choose.

## Why

Standard vector databases assume cloud-sized RAM and either load all
embeddings into memory or `mmap` the file and let the kernel decide what
stays resident. On a Raspberry Pi or any RAM-constrained device, the first
approach OOMs and the second thrashes: the kernel's LRU-ish page cache has no
idea which of your pages are index hot-set and which are one-pass scan data.

TracerDB takes both jobs back from the OS:

- **Direct I/O.** Index files are opened with `O_DIRECT` on Linux
  (`fcntl(F_NOCACHE)` on macOS), so the kernel caches nothing. Every cached
  page lives in TracerDB's own buffer pool, making the memory budget a hard,
  honest number you choose at open time.
- **A user-space page cache** ([cache.h](include/tracerdb/cache.h)): a fixed
  pool of pinned/unpinned 4 KiB frames over one aligned allocation, with
  pluggable replacement policies and extent readahead.
- **LRU-K replacement** ([policy.h](include/tracerdb/policy.h)): the
  O'Neil et al. algorithm, implemented with its full machinery — backward
  K-distance, correlated reference periods, and retained (post-eviction)
  history — plus an O(log n) eviction via a lazy-invalidation heap. CLOCK and
  exact LRU are included for comparison.

## How search stays inside the budget

The index is IVF (inverted file): k-means centroids partition the vectors
into lists, each stored as a contiguous run of pages. A query

1. scans the **centroid pages** to find the `nprobe` nearest lists — a few
   hundred KiB that *every* query re-references, and
2. streams those lists' **posting pages** once each through the cache,
   maintaining a top-k heap.

That access pattern is exactly the case LRU-K was designed for: centroid
pages accumulate K spaced-out references and become "warm" (evicted last),
while posting pages — touched once, or twice in quick succession via
readahead — never leave the "cold" class (evicted first). A one-pass scan of
any size cannot displace the hot set. CLOCK and plain LRU, by contrast, let
every scan flush the centroids, which is why they re-read ~190 MiB more from
disk in the table above.

Three details of the LRU-K implementation matter in practice (each one fixed
a measured pathology during development):

- **Correlated reference period.** A readahead fill followed milliseconds
  later by the scan consuming that page would otherwise count as 2
  references and promote scan pages to warm. References inside a short
  window collapse into one.
- **Retained information period.** Freshly prefetched pages hold the most
  evictable (single-reference) histories; without a grace window a prefetch
  batch cannibalizes itself before the scan reaches it.
- **Page-level history.** Reference history is keyed by page, not frame, and
  survives eviction (bounded by a time horizon), so a hot page in a tiny
  cache can accumulate its K references across evictions.

## Layout

```
include/tracerdb/
  pager.h     direct-I/O page file (O_DIRECT / F_NOCACHE), extent reads
  cache.h     buffer pool: pin/unpin, dirty write-back, readahead, stats
  policy.h    CLOCK, LRU, LRU-K eviction (pluggable)
  format.h    on-disk index format (header / centroids / extents / lists)
  dataset.h   raw vector file, streaming reader/writer
  kmeans.h    sample-bounded k-means for the coarse quantizer
  builder.h   two-pass, memory-bounded index build
  index.h     IVF search through the cache
tools/        tracer_gen, tracer_build, tracer_query, tracer_bench
tests/        pager, cache, policy (scan resistance), index (vs brute force)
```

Everything is header-only; the library has no dependencies beyond POSIX.

## Build & run

```sh
make            # builds tools + tests (any clang/gcc with C++17)
make test       # unit + integration tests
./demo.sh       # generate -> build -> benchmark, defaults to a 2 MiB cache
```

The demo is parameterized by env vars, e.g. a 1 GB dataset searched through
16 MiB of cache:

```sh
N=2000000 NLIST=2048 CACHE_MB=16 NPROBE=24 ./demo.sh
```

Choose `nlist` ≥ the number of natural clusters you expect in the data;
recall drops when true neighbor groups straddle list boundaries. The k-means
sample auto-scales to ~48 vectors per list (override with `--sample`).

Individual tools:

```sh
./build/tracer_gen   data.vec --n 500000 --dim 128 --clusters 1000
./build/tracer_build data.vec index.tdb --nlist 2048 --cache-mb 16
./build/tracer_query index.tdb --raw data.vec --row 123 --k 10 --nprobe 8
./build/tracer_bench index.tdb --raw data.vec --cache-mb 2 \
                     --policies clock,lru,lru2
```

`tracer_bench` reports recall@k against exact ground truth (computed by a
bounded-memory streaming scan of the raw file) and the process's peak RSS, so
the "no OOM, no swap" claim is checked on every run.

## Memory accounting

Peak RSS during search ≈ cache budget + O(nlist) metadata (16 B/list extent
table) + program text/stack. The build is also bounded: k-means trains on a
capped sample, pass 1 streams the dataset keeping only per-list counts, and
pass 2 scatters vectors to their final pages through the same page cache
(assignments spill to a temp file). Building the 280 MiB index above peaks at
~28 MiB RSS; searching it peaks at ~5 MiB with the 2 MiB cache.

One build-time finding worth noting: the scatter pass re-touches *recently
started* pages (pure recency, no reuse), and LRU-K's evict-short-history-first
rule is pathological for it — 5.6% hit rate vs CLOCK's 85.6%. The builder
therefore defaults to CLOCK and search defaults to LRU-K; the policy is a
constructor argument either way.

## License

MIT — see [LICENSE](LICENSE).

## Notes for Raspberry Pi

Everything is plain POSIX + C++17 and compiles unmodified on a Pi
(`make CXX=g++`). On Linux the pager uses real `O_DIRECT`, which requires the
4 KiB-aligned buffers the cache already uses. On a Pi 3 with 1 GB RAM, a
multi-GB index searches fine with e.g. `--cache-mb 64`: RSS stays at tens of
MiB regardless of index size, because nothing outside the pool scales with
the dataset.
