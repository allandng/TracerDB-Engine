#!/bin/sh
# TracerDB end-to-end demo: semantic search over a dataset much larger than
# the memory we allow ourselves, on any machine (laptop, Raspberry Pi, ...).
#
# Tunables (env vars):
#   N        vectors            (default 500000  ~ 256 MiB raw)
#   DIM      dimensions         (default 128)
#   NLIST    IVF lists          (default 2048)
#   CACHE_MB search page cache  (default 2)
#   NPROBE   lists probed/query (default 8)
set -e
N=${N:-500000}
DIM=${DIM:-128}
NLIST=${NLIST:-2048}
CACHE_MB=${CACHE_MB:-2}
NPROBE=${NPROBE:-8}

make tools >/dev/null
mkdir -p data

echo "== 1. generating $N x $DIM clustered embeddings =="
./build/tracer_gen data/demo.vec --n "$N" --dim "$DIM" --clusters $((NLIST / 2)) --noise 0.15

echo
echo "== 2. building disk-backed IVF index (16 MiB build cache) =="
./build/tracer_build data/demo.vec data/demo.tdb --nlist "$NLIST" --cache-mb 16

echo
echo "== 3. searching through a ${CACHE_MB} MiB page cache, all policies =="
./build/tracer_bench data/demo.tdb --raw data/demo.vec \
  --queries 200 --k 10 --nprobe "$NPROBE" --cache-mb "$CACHE_MB" \
  --policies clock,lru,lru2 --gt 32
