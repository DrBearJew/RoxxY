#!/bin/bash
# Benchmark: q8_0 vs packed16-only K at 32k context
set -e

MODEL=/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M.gguf
TESTFILE=/tmp/wiki_3articles.txt
CTX=32768
BATCH=512
UBATCH=512
NGL=99
BUILD=/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github/build-rocm-rdna3-fa

echo "=== Benchmark: q8_0 vs packed16-only K at ${CTX} ==="
echo "Test file: $(wc -c < $TESTFILE) bytes"
echo ""

run_one() {
    local NAME="$1"; shift
    echo "--- $NAME ---"
    local LOG=/tmp/bench_${NAME}.log
    ( export "$@"; timeout 180 $BUILD/bin/llama-perplexity \
        -m $MODEL --no-warmup -ngl $NGL \
        -c $CTX -b $BATCH -ub $UBATCH -fit off \
        -f $TESTFILE 2>$LOG >/tmp/bench_${NAME}_out.txt
    local RC=$?
    grep "KV buffer size" $LOG | tail -1
    grep "size = " $LOG | grep "KV" | tail -1
    grep "total time" $LOG | tail -1
    grep "Final estimate" $LOG | tail -1
    tail -1 /tmp/bench_${NAME}_out.txt
    echo ""
}

run_one "q8_0_KV" --cache-type-k q8_0 --cache-type-v q8_0

run_one "packed16_K" \
    GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 \
    GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
    GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
    GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq \
    GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT=blockfa_recthist_v4_single \
    GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1 \
    GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1

echo "=== Done ==="
