#!/usr/bin/env bash
# Benchmark: q8_0 vs packed16-only K at 32k context.
# This is a bounded helper; callers must provide an existing build/model/test file.
set -euo pipefail

MODEL="${MODEL:-/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M.gguf}"
TESTFILE="${TESTFILE:-/tmp/wiki_3articles.txt}"
CTX="${CTX:-32768}"
BATCH="${BATCH:-512}"
UBATCH="${UBATCH:-512}"
NGL="${NGL:-99}"
BUILD="${BUILD:-/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github/build-rocm-rdna3-fa}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"

if [[ ! -x "$BUILD/bin/llama-perplexity" ]]; then
    echo "missing executable: $BUILD/bin/llama-perplexity" >&2
    echo "set BUILD=/path/to/build or rebuild before running" >&2
    exit 2
fi
if [[ ! -f "$MODEL" ]]; then
    echo "missing model: $MODEL" >&2
    exit 2
fi
if [[ ! -f "$TESTFILE" ]]; then
    echo "missing test file: $TESTFILE" >&2
    exit 2
fi

echo "=== Benchmark: q8_0 vs packed16-only K at ${CTX} ==="
echo "Test file: $(wc -c < "$TESTFILE") bytes"
echo ""

run_one() {
    local name="$1"
    shift
    local -a env_args=()
    local -a llama_args=()

    while (($#)); do
        if [[ "$1" == "--" ]]; then
            shift
            llama_args=("$@")
            break
        fi
        env_args+=("$1")
        shift
    done

    echo "--- $name ---"
    local log="/tmp/bench_${name}.log"
    local out="/tmp/bench_${name}_out.txt"

    env "${env_args[@]}" timeout "$TIMEOUT_SEC" "$BUILD/bin/llama-perplexity" \
        -m "$MODEL" --no-warmup -ngl "$NGL" \
        -c "$CTX" -b "$BATCH" -ub "$UBATCH" -fit off \
        -f "$TESTFILE" "${llama_args[@]}" 2>"$log" >"$out"

    grep "KV buffer size" "$log" | tail -1 || true
    grep "size = " "$log" | grep "KV" | tail -1 || true
    grep "total time" "$log" | tail -1 || true
    grep "Final estimate" "$log" | tail -1 || true
    tail -1 "$out" || true
    echo ""
}

run_one "q8_0_KV" -- --cache-type-k q8_0 --cache-type-v q8_0

run_one "packed16_K" \
    GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 \
    GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
    GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
    GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq \
    GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT=blockfa_recthist_v4_single \
    GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1 \
    GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1 \
    -- --cache-type-k q8_0 --cache-type-v q8_0

echo "=== Done ==="
