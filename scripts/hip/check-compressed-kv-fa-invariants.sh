#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
wrapper=${LLAMA_SERVER_WRAPPER:-/home/mrtrent/.local/bin/llama-server-wrapper}
swap_config=${LLAMA_SWAP_CONFIG:-/home/mrtrent/.pi/agent/llama-swap-config.yaml}

flags='TBQ4_WMMA_FATTN|COMPRESSED_KV_WMMA_FATTN'

check_absent() {
    local path=$1
    local label=$2
    if [[ ! -f $path ]]; then
        echo "FAIL: missing $label: $path" >&2
        return 1
    fi
    if grep -nE "$flags" "$path" >&2; then
        echo "FAIL: experimental compressed-KV FA flag is present in $label: $path" >&2
        return 1
    fi
    echo "PASS: no experimental compressed-KV FA flags in $label"
}

check_present() {
    local pattern=$1
    local path=$2
    local label=$3
    if ! grep -qE "$pattern" "$path"; then
        echo "FAIL: missing $label in $path" >&2
        return 1
    fi
    echo "PASS: found $label"
}

check_absent "$wrapper" "llama-server wrapper"
check_absent "$swap_config" "llama-swap config"

check_present 'getenv\("TBQ4_WMMA_FATTN"\)' "$repo/ggml/src/ggml-cuda/fattn.cu" 'TBQ4 WMMA env gate'
check_present 'getenv\("COMPRESSED_KV_WMMA_FATTN"\)' "$repo/ggml/src/ggml-cuda/fattn.cu" 'Planar/Iso compressed-KV WMMA env gate'
check_present 'BEST_FATTN_KERNEL_WMMA_COMPRESSED_KV' "$repo/ggml/src/ggml-cuda/fattn.cu" 'compressed-KV WMMA dispatch enum/use'
check_present 'BEST_FATTN_KERNEL_VEC' "$repo/ggml/src/ggml-cuda/fattn.cu" 'VEC fallback/default availability'
check_present 'ggml_cuda_flash_attn_ext_vec\(' "$repo/ggml/src/ggml-cuda/fattn.cu" 'VEC launcher availability'
check_present 'ggml_cuda_flash_attn_ext_wmma_compressed_kv' "$repo/ggml/src/ggml-cuda/fattn.cu" 'compressed-KV WMMA launcher availability'

for cmake_file in "$repo/CMakeLists.txt" "$repo/ggml/src/CMakeLists.txt" "$repo/ggml/src/ggml-hip/CMakeLists.txt"; do
    if grep -nE 'triton|Triton' "$cmake_file" >&2; then
        echo "FAIL: Triton integration found in production CMake: $cmake_file" >&2
        exit 1
    fi
done
echo "PASS: no Triton integration in production CMake"

echo "PASS: compressed-KV FA production invariants are frozen"
