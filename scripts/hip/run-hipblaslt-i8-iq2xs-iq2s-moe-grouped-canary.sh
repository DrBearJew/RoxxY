#!/usr/bin/env bash
set -euo pipefail

# Default-off route canary for IQ2_XS/IQ2_S grouped MoE hipBLASLt-I8.
# Safe by default: does nothing unless RUN_IQ2_HIPBLASLT_I8_MOE_GROUPED_CANARY=1.

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
if [[ "${RUN_IQ2_HIPBLASLT_I8_MOE_GROUPED_CANARY:-0}" != "1" ]]; then
    echo "SKIP: set RUN_IQ2_HIPBLASLT_I8_MOE_GROUPED_CANARY=1 and LLAMA_MODEL_IQ2XS/LLAMA_MODEL_IQ2S to run"
    exit 0
fi

BIN=${LLAMA_BENCH_BIN:-$ROOT/build/bin/llama-bench}
OUT_DIR=${OUT_DIR:-/tmp/hipblaslt-i8-iq2xs-iq2s-moe-grouped-canary-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
    echo "FAIL: llama-bench not executable: $BIN" >&2
    exit 2
fi

run_one() {
    local quant=$1
    local model=$2
    local env_prefix=$3
    local label=$4
    local log="$OUT_DIR/${quant}.log"
    if [[ ! -s "$model" ]]; then
        echo "FAIL: model missing or empty: $model" >&2
        exit 2
    fi
    export GGML_CUDA_HIPBLASLT_I8_DIAG=${GGML_CUDA_HIPBLASLT_I8_DIAG:-1}
    export "GGML_CUDA_HIPBLASLT_I8_${env_prefix}_MOE_GROUPED=1"
    export "GGML_CUDA_HIPBLASLT_I8_${env_prefix}_MOE_GROUPED_ALLOW_SMALL=${GGML_CUDA_HIPBLASLT_I8_ALLOW_SMALL:-1}"
    "$BIN" -m "$model" -p "${PP:-32}" -n "${TG:-1}" -ngl "${NGL:-99}" >"$log" 2>&1
    grep -q "hipblaslt_i8_${quant,,}_moe_grouped accept" "$log"
    grep -q "grouped ext $label route" "$log"
    echo "PASS: $label grouped MoE hipBLASLt-I8 route canary log=$log"
}

run_one iq2_xs "${LLAMA_MODEL_IQ2XS:?set LLAMA_MODEL_IQ2XS to an IQ2_XS MoE GGUF}" IQ2_XS IQ2_XS
run_one iq2_s  "${LLAMA_MODEL_IQ2S:?set LLAMA_MODEL_IQ2S to an IQ2_S MoE GGUF}" IQ2_S  IQ2_S
