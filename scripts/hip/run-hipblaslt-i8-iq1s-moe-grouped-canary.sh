#!/usr/bin/env bash
set -euo pipefail

# Default-off route canary for IQ1_S grouped MoE hipBLASLt-I8.
# Safe by default: does nothing unless RUN_IQ1S_HIPBLASLT_I8_MOE_GROUPED_CANARY=1.

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
if [[ "${RUN_IQ1S_HIPBLASLT_I8_MOE_GROUPED_CANARY:-0}" != "1" ]]; then
    echo "SKIP: set RUN_IQ1S_HIPBLASLT_I8_MOE_GROUPED_CANARY=1 and LLAMA_MODEL_IQ1S=<moe-iq1_s.gguf> to run"
    exit 0
fi

BIN=${LLAMA_BENCH_BIN:-$ROOT/build/bin/llama-bench}
MODEL=${LLAMA_MODEL_IQ1S:?set LLAMA_MODEL_IQ1S to an IQ1_S MoE GGUF}
OUT_DIR=${OUT_DIR:-/tmp/hipblaslt-i8-iq1s-moe-grouped-canary-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/llama-bench.log"

if [[ ! -x "$BIN" ]]; then
    echo "FAIL: llama-bench not executable: $BIN" >&2
    exit 2
fi
if [[ ! -s "$MODEL" ]]; then
    echo "FAIL: model missing or empty: $MODEL" >&2
    exit 2
fi

export GGML_CUDA_HIPBLASLT_I8_IQ1_S_MOE_GROUPED=1
export GGML_CUDA_HIPBLASLT_I8_IQ1_S_MOE_GROUPED_ALLOW_SMALL=${GGML_CUDA_HIPBLASLT_I8_IQ1_S_MOE_GROUPED_ALLOW_SMALL:-1}
export GGML_CUDA_HIPBLASLT_I8_DIAG=${GGML_CUDA_HIPBLASLT_I8_DIAG:-1}
export GGML_CUDA_HIPBLASLT_I8_IQ1_S_MOE_GROUPED_TENSOR=${GGML_CUDA_HIPBLASLT_I8_IQ1_S_MOE_GROUPED_TENSOR:-}

"$BIN" -m "$MODEL" -p "${PP:-32}" -n "${TG:-1}" -ngl "${NGL:-99}" >"$LOG" 2>&1

grep -q "hipblaslt_i8_iq1_s_moe_grouped accept" "$LOG"
grep -q "grouped ext IQ1_S route" "$LOG"

echo "PASS: IQ1_S grouped MoE hipBLASLt-I8 route canary log=$LOG"
