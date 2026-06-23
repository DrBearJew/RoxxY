#!/usr/bin/env bash
set -euo pipefail

if [ -z "${TEST_BIN:-}" ]; then
  if [ -x ./build-rocm-fixed/bin/test-backend-ops ]; then
    TEST_BIN=./build-rocm-fixed/bin/test-backend-ops
  else
    TEST_BIN=./build/bin/test-backend-ops
  fi
fi
# test-backend-ops prints ggml type names in lowercase, e.g. q8_0.
PARAM_FILTER=${PARAM_FILTER:-q8_0.*(1024|2048|512)}

export LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4=${LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4:-1}
export LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4_LOG=${LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4_LOG:-1}
export LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4_MAX_ROUTES=${LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4_MAX_ROUTES:-64}

exec "$TEST_BIN" test -o MUL_MAT_ID,MUL_MAT -p "$PARAM_FILTER" "$@"
