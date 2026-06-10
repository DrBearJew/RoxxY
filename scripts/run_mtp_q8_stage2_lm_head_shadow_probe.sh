#!/usr/bin/env bash
set -euo pipefail

if [ -z "${CLI_BIN:-}" ]; then
  if [ -x ./build-rocm-fixed/bin/llama-cli ]; then
    CLI_BIN=./build-rocm-fixed/bin/llama-cli
  else
    CLI_BIN=./build/bin/llama-cli
  fi
fi
MODEL=${MODEL:?set MODEL=/path/to/qwen-q8-output.gguf}
PROMPT=${PROMPT:-hello}
TOKENS=${TOKENS:-16}

# Shadow mode must materialize full logits and a separate direct LM_HEAD_TOP_K tensor.
# Do not enable ACTIVE by default here: active direct mode returns before full logits,
# so the shadow checker has nothing to compare against.
export LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE=${LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE:-0}
export LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE_RAW_UNSAFE=${LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE_RAW_UNSAFE:-0}
export LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW=${LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW:-1}
export LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW_REQUIRE=${LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW_REQUIRE:-1}
export LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW_LOG=${LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW_LOG:-1}
export LLAMA_MTP_FUSED_LM_HEAD_TOPK_LOG=${LLAMA_MTP_FUSED_LM_HEAD_TOPK_LOG:-1}

exec "$CLI_BIN" -m "$MODEL" -p "$PROMPT" -n "$TOKENS" "$@"
