#!/usr/bin/env bash
# Stage4.2 full row-serial FFN fallback/control.
#
# This exercises LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS=0 after the
# Stage4.2 repair.  It is a control path only: it should not be benchmarked as a
# speed candidate and it should not require serial-column route logs.
set -euo pipefail

export LLAMA_MTP_SERIAL_EQUIV_PREFIX="${LLAMA_MTP_SERIAL_EQUIV_PREFIX:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK="${LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_STAGE41_DIAG="${LLAMA_MTP_PREFIX_ROWEQ_STAGE41_DIAG:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG:-1}"
export LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH="${LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH:-0}"
export LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD="${LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD:-1}"

export LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS=0
export LLAMA_MTP_VERIFY_TRACE="${LLAMA_MTP_VERIFY_TRACE:-1}"
export LLAMA_MTP_VERIFY_COMPARE="${LLAMA_MTP_VERIFY_COMPARE:-1}"
export LLAMA_LOG_VERBOSITY="${LLAMA_LOG_VERBOSITY:-1}"

if [[ "$#" -eq 0 ]]; then
  cat <<'MSG'
Stage4.2 full row-serial FFN fallback environment exported.
Run a command after this script, for example:
  scripts/run_qwen35moe_stage42_full_serial_fallback_env.sh ./build/bin/llama-server -v ...
MSG
  exit 0
fi

exec "$@"
