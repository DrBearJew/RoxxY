#!/usr/bin/env bash
# Stage4.2 Qwen35MoE verifier containment/minimal exact wrapper.
#
# This is not a promotion-fast path.  It encodes the latest exact minimal
# bisection result:
#   - router dense matmul row-serial
#   - softmax/top-k/expert weights row-serial
#   - routed expert projections batched through serial-column MMVQ
#   - routed glue, expert aggregation, shared gate, shared FFN batched
#
# Usage:
#   scripts/run_qwen35moe_stage42_router_topk_minimal_env.sh ./llama-server -v ... 2>&1 | tee stage42-minimal.log
set -euo pipefail

export LLAMA_MTP_SERIAL_EQUIV_PREFIX="${LLAMA_MTP_SERIAL_EQUIV_PREFIX:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK="${LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_STAGE41_DIAG="${LLAMA_MTP_PREFIX_ROWEQ_STAGE41_DIAG:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG:-1}"
export LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH="${LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH:-0}"
export LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD="${LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD:-1}"

# Disable the Stage4.1 all-serial defaults, then enable only the latest exact
# minimal subset found by bisection.
export LLAMA_MTP_PREFIX_ROWEQ_STAGE41_NO_DEFAULT_SERIAL="${LLAMA_MTP_PREFIX_ROWEQ_STAGE41_NO_DEFAULT_SERIAL:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTED_GLUE="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTED_GLUE:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_AGG="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_AGG:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_WEIGHT_AGG="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_WEIGHT_AGG:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_GATE="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_GATE:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_FFN="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_FFN:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS="${LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS:-1}"

# Keep the serial-column route visible and scoped to verifier rows.
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_MAX="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_MAX:-4}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_LOG="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_LOG:-1}"

# Compare and hidden-row trace by default.  Turn these off only for pure speed
# runs after exactness has already been checked with the same prompt/seed/shape.
export LLAMA_MTP_VERIFY_TRACE="${LLAMA_MTP_VERIFY_TRACE:-1}"
export LLAMA_MTP_VERIFY_COMPARE="${LLAMA_MTP_VERIFY_COMPARE:-1}"
export LLAMA_MTP_PREFIX_HIDDEN_TRACE="${LLAMA_MTP_PREFIX_HIDDEN_TRACE:-1}"
export LLAMA_MTP_PREFIX_HIDDEN_TRACE_LAYER="${LLAMA_MTP_PREFIX_HIDDEN_TRACE_LAYER:-0}"

# High verbosity is needed for route lines emitted at GGML_LOG_INFO.
export LLAMA_LOG_VERBOSITY="${LLAMA_LOG_VERBOSITY:-1}"

if [[ "$#" -eq 0 ]]; then
  cat <<'MSG'
Stage4.2 minimal router/top-k exactness environment exported.
Run a command after this script, for example:
  scripts/run_qwen35moe_stage42_router_topk_minimal_env.sh ./build/bin/llama-server -v ...
MSG
  exit 0
fi

exec "$@"
