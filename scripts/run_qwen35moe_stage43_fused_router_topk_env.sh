#!/usr/bin/env bash
# Stage4.3 Qwen35MoE verifier fused router/top-k/weights candidate environment.
#
# This is the next target after Stage4.2 proved that router dense can use the
# small-N MMVF route with state_match=1, but the remaining top-k/weight island
# stayed row-serial and still lost the speed gate.  Stage4.3 keeps the exact
# attention/state barriers and routed projection serial-column routes, then
# reopens the router logits + softmax/group-mask/top-k/weights chain as a fused
# small-N candidate.  Promotion still requires token_match=1 and state_match=1
# before repair on the target machine.
#
# Usage:
#   scripts/run_qwen35moe_stage43_fused_router_topk_env.sh ./llama-server -v ... 2>&1 | tee stage43-fused-router-topk.log
set -euo pipefail

# Select exact token-major prefix verifier and Stage4.3 fused router/top-k path.
export LLAMA_MTP_SERIAL_EQUIV_PREFIX="${LLAMA_MTP_SERIAL_EQUIV_PREFIX:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_STAGE43_FUSED_ROUTER_TOPK="${LLAMA_MTP_PREFIX_ROWEQ_STAGE43_FUSED_ROUTER_TOPK:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK="${LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG:-1}"
export LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH="${LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH:-0}"
export LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD="${LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD:-1}"

# Stage4.3 candidate island.  Router dense and top-k/weights are both opened;
# the checker requires the fused candidate route marker before promotion.
export LLAMA_MTP_PREFIX_ROWEQ_STAGE41_NO_DEFAULT_SERIAL="${LLAMA_MTP_PREFIX_ROWEQ_STAGE41_NO_DEFAULT_SERIAL:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER_DENSE="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER_DENSE:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTED_GLUE="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTED_GLUE:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_AGG="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_AGG:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_WEIGHT_AGG="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_WEIGHT_AGG:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_GATE="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_GATE:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_FFN="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_FFN:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS="${LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS:-1}"

# Router dense MMVF remains enabled: Stage4.2 proved it can replace row-serial
# router dense while preserving state bytes.
export LLAMA_MTP_ROWEQ_ROUTER_MMVF_ACTIVE="${LLAMA_MTP_ROWEQ_ROUTER_MMVF_ACTIVE:-1}"
export LLAMA_MTP_ROWEQ_ROUTER_MMVF_LOG="${LLAMA_MTP_ROWEQ_ROUTER_MMVF_LOG:-1}"
export LLAMA_MTP_ROWEQ_ROUTER_MMVF_MAX_COLS="${LLAMA_MTP_ROWEQ_ROUTER_MMVF_MAX_COLS:-4}"
export LLAMA_MTP_ROWEQ_ROUTER_MMVF_FILTER="${LLAMA_MTP_ROWEQ_ROUTER_MMVF_FILTER:-.ffn_gate_inp.weight}"

# New Stage4.3 route marker.  This selects the explicit small-N
# GGML_OP_ROUTER_TOPK_WEIGHTS CUDA kernel; target-side validation determines
# whether that fused row-local kernel is row-equivalent for Qwen35MoE's grouped
# expert masking and weight normalization.
export LLAMA_MTP_ROWEQ_ROUTER_TOPK_FUSED_ACTIVE="${LLAMA_MTP_ROWEQ_ROUTER_TOPK_FUSED_ACTIVE:-1}"
export LLAMA_MTP_ROWEQ_ROUTER_TOPK_FUSED_LOG="${LLAMA_MTP_ROWEQ_ROUTER_TOPK_FUSED_LOG:-1}"
export LLAMA_MTP_ROWEQ_ROUTER_TOPK_FUSED_MAX_COLS="${LLAMA_MTP_ROWEQ_ROUTER_TOPK_FUSED_MAX_COLS:-4}"
export LLAMA_MTP_ROWEQ_ROUTER_TOPK_FUSED_FILTER="${LLAMA_MTP_ROWEQ_ROUTER_TOPK_FUSED_FILTER:-prefix43_ffn_moe_router_logits,prefix43_router_topk_weights_fused_logits}"

# Routed expert projection batching remains the known exact Stage4.1/4.2 part.
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_MAX="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_MAX:-4}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_LOG="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_LOG:-1}"

# Exactness tracing defaults.  Disable only for speed runs after exactness has
# already passed on the same prompt/seed/shape.
export LLAMA_MTP_VERIFY_TRACE="${LLAMA_MTP_VERIFY_TRACE:-1}"
export LLAMA_MTP_VERIFY_COMPARE="${LLAMA_MTP_VERIFY_COMPARE:-1}"
export LLAMA_MTP_PREFIX_HIDDEN_TRACE="${LLAMA_MTP_PREFIX_HIDDEN_TRACE:-1}"
export LLAMA_MTP_PREFIX_HIDDEN_TRACE_LAYER="${LLAMA_MTP_PREFIX_HIDDEN_TRACE_LAYER:-0}"
export LLAMA_LOG_VERBOSITY="${LLAMA_LOG_VERBOSITY:-1}"

if [[ "$#" -eq 0 ]]; then
  cat <<'MSG'
Stage4.3 fused router/top-k/weights candidate environment exported.
Run a command after this script, for example:
  scripts/run_qwen35moe_stage43_fused_router_topk_env.sh ./build/bin/llama-server -v ...
MSG
  exit 0
fi

exec "$@"
