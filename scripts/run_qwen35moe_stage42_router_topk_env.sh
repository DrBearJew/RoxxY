#!/usr/bin/env bash
# Stage4.2 Qwen35MoE verifier router/top-k small-N candidate environment.
#
# This is the next exact-fast experiment after Stage4.1 bisection showed that
# router/top-k/weights, not routed projection route selection, caused pre-repair
# recurrent-state drift.  The candidate keeps top-k/weight processing row-serial
# but lets the dense router matmul run as a small-N row-equivalent backend route.
#
# Usage:
#   scripts/run_qwen35moe_stage42_router_topk_env.sh ./llama-server -v ... 2>&1 | tee stage42-router-topk.log
set -euo pipefail

# Select exact token-major prefix verifier and Stage4.2 router/top-k route.
export LLAMA_MTP_SERIAL_EQUIV_PREFIX="${LLAMA_MTP_SERIAL_EQUIV_PREFIX:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK="${LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG:-1}"
export LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH="${LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH:-0}"
export LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD="${LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD:-1}"

# Stage4.2 candidate: split router dense from top-k/weights.  Router dense uses
# the backend router_mmvf_serial_columns route; top-k/weights stay row-serial to
# preserve the currently known exact order.  Other components use the minimal
# exact Stage4.1 valves so the run isolates router dense replacement.
export LLAMA_MTP_PREFIX_ROWEQ_STAGE41_NO_DEFAULT_SERIAL="${LLAMA_MTP_PREFIX_ROWEQ_STAGE41_NO_DEFAULT_SERIAL:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER_DENSE="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER_DENSE:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTED_GLUE="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTED_GLUE:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_AGG="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_AGG:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_WEIGHT_AGG="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_WEIGHT_AGG:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_GATE="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_GATE:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_FFN="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_FFN:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS="${LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS:-1}"

# Router dense backend route.  The default filter is intentionally narrow:
# blk.N.ffn_gate_inp.weight only.  Shared gate and projection routes are left to
# their Stage4.1 behavior unless explicitly requested by the caller.
export LLAMA_MTP_ROWEQ_ROUTER_MMVF_ACTIVE="${LLAMA_MTP_ROWEQ_ROUTER_MMVF_ACTIVE:-1}"
export LLAMA_MTP_ROWEQ_ROUTER_MMVF_LOG="${LLAMA_MTP_ROWEQ_ROUTER_MMVF_LOG:-1}"
export LLAMA_MTP_ROWEQ_ROUTER_MMVF_MAX_COLS="${LLAMA_MTP_ROWEQ_ROUTER_MMVF_MAX_COLS:-4}"
export LLAMA_MTP_ROWEQ_ROUTER_MMVF_FILTER="${LLAMA_MTP_ROWEQ_ROUTER_MMVF_FILTER:-.ffn_gate_inp.weight}"

# Routed expert projection batching is no longer a safe default.  It passed the
# original narrow n_tokens=3 gates but failed later width/state sequences; keep
# the repaired serial-row MoE fallback by default.  Experts can opt back in for
# diagnostics with LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS_MAX_ROWS=3.
export LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS_MAX_ROWS="${LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS_MAX_ROWS:-0}"

# Serial-column projection routes remain useful for diagnostics and for explicit
# batched routed projection experiments.
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_MAX="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_MAX:-4}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_LOG="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_LOG:-1}"

# Exactness tracing defaults.  Disable only for speed runs after the same prompt
# and shape have already passed pre-repair compare checks.
export LLAMA_MTP_VERIFY_TRACE="${LLAMA_MTP_VERIFY_TRACE:-1}"
export LLAMA_MTP_VERIFY_COMPARE="${LLAMA_MTP_VERIFY_COMPARE:-1}"
export LLAMA_MTP_PREFIX_HIDDEN_TRACE="${LLAMA_MTP_PREFIX_HIDDEN_TRACE:-1}"
export LLAMA_MTP_PREFIX_HIDDEN_TRACE_LAYER="${LLAMA_MTP_PREFIX_HIDDEN_TRACE_LAYER:-0}"
export LLAMA_LOG_VERBOSITY="${LLAMA_LOG_VERBOSITY:-1}"

if [[ "$#" -eq 0 ]]; then
  cat <<'MSG'
Stage4.2 router/top-k candidate environment exported.
Run a command after this script, for example:
  scripts/run_qwen35moe_stage42_router_topk_env.sh ./build/bin/llama-server -v ...
MSG
  exit 0
fi

exec "$@"
