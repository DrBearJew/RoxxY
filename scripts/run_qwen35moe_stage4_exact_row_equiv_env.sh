#!/usr/bin/env bash
# Stage4 Qwen35MoE verifier environment.
# Usage:
#   scripts/run_qwen35moe_stage4_exact_row_equiv_env.sh ./llama-server ...
#
# This mode is opt-in. It keeps the exact serial-equivalent prefix verifier
# backend, enables the row-equivalent layer-FFN graph, and scopes the backend
# small-N serial-column kernels that make batched FFN/MoE columns byte-equivalent
# to independent ncols_dst=1 verifier rows.
set -euo pipefail

# Exact prefix backend selection.
export LLAMA_MTP_SERIAL_EQUIV_PREFIX="${LLAMA_MTP_SERIAL_EQUIV_PREFIX:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG:-1}"

# Compatibility aliases for older local Stage4 branch names.
export LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_BATCH="${LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_BATCH:-${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH}}"
export LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_BATCH_LOG="${LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_BATCH_LOG:-${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG}}"
export LLAMA_MTP_PREFIX_EXACT_ROWEQ_BATCH="${LLAMA_MTP_PREFIX_EXACT_ROWEQ_BATCH:-${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH}}"
export LLAMA_MTP_PREFIX_EXACT_ROWEQ_BATCH_LOG="${LLAMA_MTP_PREFIX_EXACT_ROWEQ_BATCH_LOG:-${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG}}"

# Stage3 exact-tail must not steal priority when Stage4 is being validated.
export LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH="${LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH:-0}"

# Keep final output work batched unless a shadow/debug run needs row-by-row logits.
export LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD="${LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD:-1}"

# Commit shortcuts are still safe because verifier rows materialize exact rollback slots.
export LLAMA_MTP_PREFIX_ACCEPTED_ROW_ONLY_COMMIT="${LLAMA_MTP_PREFIX_ACCEPTED_ROW_ONLY_COMMIT:-1}"
export LLAMA_MTP_PREFIX_ACCEPTED_ROW_COMMIT_VERIFY_SLOTS="${LLAMA_MTP_PREFIX_ACCEPTED_ROW_COMMIT_VERIFY_SLOTS:-4}"

# Direct top1 was not the blocker; leave it off by default so output-head behavior
# is easy to compare with the proven full-logits path.
export LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE="${LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE:-0}"

# Standalone/non-server runs need the backend policy installed explicitly. The
# server also scopes these during target decode, so these defaults are harmless.
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_MAX="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_MAX:-4}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_LOG="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_LOG:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE_FILTER="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE_FILTER:-ffn_gate_inp,ffn_gate_up,ffn_gate,ffn_up,ffn_down,ffn_up_shexp,ffn_gate_shexp,ffn_down_shexp,ffn_gate_inp_shexp}"

# Match the documented Stage4 RDNA3 validation environment. These remain scoped
# to this opt-in wrapper.
export GGML_CUDA_ROCM_RDNA3_MMVQ_DOT4="${GGML_CUDA_ROCM_RDNA3_MMVQ_DOT4:-1}"
export GGML_CUDA_ROCM_RDNA3_MMVQ_DOT4_IDS="${GGML_CUDA_ROCM_RDNA3_MMVQ_DOT4_IDS:-1}"
export GGML_CUDA_ROCM_RDNA3_MMVQ_DOT4_SIDEBAND="${GGML_CUDA_ROCM_RDNA3_MMVQ_DOT4_SIDEBAND:-1}"
export GGML_CUDA_ROCM_RDNA3_MMVQ_DOT4_LOG="${GGML_CUDA_ROCM_RDNA3_MMVQ_DOT4_LOG:-1}"

# Strong validation defaults. Disable only for performance-only A/B sweeps.
export LLAMA_MTP_VERIFY_TRACE="${LLAMA_MTP_VERIFY_TRACE:-1}"
export LLAMA_MTP_VERIFY_COMPARE="${LLAMA_MTP_VERIFY_COMPARE:-1}"
export LLAMA_MTP_GDN_INPUT_TRACE_COMPARE="${LLAMA_MTP_GDN_INPUT_TRACE_COMPARE:-1}"

# Bisect valves. Default validates all transformer layers. Set FIRST/LAST to
# narrow a state mismatch without changing the rest of the verifier contract.
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FIRST="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FIRST:-0}"
export LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_FIRST_LAYER="${LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_FIRST_LAYER:-${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FIRST}}"
# Leave LAST unset by default; the graph code clamps to the last transformer layer.
if [[ -n "${LLAMA_MTP_PREFIX_ROWEQ_LAYER_LAST:-}" ]]; then
    export LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_LAST_LAYER="${LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_LAST_LAYER:-${LLAMA_MTP_PREFIX_ROWEQ_LAYER_LAST}}"
fi

if [[ "${1:-}" == "--" ]]; then
    shift
fi

exec "$@"
