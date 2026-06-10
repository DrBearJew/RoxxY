#!/usr/bin/env bash
# Stage4.1 Qwen35MoE verifier diagnostic environment.
# Usage:
#   scripts/run_qwen35moe_stage41_component_bisect_env.sh ./llama-server ...
#
# This is a diagnostic/containment mode, not a promotion default.  It keeps the
# Stage4 state-barrier schedule, but serializes the MoE glue that Stage4 left
# batched while retaining batched serial-column routed expert projections.
set -euo pipefail

# Exact prefix backend selection.
export LLAMA_MTP_SERIAL_EQUIV_PREFIX="${LLAMA_MTP_SERIAL_EQUIV_PREFIX:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_STAGE41_DIAG="${LLAMA_MTP_PREFIX_ROWEQ_STAGE41_DIAG:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG:-1}"

# Compatibility aliases for local Stage4/Stage4.1 branches.
export LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_BATCH="${LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_BATCH:-${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH}}"
export LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_BATCH_LOG="${LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_BATCH_LOG:-${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG}}"
export LLAMA_MTP_PREFIX_EXACT_ROWEQ_BATCH="${LLAMA_MTP_PREFIX_EXACT_ROWEQ_BATCH:-${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH}}"
export LLAMA_MTP_PREFIX_EXACT_ROWEQ_BATCH_LOG="${LLAMA_MTP_PREFIX_EXACT_ROWEQ_BATCH_LOG:-${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG}}"

# Stage3 exact-tail must not steal priority while Stage4.1 is being diagnosed.
export LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH="${LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH:-0}"

# Candidate A: batch only expensive routed projections.  All surrounding MoE
# components that can perturb the hidden row before the next recurrent write are
# row-serial by default; flip individual switches to bisect.
export LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS="${LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTED_GLUE="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTED_GLUE:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_AGG="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_AGG:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_GATE="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_GATE:-1}"
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_FFN="${LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_FFN:-1}"

# Keep final output work batched unless a shadow/debug run needs row-by-row logits.
export LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD="${LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD:-1}"
export LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE="${LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE:-0}"

# Commit shortcuts are still safe only if pre-repair verifier state is exact.
# Leave them enabled for the candidate run, then disable for diagnostic A/B if needed.
export LLAMA_MTP_PREFIX_ACCEPTED_ROW_ONLY_COMMIT="${LLAMA_MTP_PREFIX_ACCEPTED_ROW_ONLY_COMMIT:-1}"
export LLAMA_MTP_PREFIX_ACCEPTED_ROW_COMMIT_VERIFY_SLOTS="${LLAMA_MTP_PREFIX_ACCEPTED_ROW_COMMIT_VERIFY_SLOTS:-4}"

# Standalone/non-server runs need backend policy installed explicitly.  The server
# also scopes these during target decode.
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_MAX="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_MAX:-4}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_LOG="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_LOG:-1}"
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE_FILTER="${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE_FILTER:-ffn_gate_up,ffn_down,ffn_gate,ffn_up,ffn_gate_inp,ffn_up_shexp,ffn_gate_shexp,ffn_down_shexp,ffn_gate_inp_shexp}"

# Strong validation defaults.  Route logs require llama-cli/server high verbosity
# (for example -v), otherwise GGML_LOG_INFO route lines may be hidden.
export LLAMA_MTP_VERIFY_TRACE="${LLAMA_MTP_VERIFY_TRACE:-1}"
export LLAMA_MTP_VERIFY_COMPARE="${LLAMA_MTP_VERIFY_COMPARE:-1}"
export LLAMA_MTP_GDN_INPUT_TRACE_COMPARE="${LLAMA_MTP_GDN_INPUT_TRACE_COMPARE:-1}"

# Hidden-row hashes before the next layer's state write.  Default to layer 0
# because the real failure reproduces with only layer 0 roweq-enabled.
export LLAMA_MTP_PREFIX_HIDDEN_TRACE="${LLAMA_MTP_PREFIX_HIDDEN_TRACE:-1}"
export LLAMA_MTP_PREFIX_HIDDEN_TRACE_LAYER="${LLAMA_MTP_PREFIX_HIDDEN_TRACE_LAYER:-0}"
export LLAMA_MTP_PREFIX_SNAPSHOT_TRACE_MAX_PRINT="${LLAMA_MTP_PREFIX_SNAPSHOT_TRACE_MAX_PRINT:-0}"

# Layer bisection valves.  Default to layer 0 only because the reported failure
# already reproduces there; set LAST unset or to full depth for full sweeps.
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FIRST="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FIRST:-0}"
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_LAST="${LLAMA_MTP_PREFIX_ROWEQ_LAYER_LAST:-0}"
export LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_FIRST_LAYER="${LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_FIRST_LAYER:-${LLAMA_MTP_PREFIX_ROWEQ_LAYER_FIRST}}"
export LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_LAST_LAYER="${LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_LAST_LAYER:-${LLAMA_MTP_PREFIX_ROWEQ_LAYER_LAST}}"

exec "$@"
