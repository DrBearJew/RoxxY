#!/usr/bin/env bash
set -euo pipefail

# Source this file, or execute it with a command after `--`, to run the Stage3
# exact-tail verifier candidate.  Example:
#
#   scripts/run_qwen35moe_stage3_exact_tail_env.sh -- ./build/bin/llama-server ... 2>&1 | tee stage3.log
#
# The script intentionally does not hard-code a model path or server/CLI flags.

export LLAMA_MTP_SERIAL_EQUIV_PREFIX=1
export LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH=1
export LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD=${LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD:-1}
export LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH_LOG=${LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH_LOG:-1}

# Compare against the serial oracle and fail promotion if any pre-repair state
# compare reports state_match=0.
export LLAMA_MTP_VERIFY_COMPARE=${LLAMA_MTP_VERIFY_COMPARE:-1}
export LLAMA_MTP_VERIFY_TRACE=${LLAMA_MTP_VERIFY_TRACE:-1}
export LLAMA_MTP_CYCLE_TRACE=${LLAMA_MTP_CYCLE_TRACE:-1}

# Make the final state-free tail prefer row-equivalent column handling for MMVQ
# and MUL_MAT_ID while retaining multi-column launch shapes where supported.
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_LOG=${LLAMA_MTP_MMVQ_SERIAL_COLUMNS_LOG:-1}

# Keep known-unsafe verifier modes out of the run unless the caller explicitly
# re-exports them after sourcing this file.
unset LLAMA_MTP_TARGET_BATCH_VERIFY_UNSAFE || true
unset LLAMA_MTP_TARGET_BATCH_VERIFY_REPLAY_ACCEPTED || true

if [[ "${1:-}" == "--" ]]; then
    shift
fi

if [[ "$#" -gt 0 ]]; then
    exec "$@"
fi

cat <<'MSG'
Stage3 exact-tail verifier environment exported.
Run your llama-server or llama-cli command in this shell, or pass it after `--`.
MSG
