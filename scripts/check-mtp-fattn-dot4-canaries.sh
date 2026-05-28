#!/usr/bin/env bash
# check-mtp-fattn-dot4-canaries.sh — MTP FA DOT4 canary validation
#
# Runs server with various env combinations and validates log output.
# Requires: llama-server binary, model, curl.
#
# Usage:
#   ./scripts/check-mtp-fattn-dot4-canaries.sh [SERVER_PORT]
#
# Exit 0 if all canaries pass, exit 1 if any fail.

set -euo pipefail

PORT="${1:-18200}"
MODEL="${LLAMA_MODEL:-/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf}"
SERVER_BIN="${LLAMA_SERVER:-./build-rocm-rdna3-fa/bin/llama-server}"
LOG_DIR="/tmp/mtp-canary-logs"
PASS=0
FAIL=0

mkdir -p "$LOG_DIR"

run_server() {
    local name="$1"
    shift
    local log_file="$LOG_DIR/${name}.log"
    
    pkill -9 -f "llama-server.*port.*${PORT}" 2>/dev/null || true
    sleep 1
    
    "$@" > "$log_file" 2>&1 &
    local pid=$!
    
    for i in $(seq 1 30); do
        if curl -s "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; then
            return 0
        fi
        sleep 1
    done
    
    echo "FAIL: Server ${name} failed to start"
    kill -9 $pid 2>/dev/null || true
    return 1
}

send_request() {
    curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"local","messages":[{"role":"user","content":"Hi"}],"max_tokens":16,"temperature":0}' \
        > /dev/null 2>&1
}

stop_server() {
    pkill -9 -f "llama-server.*port.*${PORT}" 2>/dev/null || true
    sleep 1
}

check_log() {
    local name="$1"
    local pattern="$2"
    local forbidden="${3:-}"
    local log_file="$LOG_DIR/${name}.log"
    
    if grep -q "$pattern" "$log_file"; then
        if [ -n "$forbidden" ] && grep -q "$forbidden" "$log_file"; then
            echo "FAIL: ${name} — found forbidden pattern: ${forbidden}"
            ((FAIL++))
            return 1
        fi
        echo "PASS: ${name}"
        ((PASS++))
        return 0
    else
        echo "FAIL: ${name} — missing pattern: ${pattern}"
        ((FAIL++))
        return 1
    fi
}

# ── Canary A: MTP_VERIFY, nq=4, source f16 ─────────────────────────
run_server "canary_a" \
    env COMPRESSED_KV_FATTN_LOG=1 \
    GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
    GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
    GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER=1 \
    "$SERVER_BIN" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    -fa on --batch-size 2048 --ubatch-size 1024 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 \
    --parallel 1 --ctx-size 32000 --no-warmup --n-gpu-layers 99

send_request
check_log "canary_a" "mtp_verify_qk.*nq=4.*k_repr=source_f16_op_local_packed16.*impl_status=dot4_selected.*selected=rocm_q8k_dot4_kq"
stop_server

# ── Canary B: MTP_VERIFY, nq=7, source f16 ─────────────────────────
run_server "canary_b" \
    env COMPRESSED_KV_FATTN_LOG=1 \
    GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
    GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
    GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER=1 \
    "$SERVER_BIN" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    -fa on --batch-size 2048 --ubatch-size 1024 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 \
    --parallel 1 --ctx-size 32000 --no-warmup --n-gpu-layers 99

send_request
check_log "canary_b" "mtp_verify_qk.*nq=7.*k_repr=source_f16_op_local_packed16.*impl_status=dot4_selected.*selected=rocm_q8k_dot4_kq"
stop_server

# ── Canary C: MTP_VERIFY, nq=1 ─────────────────────────────────────
run_server "canary_c" \
    env COMPRESSED_KV_FATTN_LOG=1 \
    GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
    GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
    GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER=1 \
    "$SERVER_BIN" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    -fa on --batch-size 2048 --ubatch-size 1024 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 \
    --parallel 1 --ctx-size 32000 --no-warmup --n-gpu-layers 99

send_request
check_log "canary_c" "mtp_verify_qk.*nq=1.*impl_status=nq_eq_1_decode_not_recthist"
stop_server

# ── Canary D: MTP_VERIFY, nq=2 env off ─────────────────────────────
run_server "canary_d" \
    env COMPRESSED_KV_FATTN_LOG=1 \
    GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
    GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
    GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER=1 \
    "$SERVER_BIN" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    -fa on --batch-size 2048 --ubatch-size 1024 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 \
    --parallel 1 --ctx-size 32000 --no-warmup --n-gpu-layers 99

send_request
check_log "canary_d" "mtp_verify_qk.*nq=2.*impl_status=nq_eq_2_disabled"
stop_server

# ── Canary E: MTP_VERIFY, nq=2 env on ──────────────────────────────
run_server "canary_e" \
    env COMPRESSED_KV_FATTN_LOG=1 \
    GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
    GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
    GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER=1 \
    GGML_CUDA_ROCM_MTP_VERIFY_DOT4_NQ2=1 \
    "$SERVER_BIN" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    -fa on --batch-size 2048 --ubatch-size 1024 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 \
    --parallel 1 --ctx-size 32000 --no-warmup --n-gpu-layers 99

send_request
check_log "canary_e" "mtp_verify_qk.*nq=2.*dot4_role=recthist_v4_mtp_verify.*impl_status=dot4_selected.*selected=rocm_q8k_dot4_kq"
stop_server

# ── Canary F: MTP_DRAFT, any nq ────────────────────────────────────
run_server "canary_f" \
    env COMPRESSED_KV_FATTN_LOG=1 \
    GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
    GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
    GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER=1 \
    "$SERVER_BIN" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    -fa on --batch-size 2048 --ubatch-size 1024 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 \
    --parallel 1 --ctx-size 32000 --no-warmup --n-gpu-layers 99

send_request
check_log "canary_f" "mtp_draft.*impl_status=existing_policy" "mtp_draft.*dot4_selected"
stop_server

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "=== Canary Results ==="
echo "PASS: ${PASS}"
echo "FAIL: ${FAIL}"

if [ "$FAIL" -gt 0 ]; then
    echo "SOME CANARIES FAILED"
    exit 1
fi

echo "ALL CANARIES PASSED"
exit 0
