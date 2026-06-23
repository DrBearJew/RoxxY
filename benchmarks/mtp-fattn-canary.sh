#!/bin/bash
# MTP FA route hint canary harness
# Run from llama.cpp-tree-tbq4-rdna3-github/build-rocm-rdna3-fa/
#
# Usage:
#   bash benchmarks/mtp-fattn-canary.sh <model_path> [chunk_size]
#
# Environment (auto-set if not provided):
#   COMPRESSED_KV_FATTN_LOG=1   - enable FA route logging
#   MTP_DOT4_ENV=1              - enable DOT4 env gates (for canary 3)

set -euo pipefail

MODEL="${1:?Usage: $0 <model_path> [chunk_size]}"
CHUNK="${2:-4}"
SERVER="./bin/llama-server"
PORT=18089

export COMPRESSED_KV_FATTN_LOG=1

log() {
    echo "=== [$(date +%H:%M:%S)] $* ==="
}

# Start server with MTP enabled
start_server() {
    local extra_env="$1"
    local label="$2"
    log "Starting server: $label"
    
    eval "$extra_env" $SERVER \
        -m "$MODEL" \
        --host 127.0.0.1 --port $PORT \
        --ctx-size 4096 \
        --flash-attn \
        --mtp 1 \
        --n-gpu-layers 99 \
        --log-disable \
        2>&1 | head -200 &
    
    SERVER_PID=$!
    
    # Wait for server to be ready
    for i in $(seq 1 30); do
        if curl -s http://127.0.0.1:$PORT/health | grep -q "ok"; then
            log "Server ready"
            return 0
        fi
        sleep 1
    done
    log "Server failed to start"
    kill $SERVER_PID 2>/dev/null || true
    return 1
}

stop_server() {
    log "Stopping server"
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 2
}

# Send a canary prompt and capture logs
run_canary() {
    local label="$1"
    local prompt="$2"
    
    log "Canary: $label"
    
    # Capture FA logs to a temp file
    local logfile=$(mktemp /tmp/fa-canary-XXXXXX.log)
    
    # Send request
    curl -s http://127.0.0.1:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"local\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
            \"max_tokens\": 16,
            \"temperature\": 0
        }" > /dev/null 2>&1
    
    log "Logs: $logfile"
    echo "$label" >> "$logfile"
}

# ─── Canary 1: MTP_VERIFY, nq<=2 (single token decode) ──────────────
# Expected: mtp_nq_le_2, no DOT4, VEC/WMMA fallback
# This requires MTP chunk=1 or 2
log "=== Canary 1: MTP_VERIFY nq<=2 ==="
log "Expected: fattn_hint=mtp_verify dot4_candidate=0 dot4_reject_reason=mtp_nq_le_2"
log "Run with: LLAMA_MTP_PREFILL_CHUNK=1 or 2"
log ""

# ─── Canary 2: MTP_VERIFY, nq>2, DOT4 env off ──────────────────────
log "=== Canary 2: MTP_VERIFY nq>2 DOT4 env off ==="
log "Expected: fattn_hint=mtp_verify dot4_candidate=1 dot4_reject_reason=dot4_env_disabled"
log "Run with: LLAMA_MTP_PREFILL_CHUNK=4 (no DOT4 env flags)"
log ""

# ─── Canary 3: MTP_VERIFY, nq>2, DOT4 env on ───────────────────────
log "=== Canary 3: MTP_VERIFY nq>2 DOT4 env on ==="
log "Expected: fattn_hint=mtp_verify selected=q8q4_dot4_prefill"
log "Run with:"
log "  LLAMA_MTP_PREFILL_CHUNK=4"
log "  GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1"
log "  GGML_CUDA_ROCM_Q8K_DOT4_KQ=1"
log "  GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq"
log ""

# ─── Canary 4: MTP_DRAFT, any nq ────────────────────────────────────
log "=== Canary 4: MTP_DRAFT any nq ==="
log "Expected: fattn_hint=mtp_draft dot4_candidate=0 dot4_reject_reason=mtp_draft_dot4_not_enabled"
log "Note: MTP_DRAFT logs appear during autoregressive continuation"
log ""

# ─── Canary 5: Hostile test — packed16 + MTP ────────────────────────
log "=== Canary 5: Hostile packed16 + MTP ==="
log "Expected: MTP never gets I32 K"
log "If FA somehow sees I32 K: reject=mtp_packed16_k"
log "Run with:"
log "  GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1"
log "  COMPRESSED_KV_FATTN_LOG=1"
log ""

# ─── Grep commands ──────────────────────────────────────────────────
log "=== Log analysis commands ==="
echo "grep 'fa_route fattn_hint=mtp_verify' server.log"
echo "grep 'fa_route fattn_hint=mtp_draft' server.log"
echo "grep 'fattn_hint=mtp' server.log | grep 'dot4_candidate=0'"
echo "grep 'fattn_hint=mtp' server.log | grep 'dot4_candidate=1'"
echo "grep 'reject=mtp_packed16_k' server.log"
echo "grep 'ggml_cuda_fattn_log_selection.*fattn_hint=mtp_verify' server.log"
echo "grep 'ggml_cuda_fattn_log_selection.*fattn_hint=mtp_draft' server.log"
