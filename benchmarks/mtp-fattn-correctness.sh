#!/bin/bash
# MTP correctness comparison — DOT4 on/off
# Run from llama.cpp-tree-tbq4-rdna3-github/build-rocm-rdna3-fa/
#
# Usage:
#   bash benchmarks/mtp-fattn-correctness.sh <model_path>
#
# Compares:
#   1. MTP off + DOT4 off (baseline)
#   2. MTP on  + DOT4 off
#   3. MTP on  + DOT4 on
#   4. MTP on  + forced route contract

set -euo pipefail

MODEL="${1:?Usage: $0 <model_path>}"
SERVER="./bin/llama-server"
PORT=18091
PROMPT="Explain the difference between a stack and a queue in computer science."
MAX_TOKENS=64

log() {
    echo "=== [$(date +%H:%M:%S)] $* ==="
}

run_test() {
    local label="$1"
    local extra_env="$2"
    local logfile="/tmp/fa-correctness-${label}.log"
    
    log "Test: $label"
    
    eval "$extra_env" $SERVER \
        -m "$MODEL" \
        --host 127.0.0.1 --port $PORT \
        --ctx-size 4096 \
        --flash-attn \
        --n-gpu-layers 99 \
        --log-disable \
        > "$logfile" 2>&1 &
    
    SERVER_PID=$!
    
    for i in $(seq 1 30); do
        if curl -s http://127.0.0.1:$PORT/health | grep -q "ok" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    
    local output_file="/tmp/fa-correctness-${label}-output.txt"
    curl -s http://127.0.0.1:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"local\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
            \"max_tokens\": $MAX_TOKENS,
            \"temperature\": 0
        }" | jq -r '.choices[0].message.content' > "$output_file" 2>/dev/null || true
    
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 2
    
    log "Output: $output_file"
    head -c 200 "$output_file" 2>/dev/null || echo "(empty)"
    echo ""
}

log "=== MTP correctness comparison ==="
log "Model: $MODEL"
log "Prompt: $PROMPT"
log "Max tokens: $MAX_TOKENS"
log ""

# Baseline: no MTP, no DOT4
run_test "baseline" ""

# MTP on, DOT4 off
run_test "mtp_no_dot4" "LLAMA_MTP_PREFILL_CHUNK=4"

# MTP on, DOT4 on
run_test "mtp_dot4" "LLAMA_MTP_PREFILL_CHUNK=4 GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq"

log ""
log "=== Compare outputs ==="
log "diff /tmp/fa-correctness-baseline-output.txt /tmp/fa-correctness-mtp_no_dot4-output.txt"
log "diff /tmp/fa-correctness-baseline-output.txt /tmp/fa-correctness-mtp_dot4-output.txt"
log ""
log "Expected: small numeric drift, no wild divergence"
