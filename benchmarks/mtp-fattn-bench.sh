#!/bin/bash
# MTP DOT4 benchmark — chunk size sweep
# Run from llama.cpp-tree-tbq4-rdna3-github/build-rocm-rdna3-fa/
#
# Usage:
#   bash benchmarks/mtp-fattn-bench.sh <model_path>
#
# Tests DOT4 eligibility across chunk sizes 1-8

set -euo pipefail

MODEL="${1:?Usage: $0 <model_path>}"
SERVER="./bin/llama-server"
PORT=18090

export COMPRESSED_KV_FATTN_LOG=1
export GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1
export GGML_CUDA_ROCM_Q8K_DOT4_KQ=1
export GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq

log() {
    echo "=== [$(date +%H:%M:%S)] $* ==="
}

run_bench() {
    local chunk=$1
    local logfile="/tmp/fa-bench-chunk${chunk}.log"
    
    log "Chunk=$chunk -> $logfile"
    
    LLAMA_MTP_PREFILL_CHUNK=$chunk $SERVER \
        -m "$MODEL" \
        --host 127.0.0.1 --port $PORT \
        --ctx-size 4096 \
        --flash-attn \
        --mtp 1 \
        --n-gpu-layers 99 \
        --log-disable \
        > "$logfile" 2>&1 &
    
    SERVER_PID=$!
    
    # Wait for server
    for i in $(seq 1 30); do
        if curl -s http://127.0.0.1:$PORT/health | grep -q "ok" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    
    # Send test prompt
    curl -s http://127.0.0.1:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "local",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 32,
            "temperature": 0
        }' > /dev/null 2>&1 || true
    
    sleep 1
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 2
    
    log "Chunk=$chunk done"
}

log "=== MTP DOT4 benchmark sweep ==="
log "Model: $MODEL"
log "DOT4 env: GGML_CUDA_ROCM_Q8K_DOT4_KQ=1"
log ""

for chunk in 1 2 3 4 8; do
    run_bench $chunk
done

log ""
log "=== Results ==="
log "Analyze with:"
for chunk in 1 2 3 4 8; do
    echo ""
    echo "--- Chunk $chunk ---"
    echo "grep 'fa_route.*fattn_hint=' /tmp/fa-bench-chunk${chunk}.log"
    echo "grep 'fattn_hint=mtp' /tmp/fa-bench-chunk${chunk}.log | grep -o 'dot4_candidate=[0-9] dot4_reject_reason=[^ ]*'"
    echo "grep 'ggml_cuda_fattn_log_selection.*fattn_hint=' /tmp/fa-bench-chunk${chunk}.log | grep -o 'route=[^ ]*'"
done
