#!/bin/bash
# Regression canary: PREFILL_QK nq=2 must select DOT4, not TILE.
# Invariant: nq2 gate is MTP_VERIFY_QK only; PREFILL_QK bypasses.
# Broken if fa_final_select shows selected=tile for nq=2 prefill.

SERVER="./build-rocm-rdna3-fa/bin/llama-server"
MODEL="/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf"
PORT=18500
LOG="/tmp/canary-prefill-nq2.log"

echo "=== PREFILL_QK nq=2 DOT4 regression canary ==="

GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER=1 \
GGML_CUDA_ROCM_MTP_DRAFT_DOT4_DECODE=1 \
GGML_CUDA_FA_HUNT_VEC_TILE=1 \
COMPRESSED_KV_FATTN_LOG=1 \
timeout 15 "$SERVER" -m "$MODEL" -ngl 99 -c 32000 -fa 1 -n 1 --port $PORT > "$LOG" 2>&1

echo ""
echo "Positive: nq=2 should select DOT4"
DOT4_COUNT=$(grep -c 'fa_final_select.*selected=rocm_q8k_dot4_kq.*nq=2' "$LOG")
echo "  dot4=$DOT4_COUNT (must be > 0)"

echo ""
echo "Negative: nq=2 must NOT fall to TILE"
TILE_COUNT=$(grep -c 'fa_final_select.*tile.*nq=2' "$LOG")
echo "  tile=$TILE_COUNT (must be 0)"

echo ""
echo "VEC/TILE debt:"
grep -c 'fa_vec_tile_debt.*nq=2' "$LOG"
echo "  (must be 0)"

echo ""
echo "---"
if [ "$TILE_COUNT" -gt 0 ] || [ "$DOT4_COUNT" -eq 0 ]; then
    echo "FAIL: nq=2 prefill fell to TILE or DOT4 not selected (dot4=$DOT4_COUNT tile=$TILE_COUNT)"
    exit 1
else
    echo "PASS: nq=2 prefill uses DOT4"
    exit 0
fi
