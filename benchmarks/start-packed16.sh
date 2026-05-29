#!/bin/bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github
pkill -9 -f llama-server 2>/dev/null
sleep 2
export GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1
export GGML_CUDA_ROCM_Q8K_DOT4_KQ=1
export GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER=1
export GGML_CUDA_ROCM_MTP_DRAFT_DOT4_DECODE=1
export GGML_CUDA_FA_HUNT_VEC_TILE=1
export COMPRESSED_KV_FATTN_LOG=1
export RDNA2_MATMUL_OPT_V1=1
export GGML_CUDA_MMQ_MAX_X_AUTO=1
export GGML_CUDA_ROCM_QUANT_PREFILL_F16=1
export GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_NKV=40960
export GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1
export GGML_CUDA_ROCM_Q8K_DOT4_DECODE_BN=64
exec ./build-rocm-rdna3-fa/bin/llama-server \
  -m /mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-IQ4_XS-00001-of-00002.gguf \
  -ngl 99 -c 65536 -fa 1 --cache-type-v q4_0 --port 18900 \
  > /tmp/fa-packed16.log 2>&1
