# Qwen 27B local architecture smoke

Model: `/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf`
Backend: ROCm/RX 7900 XTX, `build-rocm-rdna2-fa`, q8_0 K + tbq4_0 V unless noted.

| case | prompt | gen | pp t/s | tg t/s | note |
|---|---:|---:|---:|---:|---|
| default q8/tbq4 | 128 | 16 | 313.623 | 26.680 | VEC/default |
| f16 bucket q8/tbq4 | 128 | 16 | 333.841 | 25.226 | `GGML_CUDA_ROCM_QUANT_PREFILL_F16=1`, `STABLE_BUCKET_NKV=1024` |
| symmetric tbq4/tbq4 | 32 | 1 | 108.986 | 15.922 | auto-asym guard did not visibly trigger for this 27B smoke |
| default q8/tbq4 | 1024 | 64 | 719.237 | 27.101 | VEC/default |
| f16 bucket q8/tbq4 | 1024 | 64 | 767.590 | 26.786 | `GGML_CUDA_ROCM_QUANT_PREFILL_F16=1`, `STABLE_BUCKET_NKV=4096` |
| direct loads default q8/tbq4 | 1024 | 64 | 767.777 ±55.566 | 27.405 ±0.098 | r3 after HIP `get_int_b1/get_int_b2` direct loads |
| direct loads f16 bucket q8/tbq4 | 1024 | 64 | 833.367 ±68.676 | 27.012 ±0.405 | r3, direct loads + f16 bucket |
| half-d8 default q8/tbq4 | 1024 | 64 | 796.666 ±59.330 | 27.288 ±0.222 | r3 after vecdot `half d8[]` VGPR cleanup |
| half-d8 f16 bucket q8/tbq4 | 1024 | 64 | 837.480 ±60.069 | 27.004 ±0.525 | r3, direct loads + half-d8 + f16 bucket |

At pp1024, f16 bucket improved prompt throughput by ~6.7% with decode essentially flat/slightly lower (~-1.2%). After HIP direct `get_int_b1/get_int_b2` loads, r3 default decode averaged 27.405 t/s versus the earlier single-run 27.101 t/s. The half-d8 VGPR cleanup improved prompt averages versus direct loads, with decode roughly flat/slightly lower. VRAM returned to 1% after runs.
