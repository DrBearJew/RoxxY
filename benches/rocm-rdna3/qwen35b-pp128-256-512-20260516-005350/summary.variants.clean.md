# Qwen3.6 35B-A3B IQ4_XS pp selector variants

Artifact: `/home/mrtrent/llama.cpp-mtp-tbq4-rdna3/benches/rocm-rdna3/qwen35b-pp128-256-512-20260516-005350`

Command shape: `llama-bench -p 128,256,512 -n 0 -fa 1 -ctk tbq4_0 -ctv tbq4_0 -b 1024 -ub 512 -r 5 -o jsonl`
Model: `/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`
Mode: non-MTP, ROCm/HIP, RX 7900 XTX.

| variant | pp128 tok/s | pp256 tok/s | pp512 tok/s |
|---|---:|---:|---:|
| `baseline` | 1241.3 ± 40.1 | 1910.9 ± 39.8 | 2621.7 ± 14.0 |
| `rdna2_opt` | 1245.5 ± 51.2 | 1901.8 ± 33.3 | 2622.3 ± 25.3 |
| `maxx48` | 1780.7 ± 44.5 | 2479.6 ± 25.0 | 3150.0 ± 40.5 |
| `maxx64` | 1660.1 ± 56.6 | 2401.1 ± 39.0 | 3100.2 ± 16.1 |
| `scratch16k` | 1238.5 ± 54.5 | 1904.5 ± 24.7 | 2613.6 ± 37.1 |

Env variants:
- `baseline`: no RDNA selector env
- `rdna2_opt`: `RDNA2_MATMUL_OPT_V1=1`
- `maxx48`: `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=48`
- `maxx64`: `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=64`
- `scratch16k`: `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_IQ4_XS_MMQ_SCRATCH16K=1`
