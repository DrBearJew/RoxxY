# RDNA3 MMQ selector policy: Qwen3.6-35B-A3B-IQ4_XS

GPU/backend: `gfx1100/RX 7900 XTX` / `ROCm/HIP`
Selected: `maxx48`
Env: `GGML_CUDA_MMQ_MAX_X=48 RDNA2_MATMUL_OPT_V1=1`
Cache K/V: `tbq4_0/tbq4_0`; route: `tbq4_vec`
Basis: fastest eligible weighted llama-bench throughput

## Variant scores

| variant | eligible | route | cache K/V | env | weighted tok/s | speedup vs baseline | static | notes |
|---|:---:|---|---|---|---:|---:|---|---|
| `baseline` | yes | `tbq4_vec` | `tbq4_0/tbq4_0` | `baseline/no extra env` | 2239.0 | 1.000x | no fixed MAX_X |  |
| `rdna2_opt` | yes | `tbq4_vec` | `tbq4_0/tbq4_0` | `RDNA2_MATMUL_OPT_V1=1` | 2229.2 | 0.996x | no fixed MAX_X |  |
| `maxx32` | yes | `tbq4_vec` | `tbq4_0/tbq4_0` | `GGML_CUDA_MMQ_MAX_X=32 RDNA2_MATMUL_OPT_V1=1` | 2300.5 | 1.027x | x32: fits coherent LDS and soft accumulator budget |  |
| `maxx48` | yes | `tbq4_vec` | `tbq4_0/tbq4_0` | `GGML_CUDA_MMQ_MAX_X=48 RDNA2_MATMUL_OPT_V1=1` | 2627.7 | 1.174x | x48: fits coherent LDS and soft accumulator budget | selected |
| `maxx64` | yes | `tbq4_vec` | `tbq4_0/tbq4_0` | `GGML_CUDA_MMQ_MAX_X=64 RDNA2_MATMUL_OPT_V1=1` | 2576.5 | 1.151x | x64: over soft accumulator/register budget |  |
| `maxx96` | NO | `—` | `—` | `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=96` | — | — | x96: over soft accumulator/register budget | missing prompt rows: [128, 256, 512, 1024, 2048, 4096] |
| `maxx128` | yes | `tbq4_vec` | `tbq4_0/tbq4_0` | `GGML_CUDA_MMQ_MAX_X=128 RDNA2_MATMUL_OPT_V1=1` | 2208.0 | 0.986x | x128: over soft accumulator/register budget |  |
| `scratch16k` | NO | `—` | `—` | `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_IQ4_XS_MMQ_SCRATCH16K=1` | — | — | x64: over soft accumulator/register budget | missing prompt rows: [128, 256, 512, 1024, 2048, 4096] |
| `maxx48_f16_mma` | yes | `f16_mma` | `f16/f16` | `GGML_CUDA_MMQ_MAX_X=48 RDNA2_MATMUL_OPT_V1=1` | 2272.8 | 1.015x | x48: fits coherent LDS and soft accumulator budget |  |
| `maxx48_tbq4_wmma` | yes | `tbq4_wmma` | `tbq4_0/tbq4_0` | `GGML_CUDA_MMQ_MAX_X=48 RDNA2_MATMUL_OPT_V1=1 TBQ4_WMMA_FATTN=1` | 895.9 | 0.400x | x48: fits coherent LDS and soft accumulator budget |  |
| `maxx64_f16_mma` | yes | `f16_mma` | `f16/f16` | `GGML_CUDA_MMQ_MAX_X=64 RDNA2_MATMUL_OPT_V1=1` | 2232.6 | 0.997x | x64: over soft accumulator/register budget |  |
| `maxx64_tbq4_wmma` | yes | `tbq4_wmma` | `tbq4_0/tbq4_0` | `GGML_CUDA_MMQ_MAX_X=64 RDNA2_MATMUL_OPT_V1=1 TBQ4_WMMA_FATTN=1` | 894.5 | 0.400x | x64: over soft accumulator/register budget |  |

## Static shape candidates

| mmq_x | tiles | acc B/thread | coherent LDS | double LDS | viable | reason |
|---:|---:|---:|---:|---:|:---:|---|
| 16 | 32 | 32 | 42048 | 80964 | yes | fits coherent LDS and soft accumulator budget |
| 32 | 16 | 64 | 44160 | 83076 | yes | fits coherent LDS and soft accumulator budget |
| 48 | 11 | 96 | 46272 | 85188 | yes | fits coherent LDS and soft accumulator budget |
| 64 | 8 | 128 | 48384 | 87300 | NO | over soft accumulator/register budget |
| 80 | 7 | 160 | 51520 | 90436 | NO | over soft accumulator/register budget |
| 96 | 6 | 192 | 53632 | 92548 | NO | over soft accumulator/register budget |
| 112 | 5 | 224 | 55744 | 94660 | NO | over soft accumulator/register budget |
| 128 | 4 | 256 | 57856 | 96772 | NO | over soft accumulator/register budget |

## Shell activation

```bash
# bench/server flags: -fa 1 -ctk tbq4_0 -ctv tbq4_0
export GGML_CUDA_MMQ_MAX_X=48
export RDNA2_MATMUL_OPT_V1=1
```
