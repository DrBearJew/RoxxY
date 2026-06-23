# RDNA3 MMQ selector policy: Qwen3.6-35B-A3B-IQ4_XS

GPU/backend: `gfx1100/RX 7900 XTX` / `ROCm/HIP`
Selected: `maxx48`
Env: `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=48`
Basis: fastest eligible weighted llama-bench throughput

## Variant scores

| variant | eligible | env | weighted tok/s | speedup vs baseline | static | notes |
|---|:---:|---|---:|---:|---|---|
| `baseline` | yes | `baseline/no extra env` | 2072.2 | 1.000x | no fixed MAX_X |  |
| `rdna2_opt` | yes | `RDNA2_MATMUL_OPT_V1=1` | 2071.1 | 0.999x | no fixed MAX_X |  |
| `maxx32` | NO | `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=32` | — | — | x32: fits coherent LDS and soft accumulator budget | missing prompt rows: [128, 256, 512] |
| `maxx48` | yes | `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=48` | 2653.5 | 1.281x | x48: fits coherent LDS and soft accumulator budget | selected |
| `maxx64` | NO | `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=64` | 2568.3 | 1.239x | x64: over soft accumulator/register budget | over soft accumulator/register budget |
| `maxx96` | NO | `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=96` | — | — | x96: over soft accumulator/register budget | missing prompt rows: [128, 256, 512]; over soft accumulator/register budget |
| `maxx128` | NO | `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=128` | — | — | x128: over soft accumulator/register budget | missing prompt rows: [128, 256, 512]; over soft accumulator/register budget |
| `scratch16k` | NO | `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_IQ4_XS_MMQ_SCRATCH16K=1` | 2066.1 | 0.997x | x64: over soft accumulator/register budget | over soft accumulator/register budget |

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
export RDNA2_MATMUL_OPT_V1=1
export GGML_CUDA_MMQ_MAX_X=48
```
