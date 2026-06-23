# RDNA3 FlashAttention policy: Qwen3.6-27B MTP compressed KV

GPU/backend: `gfx1100/RX 7900 XTX` / `ROCm/HIP`
Selected: `vec_compressed_kv`
Server flags: `--flash-attn on`
Cache types: `tbq4_0, planar3_0, iso3_0`
Env: `none`

## Candidates

| candidate | eligible | route | cache types | env | status | notes |
|---|:---:|---|---|---|---|---|
| `vec_compressed_kv` | yes | `kernel=vec` | `tbq4_0, planar3_0, iso3_0` | `none` | stable default | Preserves compressed KV in the FA loop; no full-cache f16 K/V temp materialization. |
| `wmma_tbq4` | NO | `kernel=wmma_tbq4` | `tbq4_0` | `TBQ4_WMMA_FATTN=1 COMPRESSED_KV_FATTN_LOG=1` | experimental opt-in | experimental opt-in path; needs standalone coherence/perf smokes before promotion; pass --allow-experimental-wmma only after fresh smokes |
| `wmma_planar_iso` | NO | `kernel=wmma_compressed_kv` | `planar3_0, iso3_0` | `COMPRESSED_KV_WMMA_FATTN=1 COMPRESSED_KV_FATTN_LOG=1` | experimental opt-in | experimental opt-in path; needs standalone coherence/perf smokes before promotion; pass --allow-experimental-wmma only after fresh smokes |
| `tile_or_full_temp_quantized` | NO | `kernel=tile/wmma/mma with f16 temp` | `tbq4_0, planar3_0, iso3_0` | `none` | rejected for long-context compressed KV | can materialize full f16 K/V temp buffers and erase compressed-KV memory savings |

## F16 MMA guardrails

These apply to the f16 MMA FlashAttention path. They do not promote compressed-KV WMMA, which remains opt-in.

| target | guardrail | policy |
|---|---|---|
| RDNA3/RDNA4 f16 MMA FA | Use 32-logical VKQ tiles with fp16 accumulation only when the head dimension is divisible by 32; head sizes 80 and 112 use 16-wide VKQ tiles with fp32 accumulation. | Keep the layout split explicit so compressed-KV WMMA experiments do not inherit the f16 MMA path by accident. |
| RDNA4 f16 MMA FA | The faster VKQ transpose path scrambles accumulator layout along the head dimension. | Require DATA_LAYOUT_I_MAJOR_SCRAMBLED/unscramble handling before storing or reusing VKQ accumulators. |
| RDNA3/RDNA4 large heads | Do not promote MMA FA for head sizes above 128 without fresh local performance evidence; tile has been faster in this range. | Prefer TILE/VEC fallback for h > 128 on RDNA-class GPUs. |
| CDNA MFMA f16 FA | MFMA can work for head sizes up to 256 with the tuned kernel parameters. | CDNA may select MMA up to h <= 256 when the batch-size crossover is met. |

## Smoke evidence

| quant | ok | contexts | kernels vec/wmma/other | prompt tok/s | decode tok/s | peak VRAM GiB | failures |
|---|:---:|---:|---:|---|---|---|---|
| `iso3_0` | yes | 131072, 204800 | 40/0/0 | iso3_0-ctx128k:366.1, iso3_0-ctx200k:392.4 | iso3_0-ctx128k:36.2, iso3_0-ctx200k:34.8 | iso3_0-ctx128k:21.10, iso3_0-ctx200k:22.36 | — |
| `planar3_0` | yes | 131072, 204800 | 40/0/0 | planar3_0-ctx128k:196.7, planar3_0-ctx200k:296.0 | planar3_0-ctx128k:32.9, planar3_0-ctx200k:34.9 | planar3_0-ctx128k:21.10, planar3_0-ctx200k:22.36 | — |
| `tbq4_0` | yes | 131072, 204800 | 40/0/0 | tbq4_0-ctx128k:257.9, tbq4_0-ctx200k:223.1 | tbq4_0-ctx128k:29.1, tbq4_0-ctx200k:24.2 | tbq4_0-ctx128k:21.50, tbq4_0-ctx200k:23.04 | — |

## Activation

```bash
# server flags
--flash-attn on --cache-type-k tbq4_0 --cache-type-v tbq4_0
```
