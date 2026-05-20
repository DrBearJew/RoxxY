# ROCm MTP q8/tbq4 f16+MMQ status report (2026-05-20)

## Current recommendation

Keep the patched build for now and run 27B MTP with chunk/ubatch **1024**:

```bash
LLAMA_MTP_PREFILL_CHUNK=1024
LLAMA_MTP_PREFILL_FORCE_MMQ=1
GGML_CUDA_ROCM_QUANT_PREFILL_F16=1
GGML_CUDA_ROCM_QUANT_PREFILL_F16_MAX_MIB=1024
GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_ALLOC=1
GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_NKV=40960
TBQ4_COOP_SET_ROWS=1
TBQ4_LAYER_ADAPTIVE=7
COMPRESSED_KV_FATTN_LOG=1
```

Server shape used for the current 27B target:

```text
--spec-type draft-mtp --spec-draft-n-max 3 --parallel 1
--cache-type-k q8_0 --cache-type-v tbq4_0
--batch-size 1024 --ubatch-size 1024
-c 40960
```

`2048` is not the current target. It is safe after stable f16 temp allocation, but slower and uses higher loaded VRAM than `1024`.

## Change description

### TBQ4 conversion hooks

Files:

- `ggml/src/ggml-cuda/convert.cu`
- `ggml/src/ggml-cuda/tbq4-cuda.cuh`

Changes:

- Added TBQ4 full-block dequant hooks to `ggml_get_to_fp16_cuda()` and `ggml_get_to_fp32_cuda()`.
- Templated `k_tbq4_dequant_full` so it can write either `float` or `half` output.
- This lets the opt-in f16 FlashAttention route consume TBQ4 V-cache through the existing conversion interface.

### ROCm quantized-KV f16 prefill gate

File:

- `ggml/src/ggml-cuda/fattn.cu`

Changes:

- Kept HIP/ROCm quantized K/V on the existing vector route by default.
- Added explicit opt-in for f16-temp prefill route on quantized K/V.
- Added per-op f16 temp cap check; default cap is `1024 MiB`.
- Added TBQ4 contiguity guards because the current TBQ4 full-block conversion hook supports contiguous tensors only.

### Stable f16 temp allocation

File:

- `ggml/src/ggml-cuda/fattn-common.cuh`

Changes:

- Added opt-in stable f16 K/V temporary allocation for HIP/ROCm quantized K/V.
- The HIP legacy pool caches allocations by size. During MTP prefill, `nkv` grows chunk-by-chunk, so exact-sized f16 temps cause the pool to retain a ladder of intermediate buffer sizes.
- `GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_ALLOC=1` rounds f16 temp allocation to a stable max-`nkv` size and lets the pool reuse one scratch size.

### Harness

File:

- `scripts/hip/run-mtp-f16-mmq-vram-sweep.py`

Changes:

- Adds a server-based MTP sweep harness with route logging, VRAM sampling, chunk/ubatch contract checks, and f16/MMQ/stable-allocation toggles.

## Experimental flags

### Active / recommended for current 27B path

| flag | value | purpose |
|---|---:|---|
| `LLAMA_MTP_PREFILL_CHUNK` | `1024` | MTP draft prefill chunk size; must match `--ubatch-size`. |
| `LLAMA_MTP_PREFILL_FORCE_MMQ` | `1` | Avoids large MTP draft-prefill matmul temp spikes on ROCm. |
| `GGML_CUDA_ROCM_QUANT_PREFILL_F16` | `1` | Enables opt-in f16-temp FlashAttention prefill route for quantized K/V. |
| `GGML_CUDA_ROCM_QUANT_PREFILL_F16_MAX_MIB` | `1024` | Per-op f16 temp cap for quantized-KV prefill route. |
| `GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_ALLOC` | `1` | Reuses stable-sized f16 K/V temp buffers instead of caching every growing `nkv` size. |
| `GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_NKV` | `40960` | Stable max `nkv` allocation target for the 40k context run. |
| `TBQ4_COOP_SET_ROWS` | `1` | Existing TBQ4 set-rows tuning used in target artifacts. |
| `TBQ4_LAYER_ADAPTIVE` | `7` | Existing layer-adaptive TBQ4 tuning used in target artifacts. |
| `COMPRESSED_KV_FATTN_LOG` | `1` | Emits route selection logs for artifact validation. |

### Backward-compatible aliases accepted by the gate

- `GGML_CUDA_ROCM_QUANT_PREFILL_MMA`
- `GGML_CUDA_ROCM_QUANT_PREFILL_WMMA`
- `TBQ4_PREFILL_WMMA`
- `GGML_CUDA_ROCM_QUANT_PREFILL_MMA_MAX_MIB`

### Not promoted / removed from active path

- `GGML_CUDA_ROCM_FATTN_TILE_COLS`
- `GGML_CUDA_ROCM_FATTN_FORCE_MMA_F16`
- rocWMMA forced observer paths
- legacy compressed-KV WMMA envs as wrapper defaults (`TBQ4_WMMA_FATTN`, `COMPRESSED_KV_WMMA_FATTN`)

These were either unsafe, unstable, or regressive in the tested ROCm stack.

## Bugs / findings

1. **TBQ4 lacked f16/f32 conversion hooks for the f16 FlashAttention path.**
   - Symptom: TBQ4 could not use the generic quantized-KV f16 conversion route cleanly.
   - Fix: add TBQ4 dequant hooks in `convert.cu` and templated TBQ4 dequant output in `tbq4-cuda.cuh`.

2. **Quantized-KV f16 prefill is fast but unsafe as a default on HIP.**
   - Symptom: full f16 K/V temps can erase compressed-KV savings and OOM at long context.
   - Fix: default remains vector route; f16 prefill requires explicit env gate and cap.

3. **MTP f16 prefill showed apparent VRAM leak / high-water growth.**
   - Root cause: HIP legacy pool cached f16 temp allocations for each growing `nkv` size.
   - Fix: opt-in stable f16 temp allocation rounded to max `nkv`.
   - 27B chunk1024 peak delta dropped from `3.213 GiB` to `0.389 GiB`.

4. **MMQ interaction matters for MTP draft prefill.**
   - `LLAMA_MTP_PREFILL_FORCE_MMQ=1` suppresses large transient VRAM spikes from the draft-prefill matmul side.

5. **rocWMMA was regressive on this stack.**
   - 27B pp2048 rocWMMA probe was around `110.91 t/s`, so it is not a useful observer or target route here.

6. **Chunk/ubatch 2048 is not promoted.**
   - Stable allocation made it safe, but 27B chunk2048 was slower than chunk1024 and carried higher loaded VRAM.

7. **Latest clean origin/master q4/q4 MTP pp512 is slightly faster but much higher peak delta.**
   - Fresh origin/master `ad2775726`, version `9261`, q4_0/q4_0 MTP pp512: `425.80 t/s`, peak delta `3.764 GiB`.
   - Current patched q8_0/tbq4_0 MTP pp512: about `408.6 t/s`, peak delta about `0.389 GiB`.
   - Decision: keep current patched build for now; the ~17 t/s pp512 loss is acceptable for much lower transient VRAM.

## Key artifacts

- `benches/server-27b-mtp-pp-sweep-20260520-183601/summary.md`
- `benches/server-27b-mtp-f16-mmq-stablealloc-comparison-20260520-1819.md`
- `/home/mrtrent/llama.cpp-ggml-original/benches/server-27b-origin-master-mtp-q4q4-pp512-probe-20260520-191006/summary.md`
