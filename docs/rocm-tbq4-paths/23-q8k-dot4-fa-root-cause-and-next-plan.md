# Q8K DOT4 FlashAttention root cause and next plan

Date: 2026-05-25

Scope: ROCm/RDNA3 q8_0-K + q4_0/tbq4-V FlashAttention prefill probes, especially
`rocm_q8k_dot4_kq`, `fused_tile8_parallel`, and `splitkv_tile8_parallel`.

## Decision

Do not promote the current q8K DOT4 FlashAttention prototypes.

The failure is not DOT4 correctness. The failure is that the prototypes optimize
raw KQ while replacing the mature FlashAttention dataflow with a narrow scalar /
vector loop. Once KQ is fused into attention, the bottleneck moves to
FlashAttention work partitioning, online-softmax/PV, V dequantization, GQA reuse,
and memory traffic.

Keep all routes unsafe and opt-in:

```bash
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1
GGML_CUDA_ROCM_Q8K_DOT4_KQ=1
GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq
```

## Local evidence

### KQ-only was not the limiting stage

Artifact: `benches/rocm-rdna3/q8k-dot4-fullfa-probe-20260525-054229/summary.tsv`

| case | KQ ms | FA/PV ms | total ms | interpretation |
| --- | ---: | ---: | ---: | --- |
| p512 | 0.197 | 6.418 | 6.655 | KQ is ~3% of route time |
| p1024 | 0.789 | 27.847 | 29.226 | KQ is ~3% of route time |
| p2048 | 3.068 | 108.340 | 111.597 | KQ is ~3% of route time |

A faster DOT4 KQ primitive cannot overcome a slow full-FA loop if the remaining
softmax/PV path dominates.

### Tile8 improved the naive fused path but still lost production A/B

Artifacts:

- `benches/rocm-rdna3/q8k-dot4-fused-tile8-parallel-20260525-073656/summary.tsv`
- `benches/rocm-rdna3/q8k-dot4-tile8p-prod-ab-20260525-075922/summary.tsv`

Micro/prompt-only medians:

| case | FA ms | prompt tok/s |
| --- | ---: | ---: |
| tile8p p512 | 3.976 | 590.9 |
| tile8p p1024 | 16.238 | 638.7 |
| tile8p p2048 | 66.733 | 589.4 |

Production-shaped A/B:

| route | p4096 prefill tok/s | decode tok/s | route hits |
| --- | ---: | ---: | --- |
| stable default | 868.5 | 28.32 | `q8k_q4v_vec,tile` |
| DOT4 tile8p | 566.4 | 27.56 | `q8k_q4v_vec,rocm_q8k_dot4_kq` |

The small-shape tile8 improvement did not transfer to p4096 production shape.

### Split-KV did not solve the p4096 loss

Artifact: `benches/rocm-rdna3/q8k-dot4-splitkv-tile8p-20260525-084619/summary.tsv`

| case | splits | FA ms | prefill tok/s |
| --- | ---: | ---: | ---: |
| p4096 | 1 | 55.836 | 547.2 |
| p4096 | 2 | 55.473 | 549.9 |
| p4096 | 4 | 55.870 | 549.4 |
| p4096 | 8 | 54.108 | 550.0 |

Production-shaped A/B:

| route | splits | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: |
| stable default | n/a | 882.7 | 28.26 |
| splitkv auto | 4 | 547.9 | 28.20 |
| splitkv s8 | 8 | 550.9 | 28.26 |

Splitting K/V increased block parallelism but added partial-output and combine
traffic. It did not address the root cause.

## External-source constraints

### FlashAttention-2 guidance

Sources:

- Stanford CRFM FlashAttention-2 blog: https://crfm.stanford.edu/2023/07/17/flash2.html
- FlashAttention-2 paper/code: https://github.com/Dao-AILab/flash-attention

Relevant constraints:

1. FA2 speedups come from fewer non-matmul FLOPs, better sequence parallelism,
   and better warp/work partitioning.
2. Non-matmul FP32 work such as online-softmax rescale, exp, masking, and bounds
   checks can be much more expensive than matrix/tensor-core work.
3. FA1-style sliced-K work partitioning creates shared-memory writes,
   synchronization, and reductions. FA2 instead improves partitioning to reduce
   inter-warp communication.
4. For long sequences with small batch/head counts, FA2 parallelizes over the
   sequence dimension, but this is integrated into the full tiled algorithm and
   combine path, not bolted on as a small-tile afterthought.

### ROCm/CK guidance

Sources:

- AMD ROCm model acceleration docs: https://rocmdocs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html
- AMD CK-Tile FA2 blog: https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- ROCm FlashAttention README: https://github.com/ROCm/flash-attention/blob/a9a3170f/README.md

Relevant constraints:

1. ROCm FA2 is framed as reducing SRAM/HBM movement via tiling over Q, K, and V.
2. CK-Tile's example maps a workgroup to an output tile and uses large tiles such
   as M=128 Q rows, N=128 K rows, and K=32 head-dim chunks.
3. Q is kept in VGPRs, while K/V are moved through LDS/global-memory tile windows.
4. The official ROCm path uses CK or Triton FA2 backends, not a one-Q-row scalar
   softmax/PV loop.

### llama.cpp stable FA structure

Relevant local files:

- `ggml/src/ggml-cuda/fattn-common.cuh`
- `ggml/src/ggml-cuda/fattn-vec.cuh`
- `ggml/src/ggml-cuda/fattn-tile.cuh`
- `ggml/src/ggml-cuda/fattn.cu`

Key stable-path properties:

1. `launch_fattn` selects `parallel_blocks` from occupancy, number of SMs, and
   `ntiles_KV`; it only pays combine overhead when profitable.
2. Stable kernels already support split/parallel KV via `gridDim.y`, `dst_meta`,
   and `flash_attn_combine_results`.
3. `KV_max` mask scanning skips fully masked K/V tiles for long prompts.
4. Tile kernels use `ncols2` GQA grouping so multiple Q heads sharing one K/V
   head reuse K/V work where possible.
5. Quantized-KV VEC is deliberately kept for one/two-token decode; long prefill
   is routed to tile/MMA/f16-temp paths when explicitly allowed.

## Why our prototype loses

### 1. It uses one Q row per block

`ggml_cuda_q8k_dot4_fused_tile8_parallel_fattn_q4_0_kernel` maps:

```cpp
q_row = blockIdx.x;
hq    = blockIdx.y;
b     = blockIdx.z;
```

This gives one Q row and one Q head per block. That is closer to a decode/vector
kernel than a prefill FA2 tile. In p4096 prefill, the stable tile path can exploit
larger Q tiles and grouped heads; the prototype cannot.

### 2. It uses only eight K rows per inner tile

```cpp
GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K = 8
```

Every 8 K rows, the kernel performs softmax max/exp/sum updates, stores logits /
probs in shared memory, then dequantizes V and updates output. That makes the
loop dominated by scalar softmax/PV overhead and synchronization instead of raw
DOT4 throughput.

### 3. It does not exploit GQA reuse

For Qwen-style 27B shapes, observed route logs include:

```text
heads_q=24 heads_k=4 gqa=6
```

The prototype computes each Q head independently with:

```cpp
hk = hq / gqa_ratio;
```

but does not group multiple Q heads that share the same `hk`. Stable tile
selection can group GQA via `ncols2` when the ratio permits. For gqa=6, even a
2-way grouping avoids duplicated K/V traffic for pairs of Q heads.

### 4. It repacks K per FA call instead of using a persistent packed-K cache

The DOT4 path materializes packed K payloads into temporary storage per attention
call. Microbenchmarks showed packed16 KQ can be faster than q8block KQ, but the
runtime path still pays staging and misses persistent reuse. The `i8-analysis.md`
DXIL study supports 16-byte packed K tiles, but the production lesson is that
those tiles must be part of the attention dataflow/cache design, not just a
pre-kernel conversion.

### 5. Split-KV is useful for decode-like regimes, not this p4096 prefill loss

The FlashAttention README describes split KV/cache loading as an inference
optimization for very small query length, where KV loading is the bottleneck. Our
failing case has thousands of Q rows. It already has abundant inter-block
parallelism, so extra splits mostly duplicate per-split softmax state and add
O/LSE combine traffic.

## Next implementation plan

### Stage 0: no default changes

Keep current q8K DOT4 routes as diagnostic only. Promotion requires a
production-shaped A/B win, not a KQ-only or p<=2048 win.

Minimum promotion gate:

```text
p4096+n256 q8_0 K / q4_0 or tbq4 V, FA on:
  new route prefill >= stable default by 5%
  decode no worse than 2%
  test-backend-ops parity passes
  route logs prove intended route selection
```

### Stage 1: stop building standalone FA shells

Do not continue extending `fused_tile8_parallel` / `splitkv_tile8_parallel` as
separate attention implementations.

Instead, reuse the stable FA scaffolding:

- `launch_fattn` grid/parallel-block selection;
- `dst_meta` and `flash_attn_combine_results` combine semantics;
- `KV_max` mask skipping;
- GQA grouping decisions from tile/vec selectors;
- output layout already expected by llama.cpp.

### Stage 2: plug DOT4 into a real tiled/GQA path

Candidate design:

1. Start from the existing `fattn-vec.cuh` or `fattn-tile.cuh` control flow.
2. Add a D=256 q8_0-K DOT4 KQ primitive behind an explicit unsafe route.
3. Preserve existing per-tile online softmax and combine metadata.
4. Add GQA grouping first, before more K splitting.
5. Only then test packed-K layout choices.

This is less glamorous than a standalone kernel, but it attacks the measured
failure mode directly.

### Stage 3: packed-K shadow only if it is reused

A packed16 K shadow is still the likely data-layout lever, but it must be reused
across Q heads/tiles or built at KV write time.

Preferred options, in order:

1. Persistent shadow during KV `set_rows` / cache update for admitted q8_0 K.
2. Per-layer/per-context shadow reused across all Q heads in prefill.
3. Per-call tile pack into LDS only if it enables larger FA tiles and avoids a
   global temporary.

Avoid full per-call global repack unless the final p4096 A/B proves it pays.

### Stage 4: benchmark ladder

Required artifacts for any next prototype:

1. `test-backend-ops` route-backed correctness: causal, prompt-local, GQA6,
   GQA8, odd/tail fallback coverage.
2. KQ-only timing for regression diagnosis only.
3. Full-FA micro timings p512/p1024/p2048.
4. Production-shaped p4096+n256 A/B against stable default.
5. Route log summary showing `q8k_q4v_vec`, `tile`, `f16_temp`, and experimental
   route hits.

## Non-goals

- Do not promote `splitkv_tile8_parallel`.
- Do not add more split counts before fixing GQA/tile shape.
- Do not optimize KQ-only at the expense of full-FA timings.
- Do not change live serving defaults from this line of work.
- Do not require DOT4 route without the explicit unsafe + route contract envs.

## Short conclusion

DOT4 is the right primitive for packed INT8 KQ. The current prototype fails
because it is not a competitive FlashAttention implementation. The next attempt
must be a DOT4-enabled version of the stable tiled/GQA/combined FA dataflow, not a
faster standalone KQ loop wrapped in scalar softmax/PV.
