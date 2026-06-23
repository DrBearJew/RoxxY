# WMMA-sized DOT4 tile plan

Date: 2026-05-25

Scope: next q8_0-K INT8 attention prototype after the failed standalone q8K
DOT4 FlashAttention probes.

## Clarified design

Do not split the design into "WMMA route" versus "DOT4 route".

The target is one combined tile design:

```text
WMMA-sized macro tile / work decomposition
+ DOT4x packed-INT8 inner product
+ stable FlashAttention softmax/PV/combine scaffolding
```

Here "WMMA-sized" means the work shape, not necessarily the literal
`v_wmma_i32_16x16x16_iu8` instruction in the first prototype:

```text
Q tile: 16 rows x D256
K tile: 16 rows x D256
KQ tile: 16 x 16 f32 logits
inner dot: D256 = 64 packed i32 dot4 operations per Q/K pair
```

This is the missing middle ground between the failed paths:

- old DOT4 full-FA probes were too narrow: one Q row, tile8 K, scalar softmax/PV;
- old WMMA-I8 route was too route-wide and fragile;
- the stable VEC route is correct but does not use a 16x16 packed-KQ macro tile.

## Evidence basis

### `/home/mrtrent/Downloads/i8-analysis.md`

The external DXIL pipeline uses `dx.op.dot4AddPacked.i32` throughout:

- 24,652 native packed INT8 dot4 operations across 12 passes;
- 16-byte contiguous InitBuffer K tiles in runtime-weight passes;
- zero-init 4-chain accumulation pattern;
- 1D 64-thread groups for most passes, with one 2D `(8,8,1)` pass;
- native low-precision flag enabled.

The lesson is not "avoid WMMA-sized tiling". The lesson is:

```text
use regular 16B packed INT8 tiles and short DOT4 chains inside a larger tile
```

### Existing local microbench

Artifact: `benches/rocm-rdna3/q8q4-dot4-kq-microbench-20260524-060859/raw-result.json`

Shape: `nq=256`, `nk=1024`, `d=256`, `dot4_per_pair=64`.

| layout | ms | dot4 GOP/s | result |
| --- | ---: | ---: | --- |
| q8_0 block 34B | 0.038500 | 435.8 | baseline layout |
| packed16 payload + q8 scales | 0.019264 | 870.9 | 2.00x faster |
| packed16 payload + one row scale | 0.010444 | 1606.4 | scale-load/FMA ceiling |
| pack + packed16 KQ | 0.018502 | n/a | pack cost not dominant in this microbench |

Correctness versus quant reference was stable (`~6.6e-08` relative RMS for q8block
and packed16). This supports packed16 DOT4 as the KQ primitive direction, but the
microbench did not yet use a production-shaped 16x16 KQ tile inside FA.

### q8K DOT4 FA root-cause doc

See `docs/rocm-tbq4-paths/23-q8k-dot4-fa-root-cause-and-next-plan.md`.

The failure was not DOT4 correctness. It was the standalone FA work partitioning:
small K tiles, one Q row per block, poor GQA reuse, scalar softmax/PV, and repeated
pack/infrastructure overhead.

## Prototype contract

Build a KQ tile harness first, not a full FA route.

Required tile contract:

```text
input Q: 16 x 256 float or prequantized int8 + scales
input K: 16 x 256 q8_0 source and/or packed16 shadow + q8 scales
output: 16 x 16 f32 KQ tile
math: 64 DOT4 ops per Q/K pair, organized in regular 16B chunks / short chains
```

The first implementation may use scalar/vector `__builtin_amdgcn_sudot4` /
`ggml_cuda_dp4a`; it should still use the WMMA-sized macro tile.

## Non-goals for first prototype

- Do not build another one-Q-row standalone full-FA kernel.
- Do not promote any route into daily serving.
- Do not start with literal `v_wmma_i32_16x16x16_iu8` unless the DOT4 macro-tile
  harness proves the data layout and scaling contract.
- Do not optimize V/PV or softmax yet.

## Acceptance gates

1. Standalone KQ tile correctness versus CPU/quant reference.
2. ISA/disassembly shows native DOT4-style instructions on gfx1100.
3. Bench compares at least:
   - q8_0 block layout,
   - packed16 payload + q8 scales,
   - packed16 + simplified scale ceiling.
4. Only if the 16x16 packed16 KQ tile is clearly faster than the existing KQ
   primitive should it be spliced into stable FA scaffolding.

## Integration plan if KQ tile wins

Splice into the stable tile/VEC FA flow as a KQ producer only:

```text
launch_fattn
  preserves ncols1/ncols2 GQA grouping
  preserves parallel_blocks and combine
  preserves KV_max mask skipping
  preserves output layout
  calls DOT4 16x16xD256 KQ producer
  then uses existing f32 online softmax and f16/q4 V/PV path
```

The packed-K shadow should be treated as a separate measured variable. q8_0's
34B block layout is the known tax; the packed16 microbench shows why a persistent
or cheap shadow layout is likely necessary.

## First FA-scaffold splice

Implemented after the standalone tile16 harness: `rocm_q8k_dot4_packed16_vec`.

This route intentionally does **not** add another standalone full-FA kernel. It
keeps the stable VEC `launch_fattn` path for softmax, PV, GQA reuse, parallel KV
blocks, combine, KV_max, mask, sinks, and output layout. The only experimental
change is a per-call q8_0-K shadow pack:

```text
q8_0 K 34B blocks -> [256 int8 payload bytes][8 half scales] per K row
```

The KQ dot function then reads this packed16 shadow while still using the normal
q8_1 Q quantization used by the existing q8_0 VEC route.

Required runtime gates:

```bash
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1
GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_VEC=1
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_packed16_vec
```

Initial validation:

- build: `cmake --build build-rocm-rdna2-fa --target ggml-hip test-backend-ops -j4`
- correctness/route: `test-backend-ops -b ROCm0 -o FLASH_ATTN_EXT` with D256
  q8_0-K/q4_0-V prefill filters selected `rocm_q8k_dot4_packed16_vec` and passed
  tested mask/sink/causal-tail/prompt-local variants.
- ISA: gfx1100 saved assembly contains native `v_dot4_i32_iu8 ... neg_lo:[1,1,0]`.

This remains lab-only. The per-call pack is a bridge toward a persistent packed-K
shadow, not a production promotion by itself.
