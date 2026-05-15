# Compressed-KV Triton experiments

This directory is a ROCm/Triton validation lane for compressed-KV FlashAttention ideas used by the HIP implementation. It is intentionally isolated from production `llama-server`.

## What this is

- A correctness harness for compressed KV formats: `planar3_0`, `iso3_0`, and `tbq4_0`.
- A place to test materialization, paged row mapping, QK/QKV loops, masks, varlen metadata, and segmented long-context reductions.
- A design oracle for C++/HIP refactors: prove the contract here first, then port only the minimal validated idea.

## What this is not

- Not a production runtime dependency.
- Not wired into CMake.
- Not used by the wrapper or llama-swap configs.
- Not a performance claim; scripts here are correctness gates unless they explicitly say otherwise.

## Quick start

Use the local `LLM` conda environment:

```bash
/home/mrtrent/miniconda3/envs/LLM/bin/python scripts/hip/check-triton-feasibility.py
experiments/compressed_kv_triton/run_all.sh
```

Optional aggregate JSON report:

```bash
/home/mrtrent/miniconda3/envs/LLM/bin/python experiments/compressed_kv_triton/run_all_json.py
```

Known-good local environment:

- GPU: AMD Radeon RX 7900 XTX / `gfx1100`
- Triton: `3.7.0`
- PyTorch: `2.12.0a0+rocm7.13.0a20260412`
- HIP runtime reported by torch: `7.13.60980`

## Contract under test

The experiments keep three responsibilities separate:

1. **Format decoder**
   - receives a physical compressed row and dimension offsets;
   - decodes values for one format;
   - does not know about paging or attention scheduling.
2. **Row mapper**
   - maps logical rows through block tables to physical compressed rows;
   - owns tail-row and non-monotonic page behavior;
   - is tested independently before being used by attention kernels.
3. **Attention backend**
   - owns tile shape, synchronization, masks, online softmax, QK/QKV loops, and reductions;
   - materializes only the current tile/rows, never a sequence-wide dense fp16 KV cache.

Format-domain rule:

- `planar3_0` and `iso3_0` are original-domain formats.
- `tbq4_0` is a FWHT-domain format; production C++ keeps Q pre-rotation and O inverse rotation outside the generic materializer.

## Aggregate gate

Run everything:

```bash
experiments/compressed_kv_triton/run_all.sh
```

The gate covers:

| Stage | Script | Purpose |
| --- | --- | --- |
| Environment | `scripts/hip/check-triton-feasibility.py` | Confirms Triton/torch/ROCm can compile and launch. |
| Compatibility | `compat_gate.py` | Guards Python imports, bytecode compile, and Triton API assumptions. |
| llama.cpp parity | `llama_cpp_tensor_layout_parity.py`, `llama_cpp_block_table_parity.py` | Checks KV tensor byte offsets and block-table mappings against llama.cpp slot/view rules. |
| Dispatch policy | `dispatch_policy_contract.py` | Freezes default VEC, env-gated WMMA, and mixed `TBQ4_0/Q8_0` behavior. |
| Row mapping | `paged_row_mapping_contract.py` | Checks logical-row to physical-row mapping independent of format decode. |
| Materializers | `materializers.py`, `paged_materializers.py` | Verifies compressed-row decode for contiguous and paged layouts. |
| QK | `qk_only.py`, `qk_2d_tiled.py` | Compares compressed-K dot products against dense references. |
| Softmax/masks | `online_softmax.py`, `mask_semantics.py` | Tests online softmax, causal masks, sliding windows, and tail tiles. |
| QKV | `full_qkv.py`, `qkv_2d_tiled.py` | Validates end-to-end attention output for tiled compressed KV. |
| Metadata | `varlen_qkv.py` | Tests variable-length sequence metadata and GQA head mapping. |
| Long context | `segmented_qkv.py`, `compare_2d_segmented.py` | Tests segmented reduction semantics without making timing claims. |
| Domain parity | `tbq4_domain_parity.py`, `planar_iso_domain_parity.py` | Verifies TBQ4 FWHT Q/O rotation policy and Planar/Iso original-domain policy. |
| Autotune metadata | `autotune_metadata.py` | Records fixed RDNA3 configs and tuning keys without adding dependencies. |

## Production invariants

These experiments are allowed to influence C++ code only if the production invariants still hold:

- default compressed-KV FlashAttention dispatch remains VEC;
- `TBQ4_WMMA_FATTN=1` and `COMPRESSED_KV_WMMA_FATTN=1` remain explicit opt-in gates;
- Triton/Python does not enter production CMake or `llama-server` runtime;
- wrappers and llama-swap configs do not export experimental WMMA flags;
- mixed `TBQ4_0/Q8_0` stays on VEC until a Q8 V loader and domain policy exist;
- paged/block-table C++ row mapping remains gated until llama.cpp tensor/block-table parity tests pass and a separate C++ adapter is designed.

Related production checks:

```bash
scripts/hip/check-compressed-kv-fa-invariants.sh
scripts/hip/run-compressed-kv-wmma-smokes.sh
```

## C++ paged mapper target

Do not add a production paged/block-table mapper until the llama.cpp parity gates pass. The intended C++ adapter remains narrow:

- `row_ptr(logical_row)` resolves block-table metadata to a llama.cpp KV slot;
- byte offset stays `stream_stride * stream + head_stride * head + token_stride * slot`;
- format traits still only decode a physical compressed row and element offset;
- backend code still owns tiling, masks, synchronization, MMA/WMMA, and softmax.

## Promotion rule

A C++ refactor may use this lane as evidence only when:

1. `run_all.sh` passes;
2. the production invariant checker passes;
3. default llama-swap VEC smoke still passes;
4. opt-in WMMA smokes pass when attention dispatch or materialization changed;
5. rollback remains one env/config change back to the VEC path.
