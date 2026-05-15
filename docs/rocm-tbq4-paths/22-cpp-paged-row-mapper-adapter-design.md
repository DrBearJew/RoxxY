# C++ paged/block-table row-mapper adapter design

## Scope assumption

This slice designs the next compressed-KV FlashAttention step: a narrow C++ paged/block-table row-mapper adapter. It does **not** change production dispatch, runtime behavior, CMake, wrappers, llama-swap, or kernel selection.

## Objective

Design and contract-test a C++ row-mapper adapter that can map logical paged/block-table rows to existing llama.cpp KV slot byte offsets while preserving the default VEC route and compressed-KV savings.

## Non-goals

- Do not enable paged/block-table mapping in production by default.
- Do not change `ggml/src/ggml-cuda/fattn.cu` dispatch policy.
- Do not enable `TBQ4_WMMA_FATTN` or `COMPRESSED_KV_WMMA_FATTN` by default.
- Do not add Triton/Python to CMake, wrappers, llama-swap, or `llama-server` runtime.
- Do not make performance claims.
- Do not touch `docs/needle-action-drafter/`.

## Source contracts scouted

### llama.cpp KV view/index contract

Source: `src/llama-kv-cache.cpp`

- `llama_kv_cache::get_k/get_v` create active attention views with `n_kv` tokens.
- Stream stride is still based on cache capacity from `get_size()`, not active `n_kv`.
- TBQ K/V uses a merged 3D view; non-TBQ compressed formats use head-explicit 4D views.
- For FlashAttention/non-transposed V, `set_input_k_idxs` and `set_input_v_idxs` both map rows as:

```text
absolute_global_row = sinfo.strm[s] * get_size() + sinfo.idxs[s][i]
```

### Triton parity contract

Sources:

- `experiments/compressed_kv_triton/llama_cpp_kv_layout.py`
- `experiments/compressed_kv_triton/llama_cpp_block_table_parity.py`
- `experiments/compressed_kv_triton/paged_materializers.py`
- `experiments/compressed_kv_triton/qkv_2d_tiled.py`
- `experiments/compressed_kv_triton/varlen_qkv.py`

Existing parity already covers:

- active `n_kv < cache_size`;
- nonzero stream windows via `stream_base`;
- non-monotonic block tables;
- tail blocks;
- cross-stream isolation;
- slot-info sequences that cannot be represented by fixed-size block tables;
- TBQ4, Planar3, and Iso3 sharing row-mapping semantics while keeping separate format/domain decode.

### C++ materializer contract

Source: `ggml/src/ggml-cuda/fattn-compressed-kv.cuh`

- Format traits own compressed decode and domain metadata.
- The current production C++ mapper is `ggml_cuda_fattn_contiguous_row_mapper` with `row_ptr(logical_row)`.
- `ggml_cuda_fattn_materialize_compressed_rows_f16` accepts a mapper object and must not know whether rows are contiguous or paged.

## Adapter contract before code

A future paged adapter must provide the same narrow primitive as the contiguous mapper:

```c++
struct ggml_cuda_fattn_block_table_row_mapper {
    const char * base_ptr;        // pre-biased to tensor/view base
    const int32_t * block_table;  // physical block ids
    int64_t row_stride_bytes;     // one compressed physical row
    int32_t block_size;           // rows per page/block
    int32_t rows;                 // active logical rows

    __device__ const char * row_ptr(int logical_row) const;
};
```

Required behavior:

1. `logical_row` must be within active rows: `0 <= logical_row < rows`.
2. `block_id = logical_row / block_size`.
3. `in_block = logical_row % block_size`.
4. `physical_slot = block_table[block_id] * block_size + in_block`.
5. `row_ptr(logical_row) = base_ptr + physical_slot * row_stride_bytes`.
6. Invalid metadata must be rejected before kernel launch where possible.
7. Out-of-range rows inside a tile must materialize zero through existing `valid_rows`, not by reading invalid block-table entries.

## Offset contract

The C++ adapter must preserve the byte-offset identity modeled by the Triton parity lane:

```text
byte_offset = stream_stride * absolute_stream
            + head_stride   * kv_head
            + token_stride  * physical_slot
```

Important distinction:

- `token_stride` and active bounds use `n_kv` semantics in the attention view.
- `stream_stride` and set-rows global indices use cache capacity from `get_size()`.

Do not conflate `n_kv` with cache size.

## Responsibility split

- Row mapper owns logical row to physical slot/pointer mapping.
- Format traits own compressed row decode and format-domain metadata.
- TBQ4 launchers own explicit FWHT-domain Q/O rotation.
- Planar/Iso launchers remain original-domain.
- Backend owns tile shape, masks, synchronization, MMA/WMMA, online softmax, and reductions.
- Dispatch remains owned by `fattn.cu` and unchanged by this design slice.

## Tests/contracts before future production code

Before adding a production C++ paged mapper, the following must pass or be extended:

1. `llama_cpp_block_table_parity.py` proves byte-offset identity for non-monotonic/tail/multi-stream cases.
2. `paged_row_mapping_contract.py` proves logical-row to physical-row mapping independent of format decode.
3. `paged_materializers.py` proves compressed-row decode over paged physical rows.
4. `qkv_2d_tiled.py` and `varlen_qkv.py` prove tiled attention can consume paged materializers.
5. `dispatch_policy_contract.py` proves default VEC and env-gated WMMA policy remains unchanged.

Any new C++ metadata shape should first be mirrored in the Python parity lane, then ported only after the parity lane passes.

## Promotion phases

1. **Design-only checkpoint**: this document; no production behavior change.
2. **Parity extension**: add missing edge cases only if source scouting finds a gap.
3. **Header-only prototype**: add a disabled/unused C++ mapper type next to the contiguous mapper; no dispatch path uses it.
4. **Opt-in integration**: wire the mapper only behind an explicit experimental gate after parity/build validation.
5. **Runtime smoke**: run one-server-at-a-time smokes only if dispatch/materialization behavior changes.

## Validation ladder for this slice

Because this slice is documentation-only, required validation is:

```bash
experiments/compressed_kv_triton/run_all.sh
/home/mrtrent/miniconda3/envs/LLM/bin/python experiments/compressed_kv_triton/run_all_json.py
scripts/hip/check-compressed-kv-fa-invariants.sh
git diff --check
```

No C++ build or server smoke is required unless a future slice changes production C++ or runtime dispatch/materialization.

## Review checklist

- No production source changed.
- No CMake/wrapper/server Triton dependency introduced.
- Default route remains VEC.
- WMMA gates remain opt-in.
- Mixed `TBQ4_0/Q8_0` remains VEC.
- Mixed Planar/Iso remains rejected by default.
- `n_kv` and cache capacity remain distinct.
- `docs/needle-action-drafter/` remains untouched.
