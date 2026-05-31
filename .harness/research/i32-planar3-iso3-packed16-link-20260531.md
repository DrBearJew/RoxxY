# I32 packed16 DOT4 + planar3_0 / iso3_0 V link — 2026-05-31

## Goal

Link the remaining 3-bit V formats into the I32 packed16 DOT4/MMQ route, keeping K as physical I32 packed16 and omitting logical K CLI flags in runtime smokes.

## Code changes

Changed:

```text
ggml/src/ggml-cuda/fattn-packed16-dot4-mmq.cuh
ggml/src/ggml-cuda/fattn.cu
```

### `fattn-packed16-dot4-mmq.cuh`

Added V variants:

```text
PACKED16_DOT4_MMQ_V_PLANAR3_0
PACKED16_DOT4_MMQ_V_ISO3_0
```

Extended the packed16 DOT4/MMQ V predicate to admit:

```text
planar3_0
iso3_0
```

Added scalar V decoders:

```text
pdmq_decode_v_planar3_0
pdmq_decode_v_iso3_0
```

Implementation notes:

- Planar3 unpacks 2-bit payload + 1-bit sign and applies inverse 2D Givens rotation using `PI_COS` / `PI_SIN`.
- Iso3 unpacks the same 3-bit layout and applies inverse quaternion rotation using `PI_QW/QX/QY/QZ`.
- Both use `PI_CENTROIDS_3BIT` and multiply by the per-block f16 norm.
- Unlike TBQ4, no output-wide inverse rotation is needed because planar/iso inverse rotation is local during V dequantization.
- `nq==1` support is enabled for these V types so decode does not fall into q8k DOT4 KQ, whose V loaders do not support planar3/iso3.

### `fattn.cu`

Added `v_requires_dot4_mmq` for:

```text
tbq4_0
planar3_0
iso3_0
```

For those V types:

- `nq==1` routes to `rocm_packed16_dot4_mmq` when supported.
- packed16 WMMA auto-selection is bypassed because the WMMA packed16 path is not wired for these V loaders.
- route labels distinguish:
  - `dot4_mmq_tbq4`
  - `dot4_mmq_planar3`
  - `dot4_mmq_iso3`

## Build verification

Build passed:

```bash
cmake --build build-rocm-fixed --target llama-server -j 8
```

Build log:

```text
.harness/tmp/i32-planar3-iso3-link-rebuild-20260531-101049.log
```

## Runtime smoke

Artifact:

```text
.harness/tmp/i32-planar3-iso3-packed16-clean-20260531-101613
```

Both launches omitted K cache-type CLI flags and used only `--cache-type-v <type>`.

### planar3_0

```json
{
  "k_cli_omitted": true,
  "ok_json": true,
  "predicted_per_second": 35.42040702475222,
  "predicted_n": 32,
  "draft_n": 27,
  "draft_n_accepted": 22,
  "route_count": 259,
  "errors": []
}
```

Route tail:

```text
selected=rocm_packed16_dot4_mmq K=i32 V=planar3_0
```

### iso3_0

```json
{
  "k_cli_omitted": true,
  "ok_json": true,
  "predicted_per_second": 39.90706641907657,
  "predicted_n": 32,
  "draft_n": 25,
  "draft_n_accepted": 22,
  "route_count": 192,
  "errors": []
}
```

Route tail:

```text
selected=rocm_packed16_dot4_mmq K=i32 V=iso3_0
```

No lingering processes after the smokes:

```text
no llama-server
no llama-bench
```

## Prior failed smoke and fix

First planar3/iso3 smoke selected packed16 DOT4/MMQ for prefill but crashed when `decode_qk nq=1` fell into `rocm_q8k_dot4_kq`, whose V assertion only accepts f16/q4_0/q8_0. Fix was to let planar3/iso3 use packed16 DOT4/MMQ for `nq==1`, same as TBQ4.

## Status

Linked and smoke-tested. Still experimental pending a dedicated reference correctness canary for I32 packed16 + planar3/iso3 V across more shapes.
