# I32 packed16 DOT4 + q4_0 V route-require fix — 2026-05-31

## Goal

Validate the intended q4 V target for the packed16 I32 K FlashAttention lane:

```text
QK: packed16 I32 K + on-the-fly int8 Q, DOT4 accumulation
PV: q4 V decode feeding P @ V only
out: f32 accumulation
```

Do not describe the target as i16 V.  The useful target is packed q4 V payload plus f16 scales:

```text
32 V values = 16B q4 payload + 2B f16 scale = 18B = 4.5 bits/value
K packed16/i8 = 34B/32 = 8.5 bits/value
combined K+V = 13 bits/value-pair
```

## Root cause

Before this fix, `GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq` did not fully force q4_0 decode onto the packed16 DOT4/MMQ kernel.

Two gates were responsible:

1. `ggml_cuda_packed16_dot4_mmq_supported(...)` returned false for `nq <= 1` unless V was one of the experimental V formats that had no q8k loader (`tbq4_0`, `planar3_0`, `iso3_0`).
2. `fattn.cu` only honored the route-require block for `Q->ne[1] > 1` or `V == tbq4_0`.

So q4_0 prefill selected:

```text
selected=rocm_packed16_dot4_mmq K=i32 V=q4_0
```

but q4_0 decode silently fell back to:

```text
selected=rocm_q8k_dot4_kq K=i32 V=q4_0
```

That made route-require logs ambiguous and prevented clean validation of q4 V entirely inside the packed16 DOT4/MMQ attention family.

## Fix

Changed:

```text
ggml/src/ggml-cuda/fattn-packed16-dot4-mmq.cuh
ggml/src/ggml-cuda/fattn.cu
README.md
```

### `fattn-packed16-dot4-mmq.cuh`

`nq <= 1` decode remains on the older BN64/split-K DOT4 path by default for established V types, but a route-require opt-in now allows packed16 DOT4/MMQ support for q4_0/q8_0/f16 decode too:

```text
if (Q->ne[1] <= 1 && !v_needs_mmq_decode && !ggml_cuda_packed16_dot4_mmq_route_required()) return false;
```

### `fattn.cu`

`GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq` is now a hard contract for the I32 packed16 branch, including `nq == 1` decode.

If the route is required but not supported, the code aborts instead of silently choosing another kernel.

### `README.md`

The top “PROPER STARTING OPTIONS” block now uses q4_0 as the primary V target and states the q4 architecture explicitly:

```text
--cache-type-v q4_0       # primary packed q4 V target: 18B/32 values = 4.5 bits/value
```

## Before/after smoke

### Before fix

Artifact:

```text
.harness/tmp/i32-q4v-packed16-proceed-20260531-102418
```

Summary:

```json
{
  "k_cli_omitted": true,
  "ok_json": true,
  "predicted_per_second": 30.475087568259433,
  "predicted_n": 32,
  "draft_n": 28,
  "draft_n_accepted": 21,
  "selected_counts": {
    "rocm_packed16_dot4_mmq": 275,
    "rocm_q8k_dot4_kq": 36
  },
  "errors": []
}
```

q8k fallback lines were all `nq=1` decode/none selections:

```text
selected=rocm_q8k_dot4_kq nq=1 K=i32 V=q4_0
```

### After fix

Artifact:

```text
.harness/tmp/i32-q4v-packed16-route-require-clean-20260531-103129
```

Summary:

```json
{
  "k_cli_omitted": true,
  "ok_json": true,
  "predicted_per_second": 44.21452889419463,
  "predicted_n": 32,
  "draft_n": 25,
  "draft_n_accepted": 22,
  "route_count": 191,
  "selected_counts": {
    "rocm_packed16_dot4_mmq": 191
  },
  "nq1_non_mmq": [],
  "errors": []
}
```

No K cache-type CLI flag was used.

No lingering processes after the run:

```text
no llama-server
no llama-bench
```

## Build/checks

Build passed:

```bash
cmake --build build-rocm-fixed --target llama-server -j 8
```

Build log:

```text
.harness/tmp/i32-q4v-route-require-rebuild-20260531-102608.log
```

DOT4 harness passed:

```bash
./build-rocm-fixed/bin/test-dot4-harness
```

Harness log:

```text
.harness/tmp/i32-q4v-route-require-dot4-harness-20260531-103151.log
```

Tail:

```text
═══ OVERALL: ALL PASS ═══
```

## Status

The route-require contract is fixed for q4_0 V on the I32 packed16 DOT4/MMQ path.  This validates q4_0 V in the packed16 DOT4/MMQ attention family for the smoke shape.

This is still not a new separate I32-sideband V cache type; it uses the existing q4_0 KV cache payload/scales while keeping QK on packed16 I32 K and PV inside the requested DOT4/MMQ kernel when route-required.
