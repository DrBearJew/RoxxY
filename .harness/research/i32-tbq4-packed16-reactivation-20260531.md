# I32/TBQ4 packed16 DOT4 reactivation note — 2026-05-31

## Intent

Reactivate I32 packed16 K + TBQ4 V as an explicit experimental option, without changing the established packed16 default route for V=f16/q8_0/q4_0.

## Safety

No model server or bench process was running before or after the change.

```text
no llama-server
no llama-bench
```

TBQ4 V remains opt-in behind:

```bash
GGML_CUDA_ROCM_PACKED16_TBQ4_V=1
```

Without that env, K=I32 + V=TBQ4 still fails the packed16 V-type gate instead of silently entering an untested route.

## Code changes

Changed:

```text
ggml/src/ggml-cuda/fattn-packed16-dot4-mmq.cuh
ggml/src/ggml-cuda/fattn.cu
```

### `fattn-packed16-dot4-mmq.cuh`

Added optional `PACKED16_DOT4_MMQ_V_TBQ4_0` support:

- `GGML_CUDA_ROCM_PACKED16_TBQ4_V=1` opt-in helper.
- V-type predicate now admits TBQ4 only when the opt-in is set.
- Scalar TBQ4 V decoder mirrors existing q8tbq4 prefill dequant:
  - `block_tbq4_0`
  - `QK_TBQ4`
  - `d_tbq4_centroids[idx] * d`
- Host dispatch now has TBQ4 switch cases for GQA1 and GQA2 launch macros.
- Output is inverse-rotated after the packed16 DOT4 MMQ kernel when `V->type == GGML_TYPE_TBQ4_0`, matching the existing q8tbq4 DOT4 prefill path.
- For TBQ4 only, the support predicate allows `nq == 1` so decode/small-Q can use this kernel rather than falling into q8k_dot4_kq, which does not accept TBQ4 V.

### `fattn.cu`

The I32 packed16 gate now uses the shared packed16 V-type predicate instead of a hardcoded `{f16,q8_0,q4_0}` list.

Route selection changes:

- `GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq` can select TBQ4 when the TBQ4 opt-in is set.
- `nq == 1` with TBQ4 V selects `BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ` when supported, instead of the q8k decode route.
- TBQ4 V forces the DOT4 MMQ route and does not auto-select packed16 WMMA, because packed16 WMMA has not been extended to TBQ4 V.

## Verification

Build passed:

```bash
cmake --build build-rocm-fixed --target llama-server -j 8
```

Build log:

```text
.harness/tmp/i32-tbq4-packed16-build-20260531-092809.log
```

Existing DOT4 correctness harness passed after the patch:

```bash
build-rocm-fixed/bin/test-dot4-harness
```

Result:

```text
═══ OVERALL: ALL PASS ═══
```

## Runtime smoke

Clean full-model/runtime smoke after the correction to omit logical K CLI flags:

```text
.harness/tmp/i32-tbq4-packed16-clean-runtime-20260531-095107
```

Command env:

```bash
LLAMA_MTP_ENABLE_FA=1
LLAMA_MTP_PREFILL_CHUNK=1024
LLAMA_MTP_PREFILL_FORCE_MMQ=1
GGML_CUDA_ROCM_QUANT_PREFILL_F16=1
LLAMA_MTP_DISABLE_PACKED16_FA=0
GGML_CUDA_ROCM_PACKED16_TBQ4_V=1
GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=1
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq
COMPRESSED_KV_FATTN_LOG=1
```

Server args intentionally omit a K cache-type CLI flag. The captured launch args are:

```text
--device ROCm0
-m MODEL
--host 127.0.0.1 --port PORT
--no-webui --jinja --chat-template-file TEMPLATE
--flash-attn on --cache-type-v tbq4_0
--ctx-size 40960 --batch-size 1024 --ubatch-size 1024
--parallel 1 --no-warmup
--spec-type draft-mtp --spec-default --spec-draft-n-max 3 --spec-draft-p-min 0
--spec-draft-prio 2 --spec-draft-prio-batch 2
```

Result:

```json
{
  "k_cli_omitted": true,
  "ok_json": true,
  "predicted_per_second": 45.23581998405437,
  "predicted_n": 32,
  "draft_n": 25,
  "draft_n_accepted": 22,
  "acceptance_log_last": ["0.88000", "22", "25"],
  "errors": []
}
```

Selector evidence includes repeated successful I32/TBQ4 packed16 DOT4 selection:

```text
fa_final_select: inst=none       selected=rocm_packed16_dot4_mmq nq=1 nk=256 d=256 K=i32 V=tbq4_0 final=1
fa_final_select: inst=none       selected=rocm_packed16_dot4_mmq nq=4 nk=256 d=256 K=i32 V=tbq4_0 final=1
fa_final_select: inst=prefill_qk selected=rocm_packed16_dot4_mmq nq=2 nk=256 d=256 K=i32 V=tbq4_0 final=1
```

No runtime signatures were found for:

```text
I32 K rejected
GGML_ASSERT
fatal error
ABORT
unsupported V type
required rocm_packed16_dot4_mmq route was not selected
failed to decode
failed to process speculative batch
```

Post-run process check:

```text
no llama-server
no llama-bench
```

Earlier runtime artifacts that used stale logical-K CLI framing are not acceptance artifacts for this lane.

## 5x128 sustained smoke

A 5-request, 128-token-per-request clean run was executed with the same K-CLI-omitted launch shape:

```text
.harness/tmp/i32-tbq4-packed16-clean-5x128-20260531-095502
```

Summary:

```json
{
  "k_cli_omitted": true,
  "n_requests": 5,
  "tok_s_median": 48.934227810239186,
  "tok_s_mean": 48.97523035427806,
  "tok_s_min": 39.79717122950562,
  "tok_s_max": 57.91042885387118,
  "accept_rate_median": 0.8548387096774194,
  "accept_rate_mean": 0.8391054636442346,
  "route_i32_tbq4_count": 3098,
  "errors": []
}
```

Per-request tok/s:

```text
41.89
39.80
56.35
57.91
48.93
```

Per-request draft acceptance:

```text
91/106 = 0.8585
97/127 = 0.7638
105/119 = 0.8824
106/124 = 0.8548
102/122 = 0.8361
```

Route tail remained on:

```text
selected=rocm_packed16_dot4_mmq K=i32 V=tbq4_0
```

No runtime signatures were found for I32 rejection, assert, fatal, abort, unsupported V type, route-require failure, decode failure, or speculative batch failure.

## Still experimental

The route no longer crashes, repeatedly selects `K=i32 V=tbq4_0`, and sustained median was ~48.93 tok/s in this 5x128 smoke. It is still experimental until a dedicated correctness/reference canary compares I32/TBQ4 output quality across more shapes.
