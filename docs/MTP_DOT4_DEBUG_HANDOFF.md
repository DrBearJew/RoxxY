# MTP / DOT4 FlashAttention Debug Handoff

## Status

- [DONE:3] DOT4 was forced successfully with:
  - `GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq`
  - `GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1`
  - `GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER=1`
  - observed route: `selected=583 name=rocm_q8k_dot4_kq`.
- [DONE:5] The harness already computes both outputs plus `max_abs_diff`, `mean_abs_diff`, NaN/Inf, norms, cosine, and first-8 values. The first forced run was invalid because the reference leg inherited the forced DOT4 route; the harness was patched to isolate DOT4 and REF environments.
- [DONE:7] QK logit debug prints were added for DOT4 recthist-v4 single kernel under `GGML_CUDA_DOT4_DEBUG=1`.
- [DONE:8] Softmax `m/l` debug prints were added under `GGML_CUDA_DOT4_DEBUG=1`.
- [DONE:9] Attention output `O` / final `dst[0..7]` debug prints were added under `GGML_CUDA_DOT4_DEBUG=1`.
- [DONE:10] Harness skeleton exists at `tests/test-dot4-harness.cpp` and now supports `DOT4_HARNESS_TEST=N`.
- [DONE:11] Debug guards were added in `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu`.
- [PENDING:12] Rebuild and run test 1 with debug enabled.
- [PENDING:15] Move to test 2 only after test 1 passes or the first divergence is identified.

## Important caveat

I stopped before rebuilding after adding the debug-guard parameter to `ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel`. Build may expose a typo; fix compile errors before trusting runtime results.

## Files touched for this handoff

- `tests/test-dot4-harness.cpp`
  - Fixed f16 data generation to use `ggml_fp32_to_fp16_row` instead of bfloat-like bit truncation.
  - Fixed causal mask conversion to real fp16 `-inf`.
  - DOT4 leg now sets force envs.
  - REF leg now clears route contracts and disables q8k DOT4.
  - Added `DOT4_HARNESS_TEST=N` filter.
- `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu`
  - Added `GGML_CUDA_DOT4_DEBUG=1` host flag.
  - Added debug parameter to the recthist-v4 single kernel.
  - Added device prints for QK logits, softmax `m/l`, and output `dst`.

## Rebuild

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github
cmake --build build-rocm-fixed --target test-dot4-harness -j 12 2>&1 | tee /tmp/dot4-harness-build.log
```

If the build fails, inspect only the first compile error. The likely class of issue is a missed launch call after adding the `debug` kernel argument.

## Test 1: recthist, nq=2, f16 V, debug on

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github
DOT4_HARNESS_TEST=1 \
GGML_CUDA_DOT4_DEBUG=1 \
COMPRESSED_KV_FATTN_LOG=1 \
./build-rocm-fixed/bin/test-dot4-harness 2>&1 | tee /tmp/dot4-test1-debug.log
```

Expected route for the DOT4 leg:

```text
FATTN COMPUTE SELECT selected=583 name=rocm_q8k_dot4_kq
fa_dot4_launch: fa_inst=4 nq=2 nk=256 d=256 K=f16 V=f16
```

Expected route for the REF leg:

```text
selected=100 name=vec
# or selected=200 tile / selected=300 wmma_f16 depending on shape
# but NOT selected=583 rocm_q8k_dot4_kq
```

Expected debug lines:

```text
dot4_debug_qk: q=0 h=0 k=0 logit=... scale=... q_offset=...
dot4_debug_ml: q=0 h=0 k0=... m=... l=...
dot4_debug_o: q=0 h=0 d=0 m=... l=... dst=... bad=0
```

## Interpretation

Use this decision tree:

1. DOT4 route is not selected in DOT4 leg
   - Check `GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq` is being set inside harness.
   - Check `GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER=1` is set for source-f16 K.
2. REF route is still DOT4
   - Harness isolation failed; REF leg must set `GGML_CUDA_ROCM_Q8K_DOT4_KQ=0` and unset `GGML_CUDA_FA_ROUTE_REQUIRE`.
3. QK debug diverges from CPU reference
   - Suspect Q quant scale, K packed16 scale/layout, attention scale, or q/k head indexing.
4. QK looks sane, `m/l` bad
   - Suspect softmax denominator update, masked logits included/excluded incorrectly, or causal `q_offset`.
5. QK and `m/l` sane, output bad
   - Suspect V stride/layout, V f16 load, V accumulation, or final dst layout.
6. `bad=1` in `dot4_debug_o`
   - Treat as hard failure: NaN/Inf, `l <= 0`, or insane output magnitude.

## Test progression

Only proceed in this order:

```bash
# 1. recthist nq=2, V=f16
DOT4_HARNESS_TEST=1 GGML_CUDA_DOT4_DEBUG=1 ./build-rocm-fixed/bin/test-dot4-harness

# 2. recthist nq=4, V=f16
DOT4_HARNESS_TEST=2 GGML_CUDA_DOT4_DEBUG=1 ./build-rocm-fixed/bin/test-dot4-harness

# 3. recthist nq=2, nk=1024, V=f16
DOT4_HARNESS_TEST=3 GGML_CUDA_DOT4_DEBUG=1 ./build-rocm-fixed/bin/test-dot4-harness

# 4. decode nq=1, nk=1024, V=f16
DOT4_HARNESS_TEST=4 GGML_CUDA_DOT4_DEBUG=1 ./build-rocm-fixed/bin/test-dot4-harness

# 5. decode nq=1, nk=64, V=f16
DOT4_HARNESS_TEST=5 GGML_CUDA_DOT4_DEBUG=1 ./build-rocm-fixed/bin/test-dot4-harness
```

Do not move to q8/q4 V until all f16 V cases are clean.

## Pass condition for test 1

- DOT4 leg selected `rocm_q8k_dot4_kq`.
- REF leg selected non-DOT4.
- No NaN/Inf.
- `dot4_debug_o ... bad=0` for all first 8 printed dims.
- `max_abs_diff < 1e-2` preferred; investigate if `max_abs_diff >= 1e-1`.
- CPU reference first-8 values agree with REF first-8; if CPU is off, fix harness layout before kernel debugging.
