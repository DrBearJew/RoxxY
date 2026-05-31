# ROCprof token-only no-spec profile — 2026-05-31

## Scope

Profiled canonical `llama-server` no-spec decode after the gated token-only backend sampling path removed recurring full-vocab logits D2H readback.

Runtime gates used for the successful ROCprof server run:

```bash
LLAMA_ARG_BACKEND_SAMPLING=1
LLAMA_BACKEND_GREEDY_FASTPATH=1
LLAMA_BACKEND_SAMPLING_TOKEN_ONLY=1
LLAMA_SKIP_REDUNDANT_SYNCHRONIZE=1
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1
GGML_CUDA_ROCM_Q8K_DOT4_KQ=1
GGML_CUDA_ROCM_Q8K_DOT4_KQ_AUTO=1
LLAMA_MTP_PREFILL_CHUNK=1024
```

Artifact:

```text
.harness/tmp/rocprof-token-only-server-20260531-072532/
```

Server timing under profiler overhead:

```text
prompt eval time = 245.51 ms / 5 tokens = 20.37 tok/s
eval time        = 1108.42 ms / 32 tokens = 28.87 tok/s
```

## ROCprof domain stats

```text
HIP_API          5754.08 ms, 62.69%, 26894 calls
MEMORY_COPY      2438.27 ms, 26.56%, 1121 calls
KERNEL_DISPATCH   986.24 ms, 10.74%, 65779 calls
```

Note: this run includes model load and first-request reserve, so memory-copy totals are dominated by model/load-time H2D. Decode-loop copy/sync facts should still use Track A's copy-sync trace.

## Top kernel dispatches by total GPU duration

```text
37.03% 365.20 ms calls=3584 avg=101.90 us  mul_mat_vec_q<type=12, ncols=1, fused=true>
17.29% 170.53 ms calls=8192 avg=20.82 us   mul_mat_vec_q<type=12, ncols=1, fused=false>
 9.40%  92.67 ms calls=1024 avg=90.50 us   mul_mat_vec_q<type=14, ncols=1, fused=true>
 8.11%  79.94 ms calls=1057 avg=75.63 us   mul_mat_vec_q<type=14, ncols=1, fused=false>
 5.79%  57.13 ms calls=432  avg=132.25 us  mul_mat_vec_q<type=12, ncols=4, fused=false>
 3.35%  33.05 ms calls=432  avg=76.50 us   mul_mat_vec_q<type=12, ncols=2, fused=false>
 3.05%  30.10 ms calls=14849 avg=2.03 us   quantize_q8_1
 2.24%  22.13 ms calls=3297 avg=6.71 us    k_get_rows_float
 2.01%  19.87 ms calls=4385 avg=4.53 us    rms_norm_f32
 0.93%   9.13 ms calls=1632 avg=5.59 us    gated_delta_net_cuda
```

The dominant GPU time remains MMVQ quantized matvec, especially type 12 and type 14. This supports returning to the planned RDNA3 Q4_K/Q6_K matvec kernel work rather than further optimizing logits readback.

## HIP API stats

```text
hipMemcpyAsync        4137.80 ms, 71.91%, 1541 calls
hipStreamSynchronize  1162.35 ms, 20.20%, 1753 calls
hipLaunchKernel        214.52 ms,  3.73%, 9958 calls
```

Again, this is full process profile including model loading; Track A per-token traces remain the better source for recurring decode-loop copy volume.

## Safety issue found and fixed

While trying attach-style profiling, a prompt-only `n_predict=1` completion with `LLAMA_SKIP_REDUNDANT_SYNCHRONIZE=1` and backend sampling could consume a stale async sampled-token D2H result and crash in `common_token_to_piece()` with an out-of-range token.

Repro artifact before fix:

```text
.harness/tmp/fastpath-warmup-repro-20260531-072814/
```

Fix:

```text
src/llama-context.cpp
```

`LLAMA_SKIP_REDUNDANT_SYNCHRONIZE` no longer skips synchronization when backend sampling output buffers exist (`sampling.sampled.has_data()`). This preserves correctness for async sampled-token D2H copies.

Verification after fix:

```text
.harness/tmp/fix-warmup-skip-20260531-072952/
PROMPT=warmup N_PREDICT=1 with LLAMA_SKIP_REDUNDANT_SYNCHRONIZE=1: PASS
```

Combined token-only + scheduler gates after fix:

```text
.harness/tmp/fix-combined-gates-nospec-20260531-073011/
predicted_per_second=32.36198498302766
ggml_backend_cuda_synchronize=534 calls / 64 generated
D2H greedy_argmax=64 calls, 256 bytes total
```

## Conclusion

After removing full-vocab D2H readback and reducing visible syncs, throughput remains roughly flat. The best next optimization target is the existing B2 plan: RDNA3-specific quantized matvec work for `mul_mat_vec_q` (Q4_K first, Q6_K next), because ROCprof shows MMVQ dominates GPU dispatch time.
