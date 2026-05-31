# Sync Reduction Gates — 2026-05-31

## Goal

Follow-up to Track A and backend token-only readback work. The recurring full-vocab D2H readback was removed in the gated no-spec path, leaving backend synchronization count as the main visible bottleneck.

## Experimental gates added

All are opt-in and default-off:

```bash
LLAMA_SKIP_REDUNDANT_SYNCHRONIZE=1
GGML_SCHED_SKIP_INPUT_COPY_PRE_SYNC=1
GGML_SCHED_ASYNC_INPUT_COPY=1
```

Used with the existing gated token-only backend sampling path:

```bash
LLAMA_ARG_BACKEND_SAMPLING=1
LLAMA_BACKEND_GREEDY_FASTPATH=1
LLAMA_BACKEND_SAMPLING_TOKEN_ONLY=1
```

### `LLAMA_SKIP_REDUNDANT_SYNCHRONIZE`

Skips `llama_context::synchronize()` when there is no queued work:

```text
n_queued_tokens == 0 && t_compute_start_us == 0
```

This removes redundant getter-triggered context synchronizations after an explicit synchronize has already completed.

### `GGML_SCHED_SKIP_INPUT_COPY_PRE_SYNC`

In the ggml scheduler input-copy path, skips the pre-copy backend synchronize for graph input tensors when no scheduler event is available.

Rationale: in the measured server decode loop, each token is synchronized before the next decode step, so the pre-copy stream sync is redundant for this narrow path.

### `GGML_SCHED_ASYNC_INPUT_COPY`

Uses backend `set_tensor_async` for host graph-input copies when available, avoiding synchronous `buffer_set_tensor` for several tiny recurring inputs.

## Verification runs

All runs used:

```bash
MODE=perf CASES=nospec N_PREDICT=64
LLAMA_ARG_BACKEND_SAMPLING=1
LLAMA_BACKEND_GREEDY_FASTPATH=1
LLAMA_BACKEND_SAMPLING_TOKEN_ONLY=1
GGML_CUDA_COPY_SYNC_TRACE=1
GGML_CUDA_COPY_SYNC_TRACE_FLUSH_EVERY=128
```

### Token-only baseline

Artifact:

```text
.harness/tmp/backend-token-only-nospec-20260531-071750/
```

Result:

```text
predicted_per_second=32.66251002203188
ggml_backend_cuda_synchronize=994 calls / 64 generated
D2H greedy_argmax=64 calls, 256 bytes total
```

### Skip redundant context synchronize

Artifact:

```text
.harness/tmp/skip-redundant-sync-nospec-20260531-072024/
```

Result:

```text
predicted_per_second=33.09036046777361
ggml_backend_cuda_synchronize=929 calls / 64 generated
D2H greedy_argmax=64 calls, 256 bytes total
```

### Skip input pre-sync + async input copy

Artifact:

```text
.harness/tmp/async-input-copy-nospec-20260531-072223/
```

Result:

```text
predicted_per_second=32.25534947411181
ggml_backend_cuda_synchronize=469 calls / 64 generated
D2H greedy_argmax=64 calls, 256 bytes total
```

Input-copy trace changed several tiny H2D rows from synchronous buffer copies to async backend copies:

```text
attn_inp_kq_mask: ggml_backend_cuda_set_tensor_async, syncs=0
inp_pos:          ggml_backend_cuda_set_tensor_async, syncs=0
attn_inp_k_idxs:  ggml_backend_cuda_set_tensor_async, syncs=0
attn_inp_v_idxs:  ggml_backend_cuda_set_tensor_async, syncs=0
inp_out_ids:      ggml_backend_cuda_set_tensor_async, syncs=0
logit_bias:       ggml_backend_cuda_set_tensor_async, syncs=0
logit_idxs:       ggml_backend_cuda_set_tensor_async, syncs=0
```

## Conclusion

The gates reduce visible sync/copy synchronization substantially:

```text
994 -> 929 -> 469 ggml_backend_cuda_synchronize calls / 64 generated tokens
```

However, measured throughput did not improve materially in these short server traces and became slightly worse with async input copy. Keep these gates experimental/off by default. The remaining performance bottleneck is likely not just host-visible synchronization count; next work should profile kernel timeline/occupancy after token-only readback removal.
