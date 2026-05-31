# Backend Greedy Token-Only Fastpath — 2026-05-31

## Goal

Reduce no-spec decode host/device readback identified by Track A:

```text
result_output / sampled logits D2H ~= 993,280 bytes per eval
```

## Experimental gates

This patch is opt-in only and requires backend sampling to be enabled:

```bash
LLAMA_ARG_BACKEND_SAMPLING=1
LLAMA_BACKEND_GREEDY_FASTPATH=1
LLAMA_BACKEND_SAMPLING_TOKEN_ONLY=1
```

Behavior:

- `LLAMA_BACKEND_GREEDY_FASTPATH=1`
  - for deterministic `temperature <= 0`, `mirostat == 0`, `n_probs == 0`, no grammar/reasoning budget, no active repetition/DRY/adaptive/top-n-sigma penalties;
  - builds a simplified backend sampler chain: existing logit-bias sampler, then backend greedy argmax;
  - skips unsupported ROCm TOP_K/ARGSORT filter samplers that cannot change the greedy argmax.
- `LLAMA_BACKEND_SAMPLING_TOKEN_ONLY=1`
  - when backend sampling produced `data.sampled`, only the sampled token is copied back;
  - suppresses sampled logits/probs/candidates readback payloads.

## Verification

Build:

```bash
cmake --build build-rocm-fixed -j2 --target llama-server
```

Server no-spec trace:

```text
.harness/tmp/backend-token-only-nospec-20260531-071750/
```

Command shape:

```bash
OUT_DIR=.harness/tmp/backend-token-only-nospec-20260531-071750 \
MODE=perf CASES=nospec N_PREDICT=64 BASE_PORT=19230 \
EXTRA_ENV="LLAMA_ARG_BACKEND_SAMPLING=1 LLAMA_BACKEND_GREEDY_FASTPATH=1 LLAMA_BACKEND_SAMPLING_TOKEN_ONLY=1 GGML_CUDA_COPY_SYNC_TRACE=1 GGML_CUDA_COPY_SYNC_TRACE_FLUSH_EVERY=128 GGML_CUDA_COPY_SYNC_TRACE_FILE=.../copy-sync.tsv" \
benchmarks/mtp-speed-probe.sh
```

Result:

```text
predicted_n=64
predicted_per_second=32.66251002203188
errors=""
```

## Copy/sync result

Before token-only fastpath, backend sampling still copied full sampled logits:

```text
copy D2H ggml_backend_cuda_get_tensor_async node_3678 64 calls 63,569,920 bytes avg 993,280
copy D2H ggml_backend_cuda_get_tensor_async greedy_argmax 64 calls 256 bytes avg 4
```

After token-only fastpath:

```text
copy D2H ggml_backend_cuda_get_tensor_async greedy_argmax 64 calls 256 bytes avg 4
```

The recurring full-vocab D2H readback was eliminated for the gated deterministic no-spec path.

## Remaining bottleneck

The run still shows high synchronization count:

```text
ggml_backend_cuda_synchronize: 994 calls / 64 generated tokens
```

Throughput did not materially improve in this short trace, so the next target is sync reduction rather than D2H bandwidth.
