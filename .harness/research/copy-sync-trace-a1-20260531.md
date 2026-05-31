# Copy/Sync Trace A1 — RDNA3 target decode

## Scope

Temporary env-gated instrumentation was added in `ggml/src/ggml-cuda/ggml-cuda.cu` to aggregate CUDA/HIP backend copies and synchronizations.

Enable with:

```bash
GGML_CUDA_COPY_SYNC_TRACE=1
GGML_CUDA_COPY_SYNC_TRACE_FILE=/path/to/copy-sync.tsv
```

Columns:

```text
kind, direction, site, tensor, calls, bytes, syncs, avg_bytes
```

## Verification run

Command shape:

```bash
GGML_CUDA_COPY_SYNC_TRACE=1 \
GGML_CUDA_COPY_SYNC_TRACE_FILE=.harness/tmp/.../copy-sync.tsv \
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
GGML_CUDA_ROCM_Q8K_DOT4_KQ_AUTO=1 \
build-rocm-fixed/bin/llama-bench \
  -m /mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf \
  --device ROCm0 -fa 1 -ctk f16 -ctv q4_0 \
  -b 2048 -ub 1024 -p 13 -n 64 -r 1 -o json
```

Artifact directory:

```text
.harness/tmp/copy-sync-trace-llama-bench-20260531-065759/
```

Benchmark result:

```text
n_prompt=13 avg_ts=169.17 tok/s
n_gen=64   avg_ts=32.63 tok/s
```

## Important caveat

The top H2D rows by bytes are one-time model load copies, not decode-loop copies. For decode/runtime work, sort by call count and ignore one-time `ggml_backend_cuda_buffer_set_tensor` model weight rows.

## Top recurring rows by calls

```text
sync stream ggml_backend_cuda_synchronize (unnamed): calls=810 syncs=810
D2H  result_output via ggml_backend_cuda_get_tensor_async: calls=67 bytes=66,549,760 avg=993,280
H2D  ROCm0#model.input_embed#0 via ggml_backend_cuda_buffer_set_tensor: calls=67 bytes=1,863,680 avg=27,816 syncs=67
H2D  ROCm0#attn_inp_kq_mask#0 via ggml_backend_cuda_buffer_set_tensor: calls=67 bytes=93,184 avg=1,391 syncs=67
H2D  ROCm0#leaf_55#0 via ggml_backend_cuda_buffer_set_tensor: calls=67 bytes=1,456 avg=21.7 syncs=67
H2D  ROCm0#leaf_60#0 via ggml_backend_cuda_buffer_set_tensor: calls=67 bytes=728 avg=10.9 syncs=67
H2D  ROCm0#leaf_61#0 via ggml_backend_cuda_buffer_set_tensor: calls=67 bytes=728 avg=10.9 syncs=67
H2D  ROCm0# (view)#0 via ggml_backend_cuda_buffer_set_tensor: calls=67 bytes=268 avg=4.0 syncs=67
H2D  ROCm0#leaf_1001#0 via ggml_backend_cuda_buffer_set_tensor: calls=65 bytes=260 avg=4.0 syncs=65
```

## Decode-loop interpretation

For the 64-token generation benchmark, the recurring pattern is approximately:

```text
~7-8 H2D input/metadata copies per decode step
~29 KiB H2D per decode step
~1 D2H result_output copy per decode step, ~970 KiB each
~12 backend stream synchronizations per decode step
```

The largest recurring transfer is not H2D; it is:

```text
result_output D2H ~= 993 KiB/eval
```

The most suspicious sync source is:

```text
ggml_backend_cuda_synchronize: 810 calls for this run
```

## Immediate optimization targets

1. **Logits/output readback volume**
   - `result_output` copies ~993 KiB per eval.
   - Check whether no-spec greedy/server route must read full logits every token or can reduce to sampled-token/top-k path.

2. **Synchronous graph input copies**
   - `ggml_backend_cuda_buffer_set_tensor` rows show small H2D inputs are synchronous (`syncs == calls`).
   - Candidate inputs: token/embedding selection, KQ mask, pos/out ids/small metadata.
   - Next step: map `leaf_*` names to graph inputs by naming `build_inp_pos`, `build_inp_out_ids`, and related graph input tensors or adding higher-level `ggml_backend_tensor_set` trace.

3. **Backend synchronize fan-out**
   - `ggml_backend_cuda_synchronize` dominates sync count.
   - Next step: instrument caller path in `ggml_backend_sched_synchronize()` / `llama_context::synchronize()` to distinguish mandatory sampling/logits sync from redundant scheduler/backend syncs.

## Acceptance status

```text
copies/token identified: yes, approximately 7-8 recurring H2D copies/eval plus 1 D2H result copy/eval
syncs/token identified: yes, approximately 12 stream synchronizations/eval in llama-bench trace
top call sites identified: yes
```
