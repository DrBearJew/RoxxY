# Validation — correctness and benchmark plan

## Correctness
- Compare three paths on same prompt/model:
  1. non-fused TBQ4 baseline
  2. new rocWMMA TBQ4 fused path
  3. q8_0 or f16 KV reference if available
- First validate short contexts: `512`, `2048`, `4096`.
- Check no NaN, no divergent repetition, and close logits/output text.

## Build
```bash
cd /tmp/llama.cpp-mtp
cmake --build build-rocm-tq --target llama-server -j8
```

## Runtime benchmark
Use safe prefill settings first:
```text
-b 512 -ub 128 -ctk tbq4_0 -ctv tbq4_0
```

## Pass criteria
- Build passes on `/opt/rocm-7.2.3`.
- TBQ4 fused path is selected only on AMD rocWMMA TBQ4 case.
- Speed beats current AMD non-fused TBQ4 baseline: ~20.8 tok/s.
- No VRAM pool abort during 16k smoke test.
