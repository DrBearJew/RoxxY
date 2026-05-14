# AMD ROCm notes — Indras MTP + TBQ4/RotorQuant

This branch is based on `Indras-Mirror/llama.cpp-mtp` and adds the minimal HIP compatibility fixes needed to build on AMD ROCm / gfx1100.

Tested configure/build target:

- GPU: RX 7900 XTX (`gfx1100`)
- ROCm: `/opt/rocm-7.2.3`
- Target: `llama-server`

## Build

```bash
mkdir -p build-rocm-gfx1100
cd build-rocm-gfx1100
ROCM_PATH=/opt/rocm-7.2.3 HIP_PATH=/opt/rocm-7.2.3 cmake .. \
  -DGGML_HIP=ON \
  -DCMAKE_C_COMPILER=/opt/rocm-7.2.3/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-7.2.3/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.3/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm-7.2.3 \
  -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.3 \
  -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-server -j$(nproc)
```

## Relevant runtime flags

Indras uses upstream MTP naming and its own quant names:

```bash
--spec-type mtp
--spec-draft-n-max 3
--cache-type-k tbq4_0
--cache-type-v tbq4_0
```

Available KV cache types shown by `llama-server --help`:

- `tbq3_0`
- `tbq4_0`
- `planar3_0`
- `iso3_0`
- `planar4_0`
- `iso4_0`

## Status

- ROCm configure: PASS
- ROCm `llama-server` build: PASS
- Runtime benchmark on RX 7900 XTX: TODO

The branch is intended as the next benchmark candidate after `mtp-turboquant`.
