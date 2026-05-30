#!/usr/bin/env bash
set -euxo pipefail

rm -rf build-rocm-fixed

cmake -S . -B build-rocm-fixed \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_CUDA_FA=ON \
  -DGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm

cmake --build build-rocm-fixed -j"$(nproc)"
