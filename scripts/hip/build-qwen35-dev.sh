#!/usr/bin/env bash
set -euo pipefail

# Narrow ROCm/HIP dev build for Qwen3.5/3.6 MTP PDMQ iteration.
# Purpose: avoid compiling the broader packed16 experimental matrix while editing
# ggml/src/ggml-cuda/fattn-packed16-dot4-mmq-impl.cuh.
#
# Usage:
#   scripts/hip/build-qwen35-dev.sh                 # build llama-server
#   scripts/hip/build-qwen35-dev.sh <ninja-target>  # build a specific target
#
# Fast compile defaults to -O1 for the giant PDMQ HIP TU. Use
#   QWEN35_DEV_FAST_COMPILE=OFF
# for O3/performance-validation builds, or set QWEN35_DEV_HIP_FAST_OPT=-O1/-O2.
# Default dev matrix skips PVBlock-exact, legacy V4_130, approximate V4 PV-WMMA,
# and the old QBlock PV-DOT4 scout. QBlock itself remains in scope.
# The only live V prototype knob here is exact V4_144 PV4:
#   QWEN35_DEV_COMPILE_V4_144_PV4=ON
#   GGML_CUDA_ROCM_V4_K16D16_144_PV4=1
# Do not re-add approximate PV knobs to this script; archive failed variants as patches.
# HIP compilation is memory-heavy on 32GB hosts; default to 12 build jobs unless
# BUILD_JOBS or QWEN35_DEV_BUILD_JOBS is provided.
#
# Useful object-only target:
#   ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/fattn-packed16-dot4-mmq.cu.o

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${QWEN35_DEV_BUILD_DIR:-$repo_root/build-rocm-qwen35-dev}"
target="${1:-llama-server}"
log="${QWEN35_DEV_BUILD_LOG:-/tmp/build-qwen35-dev-${target//\//_}.log}"
build_jobs="${BUILD_JOBS:-${QWEN35_DEV_BUILD_JOBS:-12}}"

cmake -S "$repo_root" -B "$build_dir" -G Ninja \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="${GPU_TARGETS:-gfx1100}" \
  -DGGML_HIP_PDMQ_QWEN35_DEBUG_ONLY=ON \
  -DGGML_HIP_PDMQ_COMPILE_PVBLOCK_EXACT="${QWEN35_DEV_COMPILE_PVBLOCK_EXACT:-OFF}" \
  -DGGML_HIP_PDMQ_COMPILE_V4_K16D16=OFF \
  -DGGML_HIP_PDMQ_COMPILE_V4_APPROX_PV=OFF \
  -DGGML_HIP_PDMQ_COMPILE_QBLOCK_PV_DOT4=OFF \
  -DGGML_HIP_PDMQ_COMPILE_V4_144_PV4="${QWEN35_DEV_COMPILE_V4_144_PV4:-OFF}" \
  -DGGML_HIP_PDMQ_COMPILE_V4_144_PV_DOT4_I8=OFF \
  -DGGML_HIP_PDMQ_FAST_COMPILE="${QWEN35_DEV_FAST_COMPILE:-ON}" \
  -DGGML_HIP_PDMQ_FAST_COMPILE_OPT="${QWEN35_DEV_HIP_FAST_OPT:--O1}" \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DCMAKE_HIP_COMPILER_LAUNCHER="${QWEN35_DEV_HIP_COMPILER_LAUNCHER:-}"

rm -f "$log"
start=$(date +%s)
(
  set +e
  cmake --build "$build_dir" --target "$target" -j "$build_jobs" >"$log" 2>&1
  echo $? > "$log.rc"
) &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  now=$(date +%s)
  printf '[build-watch] qwen35-dev target=%s still running: %ss\n' "$target" "$((now-start))"
  interval="${BUILD_WATCH_INTERVAL:-30}"
  for ((i = 0; i < interval; ++i)); do
    sleep 1
    kill -0 "$pid" 2>/dev/null || break
  done
done
wait "$pid" || true
rc=$(cat "$log.rc" 2>/dev/null || echo 999)
end=$(date +%s)
printf '[build-watch] done target=%s rc=%s elapsed=%ss log=%s\n' "$target" "$rc" "$((end-start))" "$log"
tail -n "${BUILD_TAIL_LINES:-80}" "$log"
exit "$rc"
