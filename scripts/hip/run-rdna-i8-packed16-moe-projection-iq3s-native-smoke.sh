#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
BUILD_DIR=${BUILD_DIR:-$ROOT/build-rocm-qwen35-dev}
CXX=${CXX:-/usr/bin/c++}
OUT=${OUT:-/tmp/rdna-i8-packed16-moe-projection-iq3s-native-smoke}

cd "$ROOT"

if [[ ! -d "$BUILD_DIR/bin" ]]; then
  echo "BUILD_DIR/bin not found: $BUILD_DIR/bin" >&2
  echo "Set BUILD_DIR to an existing ROCm ggml build." >&2
  exit 2
fi

if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showmemuse --showuse --showpidgpus
fi

"$CXX" -std=c++17 -O2 \
  -I "$ROOT/ggml/include" -I "$ROOT/ggml/src" -I "$ROOT/include" \
  scripts/hip/rdna-i8-packed16-moe-projection-iq3s-native-smoke.cpp \
  -L "$BUILD_DIR/bin" -Wl,-rpath,"$BUILD_DIR/bin" \
  -lggml -lggml-base -lggml-cpu -lggml-hip \
  -o "$OUT"

"$OUT"
