#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
HIPCC=${HIPCC:-/opt/rocm-7.2.3/bin/hipcc}
ARCH=${ARCH:-gfx1100}
OUT=${OUT:-/tmp/rdna-i8-packed16-blockscaled-proof}

cd "$ROOT"
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showmemuse --showuse --showpidgpus
fi

"$HIPCC" --offload-arch="$ARCH" -O3 -std=c++17 \
  scripts/hip/rdna-i8-packed16-blockscaled-proof.hip \
  -o "$OUT"

"$OUT" "${M:-64}" "${N:-128}" "${K:-512}" "${ITERS:-100}" "${USE_A_SCALE:-1}"
"$OUT" "${M:-64}" "${N:-128}" "${K:-512}" "${ITERS:-100}" 0

if [[ "${RUN_EDGE_SWEEP:-1}" != "0" ]]; then
  "$OUT" 17 19 32 50 1
  "$OUT" 31 33 96 50 1
  "$OUT" 5 7 64 50 0
  "$OUT" 48 17 160 50 1
fi
