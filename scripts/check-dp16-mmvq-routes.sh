#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BIN=${BIN:-$ROOT/build-rocm-fixed/bin/test-backend-ops}
HARNESS=${HARNESS:-$ROOT/build-rocm-fixed/bin/test-dp16-packed16-mmvq}
OUT_DIR=${OUT_DIR:-$ROOT/.harness/tmp}
mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "missing executable: $BIN" >&2
  echo "build first: cmake --build build-rocm-fixed --target test-backend-ops -j \$(nproc)" >&2
  exit 2
fi

MUL_Q8_N1_4='type_a=q8_0,type_b=f32,m=16,n=[1-4],k=256,bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1'
MUL_Q8_N5='type_a=q8_0,type_b=f32,m=16,n=5,k=256,bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1'
MUL_Q8_K288='type_a=q8_0,type_b=f32,m=16,n=[1-4],k=288,bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1'
MUL_Q4_N1_4='type_a=q4_0,type_b=f32,m=16,n=[1-4],k=256,bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1'
MUL_Q8_IDS='q8_0'
MUL_Q8_FUSION='type=q8_0,glu_op=2,m=1,n=32,k=256,use_id=0,n_mats=16,n_used=8,b=0,with_bias=1,with_gate=0,batch_dims=\[1,1\]'

run_ok() {
  local name=$1; shift
  local log="$OUT_DIR/${name}.log"
  echo "== $name"
  "$@" >"$log" 2>&1
  echo "   log: $log"
}

run_fail() {
  local name=$1; shift
  local log="$OUT_DIR/${name}.log"
  echo "== $name"
  set +e
  "$@" >"$log" 2>&1
  local status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    echo "expected failure but command passed: $name" >&2
    echo "log: $log" >&2
    exit 1
  fi
  echo "   failed as expected status=$status log=$log"
}

require_grep() {
  local pattern=$1
  local file=$2
  if ! grep -q -- "$pattern" "$file"; then
    echo "missing pattern '$pattern' in $file" >&2
    exit 1
  fi
}

reject_grep() {
  local pattern=$1
  local file=$2
  if grep -q -- "$pattern" "$file"; then
    echo "unexpected pattern '$pattern' in $file" >&2
    exit 1
  fi
}

run_ok dp16-mmvq-q8-route \
  env GGML_CUDA_DP16_TRACE=1 GGML_CUDA_ROCM_Q8_DOT4_MMVQ=1 \
  "$BIN" test -o MUL_MAT -p "$MUL_Q8_N1_4"
require_grep 'backend=mmvq_q8_dot4' "$OUT_DIR/dp16-mmvq-q8-route.log"
require_grep 'route=rocm_q8_dot4_mmvq' "$OUT_DIR/dp16-mmvq-q8-route.log"
require_grep 'packed16=0' "$OUT_DIR/dp16-mmvq-q8-route.log"
reject_grep 'route=rocm_packed16_dot4_mmvq.*packed16=0' "$OUT_DIR/dp16-mmvq-q8-route.log"

run_ok dp16-mmvq-q8-fusion-route \
  env GGML_CUDA_DP16_TRACE=1 GGML_CUDA_ROCM_Q8_DOT4_MMVQ=1 GGML_CUDA_DP16_ROUTE_REQUIRE=rocm_q8_dot4_mmvq \
  "$BIN" test -o MUL_MAT_VEC_FUSION -p "$MUL_Q8_FUSION"
require_grep 'backend=mmvq_q8_dot4' "$OUT_DIR/dp16-mmvq-q8-fusion-route.log"
require_grep 'route=rocm_q8_dot4_mmvq' "$OUT_DIR/dp16-mmvq-q8-fusion-route.log"
require_grep 'kernel=dp16_mmvq_q8_dot4_fusion_n1_k256' "$OUT_DIR/dp16-mmvq-q8-fusion-route.log"
reject_grep 'reject=fusion_unsupported' "$OUT_DIR/dp16-mmvq-q8-fusion-route.log"

run_ok dp16-mmvq-packed16-route \
  env GGML_CUDA_DP16_TRACE=1 GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ=1 GGML_CUDA_DP16_ROUTE_REQUIRE=rocm_packed16_dot4_mmvq \
  "$BIN" test -o MUL_MAT -p "$MUL_Q8_N1_4"
require_grep 'backend=mmvq_packed16_i32_dot4' "$OUT_DIR/dp16-mmvq-packed16-route.log"
require_grep 'route=rocm_packed16_dot4_mmvq' "$OUT_DIR/dp16-mmvq-packed16-route.log"
require_grep 'layout=packed16_i32_scaled' "$OUT_DIR/dp16-mmvq-packed16-route.log"
require_grep 'packed16=1' "$OUT_DIR/dp16-mmvq-packed16-route.log"

run_fail dp16-mmvq-packed16-require-missing-sidecar \
  env GGML_CUDA_DP16_TRACE=1 GGML_CUDA_DP16_ROUTE_REQUIRE=rocm_packed16_dot4_mmvq \
  "$BIN" test -o MUL_MAT -p "$MUL_Q8_N1_4"
require_grep 'reject=packed16_weight_missing' "$OUT_DIR/dp16-mmvq-packed16-require-missing-sidecar.log"
require_grep 'fallback_disallowed=1' "$OUT_DIR/dp16-mmvq-packed16-require-missing-sidecar.log"

run_fail dp16-mmvq-packed16-n-too-large \
  env GGML_CUDA_DP16_TRACE=1 GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ=1 GGML_CUDA_DP16_ROUTE_REQUIRE=rocm_packed16_dot4_mmvq \
  "$BIN" test -o MUL_MAT -p "$MUL_Q8_N5"
require_grep 'reject=n_too_large' "$OUT_DIR/dp16-mmvq-packed16-n-too-large.log"
require_grep 'fallback_disallowed=1' "$OUT_DIR/dp16-mmvq-packed16-n-too-large.log"

if [[ -x "$HARNESS" ]]; then
  run_fail dp16-mmvq-packed16-k-not-aligned \
    env DP16_MMVQ_HARNESS_ALLOW_INVALID=1 GGML_CUDA_DP16_TRACE=1 GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ=1 GGML_CUDA_DP16_ROUTE_REQUIRE=rocm_packed16_dot4_mmvq \
    "$HARNESS" --m 16 --n 1 --k 288
  require_grep 'reject=k_not_aligned' "$OUT_DIR/dp16-mmvq-packed16-k-not-aligned.log"
  require_grep 'fallback_disallowed=1' "$OUT_DIR/dp16-mmvq-packed16-k-not-aligned.log"
else
  echo "skip dp16-mmvq-packed16-k-not-aligned: missing harness $HARNESS" >&2
fi

run_fail dp16-mmvq-q4-reject \
  env GGML_CUDA_DP16_TRACE=1 GGML_CUDA_DP16_ROUTE_REQUIRE=rocm_packed16_dot4_mmvq \
  "$BIN" test -o MUL_MAT -p "$MUL_Q4_N1_4"
require_grep 'backend=mmvq_q4_reject_only' "$OUT_DIR/dp16-mmvq-q4-reject.log"
require_grep 'reject=q4_correction_undefined' "$OUT_DIR/dp16-mmvq-q4-reject.log"
require_grep 'correction=unsupported' "$OUT_DIR/dp16-mmvq-q4-reject.log"

run_fail dp16-mmvq-packed16-ids-reject \
  env GGML_CUDA_DP16_TRACE=1 GGML_CUDA_DP16_ROUTE_REQUIRE=rocm_packed16_dot4_mmvq \
  "$BIN" test -o MUL_MAT_ID -p "$MUL_Q8_IDS"
require_grep 'reject=ids_unsupported' "$OUT_DIR/dp16-mmvq-packed16-ids-reject.log"
require_grep 'fallback_disallowed=1' "$OUT_DIR/dp16-mmvq-packed16-ids-reject.log"

run_fail dp16-mmvq-packed16-fusion-reject \
  env GGML_CUDA_DP16_TRACE=1 GGML_CUDA_DP16_ROUTE_REQUIRE=rocm_packed16_dot4_mmvq \
  "$BIN" test -o MUL_MAT_VEC_FUSION -p "$MUL_Q8_FUSION"
require_grep 'reject=fusion_unsupported' "$OUT_DIR/dp16-mmvq-packed16-fusion-reject.log"
require_grep 'fallback_disallowed=1' "$OUT_DIR/dp16-mmvq-packed16-fusion-reject.log"

run_ok dp16-mmvq-default-unchanged \
  "$BIN" test -o MUL_MAT -p "$MUL_Q8_N1_4"
reject_grep 'DP16 accept.*rocm_packed16_dot4_mmvq' "$OUT_DIR/dp16-mmvq-default-unchanged.log"
require_grep 'tests passed' "$OUT_DIR/dp16-mmvq-default-unchanged.log"

echo "DP16 MMVQ route canaries passed"
