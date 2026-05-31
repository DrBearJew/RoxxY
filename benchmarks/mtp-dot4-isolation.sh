#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github}
BIN=${BIN:-$ROOT/build-rocm-fixed/bin/llama-server}
MODEL=${MODEL:-/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf}
OUT_DIR=${OUT_DIR:-/tmp/mtp-dot4-isolation}
PROMPT=${PROMPT:-The capital of France is}
N_PREDICT=${N_PREDICT:-8}
BASE_PORT=${BASE_PORT:-18400}

mkdir -p "$OUT_DIR"
cd "$ROOT"

prompt_json() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

run_case() {
  local name=$1 port=$2 extra_env=$3
  local log="$OUT_DIR/${name}.log"
  local json="$OUT_DIR/${name}.json"
  echo "== $name =="

  env LLAMA_MTP_VALIDATE_INPUTS=1 LLAMA_MTP_TEACHER_PROBE=1 LLAMA_MTP_FA_ROUTE=1 COMPRESSED_KV_FATTN_LOG=1 $extra_env "$BIN" \
    --device ROCm0 \
    -m "$MODEL" \
    --flash-attn on --cache-type-k f16 --cache-type-v f16 \
    --ctx-size 1024 --batch-size 128 --ubatch-size 128 \
    --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0 \
    --parallel 1 --no-warmup --port "$port" > "$log" 2>&1 &
  local srv=$!

  local ready=0
  for _ in $(seq 1 90); do
    if curl -s "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q ok; then ready=1; break; fi
    if ! kill -0 "$srv" 2>/dev/null; then echo "server died: $name"; tail -120 "$log"; return 0; fi
    sleep 1
  done

  if [[ "$ready" == 1 ]]; then
    curl -s --max-time 120 "http://127.0.0.1:${port}/completion" \
      -H 'Content-Type: application/json' \
      -d "{\"prompt\":$(prompt_json "$PROMPT"),\"n_predict\":${N_PREDICT},\"temperature\":0,\"seed\":1234}" > "$json" || true
  else
    echo "server not ready: $name"
    tail -120 "$log"
  fi

  sleep 1
  kill "$srv" 2>/dev/null || true
  wait "$srv" 2>/dev/null || true

  grep -E 'MTP_INST_SELECT|ggml_cuda_select_mtp|ggml_cuda_fattn_log_instruction_route|bad_h=|bad_logits=|draft acceptance rate|statistics draft-mtp-depth|selected=|impl_status=|route=|GGML_ASSERT|error' "$log" | tail -120 || true
}

idx=0
case_run() { run_case "$1" "$((BASE_PORT + idx))" "$2"; idx=$((idx + 1)); }

# 1. Production-safe baseline: target FA on, draft FA off via set_mtp_source.
case_run safe-target-fa-draft-nonfa ""

# 2. Generic MTP FA with no special instruction. If this fails, DOT4 is not the first suspect.
case_run mtp-fa-none "LLAMA_MTP_ENABLE_FA=1 LLAMA_MTP_FA_INST=none GGML_CUDA_ROCM_Q8K_DOT4_KQ=0"

# 3. MTP verify instruction with DOT4 disabled, exercising existing FA fallback behavior.
case_run mtp-verify-existing "LLAMA_MTP_ENABLE_FA=1 LLAMA_MTP_FA_INST=verify GGML_CUDA_ROCM_Q8K_DOT4_KQ=0 GGML_CUDA_ROCM_MTP_VERIFY_DOT4_DISABLE=1"

# 4. MTP verify recthist DOT4 with source-f16 K adapter enabled and route contract explicit.
case_run mtp-verify-dot4-recthist "LLAMA_MTP_ENABLE_FA=1 LLAMA_MTP_FA_INST=verify GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 GGML_CUDA_ROCM_Q8K_DOT4_KQ_AUTO=1 GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_recthist_mtp_verify"

# 5. MTP draft-decode instruction with DOT4 disabled, exercising existing FA fallback behavior.
case_run mtp-draft-decode-existing "LLAMA_MTP_ENABLE_FA=1 LLAMA_MTP_FA_INST=draft_decode GGML_CUDA_ROCM_Q8K_DOT4_KQ=0 GGML_CUDA_ROCM_MTP_DRAFT_DECODE_DOT4_DISABLE=1"

# 6. MTP draft-decode DOT4 with decode env and route contract explicit.
case_run mtp-draft-decode-dot4 "LLAMA_MTP_ENABLE_FA=1 LLAMA_MTP_FA_INST=draft_decode GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 GGML_CUDA_ROCM_Q8K_DOT4_KQ_AUTO=1 GGML_CUDA_ROCM_MTP_DRAFT_DOT4_DECODE=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_decode_mtp_draft"

# 7. Optional hostile experiment: MTP persistent packed16 K + DOT4-MMQ/PWMMA router.
# Not part of default correctness scoring. Enable with RUN_MTP_PACKED16_MMQ=1.
if [[ "${RUN_MTP_PACKED16_MMQ:-0}" == 1 ]]; then
  case_run mtp-verify-packed16-mmq "LLAMA_MTP_ENABLE_FA=1 LLAMA_MTP_FA_INST=verify LLAMA_MTP_DISABLE_PACKED16_FA=0 GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=1 GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq"
fi

python3 "$ROOT/scripts/parse-mtp-acceptance-log.py" "$OUT_DIR"/*.log --csv "$OUT_DIR/summary.csv" > "$OUT_DIR/summary.json"
echo "Wrote $OUT_DIR/summary.csv"

if [[ "${STRICT_ROUTES:-0}" == 1 ]]; then
  status=0
  require_route() {
    local case_name=$1 pattern=$2
    local log="$OUT_DIR/${case_name}.log"
    if ! grep -Eq "$pattern" "$log"; then
      echo "STRICT_ROUTES FAIL: $case_name did not select required DOT4 route" >&2
      grep -E 'ggml_cuda_fattn_log_instruction_route|fa_final_select: inst=mtp|FATTN COMPUTE SELECT selected=' "$log" | tail -80 >&2 || true
      status=1
    fi
  }

  require_route mtp-verify-dot4-recthist 'fa_instruction=mtp_verify_qk.*selected=rocm_q8k_dot4_kq|fa_final_select: inst=mtp_verify_qk selected=rocm_q8k_dot4_kq'
  require_route mtp-draft-decode-dot4 'fa_instruction=mtp_draft_decode_qk.*selected=rocm_q8k_dot4_kq|fa_final_select: inst=mtp_draft_decode_qk selected=rocm_q8k_dot4_kq'
  if [[ "${RUN_MTP_PACKED16_MMQ:-0}" == 1 ]]; then
    require_route mtp-verify-packed16-mmq 'fa_final_select: inst=mtp_verify_qk selected=rocm_packed16_dot4_mmq .*K=i32 V=f16'
    require_route mtp-verify-packed16-mmq 'PDMQ QK probe PASSED'
  fi
  exit "$status"
fi
