#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github}
BIN=${BIN:-$ROOT/build-rocm-fixed/bin/llama-server}
MODEL=${MODEL:-/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf}
OUT_DIR=${OUT_DIR:-/tmp/mtp-packed16-route-matrix}
PROMPT=${PROMPT:-The capital of France is}
N_PREDICT=${N_PREDICT:-4}
CTX_SIZE=${CTX_SIZE:-1024}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-1024}
SPEC_DRAFT_N_MAX=${SPEC_DRAFT_N_MAX:-3}
SPEC_DRAFT_P_MIN=${SPEC_DRAFT_P_MIN:-0}
BASE_PORT=${BASE_PORT:-18820}
ROUTES=${ROUTES:-"source-dot4 packed16-mmq"}

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

  env \
    LLAMA_MTP_VALIDATE_INPUTS=1 \
    LLAMA_MTP_TEACHER_PROBE=1 \
    LLAMA_MTP_FA_ROUTE=1 \
    COMPRESSED_KV_FATTN_LOG=1 \
    $extra_env \
    "$BIN" \
      --device ROCm0 \
      -m "$MODEL" \
      --flash-attn on --cache-type-k f16 --cache-type-v f16 \
      --ctx-size "$CTX_SIZE" --batch-size "$BATCH_SIZE" --ubatch-size "$UBATCH_SIZE" \
      --spec-type draft-mtp --spec-draft-n-max "$SPEC_DRAFT_N_MAX" --spec-draft-p-min "$SPEC_DRAFT_P_MIN" \
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

  grep -E 'fa_final_select: inst=mtp_verify_qk|PACKED16 FA ROUTE|PDMQ|PWMMA v|bad_h=|bad_logits=|draft acceptance rate|statistics draft-mtp-depth|prompt eval time|eval time|total time|GGML_ASSERT|ABORT|error' "$log" | tail -160 || true
}

case_env() {
  case "$1" in
    source-dot4)
      echo "LLAMA_MTP_ENABLE_FA=1 LLAMA_MTP_FA_INST=verify GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 GGML_CUDA_ROCM_Q8K_DOT4_KQ_AUTO=1 GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_recthist_mtp_verify"
      ;;
    packed16-mmq)
      echo "LLAMA_MTP_ENABLE_FA=1 LLAMA_MTP_FA_INST=verify LLAMA_MTP_DISABLE_PACKED16_FA=0 GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=1 GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq"
      ;;
    pwmma-bm32-directv)
      echo "LLAMA_MTP_ENABLE_FA=1 LLAMA_MTP_FA_INST=verify LLAMA_MTP_DISABLE_PACKED16_FA=0 GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=0 GGML_CUDA_ROCM_PACKED16_WMMA_TILE=1 GGML_CUDA_ROCM_PACKED16_WMMA_BM=32 GGML_CUDA_ROCM_PACKED16_WMMA_IMPL=bm32_regout_directv GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_wmma_tile"
      ;;
    pwmma-bm32-stagev)
      echo "LLAMA_MTP_ENABLE_FA=1 LLAMA_MTP_FA_INST=verify LLAMA_MTP_DISABLE_PACKED16_FA=0 GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=0 GGML_CUDA_ROCM_PACKED16_WMMA_TILE=1 GGML_CUDA_ROCM_PACKED16_WMMA_BM=32 GGML_CUDA_ROCM_PACKED16_WMMA_IMPL=bm32_regout_stagev GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_wmma_tile"
      ;;
    pwmma-bm64-512t)
      echo "LLAMA_MTP_ENABLE_FA=1 LLAMA_MTP_FA_INST=verify LLAMA_MTP_DISABLE_PACKED16_FA=0 GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=0 GGML_CUDA_ROCM_PACKED16_WMMA_TILE=1 GGML_CUDA_ROCM_PACKED16_WMMA_BM=64 GGML_CUDA_ROCM_PACKED16_WMMA_IMPL=bm64_regout_directv_512t GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_wmma_tile"
      ;;
    pwmma-bm64-wavegate-directv)
      echo "LLAMA_MTP_ENABLE_FA=1 LLAMA_MTP_FA_INST=verify LLAMA_MTP_DISABLE_PACKED16_FA=0 GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=0 GGML_CUDA_ROCM_PACKED16_WMMA_TILE=1 GGML_CUDA_ROCM_PACKED16_WMMA_BM=64 GGML_CUDA_ROCM_PACKED16_WMMA_IMPL=bm64_512t_wavegate_directv GGML_CUDA_ROCM_PACKED16_WMMA_CAUSAL_SKIP=1 GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_wmma_tile"
      ;;
    *)
      echo "unknown route '$1'" >&2
      return 1
      ;;
  esac
}

idx=0
for route in $ROUTES; do
  envs=$(case_env "$route")
  run_case "$route" "$((BASE_PORT + idx))" "$envs"
  idx=$((idx + 1))
done

python3 "$ROOT/scripts/parse-mtp-acceptance-log.py" "$OUT_DIR"/*.log --csv "$OUT_DIR/summary.csv" > "$OUT_DIR/summary.json"

python3 - "$OUT_DIR" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = {Path(r["log"]).name: r for r in json.loads((out / "summary.json").read_text())}
route_re = re.compile(r"fa_final_select: inst=mtp_verify_qk selected=(\S+) nq=(\d+) nk=(\d+) d=(\d+) K=(\S+) V=(\S+)")
time_re = re.compile(r"(prompt eval|eval|total) time\s*=\s*([0-9.]+) ms")
rows = []
for log_path in sorted(out.glob("*.log")):
    if log_path.name == "run.log":
        continue
    text = log_path.read_text(errors="replace")
    routes = route_re.findall(text)
    summary_row = summary.get(log_path.name, {})
    selected_names = []
    for m in routes:
        if m[0] not in selected_names:
            selected_names.append(m[0])
    selected = "/".join(selected_names) if selected_names else "none"
    nqs = "/".join(sorted({m[1] for m in routes}, key=int)) if routes else ""
    nks = "/".join(sorted({m[2] for m in routes}, key=int)) if routes else ""
    kv = "/".join(sorted({f"K={m[4]} V={m[5]}" for m in routes})) if routes else ""
    pdmq = "rocm_packed16_dot4_mmq" in selected_names and "PDMQ QK probe PASSED" in text
    pwmma = "PWMMA v" in text
    impl = ""
    if m := re.search(r"PWMMA v[^\n]* IMPL=(\S+)", text):
        impl = m.group(1)
    times = {k.replace(" ", "_"): v for k, v in time_re.findall(text)}
    failure = "ok"
    if "GGML_ASSERT" in text or "ABORT" in text:
        failure = "assert_or_abort"
    elif "srv    send_error:" in text or "error:" in text and "request" in text:
        failure = "request_error"
    elif summary_row.get("bad_logits", 0) != 0:
        failure = "bad_logits"
    elif summary_row.get("bad_h", 0) != 0:
        failure = "bad_h"
    elif summary_row.get("checked_h", 0) == 0:
        failure = "no_hidden_state_checks"
    elif not routes:
        failure = "no_mtp_verify_route"
    elif (summary_row.get("acceptance") or {}).get("generated", 0) == 0:
        failure = "no_drafts_generated"
    rows.append({
        "case": log_path.stem,
        "failure_class": failure,
        "selected": selected,
        "nq_seen": nqs,
        "nk_seen": nks,
        "kv": kv,
        "pdmq_probe_passed": int(pdmq),
        "pwmma_seen": int(pwmma),
        "pwmma_impl": impl,
        "bad_h": summary_row.get("bad_h"),
        "checked_h": summary_row.get("checked_h"),
        "bad_logits": summary_row.get("bad_logits"),
        "accepted": (summary_row.get("acceptance") or {}).get("accepted"),
        "generated": (summary_row.get("acceptance") or {}).get("generated"),
        "rate": (summary_row.get("acceptance") or {}).get("rate"),
        "prompt_eval_ms": times.get("prompt_eval", ""),
        "eval_ms": times.get("eval", ""),
        "total_ms": times.get("total", ""),
    })
with (out / "routes.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["case"])
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {out / 'summary.csv'}")
print(f"Wrote {out / 'routes.csv'}")
PY
