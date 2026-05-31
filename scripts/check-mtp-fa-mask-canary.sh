#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github}
BIN=${BIN:-$ROOT/build-rocm-fixed/bin/llama-server}
MODEL=${MODEL:-/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf}
OUT_DIR=${OUT_DIR:-$ROOT/.harness/tmp/mtp-fa-mask-canary}
PORT=${PORT:-18650}
PROMPT=${PROMPT:-The capital of France is}
N_PREDICT=${N_PREDICT:-4}

mkdir -p "$OUT_DIR"
cd "$ROOT"

log="$OUT_DIR/mtp-fa-none.log"
json="$OUT_DIR/mtp-fa-none.json"
summary_json="$OUT_DIR/summary.json"
summary_csv="$OUT_DIR/summary.csv"
pid=""

cleanup() {
  if [[ -n "${pid:-}" ]]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

prompt_json() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

env \
  LLAMA_MTP_VALIDATE_INPUTS=1 \
  LLAMA_MTP_TEACHER_PROBE=1 \
  LLAMA_MTP_FA_ROUTE=1 \
  COMPRESSED_KV_FATTN_LOG=1 \
  LLAMA_MTP_ENABLE_FA=1 \
  LLAMA_MTP_FA_INST=none \
  GGML_CUDA_ROCM_Q8K_DOT4_KQ=0 \
  "$BIN" \
    --device ROCm0 \
    -m "$MODEL" \
    --flash-attn on --cache-type-k f16 --cache-type-v f16 \
    --ctx-size 1024 --batch-size 128 --ubatch-size 128 \
    --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0 \
    --parallel 1 --no-warmup --port "$PORT" > "$log" 2>&1 &
pid=$!

ready=0
for _ in $(seq 1 90); do
  if curl -s "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q ok; then
    ready=1
    break
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "FAIL: server died before readiness" >&2
    tail -120 "$log" >&2 || true
    exit 1
  fi
  sleep 1
done

if [[ "$ready" != 1 ]]; then
  echo "FAIL: server not ready" >&2
  tail -120 "$log" >&2 || true
  exit 1
fi

curl -s --max-time 120 "http://127.0.0.1:${PORT}/completion" \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\":$(prompt_json "$PROMPT"),\"n_predict\":${N_PREDICT},\"temperature\":0,\"seed\":1234}" > "$json"

cleanup
trap - EXIT

python3 "$ROOT/scripts/parse-mtp-acceptance-log.py" "$log" --csv "$summary_csv" > "$summary_json"

python3 - "$summary_json" "$log" <<'PY'
import json
import re
import sys

summary_path, log_path = sys.argv[1:3]
with open(summary_path, "r", encoding="utf-8") as f:
    rows = json.load(f)
if not rows:
    raise SystemExit("FAIL: parser produced no rows")
r = rows[0]
failures = []
if r.get("bad_h") != 0:
    failures.append(f"bad_h={r.get('bad_h')}")
if r.get("bad_logits") != 0:
    failures.append(f"bad_logits={r.get('bad_logits')}")
acceptance = r.get("acceptance") or {}
generated = acceptance.get("generated", 0)
accepted = acceptance.get("accepted", 0)
depth1 = (r.get("acceptance_by_depth") or {}).get("1") or {}
d1 = f"{depth1.get('accepted', 0)}/{depth1.get('generated', 0)}"
if generated <= 0:
    failures.append(f"generated={generated}")
if depth1.get("accepted", 0) <= 0 or depth1.get("generated", 0) <= 0:
    failures.append(f"d1={d1}")
with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    log = f.read()
if "GGML_ASSERT" in log:
    failures.append("GGML_ASSERT in log")
if "bad FA V layout" in log:
    failures.append("bad FA V layout in log")
if failures:
    print("FAIL: MTP FA mask canary failed: " + ", ".join(failures), file=sys.stderr)
    raise SystemExit(1)
print(
    "PASS: MTP FA mask canary "
    f"bad_h={r['bad_h']} bad_logits={r['bad_logits']} "
    f"accepted={accepted} generated={generated} d1={d1}"
)
PY

echo "Log: $log"
echo "Summary: $summary_csv"
