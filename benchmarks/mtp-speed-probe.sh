#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github}
BIN=${BIN:-$ROOT/build-rocm-fixed/bin/llama-server}
MODEL=${MODEL:-/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf}
OUT_DIR=${OUT_DIR:-$ROOT/.harness/tmp/mtp-speed-probe/$(date +%Y%m%d-%H%M%S)}
BASE_PORT=${BASE_PORT:-18920}
PROMPT=${PROMPT:-The capital of France is}
N_PREDICT=${N_PREDICT:-128}
CTX_SIZE=${CTX_SIZE:-2048}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-1024}
SPEC_DRAFT_N_MAX=${SPEC_DRAFT_N_MAX:-3}
CACHE_TYPE_K=${CACHE_TYPE_K:-f16}
CACHE_TYPE_V=${CACHE_TYPE_V:-q4_0}
MODE=${MODE:-perf}
CASES=${CASES:-nospec mtp_nonhook_p0 mtp_hook_p0 mtp_nonhook_p04 mtp_hook_p04}

mkdir -p "$OUT_DIR"
cd "$ROOT"

prompt_json() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

case_spec_args() {
  case "$1" in
    nospec) echo "" ;;
    mtp_*) echo "--spec-type draft-mtp --spec-draft-n-max $SPEC_DRAFT_N_MAX --spec-draft-p-min $(case_pmin "$1")" ;;
    *) echo "unknown case '$1'" >&2; return 1 ;;
  esac
}

case_pmin() {
  case "$1" in
    *_p0) echo "0" ;;
    *_p02) echo "0.2" ;;
    *_p04) echo "0.4" ;;
    *_p06) echo "0.6" ;;
    *_p08) echo "0.8" ;;
    *) echo "0" ;;
  esac
}

case_env() {
  local c=$1
  # DOT4 target decode flags are common to all cases; route-proof mode adds logs/strict require.
  local envs="LLAMA_MTP_PREFILL_CHUNK=$UBATCH_SIZE GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 GGML_CUDA_ROCM_Q8K_DOT4_KQ_AUTO=1"
  if [[ "$MODE" == "route" ]]; then
    envs="$envs COMPRESSED_KV_FATTN_LOG=1 LLAMA_MTP_FA_ROUTE=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq"
  fi
  if [[ "$MODE" == "debug" ]]; then
    envs="$envs LLAMA_MTP_VALIDATE_INPUTS=1 LLAMA_MTP_TEACHER_PROBE=1 LLAMA_MTP_CONF_TRACE=1"
  fi
  case "$c" in
    mtp_hook_*) envs="$envs LLAMA_MTP_HOOK_WIRE=1 LLAMA_MTP_PREFILL_FORCE_MMQ=1 GGML_CUDA_ROCM_QUANT_PREFILL_F16=1" ;;
  esac
  if [[ -n "${EXTRA_ENV:-}" ]]; then
    envs="$envs $EXTRA_ENV"
  fi
  echo "$envs"
}

kill_server() {
  local pid=${1:-}
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || return 0
      sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

run_case() {
  local c=$1 port=$2
  local log="$OUT_DIR/${c}.log"
  local json="$OUT_DIR/${c}.json"
  local meta="$OUT_DIR/${c}.cmd.txt"
  local envs args srv ready=0
  envs=$(case_env "$c")
  args=$(case_spec_args "$c")

  {
    echo "case=$c"
    echo "mode=$MODE"
    echo "port=$port"
    echo "env=$envs"
    echo "args=$args"
    echo "prompt=$PROMPT"
    echo "n_predict=$N_PREDICT"
    echo "ctx_size=$CTX_SIZE batch=$BATCH_SIZE ubatch=$UBATCH_SIZE draft_n_max=$SPEC_DRAFT_N_MAX cache_k=$CACHE_TYPE_K cache_v=$CACHE_TYPE_V"
  } > "$meta"

  echo "== $c ==" | tee -a "$OUT_DIR/run.log"

  # shellcheck disable=SC2086
  env $envs "$BIN" \
    --device ROCm0 \
    -m "$MODEL" \
    --host 127.0.0.1 --port "$port" \
    --flash-attn on --cache-type-k "$CACHE_TYPE_K" --cache-type-v "$CACHE_TYPE_V" \
    --ctx-size "$CTX_SIZE" --batch-size "$BATCH_SIZE" --ubatch-size "$UBATCH_SIZE" \
    --parallel 1 --no-warmup \
    $args > "$log" 2>&1 &
  srv=$!

  trap 'kill_server "$srv"' RETURN

  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q ok; then
      ready=1
      break
    fi
    if ! kill -0 "$srv" 2>/dev/null; then
      echo "server died before ready: $c" | tee -a "$OUT_DIR/run.log"
      tail -120 "$log" | tee -a "$OUT_DIR/run.log" || true
      return 0
    fi
    sleep 1
  done

  if [[ "$ready" != 1 ]]; then
    echo "server not ready: $c" | tee -a "$OUT_DIR/run.log"
    tail -120 "$log" | tee -a "$OUT_DIR/run.log" || true
  else
    curl -fsS --max-time 300 "http://127.0.0.1:${port}/completion" \
      -H 'Content-Type: application/json' \
      -d "{\"prompt\":$(prompt_json "$PROMPT"),\"n_predict\":${N_PREDICT},\"temperature\":0,\"seed\":1234,\"ignore_eos\":true}" \
      > "$json" || echo "curl request failed for $c" | tee -a "$OUT_DIR/run.log"
  fi

  sleep 1
  kill_server "$srv"
  wait "$srv" 2>/dev/null || true
  trap - RETURN
}

idx=0
for c in $CASES; do
  run_case "$c" "$((BASE_PORT + idx))"
  idx=$((idx + 1))
done

python3 - "$OUT_DIR" <<'PY'
import csv, json, re, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
route_re = re.compile(r"fa_final_select: inst=([^ ]+) selected=([^ ]+) nq=(\d+) nk=(\d+) d=(\d+) K=([^ ]+) V=([^ ]+)")
acc_re = re.compile(r"draft acceptance rate =\s*([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)")
depth_re = re.compile(r"statistics ([^:]+)-depth: (.*)")
depth_item_re = re.compile(r"d(\d+)=(\d+)/(\d+)")
err_words = ("GGML_ASSERT", "ABORT", "fatal error", "failed to process speculative batch", "failed to decode", "Invalid input batch")
for meta in sorted(out.glob("*.cmd.txt")):
    case = meta.stem.replace('.cmd','')
    logp = out / f"{case}.log"
    jsonp = out / f"{case}.json"
    text = logp.read_text(errors="replace") if logp.exists() else ""
    data = {}
    if jsonp.exists() and jsonp.stat().st_size:
        try:
            data = json.loads(jsonp.read_text(errors="replace"))
        except Exception as e:
            data = {"json_error": str(e), "raw": jsonp.read_text(errors="replace")[:500]}
    timings = data.get("timings") or {}
    acc = None
    for m in acc_re.finditer(text):
        acc = {"rate": float(m.group(1)), "accepted": int(m.group(2)), "generated": int(m.group(3))}
    depths = {}
    for m in depth_re.finditer(text):
        for d in depth_item_re.finditer(m.group(2)):
            depths[f"d{d.group(1)}"] = f"{d.group(2)}/{d.group(3)}"
    routes = route_re.findall(text)
    target_routes = [r for r in routes if r[0] == "decode_qk"]
    selected = "/".join(dict.fromkeys([r[1] for r in target_routes or routes]))
    kv = "/".join(dict.fromkeys([f"K={r[5]} V={r[6]}" for r in target_routes or routes]))
    row = {
        "case": case,
        "ok_json": bool(timings),
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "prompt_n": timings.get("prompt_n"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
        "log_accept_accepted": None if not acc else acc["accepted"],
        "log_accept_generated": None if not acc else acc["generated"],
        "log_accept_rate": None if not acc else acc["rate"],
        "d1": depths.get("d1"),
        "d2": depths.get("d2"),
        "d3": depths.get("d3"),
        "target_route": selected,
        "target_kv": kv,
        "hook_registered": "MTP draft head registered" in text,
        "chunking": "chunking MTP prefill" in text,
        "errors": ";".join(w for w in err_words if w in text),
    }
    rows.append(row)
if rows:
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
else:
    print("[]")
print(f"OUT_DIR={out}")
PY
