#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github}
BIN=${BIN:-$ROOT/build-rocm-fixed/bin/llama-server}
MODEL=${MODEL:-/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf}
OUT_DIR=${OUT_DIR:-/tmp/mtp-acceptance-matrix}
PROMPT=${PROMPT:-The capital of France is}
PROMPTS_FILE=${PROMPTS_FILE:-}
N_PREDICT=${N_PREDICT:-24}
CTX_SIZE=${CTX_SIZE:-1024}
BATCH_SIZE=${BATCH_SIZE:-128}
UBATCH_SIZE=${UBATCH_SIZE:-128}
BASE_PORT=${BASE_PORT:-18200}
FA_VALUES=${FA_VALUES:-"off on"}
N_MAX_VALUES=${N_MAX_VALUES:-"1 2 3"}
P_MIN_VALUES=${P_MIN_VALUES:-"0 0.2 0.4"}
RUN_DOT4_PROBES=${RUN_DOT4_PROBES:-0}
EXTRA_ENV=${EXTRA_ENV:-}

mkdir -p "$OUT_DIR"
cd "$ROOT"

mapfile -t PROMPTS < <(
  if [[ -n "$PROMPTS_FILE" ]]; then
    grep -vE '^\s*(#|$)' "$PROMPTS_FILE"
  else
    printf '%s\n' "$PROMPT"
  fi
)

prompt_json() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

run_case() {
  local prompt_id=$1 prompt=$2 name=$3 port=$4 fa=$5 nmax=$6 pmin=$7 extra_env=$8
  local log="$OUT_DIR/${name}.log"
  local json="$OUT_DIR/${name}.json"
  local prompt_txt="$OUT_DIR/${name}.prompt.txt"

  printf '%s\n' "$prompt" > "$prompt_txt"
  echo "== $name =="
  env LLAMA_MTP_VALIDATE_INPUTS=1 LLAMA_MTP_TEACHER_PROBE=1 $extra_env "$BIN" \
    --device ROCm0 \
    -m "$MODEL" \
    --flash-attn "$fa" --cache-type-k f16 --cache-type-v f16 \
    --ctx-size "$CTX_SIZE" --batch-size "$BATCH_SIZE" --ubatch-size "$UBATCH_SIZE" \
    --spec-type draft-mtp --spec-draft-n-max "$nmax" --spec-draft-p-min "$pmin" \
    --parallel 1 --no-warmup --port "$port" > "$log" 2>&1 &
  local srv=$!

  local ready=0
  for _ in $(seq 1 90); do
    if curl -s "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q ok; then ready=1; break; fi
    if ! kill -0 "$srv" 2>/dev/null; then echo "server died: $name"; tail -120 "$log"; return 1; fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then echo "server not ready: $name"; tail -120 "$log"; kill "$srv" 2>/dev/null || true; return 1; fi

  curl -s --max-time 180 "http://127.0.0.1:${port}/completion" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\":$(prompt_json "$prompt"),\"n_predict\":${N_PREDICT},\"temperature\":0,\"seed\":1234}" > "$json" || true
  sleep 1
  kill "$srv" 2>/dev/null || true
  wait "$srv" 2>/dev/null || true
}

idx=0
pidx=0
for prompt in "${PROMPTS[@]}"; do
  prompt_id=$(printf 'p%02d' "$pidx")
  pidx=$((pidx + 1))
  for fa in $FA_VALUES; do
    for nmax in $N_MAX_VALUES; do
      for pmin in $P_MIN_VALUES; do
        port=$((BASE_PORT + idx)); idx=$((idx + 1))
        run_case "$prompt_id" "$prompt" "${prompt_id}_fa-${fa}_n${nmax}_p${pmin}" "$port" "$fa" "$nmax" "$pmin" "$EXTRA_ENV"
      done
    done
  done
done

# Experimental DOT4 MTP FA route probes. Keep disabled by default: these are
# negative-control route probes, not part of safe-route acceptance scoring.
if [[ "$RUN_DOT4_PROBES" == 1 ]]; then
  for inst in verify draft_decode; do
    port=$((BASE_PORT + idx)); idx=$((idx + 1))
    run_case "dot4" "${PROMPTS[0]}" "dot4-${inst}_fa-on_n1_p0" "$port" on 1 0 \
      "LLAMA_MTP_ENABLE_FA=1 LLAMA_MTP_FA_INST=${inst} GGML_CUDA_FA_ROUTE_REQUIRE=dot4 GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 GGML_CUDA_ROCM_Q8K_DOT4_KQ_AUTO=1" || true
  done
fi

python3 "$ROOT/scripts/parse-mtp-acceptance-log.py" "$OUT_DIR"/*.log --csv "$OUT_DIR/summary.csv" > "$OUT_DIR/summary.json"
echo "Wrote $OUT_DIR/summary.csv"
