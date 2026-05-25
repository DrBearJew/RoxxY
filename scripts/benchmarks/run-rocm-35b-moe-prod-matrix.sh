#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/mrtrent/llama.cpp-mtp-tbq4-rdna3-daily}
BUILD=${BUILD:-/tmp/llama-mmq-wmma-i8-build-gfx1100.nyQBiu}
BIN=${BIN:-$BUILD/bin/llama-server}
MODEL=${MODEL:-/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-IQ4_XS-00001-of-00002.gguf}
PROMPT_SRC=${PROMPT_SRC:-$REPO/benches/rocm-rdna3/mmq-35b-moe-prod-cmd-ab-vram-20260525-072116/prompt16k.txt}
OUT=${OUT:-$REPO/benches/rocm-rdna3/mmq-35b-moe-prod-temp06-matrix-$(date +%Y%m%d-%H%M%S)}
PORT_BASE=${PORT_BASE:-18220}
CTX_SIZE=${CTX_SIZE:-40960}
BATCH_SIZE=${BATCH_SIZE:-1024}
UBATCH_SIZE=${UBATCH_SIZE:-1024}
N_PREDICT=${N_PREDICT:-4096}
TEMP=${TEMP:-0.6}
TOP_P=${TOP_P:-0.95}
SEED=${SEED:-3407}
CASES=${CASES:-no_mtp mtp_n2 mtp_n3}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
VRAM_POLL_SECS=${VRAM_POLL_SECS:-0.25}
SERVER_WAIT_SECS=${SERVER_WAIT_SECS:-300}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-2400}

mkdir -p "$OUT"
cp "$PROMPT_SRC" "$OUT/prompt16k.txt"

if [[ ! -x "$BIN" ]]; then
  echo "FAIL: missing llama-server: $BIN" >&2
  exit 2
fi
if [[ ! -f "$MODEL" ]]; then
  echo "FAIL: missing model: $MODEL" >&2
  exit 2
fi
if [[ ! -f "$PROMPT_SRC" ]]; then
  echo "FAIL: missing prompt: $PROMPT_SRC" >&2
  exit 2
fi

cat > "$OUT/meta.txt" <<EOF
out=$OUT
repo=$REPO
build=$BUILD
server_bin=$BIN
model=$MODEL
prompt_src=$PROMPT_SRC
ctx_size=$CTX_SIZE
batch_size=$BATCH_SIZE
ubatch_size=$UBATCH_SIZE
n_predict=$N_PREDICT
temperature=$TEMP
top_p=$TOP_P
seed=$SEED
cases=$CASES
EOF

case_env() {
  local case_name=$1
  CASE_ENV=(
    "RDNA2_MATMUL_OPT_V1=1"
    "GGML_CUDA_MMQ_MAX_X_AUTO=1"
    "GGML_CUDA_MMQ_ROUTE_LOG=1"
    "COMPRESSED_KV_FATTN_LOG=1"
    "GGML_CUDA_ROCM_QUANT_PREFILL_F16=1"
    "GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_NKV=$CTX_SIZE"
    "LLAMA_MTP_PREFILL_CHUNK=$UBATCH_SIZE"
  )
  SERVER_EXTRA=(--spec-type none)
  case "$case_name" in
    no_mtp)
      SERVER_EXTRA=(--spec-type none)
      ;;
    mtp_n2)
      SERVER_EXTRA=(--spec-type draft-mtp --spec-default --spec-draft-n-max 2 --spec-draft-p-min 0 --spec-draft-prio 2 --spec-draft-prio-batch 2 --cache-type-k-draft q8_0 --cache-type-v-draft tbq4_0)
      ;;
    mtp_n3)
      SERVER_EXTRA=(--spec-type draft-mtp --spec-default --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-prio 2 --spec-draft-prio-batch 2 --cache-type-k-draft q8_0 --cache-type-v-draft tbq4_0)
      ;;
    mtp_n4)
      SERVER_EXTRA=(--spec-type draft-mtp --spec-default --spec-draft-n-max 4 --spec-draft-p-min 0 --spec-draft-prio 2 --spec-draft-prio-batch 2 --cache-type-k-draft q8_0 --cache-type-v-draft tbq4_0)
      ;;
    force_mtp_n2)
      CASE_ENV+=("LLAMA_MTP_PREFILL_FORCE_MMQ=1")
      SERVER_EXTRA=(--spec-type draft-mtp --spec-default --spec-draft-n-max 2 --spec-draft-p-min 0 --spec-draft-prio 2 --spec-draft-prio-batch 2 --cache-type-k-draft q8_0 --cache-type-v-draft tbq4_0)
      ;;
    force_mtp_n3)
      CASE_ENV+=("LLAMA_MTP_PREFILL_FORCE_MMQ=1")
      SERVER_EXTRA=(--spec-type draft-mtp --spec-default --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-prio 2 --spec-draft-prio-batch 2 --cache-type-k-draft q8_0 --cache-type-v-draft tbq4_0)
      ;;
    *)
      echo "FAIL: unknown case '$case_name'" >&2
      return 2
      ;;
  esac
}

run_case() {
  local case_name=$1
  local idx=$2
  case_env "$case_name"
  local case_dir="$OUT/$case_name"
  local port=$((PORT_BASE + idx))
  mkdir -p "$case_dir"
  printf '%s\n' "${CASE_ENV[@]}" > "$case_dir/env.txt"
  printf '%q ' "$BIN" --port "$port" --model "$MODEL" --device ROCm0 --n-gpu-layers 99 --ctx-size "$CTX_SIZE" \
    --cache-type-k q8_0 --cache-type-v tbq4_0 --flash-attn on --batch-size "$BATCH_SIZE" --ubatch-size "$UBATCH_SIZE" \
    --cache-ram 0 --no-mmap --mlock --ignore-eos --no-webui --no-warmup --parallel 1 --temp "$TEMP" --top-p "$TOP_P" --seed "$SEED" \
    "${SERVER_EXTRA[@]}" > "$case_dir/cmd.sh"
  printf '\n' >> "$case_dir/cmd.sh"

  echo "=== $case_name port=$port ===" | tee -a "$OUT/run.log"
  (
    for assignment in "${CASE_ENV[@]}"; do export "$assignment"; done
    export HIP_VISIBLE_DEVICES
    export LD_LIBRARY_PATH="$BUILD/bin:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
    exec "$BIN" --port "$port" --model "$MODEL" --device ROCm0 --n-gpu-layers 99 --ctx-size "$CTX_SIZE" \
      --cache-type-k q8_0 --cache-type-v tbq4_0 --flash-attn on --batch-size "$BATCH_SIZE" --ubatch-size "$UBATCH_SIZE" \
      --cache-ram 0 --no-mmap --mlock --ignore-eos --no-webui --no-warmup --parallel 1 --temp "$TEMP" --top-p "$TOP_P" --seed "$SEED" \
      "${SERVER_EXTRA[@]}" \
      >"$case_dir/server.log" 2>&1
  ) &
  local pid=$!
  echo "$pid" > "$case_dir/pid.txt"

  (
    while kill -0 "$pid" 2>/dev/null; do
      printf '%s\t' "$(date +%s.%N)"
      rocm-smi --showmeminfo vram --csv 2>/dev/null | tr '\n' ' ' || true
      printf '\n'
      sleep "$VRAM_POLL_SECS"
    done
  ) > "$case_dir/vram_samples.tsv" &
  local vpid=$!

  cleanup_case() {
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    kill "$vpid" 2>/dev/null || true
    wait "$vpid" 2>/dev/null || true
  }

  local ready=0
  for _ in $(seq 1 "$SERVER_WAIT_SECS"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "FAIL: server died for $case_name" | tee -a "$OUT/run.log"
      tail -160 "$case_dir/server.log" | tee -a "$OUT/run.log"
      cleanup_case
      return 11
    fi
    if curl -fsS "http://127.0.0.1:$port/health" >"$case_dir/health.json" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    echo "FAIL: server did not become ready for $case_name" | tee -a "$OUT/run.log"
    tail -160 "$case_dir/server.log" | tee -a "$OUT/run.log"
    cleanup_case
    return 12
  fi

  python3 - <<'PY' "$port" "$case_dir" "$TEMP" "$TOP_P" "$SEED" "$N_PREDICT" "$REQUEST_TIMEOUT"
import json, pathlib, sys, time, urllib.request
port, case_dir, temp, top_p, seed, n_predict, timeout = sys.argv[1:]
case_dir = pathlib.Path(case_dir)
prompt = (case_dir.parent / 'prompt16k.txt').read_text()
def post(path, payload, timeout_s):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=timeout_s).read().decode())
tok = post('/tokenize', {'content': prompt, 'add_special': False}, 900)
ids = tok.get('tokens') or tok.get('ids') or []
ntok = len(ids)
(case_dir/'tokenize.json').write_text(json.dumps({'prompt_tokenize_count': ntok}, indent=2))
payload = {
    'prompt': prompt,
    'n_predict': int(n_predict),
    'temperature': float(temp),
    'top_p': float(top_p),
    'seed': int(seed),
    'cache_prompt': False,
    'stream': False,
    'ignore_eos': True,
}
t0 = time.time()
resp = post('/completion', payload, int(timeout))
wall = time.time() - t0
resp['_wall_s'] = wall
resp['_prompt_tokens_input'] = ntok
(case_dir/'response.json').write_text(json.dumps(resp, indent=2))
t = resp.get('timings', {})
summary = {
    'case': case_dir.name,
    'prompt_tokenize_count': ntok,
    'tokens_evaluated': resp.get('tokens_evaluated'),
    'tokens_predicted': resp.get('tokens_predicted'),
    'prompt_tps': t.get('prompt_per_second'),
    'gen_tps': t.get('predicted_per_second'),
    'prompt_ms': t.get('prompt_ms'),
    'gen_ms': t.get('predicted_ms'),
    'wall_s': wall,
    'draft_n': t.get('draft_n'),
    'draft_n_accepted': t.get('draft_n_accepted'),
    'temperature': float(temp),
    'top_p': float(top_p),
    'seed': int(seed),
}
(case_dir/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
PY
  local rc=$?
  cleanup_case

  python3 - <<'PY' "$case_dir" || true
import json, pathlib, re, sys
case_dir=pathlib.Path(sys.argv[1])
peak=None
for line in (case_dir/'vram_samples.tsv').read_text(errors='replace').splitlines():
    # rocm-smi --csv reports VRAM in bytes on this host.
    nums=[int(x) for x in re.findall(r'(?<![0-9])([0-9]{8,})(?![0-9])', line)]
    if nums:
        used=max(nums[1::2] or nums) if len(nums) >= 2 else max(nums)
        peak=used if peak is None else max(peak, used)
out={'peak_vram_bytes': peak, 'peak_vram_mib': (peak/1024/1024 if peak is not None else None), 'peak_vram_gib': (peak/1024/1024/1024 if peak is not None else None)}
(case_dir/'vram_summary.json').write_text(json.dumps(out, indent=2)+'\n')
PY
  echo "=== $case_name rc=$rc ===" | tee -a "$OUT/run.log"
  return "$rc"
}

idx=0
rc_total=0
for case_name in $CASES; do
  if ! run_case "$case_name" "$idx"; then
    rc_total=1
  fi
  idx=$((idx+1))
  sleep 3
done

python3 - <<'PY' "$OUT"
import json, pathlib, re, sys
out=pathlib.Path(sys.argv[1])
rows=[]
for s in sorted(out.glob('*/summary.json')):
    d=json.loads(s.read_text())
    v=s.with_name('vram_summary.json')
    if v.exists():
        d.update(json.loads(v.read_text()))
    server=s.with_name('server.log')
    if server.exists():
        txt=server.read_text(errors='replace')
        d['mmq_route_lines']=len(re.findall(r'get_mmq_x_max_host:', txt))
        d['f16_route_lines']=len(re.findall(r'rocm_quant_prefill_f16|kernel=tile route=tile Q=f32 K=q8_0 V=tbq4_0', txt))
        d['warnings']=len(re.findall(r'WARN|warning|cannot meet|out of memory|OOM', txt, flags=re.I))
    rows.append(d)
(out/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
lines=['# 35B MoE ROCm temp0.6 production matrix','',f'Artifact: `{out}`','', '| case | prompt toks | pred toks | prompt tok/s | gen tok/s | draft accept | peak VRAM GiB | route logs | warnings |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    da=r.get('draft_n_accepted')
    dn=r.get('draft_n')
    draft=f'{da}/{dn}' if da is not None or dn is not None else ''
    routes=f"mmq {r.get('mmq_route_lines',0)}, f16 {r.get('f16_route_lines',0)}"
    lines.append(f"| `{r.get('case')}` | {r.get('prompt_tokenize_count','')} | {r.get('tokens_predicted','')} | {float(r.get('prompt_tps') or 0):.2f} | {float(r.get('gen_tps') or 0):.2f} | {draft} | {float(r.get('peak_vram_gib') or 0):.2f} | {routes} | {r.get('warnings',0)} |")
(out/'summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY

echo "summary: $OUT/summary.md"
exit "$rc_total"
