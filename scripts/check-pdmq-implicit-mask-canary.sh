#!/usr/bin/env bash
set -euo pipefail

# Bounded graph-level canary for LLAMA_FA_IMPLICIT_CAUSAL_MASK on PDMQ compressed-K routes.
# It verifies packed8_q4 can drop the dense FA KQ mask, and that explicit disable
# switches keep the dense mask. This is a correctness/contract canary, not a
# promotion benchmark.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
BUILD_DIR=${BUILD_DIR:-build-rocm-qwen35-dev}
BIN=${BIN:-$ROOT/$BUILD_DIR/bin/llama-server}
MODEL=${MODEL:-/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf}
OUT_DIR=${OUT_DIR:-$ROOT/.harness/tmp/pdmq-implicit-mask-canary-$(date +%Y%m%d-%H%M%S)}
PORT_BASE=${PORT_BASE:-39710}
DEVICE=${DEVICE:-ROCm0}
PROMPT_LINES=${PROMPT_LINES:-64}
N_PREDICT=${N_PREDICT:-1}
CTX_SIZE=${CTX_SIZE:-4096}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-2048}
TRACE_LIMIT=${TRACE_LIMIT:-24}
DRY_RUN=${DRY_RUN:-0}
# Space-separated subset: dense implicit format_none packed16_disable
CASES=${CASES:-dense implicit format_none packed16_disable}

if [[ ! -x "$BIN" ]]; then
    echo "FAIL: llama-server not executable: $BIN" >&2
    exit 2
fi
if [[ ! -r "$MODEL" ]]; then
    echo "FAIL: model not readable: $MODEL" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
cd "$ROOT"

python3 - "$PROMPT_LINES" "$N_PREDICT" > "$OUT_DIR/request.json" <<'PY'
import json
import sys
lines = int(sys.argv[1])
n_predict = int(sys.argv[2])
prompt = ''.join(
    f'Observation {i:04d}: packed8 same-route implicit causal mask canary; compare mask bytes only.\n'
    for i in range(1, lines + 1)
)
print(json.dumps({
    'prompt': prompt,
    'n_predict': n_predict,
    'temperature': 0,
    'seed': 1,
    'stream': False,
    'cache_prompt': False,
    'ignore_eos': True,
}))
PY

run_case() {
    local label=$1
    local implicit=$2
    local pdmq_format=$3
    local packed16_disable=$4
    local port=$5
    local case_dir="$OUT_DIR/$label"
    mkdir -p "$case_dir"

    {
        echo "ROOT=$ROOT"
        echo "BIN=$BIN"
        echo "MODEL=$MODEL"
        echo "PORT=$port"
        echo "LLAMA_FA_IMPLICIT_CAUSAL_MASK=$implicit"
        echo "LLAMA_FA_IMPLICIT_CAUSAL_MASK_TRACE=1"
        echo "LLAMA_FA_IMPLICIT_CAUSAL_MASK_TRACE_LIMIT=$TRACE_LIMIT"
        echo "LLAMA_F16_FA_KQ_MASK=1"
        echo "GGML_CUDA_ROCM_PDMQ_K_CACHE=1"
        echo "GGML_CUDA_ROCM_PDMQ_K_FORMAT=$pdmq_format"
        if [[ "$packed16_disable" == "1" ]]; then
            echo "GGML_CUDA_ROCM_PACKED16_DISABLE=1"
        fi
        echo "$BIN --device $DEVICE -m $MODEL --host 127.0.0.1 --port $port --flash-attn on --cache-type-k q4_0 --cache-type-v f16 --ctx-size $CTX_SIZE --batch-size $BATCH_SIZE --ubatch-size $UBATCH_SIZE --parallel 1 --no-warmup"
    } > "$case_dir/command.txt"

    if [[ "$DRY_RUN" != "0" ]]; then
        echo "DRY_RUN: $label command written to $case_dir/command.txt"
        return 0
    fi

    local pid=""
    cleanup_case() {
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    }
    trap cleanup_case RETURN

    local -a env_args=(
        "LLAMA_FA_IMPLICIT_CAUSAL_MASK=$implicit"
        "LLAMA_FA_IMPLICIT_CAUSAL_MASK_TRACE=1"
        "LLAMA_FA_IMPLICIT_CAUSAL_MASK_TRACE_LIMIT=$TRACE_LIMIT"
        "LLAMA_F16_FA_KQ_MASK=1"
        "GGML_CUDA_ROCM_PDMQ_K_CACHE=1"
        "GGML_CUDA_ROCM_PDMQ_K_FORMAT=$pdmq_format"
    )
    if [[ "$packed16_disable" == "1" ]]; then
        env_args+=("GGML_CUDA_ROCM_PACKED16_DISABLE=1")
    fi

    env "${env_args[@]}" \
        "$BIN" \
            --device "$DEVICE" \
            -m "$MODEL" \
            --host 127.0.0.1 \
            --port "$port" \
            --flash-attn on \
            --cache-type-k q4_0 \
            --cache-type-v f16 \
            --ctx-size "$CTX_SIZE" \
            --batch-size "$BATCH_SIZE" \
            --ubatch-size "$UBATCH_SIZE" \
            --parallel 1 \
            --no-warmup \
        > "$case_dir/server.stdout.log" 2> "$case_dir/server.stderr.log" &
    pid=$!

    local ready=0
    for _ in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:$port/health" > "$case_dir/health.ok" 2> "$case_dir/health.err"; then
            ready=1
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "server died" > "$case_dir/status.txt"
            break
        fi
        sleep 1
    done
    echo "ready=$ready" > "$case_dir/status.txt"
    if [[ "$ready" != "1" ]]; then
        tail -120 "$case_dir/server.stderr.log" > "$case_dir/server.stderr.tail.txt" || true
        return 3
    fi

    set +e
    curl -sS --max-time 180 "http://127.0.0.1:$port/completion" \
        -H 'Content-Type: application/json' \
        --data-binary "@$OUT_DIR/request.json" \
        -o "$case_dir/response.json" \
        -w '%{http_code}\n' > "$case_dir/http_code.txt" 2> "$case_dir/curl.stderr.log"
    local curl_rc=$?
    set -e
    echo "$curl_rc" > "$case_dir/curl.exit_code.txt"

    cleanup_case
    trap - RETURN

    python3 - "$case_dir" "$label" <<'PY' > "$case_dir/trace-summary.json"
import hashlib
import json
import pathlib
import re
import sys

case = pathlib.Path(sys.argv[1])
label = sys.argv[2]
stderr_log = (case / 'server.stderr.log').read_text(errors='replace') if (case / 'server.stderr.log').exists() else ''
stdout_log = (case / 'server.stdout.log').read_text(errors='replace') if (case / 'server.stdout.log').exists() else ''
log = stderr_log + '\n' + stdout_log
resp_text = (case / 'response.json').read_text(errors='replace') if (case / 'response.json').exists() else ''
try:
    resp = json.loads(resp_text) if resp_text else {}
except Exception:
    resp = {}

inputs = []
for line in log.splitlines():
    if 'LLAMA_FA_IMPLICIT_MASK_TRACE: input' not in line:
        continue
    m = re.search(
        r'use_implicit=(\d+).*dense_mask_bytes_nominal=(\d+).*actual_mask_bytes=(\d+).*actual_fa_mask_bytes=(\d+).*'
        r'n_kv=(\d+) n_tokens=(\d+).*synthetic=(\d+) meta=\(([^)]*)\)',
        line,
    )
    if m:
        inputs.append({
            'use_implicit': int(m.group(1)),
            'nominal': int(m.group(2)),
            'actual_mask_bytes': int(m.group(3)),
            'actual_fa_mask_bytes': int(m.group(4)),
            'n_kv': int(m.group(5)),
            'n_tokens': int(m.group(6)),
            'synthetic': int(m.group(7)),
            'meta': m.group(8),
            'line': line,
        })

def first_float(pattern):
    m = re.search(pattern, log)
    return float(m.group(1)) if m else None

def first_int(pattern):
    m = re.search(pattern, log)
    return int(m.group(1)) if m else None

prompt_tokens = first_int(r'prompt eval time =\s+[0-9.]+ ms /\s+(\d+) tokens')
request_inputs = [x for x in inputs if x['n_tokens'] == prompt_tokens]
request_input = request_inputs[-1] if request_inputs else None
summary = {
    'label': label,
    'http_code': (case / 'http_code.txt').read_text(errors='replace').strip() if (case / 'http_code.txt').exists() else None,
    'curl_rc': (case / 'curl.exit_code.txt').read_text(errors='replace').strip() if (case / 'curl.exit_code.txt').exists() else None,
    'asserts': log.count('GGML_ASSERT') + log.count('ggml_abort') + log.count('requires BM=64') + log.count('synthetic implicit KQ mask metadata is reservation-only'),
    'input_count': len(inputs),
    'implicit_inputs': sum(1 for x in inputs if x['use_implicit'] == 1),
    'dense_inputs': sum(1 for x in inputs if x['use_implicit'] == 0 and x['actual_fa_mask_bytes'] > 0),
    'prompt_tokens': prompt_tokens,
    'request_input': request_input,
    'rocm_compute_buffer_mib': first_float(r'ROCm0 compute buffer size =\s+([0-9.]+) MiB'),
    'host_compute_buffer_mib': first_float(r'ROCm_Host compute buffer size =\s+([0-9.]+) MiB'),
    'prompt_eval_ms': first_float(r'prompt eval time =\s+([0-9.]+) ms'),
    'prompt_tps': first_float(r'prompt eval time =\s+[0-9.]+ ms /\s+\d+ tokens \([^,]+,\s+([0-9.]+) tokens per second\)'),
    'content': resp.get('content'),
    'content_sha8': hashlib.sha256((resp.get('content') or '').encode()).hexdigest()[:8],
}
print(json.dumps(summary, indent=2))

failures = []
if summary['http_code'] != '200':
    failures.append(f"http_code={summary['http_code']}")
if summary['curl_rc'] != '0':
    failures.append(f"curl_rc={summary['curl_rc']}")
if summary['asserts']:
    failures.append(f"asserts={summary['asserts']}")
if not request_input:
    failures.append('missing request prefill trace')
elif label == 'implicit':
    if request_input['use_implicit'] != 1 or request_input['actual_mask_bytes'] != 0 or request_input['actual_fa_mask_bytes'] != 0:
        failures.append('implicit request did not drop dense mask')
elif label in {'dense', 'format_none', 'packed16_disable'}:
    if request_input['use_implicit'] != 0 or request_input['actual_mask_bytes'] <= 0 or request_input['actual_fa_mask_bytes'] <= 0:
        failures.append(f'{label} request did not keep dense mask')
if failures:
    raise SystemExit('FAIL ' + label + ': ' + ', '.join(failures))
PY
}

idx=0
for case_name in $CASES; do
    idx=$((idx + 1))
    port=$((PORT_BASE + idx))
    case "$case_name" in
        dense)
            run_case dense 0 packed8_q4 0 "$port"
            ;;
        implicit)
            run_case implicit 1 packed8_q4 0 "$port"
            ;;
        format_none)
            run_case format_none 1 none 0 "$port"
            ;;
        packed16_disable)
            run_case packed16_disable 1 packed8_q4 1 "$port"
            ;;
        *)
            echo "FAIL: unknown case '$case_name'" >&2
            exit 2
            ;;
    esac
done

python3 - "$OUT_DIR" $CASES <<'PY' > "$OUT_DIR/summary.json"
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
case_names = sys.argv[2:]
rows = []
for name in case_names:
    path = out / name / 'trace-summary.json'
    if path.exists():
        rows.append(json.loads(path.read_text()))
by_label = {r['label']: r for r in rows}
summary = {
    'artifact_dir': str(out),
    'cases': rows,
}
if 'dense' in by_label and 'implicit' in by_label:
    dense = by_label['dense']
    implicit = by_label['implicit']
    d = dense.get('request_input') or {}
    i = implicit.get('request_input') or {}
    saved = (d.get('actual_fa_mask_bytes') or 0) - (i.get('actual_fa_mask_bytes') or 0)
    summary['same_route_ab'] = {
        'mask_bytes_saved': saved,
        'mask_mib_saved': round(saved / 1048576, 3),
        'rocm_compute_mib_delta_implicit_minus_dense': None if dense.get('rocm_compute_buffer_mib') is None or implicit.get('rocm_compute_buffer_mib') is None else round(implicit['rocm_compute_buffer_mib'] - dense['rocm_compute_buffer_mib'], 2),
        'host_compute_mib_delta_implicit_minus_dense': None if dense.get('host_compute_buffer_mib') is None or implicit.get('host_compute_buffer_mib') is None else round(implicit['host_compute_buffer_mib'] - dense['host_compute_buffer_mib'], 2),
        'prompt_tps_ratio_implicit_over_dense': None if not dense.get('prompt_tps') or not implicit.get('prompt_tps') else round(implicit['prompt_tps'] / dense['prompt_tps'], 4),
        'content_equal': dense.get('content') == implicit.get('content'),
    }
print(json.dumps(summary, indent=2))
PY

if [[ "$DRY_RUN" == "0" ]]; then
    python3 - "$OUT_DIR/summary.json" <<'PY'
import json
import sys
s = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
print('PASS: PDMQ implicit mask canary')
for row in s['cases']:
    req = row.get('request_input') or {}
    print(
        f"  {row['label']}: http={row['http_code']} use_implicit={req.get('use_implicit')} "
        f"mask={req.get('actual_fa_mask_bytes')} prompt_tps={row.get('prompt_tps')}"
    )
if 'same_route_ab' in s:
    ab = s['same_route_ab']
    print(
        f"  same-route A/B: saved={ab['mask_mib_saved']} MiB "
        f"rocm_delta={ab['rocm_compute_mib_delta_implicit_minus_dense']} MiB "
        f"host_delta={ab['host_compute_mib_delta_implicit_minus_dense']} MiB "
        f"prefill_ratio={ab['prompt_tps_ratio_implicit_over_dense']}"
    )
print(f"Summary: {sys.argv[1]}")
PY
else
    echo "DRY_RUN: summary skeleton written under $OUT_DIR"
fi
