#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
server=${LLAMA_SERVER_BIN:-/home/mrtrent/.local/bin/llama-server-wrapper}
server_process_pattern=${LLAMA_SERVER_PROCESS_PATTERN:-$repo/build-rocm/bin/llama-server}
model=${LLAMA_MODEL:-/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf}
template=${LLAMA_CHAT_TEMPLATE:-/home/mrtrent/.pi/agent/qwen36-merged-template.jinja}
rocm_smi=${ROCM_SMI:-/opt/rocm-7.2.3/bin/rocm-smi}
ctx_size=${CTX_SIZE:-8192}
max_tokens=${MAX_TOKENS:-32}
port_base=${PORT_BASE:-16160}
startup_wait=${STARTUP_WAIT_SECONDS:-180}
cleanup_wait=${CLEANUP_WAIT_SECONDS:-120}
vram_limit=${VRAM_LIMIT_BYTES:-3000000000}
logdir=${LOGDIR:-/tmp/compressed_kv_wmma_smokes_$(date +%Y%m%d_%H%M%S)}
prompt=${PROMPT:-$'3+3=6\n4+4='}

mkdir -p "$logdir"

vram_used() {
    "$rocm_smi" --showmeminfo vram 2>/dev/null | awk -F: '/VRAM Total Used Memory \(B\)/ { v=$NF; gsub(/^[ \t]+|[ \t]+$/, "", v); print v; exit }'
}

wait_vram_below() {
    local label=$1
    local used
    for _ in $(seq 1 "$cleanup_wait"); do
        used=$(vram_used || true)
        if [[ -n ${used:-} && $used -lt $vram_limit ]]; then
            echo "PASS: VRAM below limit after $label: $used bytes"
            return 0
        fi
        sleep 1
    done
    used=$(vram_used || true)
    echo "FAIL: VRAM did not return below $vram_limit bytes after $label; current=${used:-unknown}" >&2
    return 1
}

check_no_server() {
    local found
    found=$(pgrep -af "$server_process_pattern" || true)
    if [[ -n $found ]]; then
        echo "FAIL: existing llama-server process detected; unload/stop it before standalone WMMA smokes" >&2
        echo "$found" >&2
        return 1
    fi
}

cleanup_group() {
    local pid=${1:-}
    if [[ -z $pid ]]; then
        return 0
    fi
    kill -TERM -- "-$pid" 2>/dev/null || true
    sleep 4
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

wait_health() {
    local port=$1
    local health_file=$2
    for i in $(seq 1 "$startup_wait"); do
        if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >"$health_file" 2>&1; then
            echo "PASS: health ready on port $port after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "FAIL: health did not become ready on port $port" >&2
    return 1
}

completion_request() {
    local port=$1
    local response_file=$2
    local body_file=$3
    python3 - "$prompt" "$max_tokens" >"$logdir/payload.json" <<'PY'
import json, sys
prompt = sys.argv[1]
max_tokens = int(sys.argv[2])
print(json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0}))
PY
    curl -sS --max-time 180 -w '\nHTTP_STATUS:%{http_code}\n' \
        -H 'Content-Type: application/json' \
        -d @"$logdir/payload.json" \
        "http://127.0.0.1:${port}/v1/completions" >"$response_file"
    local status
    status=$(tail -n 1 "$response_file" | sed 's/^HTTP_STATUS://')
    sed '$d' "$response_file" >"$body_file"
    if [[ $status != 200 ]]; then
        echo "FAIL: completion HTTP status $status on port $port" >&2
        tail -n 40 "$response_file" >&2
        return 1
    fi
}

validate_completion() {
    local body_file=$1
    local completion_file=$2
    python3 - "$body_file" "$completion_file" <<'PY'
import json, sys
body_path, completion_path = sys.argv[1:3]
data = json.load(open(body_path, encoding="utf-8"))
text = data["choices"][0].get("text", "")
open(completion_path, "w", encoding="utf-8").write(text)
if not text.strip():
    raise SystemExit("empty completion")
if "33333" in text:
    raise SystemExit(f"garbage completion: {text!r}")
print(repr(text))
PY
}

run_one() {
    local name=$1
    local cache_type=$2
    local gate=$3
    local expected_kernel=$4
    local port=$5
    local log="$logdir/${name}.server.log"
    local response="$logdir/${name}.response.txt"
    local body="$logdir/${name}.body.json"
    local completion="$logdir/${name}.completion.txt"
    local health="$logdir/${name}.health.txt"
    local pid=""

    echo "=== $name: cache=$cache_type gate=$gate expected=$expected_kernel port=$port ==="
    check_no_server
    wait_vram_below "preflight $name"

    setsid bash -c 'ulimit -c 0; exec env "$@"' _ \
        "$gate=1" COMPRESSED_KV_FATTN_LOG=1 \
        "$server" \
        --port "$port" \
        --model "$model" \
        --ctx-size "$ctx_size" \
        --flash-attn on \
        --no-context-shift \
        --host 127.0.0.1 \
        --no-webui \
        --jinja \
        --chat-template-file "$template" \
        --no-mmap \
        --mlock \
        --threads 12 \
        --batch-size 1024 \
        --ubatch-size 512 \
        --cache-type-k "$cache_type" \
        --cache-type-v "$cache_type" \
        --cache-ram 128 \
        --spec-type mtp \
        --spec-draft-n-max 3 \
        --parallel 1 \
        --no-warmup \
        >"$log" 2>&1 &
    pid=$!

    if ! wait_health "$port" "$health"; then
        cleanup_group "$pid"
        wait_vram_below "failed startup cleanup $name" || true
        return 1
    fi
    if ! completion_request "$port" "$response" "$body"; then
        cleanup_group "$pid"
        wait_vram_below "failed request cleanup $name" || true
        return 1
    fi
    echo -n "completion: "
    if ! validate_completion "$body" "$completion"; then
        cleanup_group "$pid"
        wait_vram_below "failed completion cleanup $name" || true
        return 1
    fi

    if ! grep -q "kernel=$expected_kernel" "$log"; then
        echo "FAIL: expected route kernel=$expected_kernel not found in $log" >&2
        grep -n 'ggml_cuda_fattn_log_selection' "$log" | tail -40 >&2 || true
        cleanup_group "$pid"
        wait_vram_below "failed route cleanup $name" || true
        return 1
    fi
    echo "PASS: route log contains kernel=$expected_kernel"

    cleanup_group "$pid"
    pid=""
    wait_vram_below "cleanup $name"
    check_no_server
    echo "PASS: $name smoke complete"
}

main() {
    if [[ ! -x $server ]]; then
        echo "FAIL: missing server executable/wrapper: $server" >&2
        return 1
    fi
    if [[ ! -f $model ]]; then
        echo "FAIL: missing model: $model" >&2
        return 1
    fi
    if [[ ! -f $template ]]; then
        echo "FAIL: missing template: $template" >&2
        return 1
    fi
    if [[ ! -x $rocm_smi ]]; then
        echo "FAIL: missing rocm-smi: $rocm_smi" >&2
        return 1
    fi

    echo "logdir=$logdir"
    echo "ctx_size=$ctx_size max_tokens=$max_tokens vram_limit=$vram_limit"

    run_one planar3 planar3_0 COMPRESSED_KV_WMMA_FATTN wmma_compressed_kv "$((port_base + 1))"
    run_one iso3    iso3_0    COMPRESSED_KV_WMMA_FATTN wmma_compressed_kv "$((port_base + 2))"
    run_one tbq4    tbq4_0    TBQ4_WMMA_FATTN          wmma_tbq4          "$((port_base + 3))"

    echo "PASS: compressed-KV opt-in WMMA smoke suite complete"
    echo "logs: $logdir"
}

main "$@"
