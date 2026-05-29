#!/usr/bin/env bash
# check-mtp-route-contract-canaries.sh — route contract validation
#
# Verifies that GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq
# accepts legal MTP verify paths and rejects illegal ones.
#
# Usage: ./scripts/check-mtp-route-contract-canaries.sh [PORT]
# Prerequisite: the full canary script should already have verified
# positive DOT4 selection; this script focuses on edge cases.

set -euo pipefail

PORT="${1:-18200}"
MODEL="${LLAMA_MODEL:-/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf}"
SERVER_BIN="${LLAMA_SERVER:-./build-rocm-rdna3-fa/bin/llama-server}"
LOG_DIR="/tmp/mtp-rc-canary-logs"
PASS=0
FAIL=0

mkdir -p "$LOG_DIR"

# ── helpers ────────────────────────────────────────────────────────

run_server() {
    local name="$1"; shift
    local log="$LOG_DIR/${name}.log"
    pkill -9 -f "llama-server.*port.*${PORT}" 2>/dev/null || true
    sleep 1
    "$@" > "$log" 2>&1 &
    local pid=$!
    for i in $(seq 1 60); do
        if curl -s "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; then
            echo "  server ready (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "  FAIL: server did not start"
    kill -9 $pid 2>/dev/null || true
    return 1
}

send() {
    curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"local","messages":[{"role":"user","content":"Hi"}],"max_tokens":8,"temperature":0}' \
        > /dev/null 2>&1
}

stop() {
    pkill -9 -f "llama-server.*port.*${PORT}" 2>/dev/null || true
    sleep 1
}

check() {
    local name="$1" pattern="$2" forbidden="${3:-}"
    local log="$LOG_DIR/${name}.log"
    if grep -q "$pattern" "$log"; then
        if [ -n "$forbidden" ] && grep -q "$forbidden" "$log"; then
            echo "FAIL: ${name} — forbidden: ${forbidden}"
            ((FAIL++)); return 1
        fi
        echo "PASS: ${name}"
        ((PASS++)); return 0
    fi
    echo "FAIL: ${name} — missing: ${pattern}"
    # diagnostic: show relevant log lines
    grep -E 'fa_inst=mtp|route_contract|impl_status|dot4' "$log" | head -10
    ((FAIL++)); return 1
}

# ── base env (all canaries share these) ────────────────────────────

BASE_ENV=(
    env
    COMPRESSED_KV_FATTN_LOG=1
    GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1
    GGML_CUDA_ROCM_Q8K_DOT4_KQ=1
    GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER=1
)
SRV_ARGS=(--model "$MODEL" --host 127.0.0.1 --port "$PORT"
    -fa on --batch-size 2048 --ubatch-size 1024
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0
    --parallel 1 --ctx-size 32000 --no-warmup --n-gpu-layers 99)

echo "=== Route-Contract Canaries ==="

# ── RC-1: route_require unset → DOT4 never engages (env_disabled) ─
echo "RC-1: route_require absent → env_disabled"
run_server "rc1" "${BASE_ENV[@]}" "$SERVER_BIN" "${SRV_ARGS[@]}"
send
stop
check "rc1" "fa_inst=mtp_verify_qk.*nq=4.*impl_status=env_disabled"

# ── RC-2: route_require set → DOT4 fires for nq=4 ──────────────────
echo "RC-2: route_require present → dot4_selected nq=4"
run_server "rc2" "${BASE_ENV[@]}" \
    GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq \
    "$SERVER_BIN" "${SRV_ARGS[@]}"
send
stop
check "rc2" "fa_inst=mtp_verify_qk.*nq=4.*impl_status=dot4_selected"

# ── RC-3: route_require set, nq=2 w/o NQ2 → nq_eq_2_disabled ─────
echo "RC-3: route_require present, nq=2 no NQ2 env → nq_eq_2_disabled"
run_server "rc3" "${BASE_ENV[@]}" \
    GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq \
    "$SERVER_BIN" "${SRV_ARGS[@]}"
send
stop
check "rc3" "fa_inst=mtp_verify_qk.*nq=2.*impl_status=nq_eq_2_disabled"

# ── RC-4: route_require set, nq=2 WITH NQ2 → dot4_selected ────────
echo "RC-4: route_require present, nq=2 + NQ2 env → dot4_selected"
run_server "rc4" "${BASE_ENV[@]}" \
    GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq \
    GGML_CUDA_ROCM_MTP_VERIFY_DOT4_NQ2=1 \
    "$SERVER_BIN" "${SRV_ARGS[@]}"
send
stop
check "rc4" "fa_inst=mtp_verify_qk.*nq=2.*impl_status=dot4_selected"

# ── RC-5: route_require set, nq=1 → decode fallback ────────────────
echo "RC-5: route_require present, nq=1 → decode (NOT recthist)"
run_server "rc5" "${BASE_ENV[@]}" \
    GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq \
    "$SERVER_BIN" "${SRV_ARGS[@]}"
send
stop
check "rc5" "fa_inst=mtp_verify_qk.*nq=1.*impl_status=nq_eq_1_decode_not_recthist"

# ── RC-6: MTP_DRAFT never candidates DOT4, even with route_require ─
echo "RC-6: route_require present, MTP_DRAFT → existing_policy"
run_server "rc6" "${BASE_ENV[@]}" \
    GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq \
    "$SERVER_BIN" "${SRV_ARGS[@]}"
send
stop
check "rc6" "fa_inst=mtp_draft.*impl_status=existing_policy" "fa_inst=mtp_draft.*dot4_selected"

# ── RC-7: route_require set WITH k_repr_candidate for rejections ───
echo "RC-7: k_repr_candidate in rejection log, k_repr only on selected"
run_server "rc7" "${BASE_ENV[@]}" \
    GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq \
    "$SERVER_BIN" "${SRV_ARGS[@]}"
send
stop
check "rc7" "mtp_verify_qk.*nq=1.*k_repr_candidate=source_f16_op_local_packed16.*impl_status=nq_eq_1_decode"

# ── RC-8: route_require set, nq=4 → k_repr (NOT candidate) ────────
echo "RC-8: k_repr (not candidate) when DOT4 selected"
run_server "rc8" "${BASE_ENV[@]}" \
    GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq \
    "$SERVER_BIN" "${SRV_ARGS[@]}"
send
stop
check "rc8" "mtp_verify_qk.*nq=4.*k_repr=source_f16_op_local_packed16.*impl_status=dot4_selected" \
    "mtp_verify_qk.*nq=4.*k_repr_candidate"

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "=== Route-Contract Results ==="
echo "PASS: ${PASS}"
echo "FAIL: ${FAIL}"
[ "$FAIL" -eq 0 ] && echo "ALL CANARIES PASSED" || echo "SOME CANARIES FAILED"
exit $FAIL
