#!/usr/bin/env bash
set -u -o pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"

MODEL=${MODEL:-/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}
BENCH=${BENCH:-$repo/build-rocm/bin/llama-bench}
PYTHON=${PYTHON:-/home/mrtrent/miniconda3/envs/LLM/bin/python}
OUT_DIR=${OUT_DIR:-$repo/benches/rocm-rdna3/mmq-cap-sweep-$(date +%Y%m%d-%H%M%S)}
PROMPTS=${PROMPTS:-128,256,512,1024,2048,4096}
REPS=${REPS:-3}
BATCH=${BATCH:-1024}
UBATCH=${UBATCH:-512}
SLEEP_SECS=${SLEEP_SECS:-5}
DEFAULT_CASES=${DEFAULT_CASES:-baseline rdna2_opt maxx32 maxx48 maxx64 maxx128}
FA_EXPERIMENT_CASES=${FA_EXPERIMENT_CASES:-maxx48_f16_mma maxx64_f16_mma maxx48_tbq4_wmma maxx64_tbq4_wmma}
if [[ ${INCLUDE_FA_EXPERIMENTS:-0} == 1 && -z ${CASES+x} ]]; then
  CASES="$DEFAULT_CASES $FA_EXPERIMENT_CASES"
else
  CASES=${CASES:-$DEFAULT_CASES}
fi
POLICY_ARGS=${POLICY_ARGS:-}
CACHE_TYPE_K=${CACHE_TYPE_K:-tbq4_0}
CACHE_TYPE_V=${CACHE_TYPE_V:-$CACHE_TYPE_K}
EXTRA_BENCH_ARGS=${EXTRA_BENCH_ARGS:-}
read -r -a EXTRA_BENCH_ARGS_ARR <<< "$EXTRA_BENCH_ARGS"

COMMON_BASE=(
  -m "$MODEL"
  -p "$PROMPTS" -n 0
  -fa 1
  -b "$BATCH" -ub "$UBATCH"
  -r "$REPS" -o jsonl
)

mkdir -p "$OUT_DIR"

if [[ ! -x $BENCH ]]; then
  echo "FAIL: missing llama-bench: $BENCH" >&2
  exit 2
fi
if [[ ! -f $MODEL ]]; then
  echo "FAIL: missing model: $MODEL" >&2
  exit 2
fi

case_setup() {
  local name=$1
  local base=$name
  CASE_ROUTE="tbq4_vec"
  CASE_CTK="$CACHE_TYPE_K"
  CASE_CTV="$CACHE_TYPE_V"
  CASE_MMQ_X=""
  CASE_ENV=()
  CASE_NOTE="default compressed-KV VEC FlashAttention route"

  case "$base" in
    *_f16_mma|*_f16_fa)
      CASE_ROUTE="f16_mma"
      CASE_CTK="f16"
      CASE_CTV="f16"
      CASE_NOTE="f16 KV route to exercise the f16 MMA/WMMA FlashAttention dispatch"
      base=${base%_f16_mma}
      base=${base%_f16_fa}
      ;;
    *_tbq4_wmma|*_tbq4_wmma_fa)
      CASE_ROUTE="tbq4_wmma"
      CASE_CTK="tbq4_0"
      CASE_CTV="tbq4_0"
      CASE_ENV+=("TBQ4_WMMA_FATTN=1")
      CASE_NOTE="experimental direct TBQ4 rocWMMA FlashAttention route; canary before promotion"
      base=${base%_tbq4_wmma}
      base=${base%_tbq4_wmma_fa}
      ;;
    *_planar_wmma|*_planar_wmma_fa)
      CASE_ROUTE="planar_wmma"
      CASE_CTK="planar3_0"
      CASE_CTV="planar3_0"
      CASE_ENV+=("COMPRESSED_KV_WMMA_FATTN=1")
      CASE_NOTE="experimental Planar3 rocWMMA FlashAttention route; canary before promotion"
      base=${base%_planar_wmma}
      base=${base%_planar_wmma_fa}
      ;;
    *_iso_wmma|*_iso_wmma_fa)
      CASE_ROUTE="iso_wmma"
      CASE_CTK="iso3_0"
      CASE_CTV="iso3_0"
      CASE_ENV+=("COMPRESSED_KV_WMMA_FATTN=1")
      CASE_NOTE="experimental Iso3 rocWMMA FlashAttention route; canary before promotion"
      base=${base%_iso_wmma}
      base=${base%_iso_wmma_fa}
      ;;
    *_tbq4_vec)
      CASE_ROUTE="tbq4_vec"
      CASE_CTK="tbq4_0"
      CASE_CTV="tbq4_0"
      CASE_NOTE="explicit compressed-KV VEC FlashAttention route"
      base=${base%_tbq4_vec}
      ;;
  esac

  CASE_BASE="$base"
  case "$base" in
    baseline)
      ;;
    rdna2_opt)
      CASE_ENV+=("RDNA2_MATMUL_OPT_V1=1")
      ;;
    scratch16k)
      CASE_ENV+=("RDNA2_MATMUL_OPT_V1=1" "GGML_CUDA_IQ4_XS_MMQ_SCRATCH16K=1")
      CASE_MMQ_X="64"
      ;;
    native)
      CASE_ENV+=("RDNA2_MATMUL_OPT_V1=1" "GGML_CUDA_MMQ_MAX_X=128")
      CASE_MMQ_X="128"
      ;;
    maxx*)
      local cap=${base#maxx}
      if [[ ! $cap =~ ^[0-9]+$ ]]; then
        echo "FAIL: bad MAX_X case '$name'" >&2
        return 2
      fi
      CASE_ENV+=("RDNA2_MATMUL_OPT_V1=1" "GGML_CUDA_MMQ_MAX_X=$cap")
      CASE_MMQ_X="$cap"
      ;;
    *)
      echo "FAIL: unknown case '$name'" >&2
      return 2
      ;;
  esac
}

write_case_meta() {
  local name=$1
  "$PYTHON" - "$OUT_DIR/${name}.meta.json" "$name" "$CASE_BASE" "$CASE_ROUTE" "$CASE_CTK" "$CASE_CTV" "${CASE_MMQ_X:-}" "$CASE_NOTE" "${CASE_ENV[@]}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
name, base, route, ctk, ctv, mmq_x, note = sys.argv[2:9]
env = {}
for item in sys.argv[9:]:
    if "=" in item:
        k, v = item.split("=", 1)
        env[k] = v
meta = {
    "name": name,
    "base": base,
    "route": route,
    "cache_type_k": ctk,
    "cache_type_v": ctv,
    "flash_attn": True,
    "mmq_x": int(mmq_x) if mmq_x else None,
    "env": env,
    "note": note,
}
path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_case() {
  local name=$1
  local log="$OUT_DIR/${name}.jsonl.log"
  case_setup "$name" || return 2
  write_case_meta "$name"
  echo "=== ${name} ===" | tee "$log"
  echo "route=${CASE_ROUTE} ctk=${CASE_CTK} ctv=${CASE_CTV} env=${CASE_ENV[*]:-none}" | tee -a "$log"
  (
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
    export LD_LIBRARY_PATH="$repo/build-rocm/bin:/opt/rocm-7.2.3/lib:/opt/amdgpu/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    unset TBQ4_COOP_SET_ROWS TBQ4_INNERQ TBQ4_LAYER_ADAPTIVE GGML_CUDA_FORCE_MMQ GGML_CUDA_FORCE_CUBLAS
    unset LLAMA_MTP_PREFILL_FORCE_MMQ LLAMA_MTP_PREFILL_CHUNK TBQ4_WMMA_FATTN COMPRESSED_KV_WMMA_FATTN
    unset RDNA2_MATMUL_OPT_V1 GGML_CUDA_MMQ_MAX_X GGML_CUDA_IQ4_XS_MMQ_SCRATCH16K
    for assignment in "${CASE_ENV[@]}"; do
      export "$assignment"
    done
    "$BENCH" "${COMMON_BASE[@]}" -ctk "$CASE_CTK" -ctv "$CASE_CTV" "${EXTRA_BENCH_ARGS_ARR[@]}"
  ) 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  echo "=== ${name} rc=${rc} ===" | tee -a "$log"
  return "$rc"
}

rc_total=0
for case_name in $CASES; do
  run_case "$case_name" || rc_total=1
  sleep "$SLEEP_SECS"
done

"$PYTHON" - "$OUT_DIR" $CASES <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
case_names = sys.argv[2:]
all_cases = {}
variant_rc = {}
variant_meta = {}
for name in case_names:
    meta_path = out / f"{name}.meta.json"
    if meta_path.exists():
        variant_meta[name] = json.loads(meta_path.read_text(encoding="utf-8"))

    log = out / f"{name}.jsonl.log"
    rows = []
    rc = 1
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("n_gen") == 0:
                    rows.append(obj)
            if line.startswith(f"=== {name} rc="):
                try:
                    rc = int(line.split(" rc=", 1)[1].split()[0])
                except Exception:
                    rc = 1
    rows.sort(key=lambda x: x.get("n_prompt", 0))
    all_cases[name] = rows
    variant_rc[name] = rc

summary = {"artifact": str(out), "variant_rc": variant_rc, "variant_meta": variant_meta, "cases": all_cases}
(out / "summary.caps.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

prompts = sorted({int(r.get("n_prompt", 0)) for rows in all_cases.values() for r in rows if r.get("n_prompt")})
prompt_s = ",".join(map(str, prompts)) if prompts else "${PROMPTS}"
lines = [
    "# RDNA3 MMQ cap / FlashAttention sweep",
    "",
    f"Artifact: `{out}`",
    "",
    "Base command shape: `llama-bench -p " + prompt_s + " -n 0 -fa 1 -b ${BATCH:-1024} -ub ${UBATCH:-512} -r ${REPS:-3} -o jsonl`",
    "",
    "| variant | route | cache K/V | env | " + " | ".join(f"pp{p} tok/s" for p in prompts) + " | rc |",
    "|---|---|---|---|" + "---:|" * len(prompts) + "---:|",
]
for name in case_names:
    meta = variant_meta.get(name, {})
    env = meta.get("env") or {}
    env_s = " ".join(f"{k}={v}" for k, v in env.items()) or "none"
    cache_s = f"{meta.get('cache_type_k', '?')}/{meta.get('cache_type_v', '?')}"
    route_s = meta.get("route", "?")
    vals = []
    rows_by_pp = {int(r.get("n_prompt", 0)): r for r in all_cases.get(name, [])}
    for pp in prompts:
        r = rows_by_pp.get(pp)
        vals.append(f"{r['avg_ts']:.1f} ± {r.get('stddev_ts', 0.0):.1f}" if r else "missing")
    lines.append(f"| `{name}` | `{route_s}` | `{cache_s}` | `{env_s}` | " + " | ".join(vals) + f" | {variant_rc.get(name, 1)} |")
(out / "summary.caps.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

"$PYTHON" scripts/hip/rdna3-mmq-policy.py \
  --summary "$OUT_DIR/summary.caps.json" \
  --out-dir "$OUT_DIR/policy" \
  --require-summary \
  --allow-over-budget \
  $POLICY_ARGS

echo "policy: $OUT_DIR/policy/policy.md"
exit "$rc_total"
