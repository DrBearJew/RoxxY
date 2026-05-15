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
CASES=${CASES:-baseline rdna2_opt maxx32 maxx48 maxx64 maxx128}
POLICY_ARGS=${POLICY_ARGS:-}

COMMON=(
  -m "$MODEL"
  -p "$PROMPTS" -n 0
  -fa 1 -ctk tbq4_0 -ctv tbq4_0
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

run_case() {
  local name=$1
  local log="$OUT_DIR/${name}.jsonl.log"
  echo "=== ${name} ===" | tee "$log"
  (
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
    export LD_LIBRARY_PATH="$repo/build-rocm/bin:/opt/rocm-7.2.3/lib:/opt/amdgpu/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    unset TBQ4_COOP_SET_ROWS TBQ4_INNERQ TBQ4_LAYER_ADAPTIVE GGML_CUDA_FORCE_MMQ GGML_CUDA_FORCE_CUBLAS
    unset LLAMA_MTP_PREFILL_FORCE_MMQ LLAMA_MTP_PREFILL_CHUNK TBQ4_WMMA_FATTN COMPRESSED_KV_WMMA_FATTN
    case "$name" in
      baseline)
        ;;
      rdna2_opt)
        export RDNA2_MATMUL_OPT_V1=1
        unset GGML_CUDA_MMQ_MAX_X
        ;;
      maxx32)
        export RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=32
        ;;
      maxx48)
        export RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=48
        ;;
      maxx64)
        export RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=64
        ;;
      maxx96)
        export RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=96
        ;;
      maxx128|native)
        export RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=128
        ;;
      *)
        echo "FAIL: unknown case '$name'" >&2
        exit 2
        ;;
    esac
    "$BENCH" "${COMMON[@]}"
  ) 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  echo "=== ${name} rc=${rc} ===" | tee -a "$log"
  return "$rc"
}

rc_total=0
for case_name in $CASES; do
  run_case "$case_name" || rc_total=1
  sleep 5
done

"$PYTHON" - "$OUT_DIR" $CASES <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
case_names = sys.argv[2:]
all_cases = {}
variant_rc = {}
for name in case_names:
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
                    rc = int(line.rsplit("=", 1)[1].split()[0])
                except Exception:
                    rc = 1
    rows.sort(key=lambda x: x.get("n_prompt", 0))
    all_cases[name] = rows
    variant_rc[name] = rc

summary = {"artifact": str(out), "variant_rc": variant_rc, "cases": all_cases}
(out / "summary.caps.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

prompts = sorted({int(r.get("n_prompt", 0)) for rows in all_cases.values() for r in rows if r.get("n_prompt")})
lines = [
    "# RDNA3 MMQ cap sweep",
    "",
    f"Artifact: `{out}`",
    "",
    "Command shape: `llama-bench -p " + ",".join(map(str, prompts)) + " -n 0 -fa 1 -ctk tbq4_0 -ctv tbq4_0 -b ${BATCH:-1024} -ub ${UBATCH:-512} -r ${REPS:-3} -o jsonl`",
    "",
    "| variant | " + " | ".join(f"pp{p} tok/s" for p in prompts) + " | rc |",
    "|---|" + "---:|" * len(prompts) + "---:|",
]
for name in case_names:
    vals = []
    rows_by_pp = {int(r.get("n_prompt", 0)): r for r in all_cases.get(name, [])}
    for pp in prompts:
        r = rows_by_pp.get(pp)
        vals.append(f"{r['avg_ts']:.1f} ± {r.get('stddev_ts', 0.0):.1f}" if r else "missing")
    lines.append(f"| `{name}` | " + " | ".join(vals) + f" | {variant_rc.get(name, 1)} |")
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
