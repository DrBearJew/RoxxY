#!/usr/bin/env bash
set -euo pipefail

# tbq4-post-v3-acceptance.sh
#
# Runs a llama-bench matrix matching the user-provided original TBQ4 baseline
# and fails when post-v3 performance drops below the configured ratio.
#
# Defaults are intentionally env-overridable because the production v3 binary / 
# branch may live outside this checkout.

ROOT=${ROOT:-/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github}
LLAMA_BENCH=${LLAMA_BENCH:-$ROOT/build-rocm-fixed/bin/llama-bench}
MODEL=${MODEL:-/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf}
OUT_DIR=${OUT_DIR:-$ROOT/.harness/tmp/tbq4-post-v3-acceptance/$(date +%Y%m%d-%H%M%S)}

# Original table shape.
PROMPT_TOKENS=${PROMPT_TOKENS:-2048}
GEN_TOKENS=${GEN_TOKENS:-32}
DEPTHS=${DEPTHS:-"0 4096 8192 16384 32768"}

# Runtime defaults; override to match the production v3 setup exactly.
DEVICE=${DEVICE:-ROCm0}
N_GPU_LAYERS=${N_GPU_LAYERS:-99}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-1024}
CACHE_TYPE_K=${CACHE_TYPE_K:-f16}
CACHE_TYPE_V=${CACHE_TYPE_V:-q4_0}
FLASH_ATTN=${FLASH_ATTN:-1}
REPETITIONS=${REPETITIONS:-5}
EXTRA_ENV=${EXTRA_ENV:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

# Acceptance threshold relative to original baseline.
MIN_RATIO=${MIN_RATIO:-0.95}

mkdir -p "$OUT_DIR"
cd "$ROOT"

if [[ ! -x "$LLAMA_BENCH" ]]; then
  echo "ERROR: LLAMA_BENCH is not executable: $LLAMA_BENCH" >&2
  echo "Set LLAMA_BENCH=/path/to/production/v3/llama-bench" >&2
  exit 2
fi

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: MODEL not found: $MODEL" >&2
  echo "Set MODEL=/path/to/TBQ4/model.gguf" >&2
  exit 2
fi

DEPTH_CSV=$(python3 -c 'import sys; print(",".join(sys.argv[1:]))' $DEPTHS)

META="$OUT_DIR/run.meta.txt"
RAW_JSONL="$OUT_DIR/llama-bench.jsonl"
SUMMARY_JSON="$OUT_DIR/summary.json"
SUMMARY_CSV="$OUT_DIR/summary.csv"
SUMMARY_MD="$OUT_DIR/summary.md"

{
  echo "root=$ROOT"
  echo "llama_bench=$LLAMA_BENCH"
  echo "model=$MODEL"
  echo "out_dir=$OUT_DIR"
  echo "prompt_tokens=$PROMPT_TOKENS gen_tokens=$GEN_TOKENS depths=$DEPTHS"
  echo "device=$DEVICE ngl=$N_GPU_LAYERS batch=$BATCH_SIZE ubatch=$UBATCH_SIZE cache_k=$CACHE_TYPE_K cache_v=$CACHE_TYPE_V fa=$FLASH_ATTN repetitions=$REPETITIONS"
  echo "min_ratio=$MIN_RATIO"
  echo "extra_env=$EXTRA_ENV"
  echo "extra_args=$EXTRA_ARGS"
} > "$META"

# shellcheck disable=SC2086
if [[ -n "$EXTRA_ENV" ]]; then
  env $EXTRA_ENV "$LLAMA_BENCH" \
    -m "$MODEL" \
    -dev "$DEVICE" \
    -ngl "$N_GPU_LAYERS" \
    -fa "$FLASH_ATTN" \
    -ctk "$CACHE_TYPE_K" \
    -ctv "$CACHE_TYPE_V" \
    -b "$BATCH_SIZE" \
    -ub "$UBATCH_SIZE" \
    -p "$PROMPT_TOKENS" \
    -n "$GEN_TOKENS" \
    -d "$DEPTH_CSV" \
    -r "$REPETITIONS" \
    -o jsonl \
    $EXTRA_ARGS > "$RAW_JSONL" 2> "$OUT_DIR/llama-bench.stderr"
else
  # shellcheck disable=SC2086
  "$LLAMA_BENCH" \
    -m "$MODEL" \
    -dev "$DEVICE" \
    -ngl "$N_GPU_LAYERS" \
    -fa "$FLASH_ATTN" \
    -ctk "$CACHE_TYPE_K" \
    -ctv "$CACHE_TYPE_V" \
    -b "$BATCH_SIZE" \
    -ub "$UBATCH_SIZE" \
    -p "$PROMPT_TOKENS" \
    -n "$GEN_TOKENS" \
    -d "$DEPTH_CSV" \
    -r "$REPETITIONS" \
    -o jsonl \
    $EXTRA_ARGS > "$RAW_JSONL" 2> "$OUT_DIR/llama-bench.stderr"
fi

python3 - "$RAW_JSONL" "$SUMMARY_JSON" "$SUMMARY_CSV" "$SUMMARY_MD" "$MIN_RATIO" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

raw_path = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])
summary_md = Path(sys.argv[4])
min_ratio = float(sys.argv[5])

# User-provided original TBQ4 baseline, t/s.
BASELINE = {
    ("pp", 0): 732.33,
    ("tg", 0): 60.26,
    ("pp", 4096): 726.03,
    ("tg", 4096): 59.29,
    ("pp", 8192): 700.15,
    ("tg", 8192): 58.97,
    ("pp", 16384): 661.88,
    ("tg", 16384): 51.63,
    ("pp", 32768): 589.57,
    ("tg", 32768): 51.52,
}


def load_rows(path: Path):
    text = path.read_text(errors="replace").strip()
    if not text:
        return []
    # llama-bench -o jsonl emits one JSON object per line. Be tolerant of json arrays.
    if text[0] == "[":
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Ignore non-JSON progress/noise lines if any leak to stdout.
            continue
    return rows


def infer_kind(row):
    test = str(row.get("test") or row.get("name") or "").lower()
    n_prompt = int(row.get("n_prompt") or 0)
    n_gen = int(row.get("n_gen") or 0)
    if "pp" in test and "tg" not in test:
        return "pp"
    if "tg" in test:
        return "tg"
    if n_prompt > 0 and n_gen == 0:
        return "pp"
    if n_gen > 0 and n_prompt == 0:
        return "tg"
    # llama-bench sometimes keeps both params populated; use test string if present.
    if n_prompt >= 1024 and n_gen <= 0:
        return "pp"
    if n_gen > 0:
        return "tg"
    return "unknown"


def avg_ts(row):
    for key in ("avg_ts", "t/s", "tokens_per_second"):
        if key in row and row[key] is not None:
            return float(row[key])
    raise KeyError(f"no throughput field in row keys={sorted(row.keys())}")

rows = load_rows(raw_path)
seen = {}
for row in rows:
    kind = infer_kind(row)
    if kind not in {"pp", "tg"}:
        continue
    depth = int(row.get("n_depth") or row.get("depth") or 0)
    key = (kind, depth)
    if key not in BASELINE:
        continue
    seen[key] = row

summary = []
failed = False
for key in sorted(BASELINE, key=lambda k: (k[1], 0 if k[0] == "pp" else 1)):
    kind, depth = key
    baseline = BASELINE[key]
    threshold = baseline * min_ratio
    row = seen.get(key)
    if row is None:
        actual = None
        ratio = None
        status = "MISSING"
        failed = True
    else:
        actual = avg_ts(row)
        ratio = actual / baseline if baseline else math.nan
        status = "PASS" if actual >= threshold else "FAIL"
        failed = failed or status != "PASS"
    label = f"{'pp2048' if kind == 'pp' else 'tg32'}" + ("" if depth == 0 else f" @ d{depth}")
    summary.append({
        "test": label,
        "kind": kind,
        "depth": depth,
        "baseline_ts": baseline,
        "actual_ts": actual,
        "min_ratio": min_ratio,
        "threshold_ts": threshold,
        "actual_ratio": ratio,
        "status": status,
    })

summary_json.write_text(json.dumps(summary, indent=2))
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

headers = ["test", "baseline t/s", "actual t/s", "ratio", "threshold", "status"]
lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([":--", "--:", "--:", "--:", "--:", ":--"]) + "|"]
for r in summary:
    actual = "" if r["actual_ts"] is None else f"{r['actual_ts']:.2f}"
    ratio = "" if r["actual_ratio"] is None else f"{100*r['actual_ratio']:.1f}%"
    lines.append(f"| {r['test']} | {r['baseline_ts']:.2f} | {actual} | {ratio} | {r['threshold_ts']:.2f} | {r['status']} |")
summary_md.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if failed else 0)
PY

echo "OUT_DIR=$OUT_DIR"
