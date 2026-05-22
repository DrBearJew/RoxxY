#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(git -C "$SCRIPT_DIR/../.." rev-parse --show-toplevel)}
BUILD_DIR=${BUILD_DIR:-$REPO/build-rocm-rdna2-fa}
LLAMA_DEBUG=${LLAMA_DEBUG:-$BUILD_DIR/bin/llama-debug}
MODEL=${MODEL:-/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-IQ4_XS-00001-of-00002.gguf}
OUT=${OUT:-$REPO/benches/rocm-rdna3/q8q4-wmma-i8-long384-repro-matrix-$(date +%Y%m%d-%H%M%S)}
REPS=${REPS:-2}
THREADS=${THREADS:-12}
CTX=${CTX:-1536}
BATCH=${BATCH:-1536}
UBATCH=${UBATCH:-1024}
MAX_REL_RMS=${MAX_REL_RMS:-0.10}
MAX_REPRO_REL_RMS=${MAX_REPRO_REL_RMS:-1e-7}
PROMPT_FILE=${PROMPT_FILE:-}

if [[ ! -x "$LLAMA_DEBUG" ]]; then
  echo "error: llama-debug not executable: $LLAMA_DEBUG" >&2
  exit 2
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: model not found: $MODEL" >&2
  echo "set MODEL=/path/to/model.gguf" >&2
  exit 2
fi

mkdir -p "$OUT"
if [[ -z "$PROMPT_FILE" ]]; then
  PROMPT_FILE="$OUT/prompts/long_384_notes.txt"
  mkdir -p "$(dirname "$PROMPT_FILE")"
  python3 - "$PROMPT_FILE" <<'PY'
import sys
from pathlib import Path
phrase = (
    "Layer-filtered routing compares logits against a disabled baseline, "
    "records top1 parity, and rejects unsafe defaults when MoE amplification "
    "moves margins. "
)
prompt = "Summarize these benchmark notes in three bullets. " + phrase * 32
Path(sys.argv[1]).write_text(prompt)
PY
fi

{
  echo "artifact=$OUT"
  echo "commit=$(git -C "$REPO" rev-parse HEAD)"
  echo "dirty=$(git -C "$REPO" status --short --untracked-files=no | wc -l)"
  echo "started=$(date -Is)"
  echo "purpose=Q8Q4_WMMA_I8 long_384_notes reproducibility/top1 gate"
  echo "model=$MODEL"
  echo "prompt=$PROMPT_FILE"
  echo "reps=$REPS ctx=$CTX batch=$BATCH ubatch=$UBATCH threads=$THREADS"
  echo "max_rel_rms=$MAX_REL_RMS max_repro_rel_rms=$MAX_REPRO_REL_RMS"
  echo "preexisting_q8q4_env_begin"
  env | sort | rg 'GGML_CUDA_ROCM_Q8Q4_WMMA_I8|LLAMA_DEBUG_TENSOR_DUMP_DIR' || true
  echo "preexisting_q8q4_env_end"
} | tee "$OUT/run.log"

run_one() {
  local rep=$1
  local mode=$2
  shift 2
  local logits_dir="$OUT/logits-${rep}-${mode}"
  mkdir -p "$logits_dir"
  echo "== rep=$rep mode=$mode $(date -Is) ==" | tee -a "$OUT/run.log"
  env \
    -u GGML_CUDA_ROCM_Q8Q4_WMMA_I8 \
    -u GGML_CUDA_ROCM_Q8Q4_WMMA_I8_UNSAFE \
    -u GGML_CUDA_ROCM_Q8Q4_WMMA_I8_QSCALE16 \
    -u GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER \
    -u GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MIN \
    -u GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MAX \
    -u GGML_CUDA_ROCM_Q8Q4_WMMA_I8_SKIP_LAYER \
    -u GGML_CUDA_ROCM_Q8Q4_WMMA_I8_SKIP_LAYERS \
    -u GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LOG \
    -u LLAMA_DEBUG_TENSOR_DUMP_DIR \
    "$@" \
    "$LLAMA_DEBUG" \
      -m "$MODEL" \
      -f "$PROMPT_FILE" \
      -c "$CTX" \
      -b "$BATCH" \
      -ub "$UBATCH" \
      -fa on \
      -ctk q8_0 \
      -ctv q4_0 \
      -ngl 99 \
      --no-mmap \
      --no-warmup \
      --threads "$THREADS" \
      --save-logits \
      --logits-output-dir "$logits_dir" \
      > "$OUT/${rep}-${mode}.stdout.log" \
      2> "$OUT/${rep}-${mode}.stderr.log"
}

for ((i = 1; i <= REPS; i++)); do
  rep="r$i"
  run_one "$rep" off \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8=0 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8_UNSAFE=0 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LOG=1
  run_one "$rep" min23 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8=1 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8_UNSAFE=1 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MIN=23 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LOG=1
  run_one "$rep" min27 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8=1 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8_UNSAFE=1 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MIN=27 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LOG=1
  run_one "$rep" min23_skip23 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8=1 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8_UNSAFE=1 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MIN=23 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8_SKIP_LAYER=23 \
    GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LOG=1
done

python3 - "$OUT" "$REPS" "$MAX_REL_RMS" "$MAX_REPRO_REL_RMS" <<'PY' | tee "$OUT/summary.tsv"
import array
import math
import pathlib
import sys

art = pathlib.Path(sys.argv[1])
reps = int(sys.argv[2])
max_rel = float(sys.argv[3])
max_repro_rel = float(sys.argv[4])
modes = ["min23", "min27", "min23_skip23"]
failures: list[str] = []

def vals(rep: str, mode: str) -> list[float]:
    p = next((art / f"logits-{rep}-{mode}").glob("*.bin"))
    a = array.array("f")
    a.frombytes(p.read_bytes())
    return list(a)

def top(v: list[float], n: int = 10) -> list[int]:
    return sorted(range(len(v)), key=lambda i: v[i], reverse=True)[:n]

def metrics(a: list[float], b: list[float]):
    d = [y - x for x, y in zip(a, b)]
    rms = math.sqrt(sum(x * x for x in d) / len(d))
    ar = math.sqrt(sum(x * x for x in a) / len(a))
    rel = rms / (ar + 1e-12)
    max_i = max(range(len(d)), key=lambda i: abs(d[i]))
    ta = top(a)
    tb = top(b)
    return {
        "rel": rel,
        "max_abs": abs(d[max_i]),
        "max_abs_idx": max_i,
        "top1_match": ta[0] == tb[0],
        "top10_intersection": len(set(ta) & set(tb)),
        "a_argmax": ta[0],
        "b_argmax": tb[0],
        "a_margin": a[ta[0]] - a[ta[1]],
        "b_margin": b[tb[0]] - b[tb[1]],
    }

def route_count(rep: str, mode: str) -> int:
    return (art / f"{rep}-{mode}.stderr.log").read_text(errors="ignore").count("route=q8q4_wmma_i8")

print("comparison\troute_count\trel_rms\tmax_abs\tmax_abs_idx\ttop1_match\ttop10_intersection\toff_argmax\tmode_argmax\toff_margin\tmode_margin")
for i in range(1, reps + 1):
    rep = f"r{i}"
    off = vals(rep, "off")
    for mode in modes:
        m = metrics(off, vals(rep, mode))
        rc = route_count(rep, mode)
        print(
            f"{rep}:{mode}_vs_off\t{rc}\t{m['rel']:.9g}\t{m['max_abs']:.9g}\t{m['max_abs_idx']}\t"
            f"{m['top1_match']}\t{m['top10_intersection']}\t{m['a_argmax']}\t{m['b_argmax']}\t"
            f"{m['a_margin']:.9g}\t{m['b_margin']:.9g}"
        )
        if rc <= 0:
            failures.append(f"{rep}:{mode} did not route")
        if not m["top1_match"]:
            failures.append(f"{rep}:{mode} top1 mismatch")
        if m["rel"] > max_rel:
            failures.append(f"{rep}:{mode} rel_rms {m['rel']:.9g} > {max_rel}")

if reps >= 2:
    for mode in ["off", *modes]:
        m = metrics(vals("r1", mode), vals("r2", mode))
        rc = route_count("r2", mode) if mode != "off" else 0
        print(
            f"r1_vs_r2:{mode}\t{rc}\t{m['rel']:.9g}\t{m['max_abs']:.9g}\t{m['max_abs_idx']}\t"
            f"{m['top1_match']}\t{m['top10_intersection']}\t{m['a_argmax']}\t{m['b_argmax']}\t"
            f"{m['a_margin']:.9g}\t{m['b_margin']:.9g}"
        )
        if m["rel"] > max_repro_rel:
            failures.append(f"r1_vs_r2:{mode} rel_rms {m['rel']:.9g} > {max_repro_rel}")
        if not m["top1_match"]:
            failures.append(f"r1_vs_r2:{mode} top1 mismatch")

status = "pass" if not failures else "fail"
(art / "gate-status.txt").write_text(status + "\n" + "\n".join(failures) + ("\n" if failures else ""))
if failures:
    for failure in failures:
        print(f"FAIL\t{failure}")
    sys.exit(1)
PY

echo "completed=$(date -Is)" | tee -a "$OUT/run.log"
echo "summary=$OUT/summary.tsv" | tee -a "$OUT/run.log"
echo "gate_status=$(cat "$OUT/gate-status.txt")" | tee -a "$OUT/run.log"
echo "$OUT"
