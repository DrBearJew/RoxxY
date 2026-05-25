#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/scripts/hip/q8k-dot4-tile16-kq-microbench.hip"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT/benches/rocm-rdna3/q8k-dot4-tile16-kq-microbench-$STAMP}"
CXX="${HIPCXX:-/opt/rocm/bin/hipcc}"
GPU="${GPU:-gfx1100}"
NQ="${NQ:-256}"
NK="${NK:-1024}"
WARMUP="${WARMUP:-20}"
REPS="${REPS:-100}"
SEED="${SEED:-12345}"

mkdir -p "$OUT_DIR"

cat > "$OUT_DIR/manifest.txt" <<EOF
source=$SRC
out_dir=$OUT_DIR
gpu=$GPU
nq=$NQ
nk=$NK
warmup=$WARMUP
reps=$REPS
seed=$SEED
EOF

set -x
"$CXX" -O3 --offload-arch="$GPU" -DRDNA3=1 "$SRC" -o "$OUT_DIR/q8k_dot4_tile16_kq_microbench" >"$OUT_DIR/compile.log" 2>&1
set +x

"$OUT_DIR/q8k_dot4_tile16_kq_microbench" --nq "$NQ" --nk "$NK" --warmup "$WARMUP" --reps "$REPS" --seed "$SEED" \
  > "$OUT_DIR/raw-result.json" 2> "$OUT_DIR/run.log"

python3 - <<'PY' "$OUT_DIR"
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
obj = json.loads((out / 'raw-result.json').read_text())
t = obj['timings']; c = obj['correctness']
with (out / 'summary.md').open('w') as f:
    f.write('# q8K DOT4 tile16 KQ microbench\n\n')
    f.write(f"- source: `{obj['kind']}`\n")
    f.write(f"- shape: nq={obj['nq']} nk={obj['nk']} d={obj['d']} tile={obj['tile_m']}x{obj['tile_n']}\n")
    f.write(f"- dot4 per Q/K pair: {obj['dot4_per_pair']}\n")
    f.write(f"- dot4 per 16x16 tile: {obj['dot4_per_16x16_tile']}\n\n")
    f.write('## Timing\n\n')
    f.write('| layout | ms | dot4 GOP/s |\n|---|---:|---:|\n')
    f.write(f"| q8_0 block 34B | {t['tile16_q8block_ms']:.6f} | {t['q8block_dot4_gops']:.3f} |\n")
    f.write(f"| packed16 payload + q8 scales | {t['tile16_packed16_ms']:.6f} | {t['packed16_dot4_gops']:.3f} |\n")
    f.write(f"| pack q8block→packed16 only | {t['pack_q8block_to_packed16_ms']:.6f} | n/a |\n")
    f.write(f"| pack + packed16 KQ | {t['pack_plus_tile16_packed16_ms']:.6f} | n/a |\n\n")
    f.write(f"- packed16 speedup vs q8block: `{t['speedup_packed16_vs_q8block']:.4f}x`\n")
    f.write(f"- pack+packed16 speedup vs q8block: `{t['speedup_pack_plus_packed16_vs_q8block']:.4f}x`\n\n")
    f.write('## Correctness\n\n')
    f.write(f"- q8block rel RMS vs ref: `{c['q8block_rel_rms_gpu_vs_ref']:.6g}`\n")
    f.write(f"- packed16 rel RMS vs ref: `{c['packed16_rel_rms_gpu_vs_ref']:.6g}`\n")
    f.write(f"- q8block max abs vs ref: `{c['q8block_max_abs_gpu_vs_ref']:.6g}`\n")
    f.write(f"- packed16 max abs vs ref: `{c['packed16_max_abs_gpu_vs_ref']:.6g}`\n")
print(out / 'summary.md')
print((out / 'summary.md').read_text())
PY
