#!/usr/bin/env bash
set -u
cd /home/mrtrent/llama.cpp-mtp-tbq4-rdna3
ART_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
BENCH=./build-rocm/bin/llama-bench
COMMON=(
  -m "$MODEL"
  -pg 128,0 -pg 256,0 -pg 512,0
  -fa 1 -ctk tbq4_0 -ctv tbq4_0
  -b 1024 -ub 512
  -r 5 -o jsonl
)
run_case() {
  local name="$1"
  local opt="$2"
  local log="$ART_DIR/${name}.jsonl.log"
  echo "=== ${name} ===" | tee "$log"
  env \
    HIP_VISIBLE_DEVICES=0 \
    ${opt:+RDNA2_MATMUL_OPT_V1=$opt} \
    LD_LIBRARY_PATH="/home/mrtrent/llama.cpp-mtp-tbq4-rdna3/build-rocm/bin:/opt/rocm-7.2.3/lib:/opt/amdgpu/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
    bash -lc 'unset TBQ4_COOP_SET_ROWS TBQ4_INNERQ TBQ4_LAYER_ADAPTIVE GGML_CUDA_MMQ_MAX_X GGML_CUDA_FORCE_MMQ GGML_CUDA_FORCE_CUBLAS LLAMA_MTP_PREFILL_FORCE_MMQ LLAMA_MTP_PREFILL_CHUNK TBQ4_WMMA_FATTN COMPRESSED_KV_WMMA_FATTN GGML_CUDA_IQ4_XS_MMQ_SCRATCH16K; "$@"' _ "${BENCH}" "${COMMON[@]}" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  echo "=== ${name} rc=${rc} ===" | tee -a "$log"
  return "$rc"
}
run_case baseline ""; rc1=$?
sleep 5
run_case rdna2_opt "1"; rc2=$?
/home/mrtrent/miniconda3/envs/LLM/bin/python - <<'PY' "$ART_DIR" "$rc1" "$rc2"
import json, sys, re
from pathlib import Path
art=Path(sys.argv[1]); rcs={'baseline': int(sys.argv[2]), 'rdna2_opt': int(sys.argv[3])}
summary={'artifact': str(art), 'rc': rcs, 'cases': {}}
for name in ['baseline','rdna2_opt']:
    rows=[]
    log=art/f'{name}.jsonl.log'
    for line in log.read_text(errors='replace').splitlines():
        if line.startswith('{'):
            try:
                obj=json.loads(line)
            except Exception:
                continue
            if obj.get('n_gen') == 0 and obj.get('n_prompt') in (128,256,512):
                rows.append(obj)
    rows.sort(key=lambda x: x.get('n_prompt',0))
    summary['cases'][name]=rows
(art/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')

def row_value(rows, pp):
    for r in rows:
        if r.get('n_prompt') == pp:
            return r

lines=[]
lines.append('# Qwen3.6 35B-A3B IQ4_XS pp128/pp256/pp512 bench')
lines.append('')
lines.append(f'Artifact: `{art}`')
lines.append('')
lines.append('Command shape: `llama-bench -pg 128,0 -pg 256,0 -pg 512,0 -fa 1 -ctk tbq4_0 -ctv tbq4_0 -b 1024 -ub 512 -r 5 -o jsonl`')
lines.append('Model: `/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`')
lines.append('Mode: non-MTP, ROCm/HIP, RX 7900 XTX.')
lines.append('')
lines.append('| pp | baseline tok/s | RDNA2_MATMUL_OPT_V1=1 tok/s | delta |')
lines.append('|---:|---:|---:|---:|')
base=summary['cases']['baseline']; opt=summary['cases']['rdna2_opt']
for pp in [128,256,512]:
    b=row_value(base, pp); o=row_value(opt, pp)
    if b and o:
        delta=(o['avg_ts']/b['avg_ts']-1)*100
        lines.append(f"| {pp} | {b['avg_ts']:.1f} ± {b['stddev_ts']:.1f} | {o['avg_ts']:.1f} ± {o['stddev_ts']:.1f} | {delta:+.1f}% |")
    else:
        lines.append(f'| {pp} | missing | missing | — |')
lines.append('')
lines.append('Raw logs:')
lines.append('- `baseline.jsonl.log`')
lines.append('- `rdna2_opt.jsonl.log`')
(art/'summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
exit $(( rc1 != 0 || rc2 != 0 ))
