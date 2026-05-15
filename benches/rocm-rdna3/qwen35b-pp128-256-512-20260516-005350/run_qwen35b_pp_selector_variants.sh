#!/usr/bin/env bash
set -u
cd /home/mrtrent/llama.cpp-mtp-tbq4-rdna3
ART_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
BENCH=./build-rocm/bin/llama-bench
COMMON=(
  -m "$MODEL"
  -p 128,256,512 -n 0
  -fa 1 -ctk tbq4_0 -ctv tbq4_0
  -b 1024 -ub 512
  -r 5 -o jsonl
)
run_case() {
  local name="$1"; shift
  local log="$ART_DIR/${name}.clean.jsonl.log"
  echo "=== ${name} clean ===" | tee "$log"
  env HIP_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH="/home/mrtrent/llama.cpp-mtp-tbq4-rdna3/build-rocm/bin:/opt/rocm-7.2.3/lib:/opt/amdgpu/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
    "$@" \
    bash -lc 'unset TBQ4_COOP_SET_ROWS TBQ4_INNERQ TBQ4_LAYER_ADAPTIVE GGML_CUDA_FORCE_MMQ GGML_CUDA_FORCE_CUBLAS LLAMA_MTP_PREFILL_FORCE_MMQ LLAMA_MTP_PREFILL_CHUNK TBQ4_WMMA_FATTN COMPRESSED_KV_WMMA_FATTN; "$@"' _ "${BENCH}" "${COMMON[@]}" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  echo "=== ${name} clean rc=${rc} ===" | tee -a "$log"
  return "$rc"
}
run_case maxx48 RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=48; rc1=$?
sleep 5
run_case maxx64 RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=64; rc2=$?
sleep 5
run_case scratch16k RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_IQ4_XS_MMQ_SCRATCH16K=1; rc3=$?
/home/mrtrent/miniconda3/envs/LLM/bin/python - <<'PY' "$ART_DIR" "$rc1" "$rc2" "$rc3"
import json, sys
from pathlib import Path
art=Path(sys.argv[1]); rcs={'maxx48': int(sys.argv[2]), 'maxx64': int(sys.argv[3]), 'scratch16k': int(sys.argv[4])}
# Load existing clean summary if present.
all_cases={}
for name in ['baseline','rdna2_opt','maxx48','maxx64','scratch16k']:
    log=art/f'{name}.clean.jsonl.log'
    rows=[]
    if log.exists():
        for line in log.read_text(errors='replace').splitlines():
            if line.startswith('{'):
                obj=json.loads(line)
                if obj.get('n_gen') == 0 and obj.get('n_prompt') in (128,256,512):
                    rows.append(obj)
    rows.sort(key=lambda x: x.get('n_prompt',0))
    all_cases[name]=rows
summary={'artifact': str(art), 'variant_rc': rcs, 'cases': all_cases}
(art/'summary.variants.clean.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')

def val(name, pp):
    hits=[r for r in all_cases.get(name,[]) if r.get('n_prompt') == pp]
    return hits[-1] if hits else None
variants=['baseline','rdna2_opt','maxx48','maxx64','scratch16k']
lines=['# Qwen3.6 35B-A3B IQ4_XS pp selector variants','',f'Artifact: `{art}`','', 'Command shape: `llama-bench -p 128,256,512 -n 0 -fa 1 -ctk tbq4_0 -ctv tbq4_0 -b 1024 -ub 512 -r 5 -o jsonl`','Model: `/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`','Mode: non-MTP, ROCm/HIP, RX 7900 XTX.','', '| variant | pp128 tok/s | pp256 tok/s | pp512 tok/s |', '|---|---:|---:|---:|']
for name in variants:
    vals=[]
    for pp in [128,256,512]:
        r=val(name,pp)
        vals.append(f"{r['avg_ts']:.1f} ± {r['stddev_ts']:.1f}" if r else 'missing')
    lines.append(f'| `{name}` | {vals[0]} | {vals[1]} | {vals[2]} |')
lines += ['', 'Env variants:', '- `baseline`: no RDNA selector env', '- `rdna2_opt`: `RDNA2_MATMUL_OPT_V1=1`', '- `maxx48`: `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=48`', '- `maxx64`: `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=64`', '- `scratch16k`: `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_IQ4_XS_MMQ_SCRATCH16K=1`']
(art/'summary.variants.clean.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
exit $(( rc1 != 0 || rc2 != 0 || rc3 != 0 ))
