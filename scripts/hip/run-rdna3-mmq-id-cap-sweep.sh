#!/usr/bin/env bash
set -u -o pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"

BUILD=${BUILD:-/tmp/llama-mmq-wmma-i8-build-gfx1100.nyQBiu}
BIN=${BIN:-$BUILD/bin/test-backend-ops}
PYTHON=${PYTHON:-python3}
OUT_DIR=${OUT_DIR:-$repo/benches/rocm-rdna3/mmq-test-backend-ops-mmid-caps-$(date +%Y%m%d-%H%M%S)}
CASES=${CASES:-default auto maxx32 maxx48 maxx64 maxx96 maxx128 cublas}
SLEEP_SECS=${SLEEP_SECS:-2}
# Match test_mul_mat_id::vars() exactly. ggml_type_name casing matters: q4_K/q6_K, not q4_k/q6_k.
PARAM=${PARAM:-type_a=(q4_0|q8_0|q4_K|q6_K|iq2_xs),type_b=f32,n_mats=(128|32),n_used=(8|4),b=0,m=(768|1792),n=(64|128|256|512),k=2048}

mkdir -p "$OUT_DIR"
printf '%s\n' "$PARAM" > "$OUT_DIR/param.txt"

if [[ ! -x $BIN ]]; then
  echo "FAIL: missing test-backend-ops: $BIN" >&2
  exit 2
fi

case_env() {
  local name=$1
  CASE_ENV=()
  case "$name" in
    default) ;;
    auto) CASE_ENV+=("GGML_CUDA_MMQ_MAX_X_AUTO=1") ;;
    maxx*)
      local cap=${name#maxx}
      if [[ ! $cap =~ ^[0-9]+$ ]]; then
        echo "FAIL: bad MAX_X case '$name'" >&2
        return 2
      fi
      CASE_ENV+=("GGML_CUDA_MMQ_MAX_X=$cap")
      ;;
    cublas) CASE_ENV+=("GGML_CUDA_FORCE_CUBLAS=1") ;;
    *) echo "FAIL: unknown case '$name'" >&2; return 2 ;;
  esac
}

rc_total=0
for case_name in $CASES; do
  case_env "$case_name" || { rc_total=1; continue; }
  echo "=== $case_name ===" | tee -a "$OUT_DIR/run.log"
  (
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
    export LD_LIBRARY_PATH="$BUILD/bin:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
    unset GGML_CUDA_MMQ_MAX_X GGML_CUDA_FORCE_CUBLAS GGML_CUDA_FORCE_MMQ LLAMA_MTP_PREFILL_FORCE_MMQ GGML_CUDA_MMQ_MAX_X_AUTO RDNA2_MATMUL_OPT_V1
    for assignment in "${CASE_ENV[@]}"; do
      export "$assignment"
    done
    "$BIN" perf --output sql -o MUL_MAT_ID -b ROCm0 -p "$PARAM"
  ) > "$OUT_DIR/${case_name}.sql" 2> "$OUT_DIR/${case_name}.err"
  rc=$?
  echo "=== $case_name rc=$rc env=${CASE_ENV[*]:-none} ===" | tee -a "$OUT_DIR/run.log"
  if [[ $rc -ne 0 ]]; then
    rc_total=1
  fi
  sed -n '1,6p' "$OUT_DIR/${case_name}.err" | tee -a "$OUT_DIR/run.log"
  sleep "$SLEEP_SECS"
done

"$PYTHON" - "$OUT_DIR" <<'PY'
import ast, csv, json, pathlib, re, sys
out=pathlib.Path(sys.argv[1])
cols=['test_time','build_commit','backend_name','op_name','op_params','test_mode','supported','passed','error_message','time_us','flops','bandwidth_gb_s','memory_kb','n_runs','device_description','backend_reg_name']
pat=re.compile(r"INSERT INTO test_backend_ops .* VALUES \((.*)\);")
rows=[]
for f in sorted(out.glob('*.sql')):
    case=f.stem
    for line in f.read_text(errors='replace').splitlines():
        m=pat.match(line)
        if not m:
            continue
        vals=ast.literal_eval('('+m.group(1)+')')
        d=dict(zip(cols, vals)); d['case']=case
        for key in ['type_a','type_b','n_mats','n_used','b','m','n','k']:
            mo=re.search(rf'{key}=([^,]+)', d['op_params'])
            if mo:
                d[key]=mo.group(1)
        rows.append(d)
(out/'summary.json').write_text(json.dumps(rows, indent=2, default=str)+'\n')
fields=['case','type_a','type_b','n_mats','n_used','b','m','n','k','time_us','flops','memory_kb','n_runs','supported','passed','error_message']
with (out/'summary.csv').open('w', newline='') as fp:
    w=csv.DictWriter(fp, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    for r in sorted(rows, key=lambda x:(x.get('type_a',''), int(x.get('m',0)), int(x.get('n',0)), x['case'])):
        w.writerow(r)

by_shape={}
for r in rows:
    if str(r.get('supported')) != '1' or str(r.get('passed')) != '1':
        continue
    key=tuple(r.get(k,'') for k in ['type_a','n_mats','n_used','m','n','k'])
    by_shape.setdefault(key, []).append(r)
lines=[
    '# RDNA3 MMQ MUL_MAT_ID cap sweep',
    '',
    f'Artifact: `{out}`',
    '',
    'Filter: `' + (out/'param.txt').read_text().strip() + '`',
    '',
    '| type | n_mats/n_used | m | n | k | best | default us | auto us | speedup vs default |',
    '|---|---:|---:|---:|---:|---|---:|---:|---:|',
]
for key, vals in sorted(by_shape.items()):
    type_a,n_mats,n_used,m,n,k=key
    vals=sorted(vals, key=lambda r: float(r['time_us']))
    best=vals[0]
    lookup={r['case']: float(r['time_us']) for r in vals}
    default=lookup.get('default')
    auto=lookup.get('auto')
    speed=(default/float(best['time_us'])) if default else None
    lines.append(f"| `{type_a}` | {n_mats}/{n_used} | {m} | {n} | {k} | `{best['case']}` {float(best['time_us']):.2f} | " +
                 (f"{default:.2f}" if default else 'missing') + ' | ' +
                 (f"{auto:.2f}" if auto else 'missing') + ' | ' +
                 (f"{speed:.2f}x" if speed else 'n/a') + ' |')
(out/'summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY

echo "summary: $OUT_DIR/summary.md"
exit "$rc_total"
