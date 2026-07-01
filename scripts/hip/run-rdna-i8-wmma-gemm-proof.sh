#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/scripts/hip/rdna-i8-wmma-gemm-proof.hip"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-/home/mrtrent/.harness/artifacts/rdna-i8-wmma-gemm-proof-$STAMP}"
CXX="${HIPCXX:-/opt/rocm/bin/hipcc}"
OBJDUMP="${LLVM_OBJDUMP:-/opt/rocm-7.2.3/lib/llvm/bin/llvm-objdump}"
GPU="${GPU:-gfx1100}"
M="${M:-32}"
N="${N:-48}"
K="${K:-64}"
WARMUP="${WARMUP:-10}"
REPS="${REPS:-50}"
CHECK="${CHECK:-1}"
BIN="$OUT_DIR/rdna_i8_wmma_gemm_proof"
HSACO="$OUT_DIR/rdna_i8_wmma_gemm_proof.hsaco"

mkdir -p "$OUT_DIR"

cat > "$OUT_DIR/manifest.txt" <<EOF_MANIFEST
source=$SRC
out_dir=$OUT_DIR
hipcxx=$CXX
objdump=$OBJDUMP
gpu=$GPU
M=$M
N=$N
K=$K
warmup=$WARMUP
reps=$REPS
check=$CHECK
EOF_MANIFEST

set -x
"$CXX" -O3 --offload-arch="$GPU" -std=c++17 -DRDNA3=1 "$SRC" -o "$BIN" >"$OUT_DIR/compile.log" 2>&1
set +x

"$BIN" "$M" "$N" "$K" "$WARMUP" "$REPS" "$CHECK" >"$OUT_DIR/run.log" 2>&1

# Best-effort ISA evidence. The host executable contains an offload image; depending
# on LLVM version, either direct offloading disassembly or an explicit --genco image
# provides the readable AMDGPU instructions.
if [ -x "$OBJDUMP" ]; then
  "$OBJDUMP" --offloading --disassemble "$BIN" >"$OUT_DIR/disasm-offloading.txt" 2>"$OUT_DIR/disasm-offloading.err" || true

  # llvm-objdump --offloading extracts the device code object next to BIN.
  # Disassemble that AMDGPU ELF directly for the actual ISA mnemonic.
  for bundle in "$OUT_DIR"/*.hipv4-amdgcn-amd-amdhsa--"$GPU"; do
    [ -f "$bundle" ] || continue
    "$OBJDUMP" -d --triple=amdgcn-amd-amdhsa --mcpu="$GPU" "$bundle" >"$OUT_DIR/disasm-bundle.txt" 2>"$OUT_DIR/disasm-bundle.err" || true
    break
  done
fi

if "$CXX" -O3 --offload-arch="$GPU" --genco -std=c++17 -DRDNA3=1 "$SRC" -o "$HSACO" >"$OUT_DIR/genco.log" 2>&1; then
  if [ -x "$OBJDUMP" ]; then
    "$OBJDUMP" -d --triple=amdgcn-amd-amdhsa --mcpu="$GPU" "$HSACO" >"$OUT_DIR/disasm-hsaco.txt" 2>"$OUT_DIR/disasm-hsaco.err" || true
  fi
fi

WMMA_COUNT=0
for f in "$OUT_DIR"/disasm-*.txt; do
  [ -f "$f" ] || continue
  c=$(grep -c "v_wmma_i32_16x16x16_iu8" "$f" || true)
  WMMA_COUNT=$((WMMA_COUNT + c))
done

PASS_LINE=$(grep -F "[rdna-i8-wmma] PASS" "$OUT_DIR/run.log" || true)
cat > "$OUT_DIR/summary.md" <<EOF_SUMMARY
# RDNA i8 WMMA GEMM proof

- source: \`$SRC\`
- gpu: \`$GPU\`
- shape: M=$M N=$N K=$K
- timing: warmup=$WARMUP reps=$REPS check=$CHECK
- binary: \`$BIN\`
- run pass: $([ -n "$PASS_LINE" ] && echo yes || echo no)
- native ISA \`v_wmma_i32_16x16x16_iu8\` count: $WMMA_COUNT

## Run log

\`\`\`text
$(cat "$OUT_DIR/run.log")
\`\`\`

## Files

- compile log: \`$OUT_DIR/compile.log\`
- run log: \`$OUT_DIR/run.log\`
- offload disassembly: \`$OUT_DIR/disasm-offloading.txt\`
- extracted bundle disassembly: \`$OUT_DIR/disasm-bundle.txt\`
- hsaco disassembly: \`$OUT_DIR/disasm-hsaco.txt\`
EOF_SUMMARY

cat "$OUT_DIR/summary.md"

if [ -z "$PASS_LINE" ]; then
  echo "rdna-i8-wmma proof run failed" >&2
  exit 1
fi
if [ "$WMMA_COUNT" -le 0 ]; then
  echo "rdna-i8-wmma proof did not find native v_wmma_i32_16x16x16_iu8 in disassembly" >&2
  exit 1
fi
