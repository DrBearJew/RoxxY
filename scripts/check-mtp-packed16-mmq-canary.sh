#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github}
OUT_DIR=${OUT_DIR:-$ROOT/.harness/tmp/mtp-packed16-mmq-canary}
BASE_PORT=${BASE_PORT:-18720}
N_PREDICT=${N_PREDICT:-4}

mkdir -p "$OUT_DIR"
cd "$ROOT"

STRICT_ROUTES=1 \
RUN_MTP_PACKED16_MMQ=1 \
MTP_KV_ARGS="--cache-type-v q4_0 --spec-draft-type-v q4_0" \
OUT_DIR="$OUT_DIR" \
BASE_PORT="$BASE_PORT" \
N_PREDICT="$N_PREDICT" \
bash "$ROOT/benchmarks/mtp-dot4-isolation.sh" > "$OUT_DIR/run.log" 2>&1

python3 - "$OUT_DIR/summary.json" "$OUT_DIR/mtp-verify-packed16-mmq.log" <<'PY'
import json
import re
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])
rows = json.loads(summary_path.read_text())
row = next((r for r in rows if r.get("log", "").endswith("mtp-verify-packed16-mmq.log")), None)
if row is None:
    raise SystemExit("FAIL: packed16/MMQ canary row missing")

failures = []
if row.get("bad_h") != 0:
    failures.append(f"bad_h={row.get('bad_h')}")
if row.get("checked_h", 0) <= 0:
    failures.append(f"checked_h={row.get('checked_h')}")
if row.get("bad_logits") != 0:
    failures.append(f"bad_logits={row.get('bad_logits')}")
acceptance = row.get("acceptance") or {}
accepted = acceptance.get("accepted", 0) or 0
generated = acceptance.get("generated", 0) or 0
if generated <= 0:
    failures.append(f"generated={generated}")
if accepted <= 0:
    failures.append(f"accepted={accepted}")
depth = row.get("acceptance_by_depth") or {}
d1 = depth.get("1") or {}
d1_s = f"{d1.get('accepted', 0)}/{d1.get('generated', 0)}"
if d1.get("generated", 0) <= 0 or d1.get("accepted", 0) <= 0:
    failures.append(f"d1={d1_s}")

log = log_path.read_text(errors="replace")
mtp_mmq = re.findall(r"fa_final_select: inst=mtp_verify_qk selected=rocm_packed16_dot4_mmq nq=(\d+) nk=(\d+) d=(\d+) K=i32 V=q4_0", log)
pdmq2_mmq = re.findall(r"PDMQ2 route=rocm_packed16_dot4_mmq .*K=i32 V=q4_0", log)
if not mtp_mmq:
    failures.append("missing mtp_verify_qk rocm_packed16_dot4_mmq K=i32 V=q4_0 selection")
if not pdmq2_mmq:
    failures.append("missing PDMQ2 rocm_packed16_dot4_mmq K=i32 V=q4_0 evidence")
else:
    nqs = {int(m[0]) for m in mtp_mmq}
    if not (nqs & {2, 4}):
        failures.append(f"missing nq=2/4 MTP MMQ selection; saw nq={sorted(nqs)}")
if "PDMQ QK probe PASSED" not in log:
    failures.append("missing PDMQ QK probe PASSED")
for needle in ["GGML_ASSERT", "required rocm_packed16_dot4_mmq route was not selected", "bad FA V layout"]:
    if needle in log:
        failures.append(f"{needle} in log")

if failures:
    print("FAIL: MTP packed16/MMQ canary failed: " + ", ".join(failures), file=sys.stderr)
    raise SystemExit(1)

nq_s = "/".join(str(x) for x in sorted({int(m[0]) for m in mtp_mmq}))
print(
    "PASS: MTP packed16/MMQ canary "
    f"bad_h={row['bad_h']} checked_h={row['checked_h']} bad_logits={row['bad_logits']} "
    f"accepted={accepted} generated={generated} d1={d1_s} nq={nq_s} route=rocm_packed16_dot4_mmq"
)
PY

echo "Log: $OUT_DIR/mtp-verify-packed16-mmq.log"
echo "Summary: $OUT_DIR/summary.csv"
