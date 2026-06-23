#!/usr/bin/env python3
"""
Generate RDNA3 WMMA layout artifacts for a BM/page attention microkernel.

This is the layout/provenance companion for a vLLM-style packed16 verifier:

  grid.x = KV page
  grid.y = KV head row-block
  block  = 4 wave32 waves

The script intentionally separates two concerns:

  * AMD matrix_calculator output is treated as the ISA/layout oracle.
  * The row/page tables map llama.cpp attention rows onto 16x16 WMMA tiles.

The generated artifact should be read before writing or reviewing kernels that
pack dot4/i8-QK and f16-PV into 16-row BM blocks.

Example:

  scripts/hip/rdna3-wmma-bm-pages-calculator.py \
    --out-dir .harness/artifacts/wmma-calculator-bm-pages-$(date +%Y%m%d-%H%M%S) \
    --nq 2 --nq 3 --nq 4 --hk 4 --gh 6 --bm 128 --row-block 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


QK_I8_INST = "v_wmma_i32_16x16x16_iu8"
PV_F16_INST = "v_wmma_f32_16x16x16_f16"
ARCH = "rdna3"


@dataclass(frozen=True)
class WmmaOutputOwner:
    phase: str
    instruction: str
    lane: int
    lane_lo: int
    lane_hi: int
    acc_reg: int
    tile_row: int
    tile_col: int
    c_expr: str


@dataclass(frozen=True)
class I8OperandEntry:
    instruction: str
    matrix: str
    tile_row: int | None
    tile_col: int | None
    tile_k: int
    reg: int
    byte: int
    bits: str
    lanes: str
    c_expr: str


@dataclass(frozen=True)
class F16OperandEntry:
    instruction: str
    matrix: str
    tile_row: int | None
    tile_col: int | None
    tile_k: int
    reg: int
    half: int
    bits: str
    lanes: str
    c_expr: str


@dataclass(frozen=True)
class RowBlockEntry:
    nq: int
    hk_count: int
    gh: int
    hk: int
    row_block_id: int
    row_block_global: int
    row_in_block: int
    logical_row: int
    q: int
    g: int
    hq: int
    valid: bool
    logits_expr: str
    output_expr: str


@dataclass(frozen=True)
class PageTileSummary:
    nq: int
    nk: int
    bm: int
    row_block: int
    hk_count: int
    gh: int
    row_count_per_hk: int
    row_blocks_per_hk: int
    pages: int
    ctas: int
    nominal_wave32_groups: int


@dataclass(frozen=True)
class WaveRowMapEntry:
    nq: int
    hk: int
    row_block_id: int
    wave: int
    lane: int
    lane_lo: int
    lane_hi: int
    acc_reg: int
    tile_row: int
    tile_col: int
    logical_row: int
    q: int
    g: int
    hq: int
    valid: bool
    d_owner: str


def bit_range(width: int, slot: int) -> str:
    lo = width * slot
    hi = lo + width - 1
    return f"[{hi}:{lo}]"


def write_csv(path: Path, rows: Iterable[object]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists() and (p / "ggml/src/ggml-cuda").exists():
            return p
    return Path.cwd().resolve()


def find_matrix_calculator(explicit: Path | None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("AMD_MATRIX_CALCULATOR")
    if env:
        candidates.append(Path(env))
    candidates.extend([
        Path("/home/mrtrent/.harness/research/amd_matrix_instruction_calculator/matrix_calculator.py"),
        Path.home() / ".harness/research/amd_matrix_instruction_calculator/matrix_calculator.py",
    ])
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def find_python(explicit: Path | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("AMD_MATRIX_CALCULATOR_PYTHON")
    if env:
        candidates.append(Path(env))
    candidates.extend([
        Path("/tmp/matrixcalc-venv/bin/python"),
        Path("/home/mrtrent/.harness/venvs/amd-matrix-calc/bin/python"),
        Path(sys.executable),
    ])
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            # Do not resolve venv launcher symlinks.  Resolving
            # /tmp/matrixcalc-venv/bin/python to /usr/bin/python3 would drop
            # the venv site-packages where tabulate is installed.
            return c
    resolved = shutil.which("python3")
    if resolved:
        return Path(resolved)
    return Path(sys.executable)


def run_amd_matrix_calculator(out_dir: Path, calc: Path | None, py: Path, include_markdown: bool) -> dict[str, object]:
    if calc is None:
        return {"status": "missing", "reason": "matrix_calculator.py not found"}

    amd_dir = out_dir / "amd_matrix_calculator"
    amd_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, object] = {"status": "ok", "calculator": str(calc), "python": str(py), "files": {}}
    files: dict[str, str] = outputs["files"]  # type: ignore[assignment]

    for tag, inst in (("qk_i8", QK_I8_INST), ("pv_f16", PV_F16_INST)):
        detail = amd_dir / f"{tag}_detail.txt"
        cmd = [str(py), str(calc), "--architecture", ARCH, "--instruction", inst, "--detail-instruction"]
        with detail.open("w") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        files[f"{tag}_detail"] = str(detail)

        for matrix in ("A", "B", "D"):
            csv_path = amd_dir / f"{tag}_{matrix}_register_layout.csv"
            cmd = [str(py), str(calc), "--architecture", ARCH, "--instruction", inst,
                   "--register-layout", f"--{matrix}-matrix", "--csv"]
            with csv_path.open("w") as f:
                subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
            files[f"{tag}_{matrix}_csv"] = str(csv_path)

            if include_markdown:
                md_path = amd_dir / f"{tag}_{matrix}_register_layout.md"
                cmd = [str(py), str(calc), "--architecture", ARCH, "--instruction", inst,
                       "--register-layout", f"--{matrix}-matrix", "--markdown"]
                with md_path.open("w") as f:
                    subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
                files[f"{tag}_{matrix}_md"] = str(md_path)

    return outputs


def i8_operand_layout() -> list[I8OperandEntry]:
    rows: list[I8OperandEntry] = []
    for m in range(16):
        for k in range(16):
            reg = k // 4
            byte = k & 3
            rows.append(I8OperandEntry(
                instruction=QK_I8_INST,
                matrix="A_Q",
                tile_row=m,
                tile_col=None,
                tile_k=k,
                reg=reg,
                byte=byte,
                bits=bit_range(8, byte),
                lanes=f"{m};{m + 16}",
                c_expr=f"a_frag[{reg}] byte {byte} = Q[row={m}, k={k}] in lanes {m},{m + 16}",
            ))
    for k in range(16):
        for n in range(16):
            reg = k // 4
            byte = k & 3
            rows.append(I8OperandEntry(
                instruction=QK_I8_INST,
                matrix="B_K",
                tile_row=None,
                tile_col=n,
                tile_k=k,
                reg=reg,
                byte=byte,
                bits=bit_range(8, byte),
                lanes=f"{n};{n + 16}",
                c_expr=f"b_frag[{reg}] byte {byte} = K[k={k}, col={n}] in lanes {n},{n + 16}",
            ))
    return rows


def f16_operand_layout() -> list[F16OperandEntry]:
    rows: list[F16OperandEntry] = []
    for m in range(16):
        for k in range(16):
            reg = k // 2
            half = k & 1
            rows.append(F16OperandEntry(
                instruction=PV_F16_INST,
                matrix="A_P",
                tile_row=m,
                tile_col=None,
                tile_k=k,
                reg=reg,
                half=half,
                bits=bit_range(16, half),
                lanes=f"{m};{m + 16}",
                c_expr=f"a_frag[{reg}] half {half} = P[row={m}, k={k}] in lanes {m},{m + 16}",
            ))
    for k in range(16):
        for n in range(16):
            reg = k // 2
            half = k & 1
            rows.append(F16OperandEntry(
                instruction=PV_F16_INST,
                matrix="B_V",
                tile_row=None,
                tile_col=n,
                tile_k=k,
                reg=reg,
                half=half,
                bits=bit_range(16, half),
                lanes=f"{n};{n + 16}",
                c_expr=f"b_frag[{reg}] half {half} = V[k={k}, col={n}] in lanes {n},{n + 16}",
            ))
    return rows


def wmma_output_owners(phase: str, instruction: str) -> list[WmmaOutputOwner]:
    rows: list[WmmaOutputOwner] = []
    for lane in range(32):
        lane_lo = lane & 15
        lane_hi = lane >> 4
        for acc_reg in range(8):
            tile_row = 2 * acc_reg + lane_hi
            tile_col = lane_lo
            rows.append(WmmaOutputOwner(
                phase=phase,
                instruction=instruction,
                lane=lane,
                lane_lo=lane_lo,
                lane_hi=lane_hi,
                acc_reg=acc_reg,
                tile_row=tile_row,
                tile_col=tile_col,
                c_expr=f"D[{tile_row}][{tile_col}] = acc[{acc_reg}] in lane {lane}",
            ))
    return rows


def row_block_entries(nqs: list[int], hk_count: int, gh: int, row_block: int) -> list[RowBlockEntry]:
    rows: list[RowBlockEntry] = []
    for nq in nqs:
        row_count = nq * gh
        rb_per_hk = math.ceil(row_count / row_block)
        for hk in range(hk_count):
            for rb in range(rb_per_hk):
                global_rb = hk * rb_per_hk + rb
                for r in range(row_block):
                    logical = rb * row_block + r
                    q = logical // gh
                    g = logical - q * gh
                    valid = q < nq and g < gh
                    hq = hk * gh + g if valid else -1
                    rows.append(RowBlockEntry(
                        nq=nq,
                        hk_count=hk_count,
                        gh=gh,
                        hk=hk,
                        row_block_id=rb,
                        row_block_global=global_rb,
                        row_in_block=r,
                        logical_row=logical,
                        q=q,
                        g=g,
                        hq=hq,
                        valid=valid,
                        logits_expr=(f"logits[q={q} * GH * BM + g={g} * BM + key_col]" if valid else "padding"),
                        output_expr=(f"dst[q={q}, hq={hq}, d]" if valid else "padding"),
                    ))
    return rows


def page_tile_summaries(nqs: list[int], nk_values: list[int], bm: int, row_block: int, hk_count: int, gh: int) -> list[PageTileSummary]:
    rows: list[PageTileSummary] = []
    for nq in nqs:
        row_count = nq * gh
        rb_per_hk = math.ceil(row_count / row_block)
        for nk in nk_values:
            pages = math.ceil(nk / bm)
            ctas = pages * hk_count * rb_per_hk
            rows.append(PageTileSummary(
                nq=nq,
                nk=nk,
                bm=bm,
                row_block=row_block,
                hk_count=hk_count,
                gh=gh,
                row_count_per_hk=row_count,
                row_blocks_per_hk=rb_per_hk,
                pages=pages,
                ctas=ctas,
                nominal_wave32_groups=ctas * 4,
            ))
    return rows


def wave_row_map(nqs: list[int], hk_count: int, gh: int, row_block: int) -> list[WaveRowMapEntry]:
    rows: list[WaveRowMapEntry] = []
    for nq in nqs:
        row_count = nq * gh
        rb_per_hk = math.ceil(row_count / row_block)
        for hk in range(hk_count):
            for rb in range(rb_per_hk):
                for wave in range(4):
                    for lane32 in range(32):
                        lane_lo = lane32 & 15
                        lane_hi = lane32 >> 4
                        for acc_reg in range(8):
                            tile_row = 2 * acc_reg + lane_hi
                            tile_col = lane_lo
                            logical = rb * row_block + tile_row
                            q = logical // gh
                            g = logical - q * gh
                            valid = q < nq and g < gh
                            hq = hk * gh + g if valid else -1
                            rows.append(WaveRowMapEntry(
                                nq=nq,
                                hk=hk,
                                row_block_id=rb,
                                wave=wave,
                                lane=lane32,
                                lane_lo=lane_lo,
                                lane_hi=lane_hi,
                                acc_reg=acc_reg,
                                tile_row=tile_row,
                                tile_col=tile_col,
                                logical_row=logical,
                                q=q,
                                g=g,
                                hq=hq,
                                valid=valid,
                                d_owner=f"wave {wave} lane {lane32} acc[{acc_reg}] -> row {tile_row}, col {tile_col}",
                            ))
    return rows


def write_readme(path: Path, summary: dict[str, object]) -> None:
    text = f"""# RDNA3 WMMA BM/page layout artifact

Generated by `scripts/hip/rdna3-wmma-bm-pages-calculator.py`.

## Purpose

This artifact records the calculator-backed lane/register contract for a BM/page
attention kernel that uses 16-row logical row blocks and 4 wave32 waves per CTA.
The intended scheduler is:

```text
grid.x = KV page
grid.y = KV head row-block
grid.z = batch
block  = 128 threads = 4 wave32 waves
```

## Matrix calculator

Status: `{summary['amd_matrix_calculator_status']}`

Calculator: `{summary.get('matrix_calculator', 'n/a')}`
Python: `{summary.get('matrix_calculator_python', 'n/a')}`

Queried instructions:

```text
QK: {QK_I8_INST}
PV: {PV_F16_INST}
```

Raw AMD outputs, if available, are under `amd_matrix_calculator/`.

## Confirmed formulas

### QK i8 WMMA

```text
A = Q[16 x 16], B = K[16 x 16], D = logits[16 x 16]
A/B registers: 4 i32 words; byte = k % 4; reg = k / 4
A rows supplied by lanes row and row+16
B cols supplied by lanes col and col+16
D[row,col] = acc[row/2] in lane col + 16*(row&1)
```

### PV f16 WMMA

```text
A = P[16 x 16], B = V[16 x 16], D = O[16 x 16]
A/B registers: 8 packed f16 words; half = k % 2; reg = k / 2
A rows supplied by lanes row and row+16
B cols supplied by lanes col and col+16
D[row,col] = acc[row/2] in lane col + 16*(row&1)
```

## Row mapping

For one KV head:

```text
logical_row = q * GH + g
hq          = hk * GH + g
```

For the default 27B shape in this artifact:

```text
GH={summary['gh']}, HK={summary['hk_count']}, row_block={summary['row_block']}, BM={summary['bm']}
```

Generated files:

```text
summary.json
page_tile_summary.csv
row_block_mapping.csv
wave_output_ownership.csv
qk_i8_operand_layout.csv
pv_f16_operand_layout.csv
wmma_output_owners.csv
amd_matrix_calculator/*
```

## Kernel review checklist

- `grid.y` must include row blocks, not only KV heads.
- A 16-row tile maps padded rows to no-op stores.
- QK logits for a valid row store as `q * GH * BM + g * BM + key_col`.
- PV output rows store back to `(q, hq, d)`.
- WMMA output owner is always `row = 2*acc + lane_hi`, `col = lane_lo`.
- Page-local softmax state must merge using max/sum-exp, not raw probability sums.
"""
    path.write_text(text)


def main() -> int:
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path)

    ap = argparse.ArgumentParser(description=__doc__)
    default_out = repo_root / ".harness/artifacts" / f"wmma-calculator-bm-pages-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ap.add_argument("--out-dir", type=Path, default=default_out)
    ap.add_argument("--matrix-calculator", type=Path, default=None, help="path to AMD matrix_calculator.py")
    ap.add_argument("--python", type=Path, default=None, help="python executable with calculator dependencies")
    ap.add_argument("--skip-amd", action="store_true", help="skip AMD matrix_calculator invocations")
    ap.add_argument("--amd-markdown", action="store_true", help="also save markdown register layouts")
    ap.add_argument("--nq", type=int, action="append", default=None, help="NQ value; repeatable; default 2,3,4")
    ap.add_argument("--nk", type=int, action="append", default=None, help="NK value; repeatable; default 256,4096,14336")
    ap.add_argument("--hk", type=int, default=4, help="KV heads")
    ap.add_argument("--gh", type=int, default=6, help="GQA ratio / query heads per KV head")
    ap.add_argument("--bm", type=int, default=128, help="KV page size")
    ap.add_argument("--row-block", type=int, default=16, help="WMMA logical rows per row block")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    nqs = args.nq if args.nq else [2, 3, 4]
    nk_values = args.nk if args.nk else [256, 4096, 14336]

    if args.row_block != 16:
        raise ValueError("RDNA3 16x16 WMMA row-block must be 16 for this artifact")
    if args.bm % 16 != 0:
        raise ValueError("BM must be a multiple of the 16-key WMMA tile")

    calc = find_matrix_calculator(args.matrix_calculator)
    py = find_python(args.python)
    amd = {"status": "skipped"} if args.skip_amd else run_amd_matrix_calculator(out_dir, calc, py, args.amd_markdown)

    qk_operands = i8_operand_layout()
    pv_operands = f16_operand_layout()
    owners = wmma_output_owners("qk_i8", QK_I8_INST) + wmma_output_owners("pv_f16", PV_F16_INST)
    rb_entries = row_block_entries(nqs, args.hk, args.gh, args.row_block)
    page_summaries = page_tile_summaries(nqs, nk_values, args.bm, args.row_block, args.hk, args.gh)
    wave_maps = wave_row_map(nqs, args.hk, args.gh, args.row_block)

    write_csv(out_dir / "qk_i8_operand_layout.csv", qk_operands)
    write_csv(out_dir / "pv_f16_operand_layout.csv", pv_operands)
    write_csv(out_dir / "wmma_output_owners.csv", owners)
    write_csv(out_dir / "row_block_mapping.csv", rb_entries)
    write_csv(out_dir / "page_tile_summary.csv", page_summaries)
    write_csv(out_dir / "wave_output_ownership.csv", wave_maps)

    amd_files = amd.get("files", {}) if isinstance(amd, dict) else {}
    summary: dict[str, object] = {
        "artifact": str(out_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "arch": ARCH,
        "qk_instruction": QK_I8_INST,
        "pv_instruction": PV_F16_INST,
        "bm": args.bm,
        "row_block": args.row_block,
        "hk_count": args.hk,
        "gh": args.gh,
        "nq_values": nqs,
        "nk_values": nk_values,
        "amd_matrix_calculator_status": amd.get("status", "unknown") if isinstance(amd, dict) else "unknown",
        "matrix_calculator": str(calc) if calc else None,
        "matrix_calculator_python": str(py),
        "amd_matrix_calculator": amd,
        "generated_files": {
            "qk_i8_operand_layout": str(out_dir / "qk_i8_operand_layout.csv"),
            "pv_f16_operand_layout": str(out_dir / "pv_f16_operand_layout.csv"),
            "wmma_output_owners": str(out_dir / "wmma_output_owners.csv"),
            "row_block_mapping": str(out_dir / "row_block_mapping.csv"),
            "page_tile_summary": str(out_dir / "page_tile_summary.csv"),
            "wave_output_ownership": str(out_dir / "wave_output_ownership.csv"),
            **{f"amd_{k}": v for k, v in amd_files.items()},
        },
        "formulas": {
            "i8_a_b": "reg=k//4; byte=k%4; lanes=row,row+16 for A and col,col+16 for B",
            "f16_a_b": "reg=k//2; half=k%2; lanes=row,row+16 for A and col,col+16 for B",
            "d_owner": "row=2*acc_reg+lane_hi; col=lane_lo; lane=col+16*(row&1)",
            "logical_row": "logical_row=q*GH+g; hq=hk*GH+g",
            "grid": "grid=(ceil(nk/BM), hk*ceil(nq*GH/16), batch); block=128 threads",
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_readme(out_dir / "README.md", summary)

    print(out_dir)
    print(json.dumps({
        "artifact": str(out_dir),
        "amd_matrix_calculator_status": summary["amd_matrix_calculator_status"],
        "page_summaries": [asdict(x) for x in page_summaries],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
