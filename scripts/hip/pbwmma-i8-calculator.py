#!/usr/bin/env python3
"""
Empirical/derived layout calculator for RDNA3 v_wmma_i32_16x16x16_iu8_w32.

Purpose
-------
This is the small in-repo calculator for the packed-i8 WMMA path.  The formulas
below are the layout contract exercised by the DOT4 shadow probes in
`fattn-packed16-wmma-builtin.cuh` / `fattn-packed16-wmma-tile.cuh`:

  * synthetic probe:  GGML_CUDA_PWMMA_I8_DOT4_SHADOW_PROBE=1
  * live-tile probe:  GGML_CUDA_PWMMA_I8_LIVE_DOT4_SHADOW=1

DOT4 is used as an oracle for byte order, signedness, and output slot mapping;
this script makes that contract explicit as CSV/JSON so future WMMA kernels can
be generated/reviewed instead of hand-waved.

The layout matches AMD's matrix_calculator for:
  arch: rdna3
  inst: v_wmma_i32_16x16x16_iu8

Outputs
-------
  a_layout.csv  : A[M,K] -> fragment register, byte, duplicate lanes
  b_layout.csv  : B[K,N] -> fragment register, byte, duplicate lanes
  d_layout.csv  : D[M,N] -> accumulator register/slot and lane
  dot4_oracle.csv : DOT4 reference words for each D[M,N]
  accessors.md  : compact formulas for C++ kernel code
  summary.json  : machine-readable contract and artifact provenance

Example
-------
  scripts/hip/pbwmma-i8-calculator.py \
    --out-dir /home/mrtrent/.harness/artifacts/pbwmma-i8-calculator-$(date +%Y%m%d-%H%M%S) \
    --compare-amd
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class OperandEntry:
    matrix: str
    m: int | None
    n: int | None
    k: int | None
    reg: int
    byte: int | None
    bits: str | None
    lanes: str
    duplicate_lanes: int
    c_expr: str
    dot4_word_expr: str | None


@dataclass(frozen=True)
class DEntry:
    matrix: str
    m: int
    n: int
    acc_reg: int
    lane: int
    lane_lo: int
    lane_hi: int
    c_expr: str


@dataclass(frozen=True)
class Dot4Entry:
    d_m: int
    d_n: int
    acc_reg: int
    lane: int
    dot4_terms: str
    q_word_expr: str
    k_word_expr: str
    wmma_acc_expr: str


def bit_range(byte: int) -> str:
    lo = byte * 8
    hi = lo + 7
    return f"[{hi}:{lo}]"


def a_layout() -> list[OperandEntry]:
    """A[M,K] for RDNA3 WMMA i8.

    The same A row is supplied by lane M and lane M+16.  Four i8 lanes live in
    one i32 register; K selects reg=K//4 and byte=K%4.
    """
    out: list[OperandEntry] = []
    for m in range(16):
        for k in range(16):
            reg = k // 4
            byte = k & 3
            lanes = [m, m + 16]
            out.append(OperandEntry(
                matrix="A",
                m=m,
                n=None,
                k=k,
                reg=reg,
                byte=byte,
                bits=bit_range(byte),
                lanes=";".join(map(str, lanes)),
                duplicate_lanes=len(lanes),
                c_expr=f"a_frag[{reg}] byte {byte} in lanes {m},{m + 16}",
                dot4_word_expr=f"q_words[{m}][{reg}] byte {byte}",
            ))
    return out


def b_layout() -> list[OperandEntry]:
    """B[K,N] for RDNA3 WMMA i8.

    The same B column N is supplied by lane N and lane N+16.  Four i8 K values
    live in one i32 register; K selects reg=K//4 and byte=K%4.
    """
    out: list[OperandEntry] = []
    for k in range(16):
        for n in range(16):
            reg = k // 4
            byte = k & 3
            lanes = [n, n + 16]
            out.append(OperandEntry(
                matrix="B",
                m=None,
                n=n,
                k=k,
                reg=reg,
                byte=byte,
                bits=bit_range(byte),
                lanes=";".join(map(str, lanes)),
                duplicate_lanes=len(lanes),
                c_expr=f"b_frag[{reg}] byte {byte} in lanes {n},{n + 16}",
                dot4_word_expr=f"k_words[{n}][{reg}] byte {byte}",
            ))
    return out


def d_layout() -> list[DEntry]:
    """D[M,N] / accumulator layout.

    Current kernel store formula:
      lane_lo = lane & 15
      lane_hi = lane >> 4
      row     = 2*acc_reg + lane_hi
      col     = lane_lo
    Therefore inverse:
      acc_reg = M // 2
      lane    = N + 16*(M & 1)
    """
    out: list[DEntry] = []
    for m in range(16):
        for n in range(16):
            acc_reg = m // 2
            lane_hi = m & 1
            lane_lo = n
            lane = lane_lo + 16 * lane_hi
            out.append(DEntry(
                matrix="D",
                m=m,
                n=n,
                acc_reg=acc_reg,
                lane=lane,
                lane_lo=lane_lo,
                lane_hi=lane_hi,
                c_expr=f"acc_i[{acc_reg}] in lane {lane} -> D[{m}][{n}]",
            ))
    return out


def dot4_oracle() -> list[Dot4Entry]:
    """Reference DOT4 words for each WMMA output D[M,N]."""
    out: list[Dot4Entry] = []
    for d in d_layout():
        terms = []
        for reg in range(4):
            terms.append(f"dot4(q_words[{d.m}][{reg}], k_words[{d.n}][{reg}])")
        out.append(Dot4Entry(
            d_m=d.m,
            d_n=d.n,
            acc_reg=d.acc_reg,
            lane=d.lane,
            dot4_terms=" + ".join(terms),
            q_word_expr=f"q_words[{d.m}][g] == q_i32[row={d.m}][d0/4 + g]",
            k_word_expr=f"k_words[{d.n}][g] == k_payload[col={d.n}][d0/4 + g]",
            wmma_acc_expr=f"lane {d.lane}: acc_i[{d.acc_reg}]",
        ))
    return out


def write_csv(path: Path, rows: Iterable[object]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_accessors(path: Path) -> None:
    path.write_text("""# RDNA3 PBWMMA i8 layout accessors

Instruction:

```text
v_wmma_i32_16x16x16_iu8_w32
```

## Fragment sizes

```cpp
using pbwmma_v4i32 = int __attribute__((ext_vector_type(4)));  // A/B
using pbwmma_v8i32 = int __attribute__((ext_vector_type(8)));  // C/D
```

## Lane decomposition

```cpp
const int lane    = threadIdx.x & 31;
const int lane_lo = lane & 15;  // column for D, row/col selector for A/B input
const int lane_hi = lane >> 4;  // low/high row parity for D
```

## A fragment, A[M,K]

```cpp
// For output rows M = lane_lo in both lanes M and M+16.
// K = 4*g + byte.
a_frag[g] byte byte  -> A[lane_lo][4*g + byte]
```

In the BM64 i8-QK kernel this is:

```cpp
a_frag[g] = q_i32[qr][(d0/4) + g];  // qr = r_base + lane_lo
```

## B fragment, B[K,N]

```cpp
// For output column N = lane_lo in both lanes N and N+16.
// K = 4*g + byte.
b_frag[g] byte byte  -> B[4*g + byte][lane_lo]
```

In the BM64 i8-QK kernel this is:

```cpp
b_frag[g] = k_payload[row * (D/4) + (d0/4) + g]; // row = k token column lane_lo
```

## D accumulator, D[M,N]

```cpp
D[2*i + lane_hi][lane_lo] = acc_i[i];
```

Inverse:

```cpp
int acc_i_index = m / 2;
int lane        = n + 16 * (m & 1);
```

## DOT4 oracle for one K16 tile

```cpp
int ref = 0;
#pragma unroll
for (int g = 0; g < 4; ++g) {
    ref = pbwmma_dot4_i8_i8(q_words[m][g], k_words[n][g], ref);
}
assert(ref == acc_i[m / 2] in lane n + 16*(m & 1));
```

This is exactly what `GGML_CUDA_PWMMA_I8_LIVE_DOT4_SHADOW=1` checks on real
Q/K tiles.
""")


def run_amd_compare(out_dir: Path) -> dict[str, str]:
    calc = Path("/home/mrtrent/.harness/research/amd_matrix_instruction_calculator/matrix_calculator.py")
    py = Path("/home/mrtrent/.harness/venvs/amd-matrix-calc/bin/python")
    if not py.exists():
        py = Path(sys.executable)
    if not calc.exists():
        return {"status": "missing", "reason": str(calc)}

    outputs: dict[str, str] = {"status": "ok"}
    for matrix in ("A", "B", "D"):
        dst = out_dir / f"amd_matrix_calculator_{matrix}.md"
        cmd = [str(py), str(calc), "-a", "rdna3", "-i", "v_wmma_i32_16x16x16_iu8",
               "--register-layout", f"-{matrix}", "--markdown"]
        with dst.open("w") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        outputs[matrix] = str(dst)
    return outputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_out = Path("/home/mrtrent/.harness/artifacts") / f"pbwmma-i8-calculator-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ap.add_argument("--out-dir", type=Path, default=default_out)
    ap.add_argument("--compare-amd", action="store_true", help="save AMD matrix_calculator markdown for A/B/D")
    ap.add_argument("--validated-artifact", action="append", default=[], help="artifact path that contains DOT4 shadow proof")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    a_rows = a_layout()
    b_rows = b_layout()
    d_rows = d_layout()
    dot4_rows = dot4_oracle()

    write_csv(out_dir / "a_layout.csv", a_rows)
    write_csv(out_dir / "b_layout.csv", b_rows)
    write_csv(out_dir / "d_layout.csv", d_rows)
    write_csv(out_dir / "dot4_oracle.csv", dot4_rows)
    write_accessors(out_dir / "accessors.md")

    amd_compare = run_amd_compare(out_dir) if args.compare_amd else {"status": "not_requested"}

    summary = {
        "instruction": "v_wmma_i32_16x16x16_iu8_w32",
        "arch": "rdna3",
        "wave_size": 32,
        "a_fragment": {"registers": 4, "bytes_per_register": 4, "layout": "A[M,K] -> lanes M,M+16; reg=K//4; byte=K%4"},
        "b_fragment": {"registers": 4, "bytes_per_register": 4, "layout": "B[K,N] -> lanes N,N+16; reg=K//4; byte=K%4"},
        "d_fragment": {"registers": 8, "layout": "D[M,N] -> lane=N+16*(M&1); acc_reg=M//2"},
        "byte_order": "little-endian i8x4 in int32 word",
        "a_signed": True,
        "b_signed": True,
        "dot4_oracle": "sum_g pbwmma_dot4_i8_i8(q_words[M][g], k_words[N][g]) == acc_i[M//2] at lane N+16*(M&1)",
        "generated_files": {
            "a_layout": str(out_dir / "a_layout.csv"),
            "b_layout": str(out_dir / "b_layout.csv"),
            "d_layout": str(out_dir / "d_layout.csv"),
            "dot4_oracle": str(out_dir / "dot4_oracle.csv"),
            "accessors": str(out_dir / "accessors.md"),
        },
        "amd_matrix_calculator": amd_compare,
        "validated_by_artifacts": args.validated_artifact,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(out_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
