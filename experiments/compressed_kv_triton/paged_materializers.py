#!/usr/bin/env python3
"""Paged/block-table materializer checks for compressed KV formats."""

from __future__ import annotations

import argparse
import json

import torch
import triton
import triton.language as tl

from compressed_kv_tl import (
    FORMATS,
    deterministic_block_table,
    iso3_values,
    paged_physical_rows,
    physicalize_inputs,
    planar3_values,
    tbq4_values,
)
from materializers import (
    load_constants,
    materialize_iso3_ref,
    materialize_planar3_ref,
    materialize_tbq4_ref,
    synthetic_inputs,
)


@triton.jit
def paged_materialize_tbq4_kernel(d, qs, centroids, block_table, out,
                                  ROWS: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
                                  BLOCK_SIZE: tl.constexpr, BLOCK_TABLE_STRIDE: tl.constexpr):
    logical_row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    row_mask = logical_row < ROWS
    mask = row_mask & (offs < D)
    physical_row = paged_physical_rows(block_table, 0, logical_row, row_mask, BLOCK_TABLE_STRIDE, BLOCK_SIZE)
    vals = tbq4_values(d, qs, centroids, physical_row, offs, mask, D)
    tl.store(out + logical_row * D + offs, vals, mask=mask)


@triton.jit
def paged_materialize_planar3_kernel(d, qs, signs, centroids, cos, sin, block_table, out,
                                     ROWS: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
                                     BLOCK_SIZE: tl.constexpr, BLOCK_TABLE_STRIDE: tl.constexpr):
    logical_row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    row_mask = logical_row < ROWS
    mask = row_mask & (offs < D)
    physical_row = paged_physical_rows(block_table, 0, logical_row, row_mask, BLOCK_TABLE_STRIDE, BLOCK_SIZE)
    vals = planar3_values(d, qs, signs, centroids, cos, sin, physical_row, offs, mask, D)
    tl.store(out + logical_row * D + offs, vals, mask=mask)


@triton.jit
def paged_materialize_iso3_kernel(d, qs, signs, centroids, qw, qx, qy, qz, block_table, out,
                                  ROWS: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
                                  BLOCK_SIZE: tl.constexpr, BLOCK_TABLE_STRIDE: tl.constexpr):
    logical_row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    row_mask = logical_row < ROWS
    mask = row_mask & (offs < D)
    physical_row = paged_physical_rows(block_table, 0, logical_row, row_mask, BLOCK_TABLE_STRIDE, BLOCK_SIZE)
    vals = iso3_values(d, qs, signs, centroids, qw, qx, qy, qz, physical_row, offs, mask, D)
    tl.store(out + logical_row * D + offs, vals, mask=mask)


def _ref(fmt: str, inputs: dict[str, torch.Tensor], constants_cpu: dict[str, torch.Tensor]) -> torch.Tensor:
    if fmt == "tbq4_0":
        return materialize_tbq4_ref(inputs["d"], inputs["qs"], constants_cpu).half().float()
    if fmt == "planar3_0":
        return materialize_planar3_ref(inputs["d"], inputs["qs"], inputs["signs"], constants_cpu).half().float()
    if fmt == "iso3_0":
        return materialize_iso3_ref(inputs["d"], inputs["qs"], inputs["signs"], constants_cpu).half().float()
    raise ValueError(fmt)


def _run_one(fmt: str, rows: int, d_head: int, block_size: int) -> dict[str, object]:
    device = torch.device("cuda")
    constants_cpu = load_constants("cpu")
    constants = {k: v.to(device) for k, v in constants_cpu.items()}
    logical_inputs = synthetic_inputs(fmt, rows, d_head, seed=8000 + rows + d_head + len(fmt))
    block_table_cpu = deterministic_block_table(rows, block_size)
    physical_inputs = physicalize_inputs(logical_inputs, block_table_cpu, block_size)

    d_gpu = physical_inputs["d"].contiguous().to(device)
    block_table = block_table_cpu.to(device)
    out = torch.empty((rows, d_head), dtype=torch.float16, device=device)
    block_d = triton.next_power_of_2(d_head)
    grid = (rows,)

    if fmt == "tbq4_0":
        paged_materialize_tbq4_kernel[grid](
            d_gpu, physical_inputs["qs"].contiguous().to(device), constants["tbq4_centroids"], block_table, out,
            ROWS=rows, D=d_head, BLOCK_D=block_d, BLOCK_SIZE=block_size, BLOCK_TABLE_STRIDE=block_table_cpu.numel())
    elif fmt == "planar3_0":
        paged_materialize_planar3_kernel[grid](
            d_gpu, physical_inputs["qs"].contiguous().to(device), physical_inputs["signs"].contiguous().to(device),
            constants["pi_centroids_3bit"], constants["pi_cos"], constants["pi_sin"], block_table, out,
            ROWS=rows, D=d_head, BLOCK_D=block_d, BLOCK_SIZE=block_size, BLOCK_TABLE_STRIDE=block_table_cpu.numel())
    elif fmt == "iso3_0":
        paged_materialize_iso3_kernel[grid](
            d_gpu, physical_inputs["qs"].contiguous().to(device), physical_inputs["signs"].contiguous().to(device),
            constants["pi_centroids_3bit"], constants["pi_qw"], constants["pi_qx"], constants["pi_qy"], constants["pi_qz"], block_table, out,
            ROWS=rows, D=d_head, BLOCK_D=block_d, BLOCK_SIZE=block_size, BLOCK_TABLE_STRIDE=block_table_cpu.numel())
    else:
        raise ValueError(fmt)

    torch.cuda.synchronize()
    ref = _ref(fmt, logical_inputs, constants_cpu)
    diff = (out.float().cpu() - ref).abs()
    max_abs = float(diff.max().item())
    tol = 1.5e-3
    return {
        "format": fmt,
        "D": d_head,
        "rows": rows,
        "block_size": block_size,
        "block_table": [int(x) for x in block_table_cpu.tolist()],
        "max_abs_err": max_abs,
        "tol": tol,
        "passed": max_abs <= tol,
    }


def run_paged_materializer_checks(rows: int = 17, block_size: int = 4, dims: tuple[int, ...] = (128, 256)) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("paged materializer checks require a HIP/CUDA-visible torch device")
    checks = [_run_one(fmt, rows, d_head, block_size) for d_head in dims for fmt in FORMATS]
    return {"result": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=17)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--dims", type=int, nargs="+", default=[128, 256])
    args = parser.parse_args()
    report = run_paged_materializer_checks(rows=args.rows, block_size=args.block_size, dims=tuple(args.dims))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
