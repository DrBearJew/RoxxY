#!/usr/bin/env python3
"""QK-only Triton prototypes using direct compressed-K materialization."""

from __future__ import annotations

import argparse
import json

import torch
import triton
import triton.language as tl

from compressed_kv_tl import iso3_values as _iso3_values
from compressed_kv_tl import planar3_values as _planar3_values
from compressed_kv_tl import tbq4_values as _tbq4_values
from materializers import (
    load_constants,
    materialize_iso3_ref,
    materialize_planar3_ref,
    materialize_tbq4_ref,
    synthetic_inputs,
)


@triton.jit
def qk_tbq4_kernel(q, d, qs, centroids, logits, K_ROWS: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    q_row = tl.program_id(0)
    k_row = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    qv = tl.load(q + q_row * D + offs, mask=mask, other=0.0).to(tl.float32)
    kv = _tbq4_values(d, qs, centroids, k_row, offs, mask, D)
    acc = tl.sum(qv * kv, axis=0)
    tl.store(logits + q_row * K_ROWS + k_row, acc)


@triton.jit
def qk_planar3_kernel(q, d, qs, signs, centroids, cos, sin, logits, K_ROWS: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    q_row = tl.program_id(0)
    k_row = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    qv = tl.load(q + q_row * D + offs, mask=mask, other=0.0).to(tl.float32)
    kv = _planar3_values(d, qs, signs, centroids, cos, sin, k_row, offs, mask, D)
    acc = tl.sum(qv * kv, axis=0)
    tl.store(logits + q_row * K_ROWS + k_row, acc)


@triton.jit
def qk_iso3_kernel(q, d, qs, signs, centroids, qw, qx, qy, qz, logits, K_ROWS: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    q_row = tl.program_id(0)
    k_row = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    qv = tl.load(q + q_row * D + offs, mask=mask, other=0.0).to(tl.float32)
    kv = _iso3_values(d, qs, signs, centroids, qw, qx, qy, qz, k_row, offs, mask, D)
    acc = tl.sum(qv * kv, axis=0)
    tl.store(logits + q_row * K_ROWS + k_row, acc)


def _k_ref(fmt: str, inputs: dict[str, torch.Tensor], constants_cpu: dict[str, torch.Tensor]) -> torch.Tensor:
    if fmt == "tbq4_0":
        return materialize_tbq4_ref(inputs["d"], inputs["qs"], constants_cpu).half().float()
    if fmt == "planar3_0":
        return materialize_planar3_ref(inputs["d"], inputs["qs"], inputs["signs"], constants_cpu).half().float()
    if fmt == "iso3_0":
        return materialize_iso3_ref(inputs["d"], inputs["qs"], inputs["signs"], constants_cpu).half().float()
    raise ValueError(fmt)


def _run_one(fmt: str, d_head: int, q_rows: int, k_rows: int) -> dict[str, object]:
    device = torch.device("cuda")
    constants_cpu = load_constants("cpu")
    constants = {k: v.to(device) for k, v in constants_cpu.items()}
    inputs = synthetic_inputs(fmt, k_rows, d_head, seed=2000 + d_head + len(fmt))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(3000 + d_head + len(fmt))
    q_cpu = torch.randn((q_rows, d_head), generator=gen, dtype=torch.float32).half()
    q = q_cpu.to(device)
    d_gpu = inputs["d"].contiguous().to(device)
    logits = torch.empty((q_rows, k_rows), dtype=torch.float32, device=device)
    grid = (q_rows, k_rows)
    block_d = triton.next_power_of_2(d_head)

    if fmt == "tbq4_0":
        qk_tbq4_kernel[grid](q, d_gpu, inputs["qs"].contiguous().to(device), constants["tbq4_centroids"], logits, K_ROWS=k_rows, D=d_head, BLOCK_D=block_d)
    elif fmt == "planar3_0":
        qk_planar3_kernel[grid](q, d_gpu, inputs["qs"].contiguous().to(device), inputs["signs"].contiguous().to(device), constants["pi_centroids_3bit"], constants["pi_cos"], constants["pi_sin"], logits, K_ROWS=k_rows, D=d_head, BLOCK_D=block_d)
    elif fmt == "iso3_0":
        qk_iso3_kernel[grid](q, d_gpu, inputs["qs"].contiguous().to(device), inputs["signs"].contiguous().to(device), constants["pi_centroids_3bit"], constants["pi_qw"], constants["pi_qx"], constants["pi_qy"], constants["pi_qz"], logits, K_ROWS=k_rows, D=d_head, BLOCK_D=block_d)
    else:
        raise ValueError(fmt)

    torch.cuda.synchronize()
    ref = q_cpu.float() @ _k_ref(fmt, inputs, constants_cpu).T
    diff = (logits.cpu() - ref).abs()
    max_abs = float(diff.max().item())
    tol = 2.0e-2
    return {"format": fmt, "D": d_head, "q_rows": q_rows, "k_rows": k_rows, "max_abs_err": max_abs, "tol": tol, "passed": max_abs <= tol}


def run_qk_checks(q_rows: int = 8, k_rows: int = 17, dims: tuple[int, ...] = (128, 256)) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("QK checks require a HIP/CUDA-visible torch device")
    checks = [_run_one(fmt, d_head, q_rows, k_rows) for d_head in dims for fmt in ("planar3_0", "iso3_0", "tbq4_0")]
    return {"result": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-rows", type=int, default=8)
    parser.add_argument("--k-rows", type=int, default=17)
    parser.add_argument("--dims", type=int, nargs="+", default=[128, 256])
    args = parser.parse_args()
    report = run_qk_checks(q_rows=args.q_rows, k_rows=args.k_rows, dims=tuple(args.dims))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
