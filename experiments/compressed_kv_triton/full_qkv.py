#!/usr/bin/env python3
"""Full QKV Triton prototype with direct compressed K/V materialization.

This is correctness-first and intentionally small: one Triton program owns one Q
row and emits the full D-dimensional output. It materializes only the current K
and V row vectors in registers while walking the KV sequence with online softmax.
"""

from __future__ import annotations

import argparse
import json

import torch
import triton
import triton.language as tl

from materializers import (
    load_constants,
    materialize_iso3_ref,
    materialize_planar3_ref,
    materialize_tbq4_ref,
    synthetic_inputs,
)
from qk_only import _iso3_values, _planar3_values, _tbq4_values


@triton.jit
def full_qkv_tbq4_kernel(q, k_d, k_qs, v_d, v_qs, centroids, out,
                         K_ROWS: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
                         SCALE: tl.constexpr, CAUSAL: tl.constexpr, Q_POS_START: tl.constexpr):
    q_row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    qv = tl.load(q + q_row * D + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m = -float("inf")
    l = 0.0

    for k_row in range(0, K_ROWS):
        kv = _tbq4_values(k_d, k_qs, centroids, k_row, offs, mask, D)
        vv = _tbq4_values(v_d, v_qs, centroids, k_row, offs, mask, D)
        score = tl.sum(qv * kv, axis=0) * SCALE
        if CAUSAL:
            score = tl.where(k_row <= Q_POS_START + q_row, score, -float("inf"))
        m_new = tl.maximum(m, score)
        alpha = tl.exp(m - m_new)
        p = tl.exp(score - m_new)
        acc = acc * alpha + p * vv
        l = l * alpha + p
        m = m_new

    outv = acc / l
    tl.store(out + q_row * D + offs, outv, mask=mask)


@triton.jit
def full_qkv_planar3_kernel(q, k_d, k_qs, k_signs, v_d, v_qs, v_signs, centroids, cos, sin, out,
                            K_ROWS: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
                            SCALE: tl.constexpr, CAUSAL: tl.constexpr, Q_POS_START: tl.constexpr):
    q_row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    qv = tl.load(q + q_row * D + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m = -float("inf")
    l = 0.0

    for k_row in range(0, K_ROWS):
        kv = _planar3_values(k_d, k_qs, k_signs, centroids, cos, sin, k_row, offs, mask, D)
        vv = _planar3_values(v_d, v_qs, v_signs, centroids, cos, sin, k_row, offs, mask, D)
        score = tl.sum(qv * kv, axis=0) * SCALE
        if CAUSAL:
            score = tl.where(k_row <= Q_POS_START + q_row, score, -float("inf"))
        m_new = tl.maximum(m, score)
        alpha = tl.exp(m - m_new)
        p = tl.exp(score - m_new)
        acc = acc * alpha + p * vv
        l = l * alpha + p
        m = m_new

    outv = acc / l
    tl.store(out + q_row * D + offs, outv, mask=mask)


@triton.jit
def full_qkv_iso3_kernel(q, k_d, k_qs, k_signs, v_d, v_qs, v_signs, centroids, qw, qx, qy, qz, out,
                         K_ROWS: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
                         SCALE: tl.constexpr, CAUSAL: tl.constexpr, Q_POS_START: tl.constexpr):
    q_row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    qv = tl.load(q + q_row * D + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m = -float("inf")
    l = 0.0

    for k_row in range(0, K_ROWS):
        kv = _iso3_values(k_d, k_qs, k_signs, centroids, qw, qx, qy, qz, k_row, offs, mask, D)
        vv = _iso3_values(v_d, v_qs, v_signs, centroids, qw, qx, qy, qz, k_row, offs, mask, D)
        score = tl.sum(qv * kv, axis=0) * SCALE
        if CAUSAL:
            score = tl.where(k_row <= Q_POS_START + q_row, score, -float("inf"))
        m_new = tl.maximum(m, score)
        alpha = tl.exp(m - m_new)
        p = tl.exp(score - m_new)
        acc = acc * alpha + p * vv
        l = l * alpha + p
        m = m_new

    outv = acc / l
    tl.store(out + q_row * D + offs, outv, mask=mask)


def _materialize_ref(fmt: str, inputs: dict[str, torch.Tensor], constants_cpu: dict[str, torch.Tensor]) -> torch.Tensor:
    if fmt == "tbq4_0":
        return materialize_tbq4_ref(inputs["d"], inputs["qs"], constants_cpu).half().float()
    if fmt == "planar3_0":
        return materialize_planar3_ref(inputs["d"], inputs["qs"], inputs["signs"], constants_cpu).half().float()
    if fmt == "iso3_0":
        return materialize_iso3_ref(inputs["d"], inputs["qs"], inputs["signs"], constants_cpu).half().float()
    raise ValueError(fmt)


def _run_one(fmt: str, d_head: int, q_rows: int, k_rows: int, causal: bool) -> dict[str, object]:
    device = torch.device("cuda")
    constants_cpu = load_constants("cpu")
    constants = {k: v.to(device) for k, v in constants_cpu.items()}
    k_inputs = synthetic_inputs(fmt, k_rows, d_head, seed=5000 + d_head + len(fmt))
    v_inputs = synthetic_inputs(fmt, k_rows, d_head, seed=6000 + d_head + len(fmt))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(7000 + d_head + len(fmt))
    q_cpu = torch.randn((q_rows, d_head), generator=gen, dtype=torch.float32).half()
    q = q_cpu.to(device)
    out = torch.empty((q_rows, d_head), dtype=torch.float32, device=device)
    grid = (q_rows,)
    block_d = triton.next_power_of_2(d_head)
    scale = d_head ** -0.5
    q_pos_start = max(0, k_rows - q_rows)

    k_d = k_inputs["d"].contiguous().to(device)
    v_d = v_inputs["d"].contiguous().to(device)

    if fmt == "tbq4_0":
        full_qkv_tbq4_kernel[grid](
            q, k_d, k_inputs["qs"].contiguous().to(device), v_d, v_inputs["qs"].contiguous().to(device), constants["tbq4_centroids"], out,
            K_ROWS=k_rows, D=d_head, BLOCK_D=block_d, SCALE=scale, CAUSAL=causal, Q_POS_START=q_pos_start)
    elif fmt == "planar3_0":
        full_qkv_planar3_kernel[grid](
            q, k_d, k_inputs["qs"].contiguous().to(device), k_inputs["signs"].contiguous().to(device),
            v_d, v_inputs["qs"].contiguous().to(device), v_inputs["signs"].contiguous().to(device),
            constants["pi_centroids_3bit"], constants["pi_cos"], constants["pi_sin"], out,
            K_ROWS=k_rows, D=d_head, BLOCK_D=block_d, SCALE=scale, CAUSAL=causal, Q_POS_START=q_pos_start)
    elif fmt == "iso3_0":
        full_qkv_iso3_kernel[grid](
            q, k_d, k_inputs["qs"].contiguous().to(device), k_inputs["signs"].contiguous().to(device),
            v_d, v_inputs["qs"].contiguous().to(device), v_inputs["signs"].contiguous().to(device),
            constants["pi_centroids_3bit"], constants["pi_qw"], constants["pi_qx"], constants["pi_qy"], constants["pi_qz"], out,
            K_ROWS=k_rows, D=d_head, BLOCK_D=block_d, SCALE=scale, CAUSAL=causal, Q_POS_START=q_pos_start)
    else:
        raise ValueError(fmt)

    torch.cuda.synchronize()

    k_ref = _materialize_ref(fmt, k_inputs, constants_cpu)
    v_ref = _materialize_ref(fmt, v_inputs, constants_cpu)
    logits = (q_cpu.float() @ k_ref.T) * scale
    if causal:
        for q_idx in range(q_rows):
            logits[q_idx, q_pos_start + q_idx + 1 :] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    ref = probs @ v_ref
    diff = (out.cpu() - ref).abs()
    max_abs = float(diff.max().item())
    tol = 2.5e-3
    return {"format": fmt, "D": d_head, "q_rows": q_rows, "k_rows": k_rows, "causal": causal, "max_abs_err": max_abs, "tol": tol, "passed": max_abs <= tol}


def run_full_qkv_checks(q_rows: int = 5, k_rows: int = 13, dims: tuple[int, ...] = (128, 256)) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("full QKV checks require a HIP/CUDA-visible torch device")
    checks = [
        _run_one(fmt, d_head, q_rows, k_rows, causal)
        for d_head in dims
        for fmt in ("planar3_0", "iso3_0", "tbq4_0")
        for causal in (False, True)
    ]
    return {"result": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-rows", type=int, default=5)
    parser.add_argument("--k-rows", type=int, default=13)
    parser.add_argument("--dims", type=int, nargs="+", default=[128, 256])
    args = parser.parse_args()
    report = run_full_qkv_checks(q_rows=args.q_rows, k_rows=args.k_rows, dims=tuple(args.dims))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
