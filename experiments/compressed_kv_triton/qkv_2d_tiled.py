#!/usr/bin/env python3
"""IBM-style 2D tiled full QKV prototype over paged compressed K/V."""

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
def _mask_scores(scores, offs_m, offs_n, m_mask, n_mask,
                 CAUSAL: tl.constexpr, Q_POS_START: tl.constexpr, SLIDING_WINDOW: tl.constexpr):
    valid = m_mask[:, None] & n_mask[None, :]
    q_abs = Q_POS_START + offs_m
    if CAUSAL:
        valid = valid & (offs_n[None, :] <= q_abs[:, None])
    if SLIDING_WINDOW > 0:
        valid = valid & (offs_n[None, :] <= q_abs[:, None]) & ((q_abs[:, None] - offs_n[None, :]) < SLIDING_WINDOW)
    return tl.where(valid, scores, -float("inf"))


@triton.jit
def qkv2d_tbq4_kernel(q, k_d, k_qs, v_d, v_qs, centroids, block_table, out,
                      Q_ROWS: tl.constexpr, K_ROWS: tl.constexpr, D: tl.constexpr,
                      BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr, BLOCK_TABLE_STRIDE: tl.constexpr,
                      SCALE: tl.constexpr, CAUSAL: tl.constexpr, Q_POS_START: tl.constexpr,
                      SLIDING_WINDOW: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < Q_ROWS
    d_mask = offs_d < D
    qv = tl.load(q + offs_m[:, None] * D + offs_d[None, :], mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    M = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, K_ROWS, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < K_ROWS
        physical_rows = paged_physical_rows(block_table, 0, offs_n, n_mask, BLOCK_TABLE_STRIDE, BLOCK_SIZE)
        K = tbq4_values(k_d, k_qs, centroids, physical_rows[None, :], offs_d[:, None], d_mask[:, None] & n_mask[None, :], D)
        V = tbq4_values(v_d, v_qs, centroids, physical_rows[:, None], offs_d[None, :], n_mask[:, None] & d_mask[None, :], D)
        S = tl.dot(qv.to(tl.float16), K.to(tl.float16), out_dtype=tl.float32) * SCALE
        S = _mask_scores(S, offs_m, offs_n, m_mask, n_mask, CAUSAL, Q_POS_START, SLIDING_WINDOW)
        M_new = tl.maximum(M, tl.max(S, axis=1))
        M_new = tl.where(M_new > -float("inf"), M_new, 0.0)
        P = tl.exp(S - M_new[:, None])
        alpha = tl.exp(M - M_new)
        acc = acc * alpha[:, None] + tl.dot(P.to(tl.float16), V.to(tl.float16), out_dtype=tl.float32)
        L = L * alpha + tl.sum(P, axis=1)
        M = M_new

    outv = tl.where(L[:, None] > 0.0, acc / L[:, None], 0.0)
    tl.store(out + offs_m[:, None] * D + offs_d[None, :], outv, mask=m_mask[:, None] & d_mask[None, :])


@triton.jit
def qkv2d_planar3_kernel(q, k_d, k_qs, k_signs, v_d, v_qs, v_signs, centroids, cos, sin, block_table, out,
                          Q_ROWS: tl.constexpr, K_ROWS: tl.constexpr, D: tl.constexpr,
                          BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_SIZE: tl.constexpr, BLOCK_TABLE_STRIDE: tl.constexpr,
                          SCALE: tl.constexpr, CAUSAL: tl.constexpr, Q_POS_START: tl.constexpr,
                          SLIDING_WINDOW: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < Q_ROWS
    d_mask = offs_d < D
    qv = tl.load(q + offs_m[:, None] * D + offs_d[None, :], mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    M = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, K_ROWS, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < K_ROWS
        physical_rows = paged_physical_rows(block_table, 0, offs_n, n_mask, BLOCK_TABLE_STRIDE, BLOCK_SIZE)
        K = planar3_values(k_d, k_qs, k_signs, centroids, cos, sin, physical_rows[None, :], offs_d[:, None], d_mask[:, None] & n_mask[None, :], D)
        V = planar3_values(v_d, v_qs, v_signs, centroids, cos, sin, physical_rows[:, None], offs_d[None, :], n_mask[:, None] & d_mask[None, :], D)
        S = tl.dot(qv.to(tl.float16), K.to(tl.float16), out_dtype=tl.float32) * SCALE
        S = _mask_scores(S, offs_m, offs_n, m_mask, n_mask, CAUSAL, Q_POS_START, SLIDING_WINDOW)
        M_new = tl.maximum(M, tl.max(S, axis=1))
        M_new = tl.where(M_new > -float("inf"), M_new, 0.0)
        P = tl.exp(S - M_new[:, None])
        alpha = tl.exp(M - M_new)
        acc = acc * alpha[:, None] + tl.dot(P.to(tl.float16), V.to(tl.float16), out_dtype=tl.float32)
        L = L * alpha + tl.sum(P, axis=1)
        M = M_new

    outv = tl.where(L[:, None] > 0.0, acc / L[:, None], 0.0)
    tl.store(out + offs_m[:, None] * D + offs_d[None, :], outv, mask=m_mask[:, None] & d_mask[None, :])


@triton.jit
def qkv2d_iso3_kernel(q, k_d, k_qs, k_signs, v_d, v_qs, v_signs, centroids, qw, qx, qy, qz, block_table, out,
                       Q_ROWS: tl.constexpr, K_ROWS: tl.constexpr, D: tl.constexpr,
                       BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_SIZE: tl.constexpr, BLOCK_TABLE_STRIDE: tl.constexpr,
                       SCALE: tl.constexpr, CAUSAL: tl.constexpr, Q_POS_START: tl.constexpr,
                       SLIDING_WINDOW: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < Q_ROWS
    d_mask = offs_d < D
    qv = tl.load(q + offs_m[:, None] * D + offs_d[None, :], mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    M = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, K_ROWS, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < K_ROWS
        physical_rows = paged_physical_rows(block_table, 0, offs_n, n_mask, BLOCK_TABLE_STRIDE, BLOCK_SIZE)
        K = iso3_values(k_d, k_qs, k_signs, centroids, qw, qx, qy, qz, physical_rows[None, :], offs_d[:, None], d_mask[:, None] & n_mask[None, :], D)
        V = iso3_values(v_d, v_qs, v_signs, centroids, qw, qx, qy, qz, physical_rows[:, None], offs_d[None, :], n_mask[:, None] & d_mask[None, :], D)
        S = tl.dot(qv.to(tl.float16), K.to(tl.float16), out_dtype=tl.float32) * SCALE
        S = _mask_scores(S, offs_m, offs_n, m_mask, n_mask, CAUSAL, Q_POS_START, SLIDING_WINDOW)
        M_new = tl.maximum(M, tl.max(S, axis=1))
        M_new = tl.where(M_new > -float("inf"), M_new, 0.0)
        P = tl.exp(S - M_new[:, None])
        alpha = tl.exp(M - M_new)
        acc = acc * alpha[:, None] + tl.dot(P.to(tl.float16), V.to(tl.float16), out_dtype=tl.float32)
        L = L * alpha + tl.sum(P, axis=1)
        M = M_new

    outv = tl.where(L[:, None] > 0.0, acc / L[:, None], 0.0)
    tl.store(out + offs_m[:, None] * D + offs_d[None, :], outv, mask=m_mask[:, None] & d_mask[None, :])


def _materialize_ref(fmt: str, inputs: dict[str, torch.Tensor], constants_cpu: dict[str, torch.Tensor]) -> torch.Tensor:
    if fmt == "tbq4_0":
        return materialize_tbq4_ref(inputs["d"], inputs["qs"], constants_cpu).half().float()
    if fmt == "planar3_0":
        return materialize_planar3_ref(inputs["d"], inputs["qs"], inputs["signs"], constants_cpu).half().float()
    if fmt == "iso3_0":
        return materialize_iso3_ref(inputs["d"], inputs["qs"], inputs["signs"], constants_cpu).half().float()
    raise ValueError(fmt)


def _reference(q_cpu: torch.Tensor, k_ref: torch.Tensor, v_ref: torch.Tensor, scale: float,
               causal: bool, q_pos_start: int, sliding_window: int) -> torch.Tensor:
    q_rows = q_cpu.shape[0]
    k_rows = k_ref.shape[0]
    logits = (q_cpu.float() @ k_ref.T) * scale
    for q_idx in range(q_rows):
        q_abs = q_pos_start + q_idx
        for k_idx in range(k_rows):
            masked = False
            if causal and k_idx > q_abs:
                masked = True
            if sliding_window > 0 and (k_idx > q_abs or q_abs - k_idx >= sliding_window):
                masked = True
            if masked:
                logits[q_idx, k_idx] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return probs @ v_ref


def _run_one(fmt: str, d_head: int, q_rows: int, k_rows: int, block_size: int,
             block_m: int, block_n: int, causal: bool, sliding_window: int) -> dict[str, object]:
    device = torch.device("cuda")
    constants_cpu = load_constants("cpu")
    constants = {k: v.to(device) for k, v in constants_cpu.items()}
    logical_k = synthetic_inputs(fmt, k_rows, d_head, seed=10000 + d_head + len(fmt))
    logical_v = synthetic_inputs(fmt, k_rows, d_head, seed=11000 + d_head + len(fmt))
    block_table_cpu = deterministic_block_table(k_rows, block_size)
    physical_k = physicalize_inputs(logical_k, block_table_cpu, block_size)
    physical_v = physicalize_inputs(logical_v, block_table_cpu, block_size)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(12000 + d_head + len(fmt))
    q_cpu = torch.randn((q_rows, d_head), generator=gen, dtype=torch.float32).half()

    q = q_cpu.contiguous().to(device)
    k_d = physical_k["d"].contiguous().to(device)
    v_d = physical_v["d"].contiguous().to(device)
    block_table = block_table_cpu.to(device)
    out = torch.empty((q_rows, d_head), dtype=torch.float32, device=device)
    grid = (triton.cdiv(q_rows, block_m),)
    block_d = triton.next_power_of_2(d_head)
    scale = d_head ** -0.5
    q_pos_start = max(0, k_rows - q_rows)

    if fmt == "tbq4_0":
        qkv2d_tbq4_kernel[grid](
            q, k_d, physical_k["qs"].contiguous().to(device), v_d, physical_v["qs"].contiguous().to(device),
            constants["tbq4_centroids"], block_table, out,
            Q_ROWS=q_rows, K_ROWS=k_rows, D=d_head, BLOCK_D=block_d, BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_SIZE=block_size, BLOCK_TABLE_STRIDE=block_table_cpu.numel(), SCALE=scale, CAUSAL=causal,
            Q_POS_START=q_pos_start, SLIDING_WINDOW=sliding_window)
    elif fmt == "planar3_0":
        qkv2d_planar3_kernel[grid](
            q, k_d, physical_k["qs"].contiguous().to(device), physical_k["signs"].contiguous().to(device),
            v_d, physical_v["qs"].contiguous().to(device), physical_v["signs"].contiguous().to(device),
            constants["pi_centroids_3bit"], constants["pi_cos"], constants["pi_sin"], block_table, out,
            Q_ROWS=q_rows, K_ROWS=k_rows, D=d_head, BLOCK_D=block_d, BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_SIZE=block_size, BLOCK_TABLE_STRIDE=block_table_cpu.numel(), SCALE=scale, CAUSAL=causal,
            Q_POS_START=q_pos_start, SLIDING_WINDOW=sliding_window)
    elif fmt == "iso3_0":
        qkv2d_iso3_kernel[grid](
            q, k_d, physical_k["qs"].contiguous().to(device), physical_k["signs"].contiguous().to(device),
            v_d, physical_v["qs"].contiguous().to(device), physical_v["signs"].contiguous().to(device),
            constants["pi_centroids_3bit"], constants["pi_qw"], constants["pi_qx"], constants["pi_qy"], constants["pi_qz"], block_table, out,
            Q_ROWS=q_rows, K_ROWS=k_rows, D=d_head, BLOCK_D=block_d, BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_SIZE=block_size, BLOCK_TABLE_STRIDE=block_table_cpu.numel(), SCALE=scale, CAUSAL=causal,
            Q_POS_START=q_pos_start, SLIDING_WINDOW=sliding_window)
    else:
        raise ValueError(fmt)

    torch.cuda.synchronize()
    k_ref = _materialize_ref(fmt, logical_k, constants_cpu)
    v_ref = _materialize_ref(fmt, logical_v, constants_cpu)
    ref = _reference(q_cpu, k_ref, v_ref, scale, causal, q_pos_start, sliding_window)
    diff = (out.cpu() - ref).abs()
    max_abs = float(diff.max().item())
    tol = 4.0e-3
    return {
        "format": fmt,
        "D": d_head,
        "q_rows": q_rows,
        "k_rows": k_rows,
        "block_m": block_m,
        "block_n": block_n,
        "block_size": block_size,
        "causal": causal,
        "sliding_window": sliding_window,
        "max_abs_err": max_abs,
        "tol": tol,
        "passed": max_abs <= tol,
    }


def run_qkv_2d_checks(q_rows: int = 8, k_rows: int = 17, block_size: int = 4,
                      block_m: int = 16, block_n: int = 16,
                      dims: tuple[int, ...] = (128, 256)) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("2D QKV checks require a HIP/CUDA-visible torch device")
    cases = ((False, 0), (True, 0), (True, 4))
    checks = [
        _run_one(fmt, d_head, q_rows, k_rows, block_size, block_m, block_n, causal, sliding_window)
        for d_head in dims
        for fmt in FORMATS
        for causal, sliding_window in cases
    ]
    return {"result": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-rows", type=int, default=8)
    parser.add_argument("--k-rows", type=int, default=17)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=16)
    parser.add_argument("--dims", type=int, nargs="+", default=[128, 256])
    args = parser.parse_args()
    report = run_qkv_2d_checks(q_rows=args.q_rows, k_rows=args.k_rows, block_size=args.block_size,
                               block_m=args.block_m, block_n=args.block_n, dims=tuple(args.dims))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
