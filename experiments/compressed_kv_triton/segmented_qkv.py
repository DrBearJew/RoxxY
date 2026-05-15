#!/usr/bin/env python3
"""Segmented 3D long-context QKV prototype over paged compressed K/V."""

from __future__ import annotations

import argparse
import json
import math

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
def _segment_valid(k_pos, q_row, seg_idx, SEGMENT_LEN: tl.constexpr, K_ROWS: tl.constexpr,
                   CAUSAL: tl.constexpr, Q_POS_START: tl.constexpr):
    seg_start: tl.constexpr = 0  # placeholder to keep expression names simple
    start = seg_idx * SEGMENT_LEN
    end = tl.minimum(start + SEGMENT_LEN, K_ROWS)
    valid = (k_pos >= start) & (k_pos < end)
    if CAUSAL:
        valid = valid & (k_pos <= Q_POS_START + q_row)
    return valid


@triton.jit
def segmented_tbq4_kernel(q, k_d, k_qs, v_d, v_qs, centroids, block_table, seg_acc, seg_m, seg_l,
                          Q_ROWS: tl.constexpr, K_ROWS: tl.constexpr, D: tl.constexpr,
                          BLOCK_D: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_TABLE_STRIDE: tl.constexpr,
                          NUM_SEGMENTS: tl.constexpr, SEGMENT_LEN: tl.constexpr,
                          SCALE: tl.constexpr, CAUSAL: tl.constexpr, Q_POS_START: tl.constexpr):
    q_row = tl.program_id(0)
    seg_idx = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    qv = tl.load(q + q_row * D + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m = -float("inf")
    l = 0.0
    for k_pos in range(0, K_ROWS):
        valid = _segment_valid(k_pos, q_row, seg_idx, SEGMENT_LEN, K_ROWS, CAUSAL, Q_POS_START)
        physical_row = paged_physical_rows(block_table, 0, k_pos, valid, BLOCK_TABLE_STRIDE, BLOCK_SIZE)
        kv = tbq4_values(k_d, k_qs, centroids, physical_row, offs, mask & valid, D)
        vv = tbq4_values(v_d, v_qs, centroids, physical_row, offs, mask & valid, D)
        score = tl.sum(qv * kv, axis=0) * SCALE
        score = tl.where(valid, score, -float("inf"))
        m_new = tl.maximum(m, score)
        m_new = tl.where(m_new > -float("inf"), m_new, 0.0)
        alpha = tl.exp(m - m_new)
        p = tl.exp(score - m_new)
        acc = acc * alpha + p * vv
        l = l * alpha + p
        m = m_new
    tl.store(seg_acc + (q_row * NUM_SEGMENTS + seg_idx) * D + offs, acc, mask=mask)
    tl.store(seg_m + q_row * NUM_SEGMENTS + seg_idx, m)
    tl.store(seg_l + q_row * NUM_SEGMENTS + seg_idx, l)


@triton.jit
def segmented_planar3_kernel(q, k_d, k_qs, k_signs, v_d, v_qs, v_signs, centroids, cos, sin, block_table, seg_acc, seg_m, seg_l,
                             Q_ROWS: tl.constexpr, K_ROWS: tl.constexpr, D: tl.constexpr,
                             BLOCK_D: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_TABLE_STRIDE: tl.constexpr,
                             NUM_SEGMENTS: tl.constexpr, SEGMENT_LEN: tl.constexpr,
                             SCALE: tl.constexpr, CAUSAL: tl.constexpr, Q_POS_START: tl.constexpr):
    q_row = tl.program_id(0)
    seg_idx = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    qv = tl.load(q + q_row * D + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m = -float("inf")
    l = 0.0
    for k_pos in range(0, K_ROWS):
        valid = _segment_valid(k_pos, q_row, seg_idx, SEGMENT_LEN, K_ROWS, CAUSAL, Q_POS_START)
        physical_row = paged_physical_rows(block_table, 0, k_pos, valid, BLOCK_TABLE_STRIDE, BLOCK_SIZE)
        kv = planar3_values(k_d, k_qs, k_signs, centroids, cos, sin, physical_row, offs, mask & valid, D)
        vv = planar3_values(v_d, v_qs, v_signs, centroids, cos, sin, physical_row, offs, mask & valid, D)
        score = tl.sum(qv * kv, axis=0) * SCALE
        score = tl.where(valid, score, -float("inf"))
        m_new = tl.maximum(m, score)
        m_new = tl.where(m_new > -float("inf"), m_new, 0.0)
        alpha = tl.exp(m - m_new)
        p = tl.exp(score - m_new)
        acc = acc * alpha + p * vv
        l = l * alpha + p
        m = m_new
    tl.store(seg_acc + (q_row * NUM_SEGMENTS + seg_idx) * D + offs, acc, mask=mask)
    tl.store(seg_m + q_row * NUM_SEGMENTS + seg_idx, m)
    tl.store(seg_l + q_row * NUM_SEGMENTS + seg_idx, l)


@triton.jit
def segmented_iso3_kernel(q, k_d, k_qs, k_signs, v_d, v_qs, v_signs, centroids, qw, qx, qy, qz, block_table, seg_acc, seg_m, seg_l,
                          Q_ROWS: tl.constexpr, K_ROWS: tl.constexpr, D: tl.constexpr,
                          BLOCK_D: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_TABLE_STRIDE: tl.constexpr,
                          NUM_SEGMENTS: tl.constexpr, SEGMENT_LEN: tl.constexpr,
                          SCALE: tl.constexpr, CAUSAL: tl.constexpr, Q_POS_START: tl.constexpr):
    q_row = tl.program_id(0)
    seg_idx = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    qv = tl.load(q + q_row * D + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m = -float("inf")
    l = 0.0
    for k_pos in range(0, K_ROWS):
        valid = _segment_valid(k_pos, q_row, seg_idx, SEGMENT_LEN, K_ROWS, CAUSAL, Q_POS_START)
        physical_row = paged_physical_rows(block_table, 0, k_pos, valid, BLOCK_TABLE_STRIDE, BLOCK_SIZE)
        kv = iso3_values(k_d, k_qs, k_signs, centroids, qw, qx, qy, qz, physical_row, offs, mask & valid, D)
        vv = iso3_values(v_d, v_qs, v_signs, centroids, qw, qx, qy, qz, physical_row, offs, mask & valid, D)
        score = tl.sum(qv * kv, axis=0) * SCALE
        score = tl.where(valid, score, -float("inf"))
        m_new = tl.maximum(m, score)
        m_new = tl.where(m_new > -float("inf"), m_new, 0.0)
        alpha = tl.exp(m - m_new)
        p = tl.exp(score - m_new)
        acc = acc * alpha + p * vv
        l = l * alpha + p
        m = m_new
    tl.store(seg_acc + (q_row * NUM_SEGMENTS + seg_idx) * D + offs, acc, mask=mask)
    tl.store(seg_m + q_row * NUM_SEGMENTS + seg_idx, m)
    tl.store(seg_l + q_row * NUM_SEGMENTS + seg_idx, l)


@triton.jit
def reduce_segments_kernel(seg_acc, seg_m, seg_l, out,
                           Q_ROWS: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
                           NUM_SEGMENTS: tl.constexpr):
    q_row = tl.program_id(0)
    offs_s = tl.arange(0, NUM_SEGMENTS)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    m = tl.load(seg_m + q_row * NUM_SEGMENTS + offs_s)
    overall_m = tl.max(m, axis=0)
    l = tl.load(seg_l + q_row * NUM_SEGMENTS + offs_s)
    weights = tl.exp(m - overall_m)
    denom = tl.sum(l * weights, axis=0)
    segv = tl.load(
        seg_acc + q_row * NUM_SEGMENTS * D + offs_s[:, None] * D + offs_d[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )
    acc = tl.sum(segv * weights[:, None], axis=0)
    outv = tl.where(denom > 0.0, acc / denom, 0.0)
    tl.store(out + q_row * D + offs_d, outv, mask=d_mask)


def _materialize_ref(fmt: str, inputs: dict[str, torch.Tensor], constants_cpu: dict[str, torch.Tensor]) -> torch.Tensor:
    if fmt == "tbq4_0":
        return materialize_tbq4_ref(inputs["d"], inputs["qs"], constants_cpu).half().float()
    if fmt == "planar3_0":
        return materialize_planar3_ref(inputs["d"], inputs["qs"], inputs["signs"], constants_cpu).half().float()
    if fmt == "iso3_0":
        return materialize_iso3_ref(inputs["d"], inputs["qs"], inputs["signs"], constants_cpu).half().float()
    raise ValueError(fmt)


def _reference(q_cpu: torch.Tensor, k_ref: torch.Tensor, v_ref: torch.Tensor, scale: float, causal: bool, q_pos_start: int) -> torch.Tensor:
    logits = (q_cpu.float() @ k_ref.T) * scale
    if causal:
        for q_idx in range(q_cpu.shape[0]):
            logits[q_idx, q_pos_start + q_idx + 1:] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return probs @ v_ref


def _run_one(fmt: str, d_head: int, q_rows: int, k_rows: int, block_size: int, num_segments: int, causal: bool) -> dict[str, object]:
    device = torch.device("cuda")
    constants_cpu = load_constants("cpu")
    constants = {k: v.to(device) for k, v in constants_cpu.items()}
    logical_k = synthetic_inputs(fmt, k_rows, d_head, seed=16000 + d_head + len(fmt))
    logical_v = synthetic_inputs(fmt, k_rows, d_head, seed=17000 + d_head + len(fmt))
    block_table_cpu = deterministic_block_table(k_rows, block_size)
    physical_k = physicalize_inputs(logical_k, block_table_cpu, block_size)
    physical_v = physicalize_inputs(logical_v, block_table_cpu, block_size)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(18000 + d_head + len(fmt))
    q_cpu = torch.randn((q_rows, d_head), generator=gen, dtype=torch.float32).half()

    q = q_cpu.contiguous().to(device)
    block_d = triton.next_power_of_2(d_head)
    scale = d_head ** -0.5
    q_pos_start = max(0, k_rows - q_rows)
    segment_len = math.ceil(k_rows / num_segments)
    seg_acc = torch.empty((q_rows, num_segments, d_head), dtype=torch.float32, device=device)
    seg_m = torch.empty((q_rows, num_segments), dtype=torch.float32, device=device)
    seg_l = torch.empty((q_rows, num_segments), dtype=torch.float32, device=device)
    out = torch.empty((q_rows, d_head), dtype=torch.float32, device=device)
    grid = (q_rows, num_segments)
    block_table = block_table_cpu.to(device)
    k_d = physical_k["d"].contiguous().to(device)
    v_d = physical_v["d"].contiguous().to(device)

    common = dict(
        Q_ROWS=q_rows, K_ROWS=k_rows, D=d_head, BLOCK_D=block_d, BLOCK_SIZE=block_size,
        BLOCK_TABLE_STRIDE=block_table_cpu.numel(), NUM_SEGMENTS=num_segments, SEGMENT_LEN=segment_len,
        SCALE=scale, CAUSAL=causal, Q_POS_START=q_pos_start,
    )

    if fmt == "tbq4_0":
        segmented_tbq4_kernel[grid](
            q, k_d, physical_k["qs"].contiguous().to(device), v_d, physical_v["qs"].contiguous().to(device),
            constants["tbq4_centroids"], block_table, seg_acc, seg_m, seg_l, **common)
    elif fmt == "planar3_0":
        segmented_planar3_kernel[grid](
            q, k_d, physical_k["qs"].contiguous().to(device), physical_k["signs"].contiguous().to(device),
            v_d, physical_v["qs"].contiguous().to(device), physical_v["signs"].contiguous().to(device),
            constants["pi_centroids_3bit"], constants["pi_cos"], constants["pi_sin"], block_table, seg_acc, seg_m, seg_l, **common)
    elif fmt == "iso3_0":
        segmented_iso3_kernel[grid](
            q, k_d, physical_k["qs"].contiguous().to(device), physical_k["signs"].contiguous().to(device),
            v_d, physical_v["qs"].contiguous().to(device), physical_v["signs"].contiguous().to(device),
            constants["pi_centroids_3bit"], constants["pi_qw"], constants["pi_qx"], constants["pi_qy"], constants["pi_qz"],
            block_table, seg_acc, seg_m, seg_l, **common)
    else:
        raise ValueError(fmt)

    reduce_segments_kernel[(q_rows,)](seg_acc, seg_m, seg_l, out, Q_ROWS=q_rows, D=d_head, BLOCK_D=block_d, NUM_SEGMENTS=num_segments)
    torch.cuda.synchronize()

    k_ref = _materialize_ref(fmt, logical_k, constants_cpu)
    v_ref = _materialize_ref(fmt, logical_v, constants_cpu)
    ref = _reference(q_cpu, k_ref, v_ref, scale, causal, q_pos_start)
    diff = (out.cpu() - ref).abs()
    max_abs = float(diff.max().item())
    tol = 4.0e-3
    return {
        "format": fmt,
        "D": d_head,
        "q_rows": q_rows,
        "k_rows": k_rows,
        "block_size": block_size,
        "num_segments": num_segments,
        "segment_len": segment_len,
        "causal": causal,
        "max_abs_err": max_abs,
        "tol": tol,
        "passed": max_abs <= tol,
    }


def run_segmented_qkv_checks(q_rows: int = 5, k_rows: int = 37, block_size: int = 4,
                             num_segments: int = 4, dims: tuple[int, ...] = (128, 256)) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("segmented QKV checks require a HIP/CUDA-visible torch device")
    checks = [
        _run_one(fmt, d_head, q_rows, k_rows, block_size, num_segments, causal)
        for d_head in dims
        for fmt in FORMATS
        for causal in (False, True)
    ]
    return {"result": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-rows", type=int, default=5)
    parser.add_argument("--k-rows", type=int, default=37)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--num-segments", type=int, default=4)
    parser.add_argument("--dims", type=int, nargs="+", default=[128, 256])
    args = parser.parse_args()
    report = run_segmented_qkv_checks(q_rows=args.q_rows, k_rows=args.k_rows, block_size=args.block_size,
                                      num_segments=args.num_segments, dims=tuple(args.dims))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
