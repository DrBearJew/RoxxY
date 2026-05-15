#!/usr/bin/env python3
"""Variable-length cu_seqlens-style QKV tests over paged compressed K/V.

This is a metadata correctness prototype: multiple sequences, different Q/KV
lengths, and GQA query-head to KV-head mapping. One Triton program owns one
(query token, query head) output row.
"""

from __future__ import annotations

import argparse
import json
import math

import torch
import triton
import triton.language as tl

from compressed_kv_tl import FORMATS, iso3_values, paged_physical_rows, planar3_values, tbq4_values
from materializers import (
    load_constants,
    materialize_iso3_ref,
    materialize_planar3_ref,
    materialize_tbq4_ref,
    synthetic_inputs,
)


@triton.jit
def _find_seq_idx(cu_q, q_token, NUM_SEQS: tl.constexpr):
    seq = tl.full((), 0, dtype=tl.int64)
    for s in range(0, NUM_SEQS):
        start = tl.load(cu_q + s)
        stop = tl.load(cu_q + s + 1)
        seq = tl.where((q_token >= start) & (q_token < stop), s, seq)
    return seq


@triton.jit
def varlen_qkv_tbq4_kernel(q, k_d, k_qs, v_d, v_qs, centroids, block_tables, cu_q, cu_k, seq_lens_k, out,
                           TOTAL_Q: tl.constexpr, NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr,
                           Q_PER_KV: tl.constexpr, NUM_SEQS: tl.constexpr, MAX_K: tl.constexpr,
                           D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                           BLOCK_TABLE_STRIDE: tl.constexpr, SCALE: tl.constexpr,
                           CAUSAL: tl.constexpr, SLIDING_WINDOW: tl.constexpr):
    q_token = tl.program_id(0)
    q_head = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    seq = _find_seq_idx(cu_q, q_token, NUM_SEQS)
    q_start = tl.load(cu_q + seq)
    q_stop = tl.load(cu_q + seq + 1)
    q_len = q_stop - q_start
    k_start = tl.load(cu_k + seq)
    k_len = tl.load(seq_lens_k + seq)
    local_q = q_token - q_start
    context_len = k_len - q_len
    q_abs = context_len + local_q
    kv_head = q_head // Q_PER_KV
    qv = tl.load(q + (q_token * NUM_Q_HEADS + q_head) * D + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m = -float("inf")
    l = 0.0
    for k_pos in range(0, MAX_K):
        valid = k_pos < k_len
        if CAUSAL:
            valid = valid & (k_pos <= q_abs)
        if SLIDING_WINDOW > 0:
            valid = valid & (k_pos <= q_abs) & ((q_abs - k_pos) < SLIDING_WINDOW)
        physical_token = paged_physical_rows(block_tables, seq, k_pos, valid, BLOCK_TABLE_STRIDE, BLOCK_SIZE)
        physical_row = physical_token * NUM_KV_HEADS + kv_head
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
    outv = tl.where(l > 0.0, acc / l, 0.0)
    tl.store(out + (q_token * NUM_Q_HEADS + q_head) * D + offs, outv, mask=mask & (q_token < TOTAL_Q))


@triton.jit
def varlen_qkv_planar3_kernel(q, k_d, k_qs, k_signs, v_d, v_qs, v_signs, centroids, cos, sin,
                              block_tables, cu_q, cu_k, seq_lens_k, out,
                              TOTAL_Q: tl.constexpr, NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr,
                              Q_PER_KV: tl.constexpr, NUM_SEQS: tl.constexpr, MAX_K: tl.constexpr,
                              D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                              BLOCK_TABLE_STRIDE: tl.constexpr, SCALE: tl.constexpr,
                              CAUSAL: tl.constexpr, SLIDING_WINDOW: tl.constexpr):
    q_token = tl.program_id(0)
    q_head = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    seq = _find_seq_idx(cu_q, q_token, NUM_SEQS)
    q_start = tl.load(cu_q + seq)
    q_stop = tl.load(cu_q + seq + 1)
    q_len = q_stop - q_start
    k_start = tl.load(cu_k + seq)
    k_len = tl.load(seq_lens_k + seq)
    local_q = q_token - q_start
    context_len = k_len - q_len
    q_abs = context_len + local_q
    kv_head = q_head // Q_PER_KV
    qv = tl.load(q + (q_token * NUM_Q_HEADS + q_head) * D + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m = -float("inf")
    l = 0.0
    for k_pos in range(0, MAX_K):
        valid = k_pos < k_len
        if CAUSAL:
            valid = valid & (k_pos <= q_abs)
        if SLIDING_WINDOW > 0:
            valid = valid & (k_pos <= q_abs) & ((q_abs - k_pos) < SLIDING_WINDOW)
        physical_token = paged_physical_rows(block_tables, seq, k_pos, valid, BLOCK_TABLE_STRIDE, BLOCK_SIZE)
        physical_row = physical_token * NUM_KV_HEADS + kv_head
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
    outv = tl.where(l > 0.0, acc / l, 0.0)
    tl.store(out + (q_token * NUM_Q_HEADS + q_head) * D + offs, outv, mask=mask & (q_token < TOTAL_Q))


@triton.jit
def varlen_qkv_iso3_kernel(q, k_d, k_qs, k_signs, v_d, v_qs, v_signs, centroids, qw, qx, qy, qz,
                           block_tables, cu_q, cu_k, seq_lens_k, out,
                           TOTAL_Q: tl.constexpr, NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr,
                           Q_PER_KV: tl.constexpr, NUM_SEQS: tl.constexpr, MAX_K: tl.constexpr,
                           D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                           BLOCK_TABLE_STRIDE: tl.constexpr, SCALE: tl.constexpr,
                           CAUSAL: tl.constexpr, SLIDING_WINDOW: tl.constexpr):
    q_token = tl.program_id(0)
    q_head = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    seq = _find_seq_idx(cu_q, q_token, NUM_SEQS)
    q_start = tl.load(cu_q + seq)
    q_stop = tl.load(cu_q + seq + 1)
    q_len = q_stop - q_start
    k_start = tl.load(cu_k + seq)
    k_len = tl.load(seq_lens_k + seq)
    local_q = q_token - q_start
    context_len = k_len - q_len
    q_abs = context_len + local_q
    kv_head = q_head // Q_PER_KV
    qv = tl.load(q + (q_token * NUM_Q_HEADS + q_head) * D + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m = -float("inf")
    l = 0.0
    for k_pos in range(0, MAX_K):
        valid = k_pos < k_len
        if CAUSAL:
            valid = valid & (k_pos <= q_abs)
        if SLIDING_WINDOW > 0:
            valid = valid & (k_pos <= q_abs) & ((q_abs - k_pos) < SLIDING_WINDOW)
        physical_token = paged_physical_rows(block_tables, seq, k_pos, valid, BLOCK_TABLE_STRIDE, BLOCK_SIZE)
        physical_row = physical_token * NUM_KV_HEADS + kv_head
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
    outv = tl.where(l > 0.0, acc / l, 0.0)
    tl.store(out + (q_token * NUM_Q_HEADS + q_head) * D + offs, outv, mask=mask & (q_token < TOTAL_Q))


def _materialize_ref(fmt: str, inputs: dict[str, torch.Tensor], constants_cpu: dict[str, torch.Tensor]) -> torch.Tensor:
    if fmt == "tbq4_0":
        return materialize_tbq4_ref(inputs["d"], inputs["qs"], constants_cpu).half().float()
    if fmt == "planar3_0":
        return materialize_planar3_ref(inputs["d"], inputs["qs"], inputs["signs"], constants_cpu).half().float()
    if fmt == "iso3_0":
        return materialize_iso3_ref(inputs["d"], inputs["qs"], inputs["signs"], constants_cpu).half().float()
    raise ValueError(fmt)


def _make_metadata(q_lens: list[int], k_lens: list[int], block_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    cu_q = torch.tensor([0, *torch.cumsum(torch.tensor(q_lens), dim=0).tolist()], dtype=torch.int64)
    cu_k = torch.tensor([0, *torch.cumsum(torch.tensor(k_lens), dim=0).tolist()], dtype=torch.int64)
    blocks_per_seq = [math.ceil(k / block_size) for k in k_lens]
    stride = max(blocks_per_seq)
    block_tables = torch.full((len(k_lens), stride), -1, dtype=torch.int64)
    physical_cursor = 0
    for seq, blocks in enumerate(blocks_per_seq):
        ids = torch.arange(physical_cursor, physical_cursor + blocks, dtype=torch.int64)
        if blocks > 1:
            ids = torch.roll(ids, shifts=1)
        block_tables[seq, :blocks] = ids
        physical_cursor += blocks
    seq_lens_k = torch.tensor(k_lens, dtype=torch.int64)
    return cu_q, cu_k, seq_lens_k, block_tables.contiguous(), physical_cursor * block_size


def _physicalize_varlen(logical: dict[str, torch.Tensor], cu_k: torch.Tensor, k_lens: list[int], block_tables: torch.Tensor,
                        block_size: int, physical_tokens: int, num_kv_heads: int) -> dict[str, torch.Tensor]:
    physical: dict[str, torch.Tensor] = {}
    for name, tensor in logical.items():
        out = torch.zeros((physical_tokens * num_kv_heads, *tensor.shape[1:]), dtype=tensor.dtype)
        for seq, k_len in enumerate(k_lens):
            for k_pos in range(k_len):
                physical_token = int(block_tables[seq, k_pos // block_size]) * block_size + k_pos % block_size
                logical_token = int(cu_k[seq]) + k_pos
                for kv_head in range(num_kv_heads):
                    out[physical_token * num_kv_heads + kv_head].copy_(tensor[logical_token * num_kv_heads + kv_head])
        physical[name] = out.contiguous()
    return physical


def _reference(q_cpu: torch.Tensor, k_ref: torch.Tensor, v_ref: torch.Tensor, q_lens: list[int], k_lens: list[int],
               num_q_heads: int, num_kv_heads: int, scale: float, causal: bool, sliding_window: int) -> torch.Tensor:
    q_per_kv = num_q_heads // num_kv_heads
    out = torch.empty_like(q_cpu, dtype=torch.float32)
    q_cursor = 0
    k_cursor = 0
    for seq, (q_len, k_len) in enumerate(zip(q_lens, k_lens, strict=True)):
        context_len = k_len - q_len
        for q_pos in range(q_len):
            q_token = q_cursor + q_pos
            q_abs = context_len + q_pos
            for q_head in range(num_q_heads):
                kv_head = q_head // q_per_kv
                rows = torch.arange(k_len)
                k_rows = (k_cursor + rows) * num_kv_heads + kv_head
                logits = (q_cpu[q_token, q_head].float() @ k_ref[k_rows].T) * scale
                valid = torch.ones(k_len, dtype=torch.bool)
                if causal:
                    valid &= rows <= q_abs
                if sliding_window > 0:
                    valid &= (rows <= q_abs) & ((q_abs - rows) < sliding_window)
                logits = torch.where(valid, logits, torch.full_like(logits, -float("inf")))
                probs = torch.softmax(logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0)
                out[q_token, q_head] = probs @ v_ref[k_rows]
        q_cursor += q_len
        k_cursor += k_len
    return out


def _run_one(fmt: str, d_head: int, q_lens: list[int], k_lens: list[int], num_q_heads: int,
             num_kv_heads: int, block_size: int, causal: bool, sliding_window: int) -> dict[str, object]:
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    device = torch.device("cuda")
    constants_cpu = load_constants("cpu")
    constants = {k: v.to(device) for k, v in constants_cpu.items()}
    total_q = sum(q_lens)
    total_k = sum(k_lens)
    cu_q, cu_k, seq_lens_k, block_tables, physical_tokens = _make_metadata(q_lens, k_lens, block_size)
    logical_rows = total_k * num_kv_heads
    logical_k = synthetic_inputs(fmt, logical_rows, d_head, seed=13000 + d_head + len(fmt))
    logical_v = synthetic_inputs(fmt, logical_rows, d_head, seed=14000 + d_head + len(fmt))
    physical_k = _physicalize_varlen(logical_k, cu_k, k_lens, block_tables, block_size, physical_tokens, num_kv_heads)
    physical_v = _physicalize_varlen(logical_v, cu_k, k_lens, block_tables, block_size, physical_tokens, num_kv_heads)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(15000 + d_head + len(fmt))
    q_cpu = torch.randn((total_q, num_q_heads, d_head), generator=gen, dtype=torch.float32).half()

    q = q_cpu.contiguous().to(device)
    out = torch.empty((total_q, num_q_heads, d_head), dtype=torch.float32, device=device)
    grid = (total_q, num_q_heads)
    block_d = triton.next_power_of_2(d_head)
    scale = d_head ** -0.5

    common = dict(
        TOTAL_Q=total_q, NUM_Q_HEADS=num_q_heads, NUM_KV_HEADS=num_kv_heads,
        Q_PER_KV=num_q_heads // num_kv_heads, NUM_SEQS=len(q_lens), MAX_K=max(k_lens),
        D=d_head, BLOCK_D=block_d, BLOCK_SIZE=block_size, BLOCK_TABLE_STRIDE=block_tables.shape[1],
        SCALE=scale, CAUSAL=causal, SLIDING_WINDOW=sliding_window,
    )
    k_d = physical_k["d"].contiguous().to(device)
    v_d = physical_v["d"].contiguous().to(device)
    block_tables_gpu = block_tables.to(device)
    cu_q_gpu = cu_q.to(device)
    cu_k_gpu = cu_k.to(device)
    seq_lens_gpu = seq_lens_k.to(device)

    if fmt == "tbq4_0":
        varlen_qkv_tbq4_kernel[grid](
            q, k_d, physical_k["qs"].contiguous().to(device), v_d, physical_v["qs"].contiguous().to(device),
            constants["tbq4_centroids"], block_tables_gpu, cu_q_gpu, cu_k_gpu, seq_lens_gpu, out, **common)
    elif fmt == "planar3_0":
        varlen_qkv_planar3_kernel[grid](
            q, k_d, physical_k["qs"].contiguous().to(device), physical_k["signs"].contiguous().to(device),
            v_d, physical_v["qs"].contiguous().to(device), physical_v["signs"].contiguous().to(device),
            constants["pi_centroids_3bit"], constants["pi_cos"], constants["pi_sin"],
            block_tables_gpu, cu_q_gpu, cu_k_gpu, seq_lens_gpu, out, **common)
    elif fmt == "iso3_0":
        varlen_qkv_iso3_kernel[grid](
            q, k_d, physical_k["qs"].contiguous().to(device), physical_k["signs"].contiguous().to(device),
            v_d, physical_v["qs"].contiguous().to(device), physical_v["signs"].contiguous().to(device),
            constants["pi_centroids_3bit"], constants["pi_qw"], constants["pi_qx"], constants["pi_qy"], constants["pi_qz"],
            block_tables_gpu, cu_q_gpu, cu_k_gpu, seq_lens_gpu, out, **common)
    else:
        raise ValueError(fmt)

    torch.cuda.synchronize()
    k_ref = _materialize_ref(fmt, logical_k, constants_cpu)
    v_ref = _materialize_ref(fmt, logical_v, constants_cpu)
    ref = _reference(q_cpu, k_ref, v_ref, q_lens, k_lens, num_q_heads, num_kv_heads, scale, causal, sliding_window)
    diff = (out.cpu() - ref).abs()
    max_abs = float(diff.max().item())
    tol = 4.0e-3
    return {
        "format": fmt,
        "D": d_head,
        "q_lens": q_lens,
        "k_lens": k_lens,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "q_per_kv": num_q_heads // num_kv_heads,
        "block_size": block_size,
        "causal": causal,
        "sliding_window": sliding_window,
        "max_abs_err": max_abs,
        "tol": tol,
        "passed": max_abs <= tol,
    }


def run_varlen_qkv_checks(dims: tuple[int, ...] = (128, 256)) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("varlen QKV checks require a HIP/CUDA-visible torch device")
    q_lens = [3, 1, 4]
    k_lens = [9, 5, 11]
    cases = ((True, 0), (True, 4))
    checks = [
        _run_one(fmt, d_head, q_lens, k_lens, num_q_heads=4, num_kv_heads=2, block_size=4,
                 causal=causal, sliding_window=sliding_window)
        for d_head in dims
        for fmt in FORMATS
        for causal, sliding_window in cases
    ]
    return {"result": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[128, 256])
    args = parser.parse_args()
    report = run_varlen_qkv_checks(dims=tuple(args.dims))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
