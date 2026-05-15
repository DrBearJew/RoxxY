#!/usr/bin/env python3
"""Triton online-softmax tile loop prototype for FA-shaped attention."""

from __future__ import annotations

import argparse
import json

import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(logits, probs, row_meta, K_ROWS: tl.constexpr, BLOCK_K: tl.constexpr, CAUSAL: tl.constexpr, Q_POS_START: tl.constexpr):
    q_row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    m = -float("inf")
    l = 0.0

    for k0 in range(0, K_ROWS, BLOCK_K):
        k = k0 + offs
        valid = k < K_ROWS
        if CAUSAL:
            valid = valid & (k <= (Q_POS_START + q_row))
        x = tl.load(logits + q_row * K_ROWS + k, mask=valid, other=-float("inf"))
        block_m = tl.max(x, axis=0)
        m_new = tl.maximum(m, block_m)
        alpha = tl.exp(m - m_new)
        p = tl.exp(x - m_new)
        l = l * alpha + tl.sum(p, axis=0)
        m = m_new

    for k0 in range(0, K_ROWS, BLOCK_K):
        k = k0 + offs
        valid = k < K_ROWS
        if CAUSAL:
            valid = valid & (k <= (Q_POS_START + q_row))
        x = tl.load(logits + q_row * K_ROWS + k, mask=valid, other=-float("inf"))
        p = tl.exp(x - m) / l
        tl.store(probs + q_row * K_ROWS + k, p, mask=k < K_ROWS)

    tl.store(row_meta + q_row * 2 + 0, m)
    tl.store(row_meta + q_row * 2 + 1, l)


def run_online_softmax_checks(q_rows: int = 8, k_rows: int = 37, block_k: int = 16) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("online softmax checks require a HIP/CUDA-visible torch device")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(4000 + q_rows + k_rows + block_k)
    logits_cpu = torch.randn((q_rows, k_rows), generator=gen, dtype=torch.float32) * 3.0
    checks: list[dict[str, object]] = []
    for causal in (False, True):
        logits = logits_cpu.to("cuda")
        probs = torch.empty_like(logits)
        meta = torch.empty((q_rows, 2), dtype=torch.float32, device="cuda")
        q_pos_start = max(0, k_rows - q_rows)
        online_softmax_kernel[(q_rows,)](logits, probs, meta, K_ROWS=k_rows, BLOCK_K=block_k, CAUSAL=causal, Q_POS_START=q_pos_start)
        torch.cuda.synchronize()

        masked = logits_cpu.clone()
        if causal:
            for q in range(q_rows):
                masked[q, q_pos_start + q + 1 :] = -float("inf")
        ref = torch.softmax(masked, dim=-1)
        if causal:
            # Triton stores zeros for invalid positions; make the reference explicit.
            ref = torch.nan_to_num(ref, nan=0.0)
        diff = (probs.cpu() - ref).abs()
        row_sum_err = (probs.cpu().sum(dim=-1) - torch.ones(q_rows)).abs().max().item()
        max_abs = float(diff.max().item())
        tol = 1.5e-6
        checks.append({
            "causal": causal,
            "q_rows": q_rows,
            "k_rows": k_rows,
            "block_k": block_k,
            "max_abs_err": max_abs,
            "row_sum_err": float(row_sum_err),
            "tol": tol,
            "passed": max_abs <= tol and row_sum_err <= tol * 4,
        })
    return {"result": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-rows", type=int, default=8)
    parser.add_argument("--k-rows", type=int, default=37)
    parser.add_argument("--block-k", type=int, default=16)
    args = parser.parse_args()
    report = run_online_softmax_checks(q_rows=args.q_rows, k_rows=args.k_rows, block_k=args.block_k)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
