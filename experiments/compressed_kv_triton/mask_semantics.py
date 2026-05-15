#!/usr/bin/env python3
"""Causal/sliding-window mask checks matching the IBM unified-attention shape."""

from __future__ import annotations

import argparse
import json

import torch
import triton
import triton.language as tl


@triton.jit
def mask_kernel(out, Q_ROWS: tl.constexpr, K_ROWS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                CONTEXT_LEN: tl.constexpr, CAUSAL: tl.constexpr, SLIDING_WINDOW: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    q = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = (q[:, None] < Q_ROWS) & (k[None, :] < K_ROWS)
    q_abs = CONTEXT_LEN + q
    if CAUSAL:
        valid = valid & (k[None, :] <= q_abs[:, None])
    if SLIDING_WINDOW > 0:
        valid = valid & (k[None, :] <= q_abs[:, None]) & ((q_abs[:, None] - k[None, :]) < SLIDING_WINDOW)
    tl.store(out + q[:, None] * K_ROWS + k[None, :], valid.to(tl.int32), mask=(q[:, None] < Q_ROWS) & (k[None, :] < K_ROWS))


def _ref(q_rows: int, k_rows: int, context_len: int, causal: bool, sliding_window: int) -> torch.Tensor:
    out = torch.zeros((q_rows, k_rows), dtype=torch.int32)
    for q in range(q_rows):
        q_abs = context_len + q
        for k in range(k_rows):
            valid = True
            if causal:
                valid = valid and k <= q_abs
            if sliding_window > 0:
                valid = valid and k <= q_abs and (q_abs - k) < sliding_window
            out[q, k] = 1 if valid else 0
    return out


def run_mask_semantics_checks(q_rows: int = 5, k_rows: int = 13, block_m: int = 8, block_n: int = 8) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("mask checks require a HIP/CUDA-visible torch device")
    checks: list[dict[str, object]] = []
    context_len = k_rows - q_rows
    for causal, sliding_window in ((False, 0), (True, 0), (True, 1), (True, 4), (True, 32)):
        out = torch.empty((q_rows, k_rows), dtype=torch.int32, device="cuda")
        grid = (triton.cdiv(q_rows, block_m), triton.cdiv(k_rows, block_n))
        mask_kernel[grid](out, Q_ROWS=q_rows, K_ROWS=k_rows, BLOCK_M=block_m, BLOCK_N=block_n,
                          CONTEXT_LEN=context_len, CAUSAL=causal, SLIDING_WINDOW=sliding_window)
        torch.cuda.synchronize()
        ref = _ref(q_rows, k_rows, context_len, causal, sliding_window)
        mismatches = int((out.cpu() != ref).sum().item())
        checks.append({
            "q_rows": q_rows,
            "k_rows": k_rows,
            "context_len": context_len,
            "causal": causal,
            "sliding_window": sliding_window,
            "valid_count": int(out.cpu().sum().item()),
            "mismatches": mismatches,
            "passed": mismatches == 0,
        })
    return {"result": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-rows", type=int, default=5)
    parser.add_argument("--k-rows", type=int, default=13)
    args = parser.parse_args()
    report = run_mask_semantics_checks(q_rows=args.q_rows, k_rows=args.k_rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
