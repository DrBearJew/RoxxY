#!/usr/bin/env python3
"""TBQ4 FWHT-domain parity checks.

TBQ4 K/V rows are stored in the signed-FWHT domain. Production launchers keep
that domain split explicit: rotate Q before attention, run attention against
FWHT-domain K/V, then inverse-rotate O when V is TBQ4.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

from llama_cpp_kv_layout import DOMAIN_FWHT, domain_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
TBQ4_HEADER = REPO_ROOT / "ggml/src/ggml-cuda/tbq4-cuda.cuh"
INV_SQRT_128 = 0.08838834764831845


def _parse_float_array(path: Path, name: str) -> torch.Tensor:
    text = path.read_text()
    match = re.search(rf"{re.escape(name)}\s*\[[^\]]+\]\s*=\s*\{{(.*?)\}};", text, re.S)
    if not match:
        raise ValueError(f"array {name!r} not found in {path}")
    vals = [float(x.rstrip("f")) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?f?", match.group(1))]
    if len(vals) != 128:
        raise ValueError(f"array {name!r} expected 128 values, got {len(vals)}")
    return torch.tensor(vals, dtype=torch.float32)


def load_tbq4_signs() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _parse_float_array(TBQ4_HEADER, "d_tbq4_wht_s1"),
        _parse_float_array(TBQ4_HEADER, "d_tbq4_wht_s2"),
    )


def fwht_128(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != 128:
        raise ValueError(f"fwht_128 expects last dim 128, got {x.shape[-1]}")
    out = x.float().clone().reshape(-1, 128)
    h = 1
    while h < 128:
        grouped = out.reshape(-1, 128 // (2 * h), 2 * h)
        a = grouped[:, :, :h].clone()
        b = grouped[:, :, h:2 * h].clone()
        grouped[:, :, :h] = a + b
        grouped[:, :, h:2 * h] = a - b
        h *= 2
    return (out * INV_SQRT_128).reshape_as(x)


def rotate_forward(x: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 128 != 0:
        raise ValueError("TBQ4 rotate_forward requires D divisible by 128")
    y = x.float().clone()
    flat = y.reshape(-1, y.shape[-1])
    for block in range(y.shape[-1] // 128):
        sl = slice(block * 128, (block + 1) * 128)
        flat[:, sl] = fwht_128(flat[:, sl] * s1) * s2
    return y


def rotate_inverse(x: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 128 != 0:
        raise ValueError("TBQ4 rotate_inverse requires D divisible by 128")
    y = x.float().clone()
    flat = y.reshape(-1, y.shape[-1])
    for block in range(y.shape[-1] // 128):
        sl = slice(block * 128, (block + 1) * 128)
        flat[:, sl] = fwht_128(flat[:, sl] * s2) * s1
    return y


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
    scores = q @ k.T / math.sqrt(q.shape[-1])
    if causal:
        q_rows = q.shape[0]
        k_rows = k.shape[0]
        q_pos = torch.arange(q_rows, dtype=torch.int64) + max(k_rows - q_rows, 0)
        k_pos = torch.arange(k_rows, dtype=torch.int64)
        mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        scores = scores.masked_fill(~mask, -torch.inf)
    probs = torch.softmax(scores, dim=-1)
    return probs @ v, scores


def _run_case(d_head: int, q_rows: int, k_rows: int, causal: bool, seed: int) -> dict[str, object]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    s1, s2 = load_tbq4_signs()

    q = torch.randn((q_rows, d_head), generator=gen, dtype=torch.float32)
    k = torch.randn((k_rows, d_head), generator=gen, dtype=torch.float32)
    v = torch.randn((k_rows, d_head), generator=gen, dtype=torch.float32)

    q_rot = rotate_forward(q, s1, s2)
    k_rot = rotate_forward(k, s1, s2)
    v_rot = rotate_forward(v, s1, s2)

    q_roundtrip = rotate_inverse(q_rot, s1, s2)
    out_ref, scores_ref = attention(q, k, v, causal=causal)
    out_rot_domain, scores_rot = attention(q_rot, k_rot, v_rot, causal=causal)
    out_contract = rotate_inverse(out_rot_domain, s1, s2)

    finite_scores = torch.isfinite(scores_ref) & torch.isfinite(scores_rot)
    finite_mask_mismatch = bool((torch.isfinite(scores_ref) != torch.isfinite(scores_rot)).any().item())
    scores_err = float((scores_ref[finite_scores] - scores_rot[finite_scores]).abs().max().item()) if finite_scores.any() else 0.0
    out_err = float((out_ref - out_contract).abs().max().item())
    roundtrip_err = float((q - q_roundtrip).abs().max().item())
    tol = 2.5e-4

    return {
        "D": d_head,
        "q_rows": q_rows,
        "k_rows": k_rows,
        "causal": causal,
        "scores_max_abs_err": scores_err,
        "finite_mask_mismatch": finite_mask_mismatch,
        "output_max_abs_err": out_err,
        "roundtrip_max_abs_err": roundtrip_err,
        "tol": tol,
        "passed": (not finite_mask_mismatch) and scores_err <= tol and out_err <= tol and roundtrip_err <= tol,
    }


def run() -> dict[str, object]:
    policy = domain_policy("tbq4_0")
    checks = [
        _run_case(d_head=d_head, q_rows=q_rows, k_rows=k_rows, causal=causal, seed=9100 + d_head + q_rows * 17 + k_rows)
        for d_head in (128, 256)
        for q_rows, k_rows in ((1, 7), (5, 13), (9, 37))
        for causal in (False, True)
    ]
    policy_ok = (
        policy["domain"] == DOMAIN_FWHT
        and policy["rotate_q_before_attention"]
        and policy["rotate_o_after_attention"]
    )
    return {
        "result": "PASS" if policy_ok and all(c["passed"] for c in checks) else "FAIL",
        "format": "tbq4_0",
        "domain_policy": policy,
        "quantization_scope": "not tested here; this test verifies FWHT-domain attention algebra and launcher rotation policy",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = run()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
