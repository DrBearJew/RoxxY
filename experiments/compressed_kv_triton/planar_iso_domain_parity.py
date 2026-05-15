#!/usr/bin/env python3
"""Planar/Iso original-domain parity checks.

Planar3 and Iso3 materializers decode original-domain values. Unlike TBQ4, they
must not acquire hidden Q pre-rotation or O inverse-rotation in the generic
compressed-KV attention path.
"""

from __future__ import annotations

import argparse
import json

import torch

from llama_cpp_kv_layout import DOMAIN_ORIGINAL, domain_policy
from materializers import (
    load_constants,
    materialize_iso3_ref,
    materialize_planar3_ref,
    synthetic_inputs,
)
from tbq4_domain_parity import attention, load_tbq4_signs, rotate_forward, rotate_inverse


FORMATS = ("planar3_0", "iso3_0")


def _materialize(fmt: str, inputs: dict[str, torch.Tensor], constants: dict[str, torch.Tensor]) -> torch.Tensor:
    if fmt == "planar3_0":
        return materialize_planar3_ref(inputs["d"], inputs["qs"], inputs["signs"], constants).float()
    if fmt == "iso3_0":
        return materialize_iso3_ref(inputs["d"], inputs["qs"], inputs["signs"], constants).float()
    raise ValueError(fmt)


def _run_case(fmt: str, d_head: int, q_rows: int, k_rows: int, causal: bool, seed: int) -> dict[str, object]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    constants = load_constants("cpu")
    s1, s2 = load_tbq4_signs()

    k_inputs = synthetic_inputs(fmt, k_rows, d_head, seed=seed + 101)
    v_inputs = synthetic_inputs(fmt, k_rows, d_head, seed=seed + 202)
    k = _materialize(fmt, k_inputs, constants)
    v = _materialize(fmt, v_inputs, constants)
    q = torch.randn((q_rows, d_head), generator=gen, dtype=torch.float32)

    out_ref, _ = attention(q, k, v, causal=causal)

    policy = domain_policy(fmt)
    q_contract = q
    out_contract_domain, _ = attention(q_contract, k, v, causal=causal)
    out_contract = out_contract_domain

    # Deliberately model the class of bug this test guards against: applying the
    # TBQ4 FWHT-domain Q/O transforms to original-domain Planar/Iso KV.
    q_wrong = rotate_forward(q, s1, s2)
    out_wrong_domain, _ = attention(q_wrong, k, v, causal=causal)
    out_wrong = rotate_inverse(out_wrong_domain, s1, s2)

    no_rotation_err = float((out_ref - out_contract).abs().max().item())
    wrong_rotation_err = float((out_ref - out_wrong).abs().max().item())
    return {
        "format": fmt,
        "D": d_head,
        "q_rows": q_rows,
        "k_rows": k_rows,
        "causal": causal,
        "domain_policy": policy,
        "no_rotation_max_abs_err": no_rotation_err,
        "wrong_tbq4_rotation_max_abs_err": wrong_rotation_err,
        "tol": 0.0,
        "wrong_rotation_signal_min": 1.0e-4,
        "passed": (
            policy["domain"] == DOMAIN_ORIGINAL
            and not policy["rotate_q_before_attention"]
            and not policy["rotate_o_after_attention"]
            and no_rotation_err == 0.0
            and wrong_rotation_err > 1.0e-4
        ),
    }


def run() -> dict[str, object]:
    checks = [
        _run_case(fmt=fmt, d_head=d_head, q_rows=q_rows, k_rows=k_rows, causal=causal, seed=7200 + len(fmt) + d_head + q_rows * 31 + k_rows)
        for fmt in FORMATS
        for d_head in (128, 256)
        for q_rows, k_rows in ((1, 7), (5, 13), (9, 37))
        for causal in (False, True)
    ]
    return {
        "result": "PASS" if all(c["passed"] for c in checks) else "FAIL",
        "contract": "Planar/Iso compressed KV materializes original-domain rows and must not use TBQ4 Q/O FWHT transforms",
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
