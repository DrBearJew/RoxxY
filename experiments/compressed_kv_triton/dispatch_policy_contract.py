#!/usr/bin/env python3
"""Compressed-KV FlashAttention dispatch policy contract.

This freezes the experimental-route boundary used by the Triton lane:
compressed KV defaults to VEC, WMMA is env-gated, and mixed TBQ4_0/Q8_0 stays
on VEC until a Q8 V loader/domain policy exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llama_cpp_kv_layout import DOMAIN_FWHT, DOMAIN_ORIGINAL, domain_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
FATTN_CU = REPO_ROOT / "ggml/src/ggml-cuda/fattn.cu"
INVARIANT_SH = REPO_ROOT / "scripts/hip/check-compressed-kv-fa-invariants.sh"


def modeled_dispatch(
    *,
    k_type: str,
    v_type: str,
    q_d: int,
    q_cols: int,
    tbq4_wmma_env: bool,
    compressed_wmma_env: bool,
    amd_wmma_available: bool = True,
) -> str:
    if q_d not in (128, 256):
        return "none"

    if k_type == "tbq4_0":
        if v_type not in {"tbq4_0", "q8_0"}:
            return "none"
        if v_type == "tbq4_0" and q_cols > 2 and tbq4_wmma_env and amd_wmma_available:
            return "wmma_tbq4"
        if v_type == "q8_0":
            return "vec"
        return "vec"

    if k_type in {"planar3_0", "iso3_0"}:
        if v_type != k_type:
            return "vec"
        if q_cols > 2 and compressed_wmma_env and amd_wmma_available:
            return "wmma_compressed_kv"
        return "vec"

    return "none"


def _source_checks() -> list[dict[str, object]]:
    fattn = FATTN_CU.read_text()
    inv = INVARIANT_SH.read_text()
    checks = [
        ("tbq4_env_gate", 'getenv("TBQ4_WMMA_FATTN")' in fattn),
        ("compressed_env_gate", 'getenv("COMPRESSED_KV_WMMA_FATTN")' in fattn),
        ("mixed_tbq4_q8_allowed", "K->type == GGML_TYPE_TBQ4_0 && V->type == GGML_TYPE_Q8_0" in fattn),
        ("mixed_tbq4_q8_vec_return", "if (v_is_q8_0)" in fattn and "BEST_FATTN_KERNEL_VEC" in fattn),
        ("compressed_wmma_same_type_only", "K->type == V->type" in fattn and "BEST_FATTN_KERNEL_WMMA_COMPRESSED_KV" in fattn),
        ("invariant_checker_guards_no_triton_cmake", "no Triton integration in production CMake" in inv),
    ]
    return [{"name": name, "passed": passed} for name, passed in checks]


def _policy_cases() -> list[dict[str, object]]:
    cases = [
        {
            "name": "tbq4_default_vec",
            "args": dict(k_type="tbq4_0", v_type="tbq4_0", q_d=128, q_cols=8, tbq4_wmma_env=False, compressed_wmma_env=False),
            "expected": "vec",
        },
        {
            "name": "tbq4_opt_in_wmma",
            "args": dict(k_type="tbq4_0", v_type="tbq4_0", q_d=128, q_cols=8, tbq4_wmma_env=True, compressed_wmma_env=False),
            "expected": "wmma_tbq4",
        },
        {
            "name": "tbq4_decode_like_small_batch_stays_vec",
            "args": dict(k_type="tbq4_0", v_type="tbq4_0", q_d=128, q_cols=2, tbq4_wmma_env=True, compressed_wmma_env=False),
            "expected": "vec",
        },
        {
            "name": "mixed_tbq4_q8_stays_vec_even_with_env",
            "args": dict(k_type="tbq4_0", v_type="q8_0", q_d=128, q_cols=8, tbq4_wmma_env=True, compressed_wmma_env=True),
            "expected": "vec",
        },
        {
            "name": "planar_default_vec",
            "args": dict(k_type="planar3_0", v_type="planar3_0", q_d=256, q_cols=8, tbq4_wmma_env=False, compressed_wmma_env=False),
            "expected": "vec",
        },
        {
            "name": "planar_opt_in_wmma",
            "args": dict(k_type="planar3_0", v_type="planar3_0", q_d=256, q_cols=8, tbq4_wmma_env=False, compressed_wmma_env=True),
            "expected": "wmma_compressed_kv",
        },
        {
            "name": "iso_default_vec",
            "args": dict(k_type="iso3_0", v_type="iso3_0", q_d=128, q_cols=8, tbq4_wmma_env=False, compressed_wmma_env=False),
            "expected": "vec",
        },
        {
            "name": "planar_iso_mixed_not_wmma",
            "args": dict(k_type="planar3_0", v_type="iso3_0", q_d=128, q_cols=8, tbq4_wmma_env=False, compressed_wmma_env=True),
            "expected": "vec",
        },
        {
            "name": "unsupported_dim_none",
            "args": dict(k_type="tbq4_0", v_type="tbq4_0", q_d=192, q_cols=8, tbq4_wmma_env=True, compressed_wmma_env=False),
            "expected": "none",
        },
    ]

    out: list[dict[str, object]] = []
    for case in cases:
        got = modeled_dispatch(**case["args"])
        out.append({
            "name": case["name"],
            "args": case["args"],
            "expected": case["expected"],
            "got": got,
            "passed": got == case["expected"],
        })
    return out


def _domain_checks() -> list[dict[str, object]]:
    expected = {
        "tbq4_0": DOMAIN_FWHT,
        "planar3_0": DOMAIN_ORIGINAL,
        "iso3_0": DOMAIN_ORIGINAL,
    }
    checks: list[dict[str, object]] = []
    for fmt, domain in expected.items():
        policy = domain_policy(fmt)
        checks.append({
            "format": fmt,
            "expected_domain": domain,
            "policy": policy,
            "passed": policy["domain"] == domain,
        })
    return checks


def run() -> dict[str, object]:
    source_checks = _source_checks()
    policy_cases = _policy_cases()
    domain_checks = _domain_checks()
    passed = all(c["passed"] for c in source_checks + policy_cases + domain_checks)
    return {
        "result": "PASS" if passed else "FAIL",
        "contract": "default VEC; WMMA env-gated; mixed TBQ4_0/Q8_0 remains VEC; Triton stays experimental",
        "source_checks": source_checks,
        "policy_cases": policy_cases,
        "domain_checks": domain_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = run()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
