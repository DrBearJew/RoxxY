#!/usr/bin/env python3
"""Aggregate JSON report mode for the compressed-KV Triton lane.

This runs the same gate set as run_all.sh and emits one machine-readable report.
It remains correctness-only: durations are execution metadata, not performance
claims.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    "scripts/hip/check-triton-feasibility.py",
    "experiments/compressed_kv_triton/compat_gate.py",
    "experiments/compressed_kv_triton/llama_cpp_tensor_layout_parity.py",
    "experiments/compressed_kv_triton/llama_cpp_block_table_parity.py",
    "experiments/compressed_kv_triton/dispatch_policy_contract.py",
    "experiments/compressed_kv_triton/paged_row_mapping_contract.py",
    "experiments/compressed_kv_triton/materializers.py",
    "experiments/compressed_kv_triton/paged_materializers.py",
    "experiments/compressed_kv_triton/qk_only.py",
    "experiments/compressed_kv_triton/qk_2d_tiled.py",
    "experiments/compressed_kv_triton/online_softmax.py",
    "experiments/compressed_kv_triton/full_qkv.py",
    "experiments/compressed_kv_triton/qkv_2d_tiled.py",
    "experiments/compressed_kv_triton/varlen_qkv.py",
    "experiments/compressed_kv_triton/mask_semantics.py",
    "experiments/compressed_kv_triton/segmented_qkv.py",
    "experiments/compressed_kv_triton/compare_2d_segmented.py",
    "experiments/compressed_kv_triton/tbq4_domain_parity.py",
    "experiments/compressed_kv_triton/planar_iso_domain_parity.py",
    "experiments/compressed_kv_triton/autotune_metadata.py",
]


def _parse_json(stdout: str) -> Any:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _collect_max_abs_err(obj: Any) -> list[float]:
    vals: list[float] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.endswith("max_abs_err") and isinstance(value, (int, float)):
                vals.append(float(value))
            vals.extend(_collect_max_abs_err(value))
    elif isinstance(obj, list):
        for value in obj:
            vals.extend(_collect_max_abs_err(value))
    return vals


def _run_script(py: str, script: str) -> dict[str, object]:
    started = time.monotonic()
    proc = subprocess.run(
        [py, script],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
        check=False,
    )
    duration = time.monotonic() - started
    parsed: Any = None
    parse_error: str | None = None
    try:
        parsed = _parse_json(proc.stdout)
    except Exception as exc:  # report parse failure without hiding command failure
        parse_error = f"{type(exc).__name__}: {exc}"

    max_abs_values = _collect_max_abs_err(parsed)
    result = parsed.get("result") if isinstance(parsed, dict) else None
    return {
        "script": script,
        "returncode": proc.returncode,
        "duration_sec": round(duration, 3),
        "result": result,
        "max_abs_err_max": max(max_abs_values) if max_abs_values else None,
        "json_parse_error": parse_error,
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
        "parsed": parsed,
        "passed": proc.returncode == 0 and parse_error is None and isinstance(parsed, dict) and result == "PASS",
    }


def run(py: str) -> dict[str, object]:
    checks = [_run_script(py, script) for script in SCRIPTS]
    return {
        "result": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "python": py,
        "repo": str(REPO_ROOT),
        "comparison": "correctness_only_no_timing_claims",
        "script_count": len(checks),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used to run each gate script")
    args = parser.parse_args()
    report = run(args.python)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
