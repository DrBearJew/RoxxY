#!/usr/bin/env python3
"""Correctness-only comparison between 2D tiled and segmented 3D prototypes."""

from __future__ import annotations

import json

from qkv_2d_tiled import run_qkv_2d_checks
from segmented_qkv import run_segmented_qkv_checks


def _max_err(report: dict[str, object]) -> float:
    return max(float(c["max_abs_err"]) for c in report["checks"])  # type: ignore[index]


def main() -> int:
    qkv_2d = run_qkv_2d_checks(q_rows=8, k_rows=17, dims=(128,))
    segmented = run_segmented_qkv_checks(q_rows=5, k_rows=37, dims=(128,))
    report = {
        "comparison": "correctness_only_no_timing",
        "qkv_2d_result": qkv_2d["result"],
        "qkv_2d_max_abs_err": _max_err(qkv_2d),
        "segmented_3d_result": segmented["result"],
        "segmented_3d_max_abs_err": _max_err(segmented),
        "interpretation": [
            "2D tiled path is the simpler prefill-like shape once paged materialization is correct.",
            "Segmented 3D path preserves the same online-softmax math while splitting longer KV ranges into independently reducible segments.",
            "No profiling claim is made by this script.",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if qkv_2d["result"] == "PASS" and segmented["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
