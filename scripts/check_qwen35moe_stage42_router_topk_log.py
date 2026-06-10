#!/usr/bin/env python3
"""Stage4.2 wrapper around the Stage4.1/4.2 Qwen35MoE verifier log checker.

By default this checks the Stage4.2 router/top-k candidate shape:
  - exact pre-repair compares unless --allow-pre-repair-state-miss is passed;
  - a Stage4.2 marker/reason;
  - router_mmvf_serial_columns route visibility;
  - serial-column routed projection route visibility;
  - target ncols if requested.

The implementation delegates generic compare/route parsing to
check_qwen35moe_stage41_component_bisect_log.py so bisection output stays in one
place.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="+", type=Path)
    parser.add_argument("--require-ncols", action="append", type=int, default=[])
    parser.add_argument("--allow-pre-repair-state-miss", action="store_true")
    parser.add_argument("--no-require-hidden-trace", action="store_true")
    parser.add_argument("--min-generation-tps", type=float, default=0.0)
    ns = parser.parse_args()

    checker = Path(__file__).with_name("check_qwen35moe_stage41_component_bisect_log.py")
    cmd = [sys.executable, str(checker)]
    cmd.extend(str(path) for path in ns.log)
    cmd.extend(["--component-profile", "none"])
    if ns.allow_pre_repair_state_miss:
        cmd.append("--allow-pre-repair-state-miss")
    if ns.no_require_hidden_trace:
        cmd.append("--no-require-hidden-trace")
    for value in ns.require_ncols:
        cmd.extend(["--require-ncols", str(value)])
    if ns.min_generation_tps > 0.0:
        cmd.extend(["--min-generation-tps", f"{ns.min_generation_tps:.6f}"])

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        return result.returncode

    text = "\n".join(path.read_text(errors="replace") for path in ns.log)
    failures: list[str] = []
    if "exact_roweq_stage42_router_topk_prefix_requested" not in text:
        failures.append("missing Stage4.2 backend reason exact_roweq_stage42_router_topk_prefix_requested")
    if "stage42_router_topk=1" not in text:
        failures.append("missing graph marker field stage42_router_topk=1")
    if "router_topk_split=1" not in text:
        failures.append("missing graph marker field router_topk_split=1")
    if "route=router_mmvf_serial_columns" not in text:
        failures.append("missing selected router_mmvf_serial_columns route")

    if failures:
        print("FAIL: Stage4.2 router/top-k log rejected", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        print("note: use llama-cli/server -v so GGML_LOG_INFO route lines are visible", file=sys.stderr)
        return 1

    print("PASS: Stage4.2 router/top-k log accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
