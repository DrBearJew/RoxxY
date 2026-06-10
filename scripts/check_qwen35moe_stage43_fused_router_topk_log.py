#!/usr/bin/env python3
"""Validate Stage4.3 Qwen35MoE fused router/top-k/weights candidate logs.

This wrapper keeps the Stage4.1 compare/route parser as the single source of
truth, then adds Stage4.3-specific requirements:
  - the Stage4.3 backend reason;
  - graph marker field stage43_router_topk_fused=1;
  - router/top-k mode identifying the fused candidate;
  - router dense MMVF route visibility;
  - explicit fused router/top-k/weights op route visibility.

Promotion mode is strict: token_match=1 and state_match=1 are required before
repair unless --allow-pre-repair-state-miss is explicitly passed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STAGE43_REASON = "exact_roweq_stage43_fused_router_topk_prefix_requested"
STAGE43_GRAPH_FIELD = "stage43_router_topk_fused=1"
STAGE43_MODE = "router_mmvf_roweq_fused_topk_weights_candidate"
STAGE43_ROUTE = "route=router_topk_weights_roweq_fused_candidate"
ROUTER_MMVF_ROUTE = "route=router_mmvf_serial_columns"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="+", type=Path)
    parser.add_argument("--require-ncols", action="append", type=int, default=[])
    parser.add_argument("--allow-pre-repair-state-miss", action="store_true")
    parser.add_argument("--no-require-hidden-trace", action="store_true")
    parser.add_argument("--no-require-router-mmvf", action="store_true")
    parser.add_argument("--no-require-fused-route", action="store_true")
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
    if STAGE43_REASON not in text:
        failures.append(f"missing Stage4.3 backend reason {STAGE43_REASON}")
    if STAGE43_GRAPH_FIELD not in text:
        failures.append(f"missing graph marker field {STAGE43_GRAPH_FIELD}")
    if STAGE43_MODE not in text:
        failures.append(f"missing router_topk_mode={STAGE43_MODE}")
    if "serial_router=0" not in text:
        failures.append("missing graph marker field serial_router=0")
    if "serial_topk_weights=0" not in text:
        failures.append("missing graph marker field serial_topk_weights=0")
    if "batch_routed_proj=1" not in text:
        failures.append("missing graph marker field batch_routed_proj=1")
    if not ns.no_require_router_mmvf and ROUTER_MMVF_ROUTE not in text:
        failures.append("missing selected router_mmvf_serial_columns route")
    if not ns.no_require_fused_route and STAGE43_ROUTE not in text:
        failures.append("missing selected router_topk_weights_roweq_fused_candidate route")

    if failures:
        print("FAIL: Stage4.3 fused router/top-k log rejected", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        print("note: use llama-cli/server -v so GGML_LOG_INFO route lines are visible", file=sys.stderr)
        print("note: use --no-require-fused-route only for graph-discovery diagnostics, not promotion", file=sys.stderr)
        return 1

    print("PASS: Stage4.3 fused router/top-k log accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
