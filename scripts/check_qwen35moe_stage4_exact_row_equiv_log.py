#!/usr/bin/env python3
"""Validate Stage4 Qwen35MoE exact row-equivalent verifier logs.

Stage4 is promotion-worthy only when all pre-repair verifier compares are exact:
  token_match=1
  state_match=1

The checker also confirms that the row-equivalent prefix graph and serial-column
backend routes were actually used.  It accepts multiple route names because the
target may choose the generic one-launch serial-column kernel, the fused
one-launch kernel, or the RDNA3 Q4_K dot4 serial-column kernel.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

COMPARE_RE = re.compile(r"\bMTP_VERIFY_COMPARE:.*")
POST_RE = re.compile(r"\bMTP_VERIFY_COMPARE_POST_REPAIR:.*")
FIELD_RE = re.compile(r"\b([A-Za-z_]+)=([^\s]+)")
NCOLS_RE = re.compile(r"\bncols_dst=([0-9]+)")
TPS_RE = re.compile(r"\bGeneration(?:\s+|:\s*)~?([0-9]+(?:\.[0-9]+)?)\s*t/s\b")
ROUTE_RE = re.compile(r"route=([A-Za-z0-9_]+)")

STAGE4_MARKERS = (
    "MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH(qwen35moe)",
    "MTP_PREFIX_EXACT_ROW_EQUIV_BATCH(qwen35moe)",
    "MTP_PREFIX_EXACT_ROWEQ_BATCH(qwen35moe)",
)

STAGE4_REASONS = (
    "exact_roweq_layer_ffn_prefix_requested",
    "exact_token_major_prefix_graph_row_equiv_batch_requested",
    "exact_token_major_prefix_graph_roweq_batch_requested",
)

SERIAL_COLUMN_ROUTES = {
    "mmvq_serial_columns",
    "mmvq_serial_columns_single_launch",
    "mmvq_serial_columns_fused_single_launch",
    "rdna3_mmvq_dot4_serial_columns",
}

REQUIRED_TENSOR_GROUPS = {
    "router": ("ffn_gate_inp",),
    "expert_gate_up": ("ffn_gate_up", "ffn_moe_gate_up"),
    "expert_down": ("ffn_down", "ffn_moe_down"),
}


def parse_fields(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in FIELD_RE.finditer(line)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="+", type=Path, help="log file(s) to validate")
    parser.add_argument("--require-ncols", type=int, action="append", default=None,
                        help="Require a selected serial-column route with this ncols_dst value; repeatable. Defaults to 4.")
    parser.add_argument("--require-graph-marker", action="store_true", default=True)
    parser.add_argument("--no-require-graph-marker", dest="require_graph_marker", action="store_false")
    parser.add_argument("--require-backend", action="store_true", default=True)
    parser.add_argument("--no-require-backend", dest="require_backend", action="store_false")
    parser.add_argument("--require-serial-columns", action="store_true", default=True)
    parser.add_argument("--no-require-serial-columns", dest="require_serial_columns", action="store_false")
    parser.add_argument("--require-tensor-groups", action="store_true", default=False,
                        help="Also require router, expert gate/up, and expert down route lines.")
    parser.add_argument("--allow-pre-repair-state-miss", action="store_true",
                        help="Diagnostics only: do not fail on state_match=0 before repair.")
    parser.add_argument("--min-generation-tps", type=float, default=0.0,
                        help="Optional local speed gate. Fails if max parsed Generation t/s is lower.")
    ns = parser.parse_args()
    if ns.require_ncols is None:
        ns.require_ncols = [4]

    lines: list[str] = []
    for path in ns.log:
        if not path.exists():
            print(f"ERROR: log not found: {path}", file=sys.stderr)
            return 2
        lines.extend(path.read_text(errors="replace").splitlines())

    compare_lines = [ln for ln in lines if COMPARE_RE.search(ln) and not POST_RE.search(ln)]
    post_lines = [ln for ln in lines if POST_RE.search(ln)]
    bad_compare: list[str] = []
    for ln in compare_lines:
        fields = parse_fields(ln)
        if fields.get("token_match") != "1" or fields.get("state_match") != "1":
            bad_compare.append(ln)

    backend_seen = any(any(reason in ln for reason in STAGE4_REASONS) for ln in lines)
    graph_marker_seen = any(any(marker in ln for marker in STAGE4_MARKERS) for ln in lines)

    routes: dict[str, int] = {}
    ncols: dict[int, int] = {}
    selected_serial_column_lines: list[str] = []
    for ln in lines:
        selected = "status=selected" in ln
        route_matches = list(ROUTE_RE.finditer(ln))
        for m in route_matches:
            route = m.group(1)
            routes[route] = routes.get(route, 0) + 1
            if selected and route in SERIAL_COLUMN_ROUTES:
                selected_serial_column_lines.append(ln)
        for m in NCOLS_RE.finditer(ln):
            value = int(m.group(1))
            ncols[value] = ncols.get(value, 0) + 1

    serial_column_seen = bool(selected_serial_column_lines) or any(route in SERIAL_COLUMN_ROUTES for route in routes)

    tps_values: list[float] = []
    for ln in lines:
        m = TPS_RE.search(ln)
        if m:
            tps_values.append(float(m.group(1)))

    print(f"compare_lines={len(compare_lines)} bad_compare_lines={len(bad_compare)} post_repair_lines={len(post_lines)}")
    print(f"backend_seen={int(backend_seen)} graph_marker_seen={int(graph_marker_seen)} serial_column_seen={int(serial_column_seen)}")
    print(f"ncols_dst_seen={dict(sorted(ncols.items()))}")
    print(f"routes_seen={dict(sorted(routes.items()))}")
    if tps_values:
        print(f"generation_tps_max={max(tps_values):.3f} generation_tps_values={tps_values}")

    failures: list[str] = []
    if not compare_lines:
        failures.append("no MTP_VERIFY_COMPARE lines found")
    if bad_compare and not ns.allow_pre_repair_state_miss:
        failures.append("one or more pre-repair compare lines are not exact")
    if ns.require_backend and not backend_seen:
        failures.append("missing Stage4 backend reason")
    if ns.require_graph_marker and not graph_marker_seen:
        failures.append("missing Stage4 graph marker")
    if ns.require_serial_columns and not serial_column_seen:
        failures.append("missing selected Stage4 serial-column backend route")
    if ns.require_serial_columns:
        for value in ns.require_ncols:
            if not any(f"ncols_dst={value}" in ln for ln in selected_serial_column_lines):
                failures.append(f"missing selected serial-column route with ncols_dst={value}")
    if ns.require_tensor_groups:
        for group, needles in REQUIRED_TENSOR_GROUPS.items():
            if not any(any(needle in ln for needle in needles) for ln in selected_serial_column_lines):
                failures.append(f"missing selected serial-column tensor group: {group}")
    if ns.min_generation_tps > 0.0 and (not tps_values or max(tps_values) < ns.min_generation_tps):
        got = max(tps_values) if tps_values else 0.0
        failures.append(f"generation_tps_max={got:.3f} < required {ns.min_generation_tps:.3f}")

    if failures:
        print("FAIL: Stage4 exact row-equivalent verifier log rejected", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        if bad_compare:
            print("first_bad_compare_lines:", file=sys.stderr)
            for ln in bad_compare[:20]:
                print(ln, file=sys.stderr)
            if len(bad_compare) > 20:
                print(f"... {len(bad_compare) - 20} more", file=sys.stderr)
        return 1

    print("PASS: token_match=1 and state_match=1 on every pre-repair compare line")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
