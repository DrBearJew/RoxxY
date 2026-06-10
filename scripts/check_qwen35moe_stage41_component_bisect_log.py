#!/usr/bin/env python3
"""Validate and summarize Stage4.1 Qwen35MoE component-bisect logs.

Promotion mode requires exact pre-repair compares:
  token_match=1
  state_match=1

Diagnostic mode can be run with --allow-pre-repair-state-miss to keep the log
checker useful while bisection is still locating the first divergent component.
The checker also extracts hidden-row hashes emitted by LLAMA_MTP_PREFIX_HIDDEN_TRACE.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

COMPARE_RE = re.compile(r"\bMTP_VERIFY_COMPARE:.*")
POST_RE = re.compile(r"\bMTP_VERIFY_COMPARE_POST_REPAIR:.*")
FIELD_RE = re.compile(r"\b([A-Za-z_]+)=([^\s]+)")
NCOLS_RE = re.compile(r"\bncols_dst=([0-9]+)")
ROUTE_RE = re.compile(r"route=([A-Za-z0-9_]+)")
TPS_RE = re.compile(r"\bGeneration(?:\s+|:\s*)~?([0-9]+(?:\.[0-9]+)?)\s*t/s\b")
HIDDEN_RE = re.compile(
    r"MTP_PREFIX_HIDDEN_TRACE:.*?layer=(-?\d+).*?row=(-?\d+).*?node=([^\s]+).*?hash=([0-9a-fA-F]+)"
)

STAGE41_MARKERS = (
    "MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH(qwen35moe)",
    "stage41=1",
    "contract=serial_attention_component_bisect_ffn",
)

STAGE41_REASONS = (
    "exact_roweq_stage43_fused_router_topk_prefix_requested",
    "exact_roweq_stage42_router_topk_prefix_requested",
    "exact_roweq_stage41_component_bisect_prefix_requested",
    "exact_roweq_layer_ffn_prefix_requested",
)

SERIAL_COLUMN_ROUTES = {
    "mmvq_serial_columns",
    "mmvq_serial_columns_single_launch",
    "mmvq_serial_columns_fused_single_launch",
    "rdna3_mmvq_dot4_serial_columns",
    "router_mmvf_serial_columns",
    "router_topk_weights_roweq_fused_candidate",
}

SAFE_ISLAND_COMPONENT_FLAGS = (
    "serial_router=1",
    "serial_topk_weights=1",
    "serial_routed_glue=1",
    "serial_expert_agg=1",
    "serial_shared_gate=1",
    "serial_shared_ffn=1",
    "batch_routed_proj=1",
)

MINIMAL_ROUTER_TOPK_COMPONENT_FLAGS = (
    "serial_router=1",
    "serial_topk_weights=1",
    "serial_routed_glue=0",
    "serial_expert_agg=0",
    "serial_shared_gate=0",
    "serial_shared_ffn=0",
    "batch_routed_proj=1",
)

FULL_SERIAL_FALLBACK_COMPONENT_FLAGS = (
    "batch_routed_proj=0",
)

COMPONENT_PROFILES = {
    "safe-island": SAFE_ISLAND_COMPONENT_FLAGS,
    "minimal-router-topk": MINIMAL_ROUTER_TOPK_COMPONENT_FLAGS,
    "full-serial-fallback": FULL_SERIAL_FALLBACK_COMPONENT_FLAGS,
}


def parse_fields(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in FIELD_RE.finditer(line)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="+", type=Path, help="log file(s) to validate")
    parser.add_argument("--require-ncols", type=int, action="append", default=[],
                        help="Require a selected serial-column route with this ncols_dst; repeatable.")
    parser.add_argument("--allow-pre-repair-state-miss", action="store_true",
                        help="Diagnostic mode: do not fail on state_match=0 before repair.")
    parser.add_argument("--require-hidden-trace", action="store_true", default=True)
    parser.add_argument("--no-require-hidden-trace", dest="require_hidden_trace", action="store_false")
    parser.add_argument("--require-serial-columns", action="store_true", default=True)
    parser.add_argument("--no-require-serial-columns", dest="require_serial_columns", action="store_false")
    parser.add_argument("--require-component-flags", action="store_true", default=False,
                        help="Backward-compatible alias for --component-profile safe-island.")
    parser.add_argument("--no-require-component-flags", dest="require_component_flags", action="store_false",
                        help="Do not require the old all-serial safe-island flag set; use this for component-off bisection logs.")
    parser.add_argument("--component-profile", choices=["none", "safe-island", "minimal-router-topk", "full-serial-fallback"], default="none",
                        help="Require a known component-flag profile in the Stage4.1/4.2 graph marker.")
    parser.add_argument("--min-generation-tps", type=float, default=0.0,
                        help="Optional local speed gate. Fails if max parsed Generation t/s is lower.")
    ns = parser.parse_args()

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

    marker_seen = any(any(marker in ln for marker in STAGE41_MARKERS) for ln in lines)
    backend_seen = any(any(reason in ln for reason in STAGE41_REASONS) for ln in lines)
    component_flag_line = next((ln for ln in lines if "MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH(qwen35moe)" in ln), "")

    routes: dict[str, int] = {}
    selected_serial_column_lines: list[str] = []
    ncols: dict[int, int] = {}
    for ln in lines:
        selected = "status=selected" in ln
        for m in ROUTE_RE.finditer(ln):
            route = m.group(1)
            routes[route] = routes.get(route, 0) + 1
            if selected and route in SERIAL_COLUMN_ROUTES:
                selected_serial_column_lines.append(ln)
        for m in NCOLS_RE.finditer(ln):
            value = int(m.group(1))
            ncols[value] = ncols.get(value, 0) + 1

    hidden: dict[tuple[int, int], list[tuple[str, str]]] = defaultdict(list)
    for ln in lines:
        m = HIDDEN_RE.search(ln)
        if m:
            layer = int(m.group(1))
            row = int(m.group(2))
            node = m.group(3)
            hsh = m.group(4).lower()
            hidden[(layer, row)].append((node, hsh))

    tps_values: list[float] = []
    for ln in lines:
        m = TPS_RE.search(ln)
        if m:
            tps_values.append(float(m.group(1)))

    print(f"compare_lines={len(compare_lines)} bad_compare_lines={len(bad_compare)} post_repair_lines={len(post_lines)}")
    print(f"backend_seen={int(backend_seen)} marker_seen={int(marker_seen)}")
    if component_flag_line:
        print(f"component_marker={component_flag_line}")
    print(f"ncols_dst_seen={dict(sorted(ncols.items()))}")
    print(f"routes_seen={dict(sorted(routes.items()))}")
    print(f"hidden_hash_rows={len(hidden)}")
    for (layer, row), values in sorted(hidden.items())[:32]:
        tail = values[-1]
        print(f"hidden layer={layer} row={row} last_node={tail[0]} last_hash={tail[1]} samples={len(values)}")
    if tps_values:
        print(f"generation_tps_max={max(tps_values):.3f} generation_tps_values={tps_values}")

    failures: list[str] = []
    if not compare_lines:
        failures.append("no MTP_VERIFY_COMPARE lines found")
    if bad_compare and not ns.allow_pre_repair_state_miss:
        failures.append("one or more pre-repair compare lines are not exact")
    if not backend_seen:
        failures.append("missing Stage4.1 backend reason")
    if not marker_seen:
        failures.append("missing Stage4.1 graph marker")
    component_profile = "safe-island" if ns.require_component_flags and ns.component_profile == "none" else ns.component_profile
    if component_profile != "none":
        for flag in COMPONENT_PROFILES[component_profile]:
            if flag not in component_flag_line:
                failures.append(f"missing {component_profile} component flag in graph marker: {flag}")
    if ns.require_serial_columns:
        if not selected_serial_column_lines and not any(route in SERIAL_COLUMN_ROUTES for route in routes):
            failures.append("missing selected serial-column backend route")
        for value in ns.require_ncols:
            if not any(f"ncols_dst={value}" in ln for ln in selected_serial_column_lines):
                failures.append(f"missing selected serial-column route with ncols_dst={value}")
    if ns.require_hidden_trace and not hidden:
        failures.append("missing MTP_PREFIX_HIDDEN_TRACE hashes")
    if ns.min_generation_tps > 0.0 and (not tps_values or max(tps_values) < ns.min_generation_tps):
        got = max(tps_values) if tps_values else 0.0
        failures.append(f"generation_tps_max={got:.3f} < required {ns.min_generation_tps:.3f}")

    if failures:
        print("FAIL: Stage4.1 component-bisect log rejected", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        if bad_compare:
            print("first_bad_compare_lines:", file=sys.stderr)
            for ln in bad_compare[:20]:
                print(ln, file=sys.stderr)
            if len(bad_compare) > 20:
                print(f"... {len(bad_compare) - 20} more", file=sys.stderr)
        print("note: route checks require llama-cli/server high verbosity, e.g. -v, because GGML_LOG_INFO route lines may be hidden", file=sys.stderr)
        print("note: use --no-require-component-flags or --component-profile minimal-router-topk for intentional component-off bisection", file=sys.stderr)
        return 1

    print("PASS: Stage4.1 log accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
