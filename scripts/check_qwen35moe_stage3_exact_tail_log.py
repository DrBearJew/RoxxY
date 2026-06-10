#!/usr/bin/env python3
"""Validate Stage3 Qwen35MoE exact-tail verifier logs.

The hard gate is every pre-repair MTP_VERIFY_COMPARE line reporting:
  token_match=1
  state_match=1

The script also reports whether it saw the Stage3 backend reason, graph marker,
and ncols_dst values in backend route logs.
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


def parse_fields(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in FIELD_RE.finditer(line)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="+", type=Path)
    ap.add_argument("--require-tail-backend", action="store_true", default=True)
    ap.add_argument("--no-require-tail-backend", dest="require_tail_backend", action="store_false")
    ap.add_argument("--require-graph-marker", action="store_true", default=False)
    ap.add_argument("--require-ncols", type=int, default=0,
                    help="Require at least one backend route log with this ncols_dst value; 0 disables.")
    ns = ap.parse_args()

    lines: list[str] = []
    for path in ns.log:
        if not path.exists():
            print(f"ERROR: log not found: {path}", file=sys.stderr)
            return 2
        lines.extend(path.read_text(errors="replace").splitlines())

    compare_lines = [ln for ln in lines if COMPARE_RE.search(ln) and not POST_RE.search(ln)]
    post_lines = [ln for ln in lines if POST_RE.search(ln)]
    bad: list[str] = []

    for ln in compare_lines:
        fields = parse_fields(ln)
        if fields.get("token_match") != "1" or fields.get("state_match") != "1":
            bad.append(ln)

    tail_backend_seen = any("exact_token_major_prefix_graph_tail_batch_requested" in ln for ln in lines)
    graph_marker_seen = any("MTP_PREFIX_EXACT_TAIL_BATCH(qwen35moe)" in ln for ln in lines)
    ncols: dict[int, int] = {}
    for ln in lines:
        for m in NCOLS_RE.finditer(ln):
            value = int(m.group(1))
            ncols[value] = ncols.get(value, 0) + 1

    print(f"compare_lines={len(compare_lines)} bad_compare_lines={len(bad)} post_repair_lines={len(post_lines)}")
    print(f"tail_backend_seen={int(tail_backend_seen)} graph_marker_seen={int(graph_marker_seen)} ncols_dst_seen={dict(sorted(ncols.items()))}")

    if len(compare_lines) == 0:
        print("ERROR: no MTP_VERIFY_COMPARE lines found", file=sys.stderr)
        return 1
    if bad:
        print("ERROR: pre-repair verifier compare failed:", file=sys.stderr)
        for ln in bad[:20]:
            print(ln, file=sys.stderr)
        if len(bad) > 20:
            print(f"... {len(bad) - 20} more", file=sys.stderr)
        return 1
    if ns.require_tail_backend and not tail_backend_seen:
        print("ERROR: Stage3 tail backend reason not seen", file=sys.stderr)
        return 1
    if ns.require_graph_marker and not graph_marker_seen:
        print("ERROR: Stage3 graph marker not seen", file=sys.stderr)
        return 1
    if ns.require_ncols and ncols.get(ns.require_ncols, 0) == 0:
        print(f"ERROR: no ncols_dst={ns.require_ncols} route log seen", file=sys.stderr)
        return 1

    print("PASS: token_match=1 and state_match=1 on every pre-repair compare line")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
