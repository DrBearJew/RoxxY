#!/usr/bin/env python3
"""Fast no-model P5Y root verify-mask trace contract probe.

Default mode validates source plus a representative P5Y trace without loading a
model. Use --trace-log to validate a separately captured live target+draft log.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SOURCE = REPO_ROOT / "common/speculative.cpp"

REQUIRED_TRACE_TOKENS = [
    "draft-jetspec p5y_root_verify_mask_runtime",
    "phase=verify_mask_runtime_ready",
    "root_verify_mask_runtime_ready=1",
    "root_tree_runtime_ready=1",
    "actual_tree_nodes=1",
    "actual_verify_mask_entries=1",
    "verify_mask_rows=1",
    "verify_mask_cols=1",
    "root_attends_self=1",
    "root_mask_row=0",
    "root_mask_col=0",
    "prefix_visible=1",
    "ancestor_only=1",
    "sibling_visible=0",
    "descendant_visible=0",
    "no_draft_head_graph=1",
    "no_mask_tensor=1",
    "no_accept=1",
    "no_token_commit=1",
    "no_hidden_kv_commit=1",
    "no_rejected_branch_discard=1",
    "no_publish=1",
    "no_visible_state_change=1",
    "no_kv_mutation=1",
    "no_draft_tokens=1",
]

FORBIDDEN_TRACE_TOKENS = [
    "accept_path_descriptor_ready=1",
    "token_commit_descriptor_ready=1",
    "hidden_kv_survivor_commit_descriptor_ready=1",
    "rejected_branch_discard_descriptor_ready=1",
    "publish_gate_descriptor_ready=1",
    "actual_accepted_nodes=1",
    "actual_committed_tokens=1",
    "actual_publish_visible_state=1",
    "tree_accept",
    "draft token",
]

SELF_TEST_TRACE = (
    "common_speculative_impl_draft_jetspec::process: draft-jetspec p5y_root_verify_mask_runtime "
    "phase=verify_mask_runtime_ready root_verify_mask_runtime_ready=1 "
    "root_verify_mask_runtime_hash=0123456789abcdef root_tree_runtime_ready=1 "
    "root_tree_runtime_hash=fedcba9876543210 actual_tree_nodes=1 "
    "actual_verify_mask_entries=1 verify_mask_rows=1 verify_mask_cols=1 "
    "root_attends_self=1 root_mask_row=0 root_mask_col=0 prefix_visible=1 "
    "ancestor_only=1 sibling_visible=0 descendant_visible=0 no_draft_head_graph=1 "
    "no_mask_tensor=1 no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 "
    "no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 "
    "no_kv_mutation=1 no_draft_tokens=1"
)


class P5YTraceProbeError(RuntimeError):
    """Raised when a probe input cannot be inspected."""


def _source() -> str:
    if not SOURCE.exists():
        raise P5YTraceProbeError(f"missing source: {SOURCE}")
    return SOURCE.read_text(encoding="utf-8", errors="replace")


def _slice(text: str, start_token: str, end_token: str) -> str:
    start = text.find(start_token)
    end = text.find(end_token, start + len(start_token))
    if start < 0 or end < 0 or end <= start:
        raise P5YTraceProbeError(f"cannot isolate slice {start_token!r}..{end_token!r}")
    return text[start:end]


def _parse_scalar(line: str, key: str) -> int | None:
    match = re.search(rf"{re.escape(key)}=(-?\d+)", line)
    return None if match is None else int(match.group(1))


def validate_trace_line(line: str) -> dict[str, Any]:
    errors: list[str] = []
    missing = [token for token in REQUIRED_TRACE_TOKENS if token not in line]
    forbidden = [token for token in FORBIDDEN_TRACE_TOKENS if token in line]
    errors.extend(f"missing P5Y trace token: {token}" for token in missing)
    errors.extend(f"forbidden P5Y trace token present: {token}" for token in forbidden)

    actual_nodes = _parse_scalar(line, "actual_tree_nodes")
    entries = _parse_scalar(line, "actual_verify_mask_entries")
    rows = _parse_scalar(line, "verify_mask_rows")
    cols = _parse_scalar(line, "verify_mask_cols")
    root_self = _parse_scalar(line, "root_attends_self")
    root_row = _parse_scalar(line, "root_mask_row")
    root_col = _parse_scalar(line, "root_mask_col")
    if actual_nodes != 1:
        errors.append("P5Y trace actual_tree_nodes must be exactly 1")
    if entries != 1:
        errors.append("P5Y trace actual_verify_mask_entries must be exactly 1")
    if rows != 1 or cols != 1:
        errors.append("P5Y trace root verify mask must be 1x1")
    if root_self != 1:
        errors.append("P5Y trace root must attend to itself")
    if root_row != 0 or root_col != 0:
        errors.append("P5Y trace root mask coordinate must be (0,0)")

    return {
        "ok": not errors,
        "errors": errors,
        "missing_tokens": missing,
        "forbidden_hits": forbidden,
        "actual_tree_nodes": actual_nodes,
        "actual_verify_mask_entries": entries,
        "verify_mask_rows": rows,
        "verify_mask_cols": cols,
        "root_attends_self": root_self,
        "root_mask_row": root_row,
        "root_mask_col": root_col,
    }


def _source_contract() -> dict[str, Any]:
    text = _source()
    impl = _slice(text, "struct common_speculative_impl_draft_jetspec", "struct common_speculative_impl_draft_mtp")
    branch = _slice(
        impl,
        "if (p5y_root_verify_mask_enabled) {\n                if (!build_root_only_verify_mask_runtime())",
        "            if (trace_taps) {\n                LOG_INF(\"%s: draft-jetspec p5x_root_tree_runtime",
    )
    build = _slice(impl, "bool build_root_only_verify_mask_runtime()", "bool build_verify_mask_descriptor()")

    errors: list[str] = []
    required_source_tokens = [
        'common_speculative_env_enabled("LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY")',
        "p5y_root_verify_mask_enabled && !p5x_root_tree_enabled",
        "if (!p5x_root_tree_enabled || !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {",
        "if (tree_build_actual_nodes_last != 1) {",
        "actual_verify_mask_entries_last = 1",
        "root_verify_mask_values[0] = 1",
        "runtime_phase = jetspec_runtime_phase::verify_mask_runtime_ready",
        "return true;",
        "no_accept=1",
        "no_token_commit=1",
        "no_publish=1",
        "no_draft_tokens=1",
    ]
    haystacks = [impl, build, branch]
    for token in required_source_tokens:
        if not any(token in haystack for haystack in haystacks):
            errors.append(f"missing P5Y source trace/negative token: {token}")

    forbidden_after_root_mask = [
        "build_accept_path_descriptor()",
        "build_token_commit_descriptor()",
        "build_hidden_kv_survivor_commit_descriptor()",
        "build_rejected_branch_discard_descriptor()",
        "build_publish_gate_descriptor()",
    ]
    for token in forbidden_after_root_mask:
        if token in branch:
            errors.append(f"P5Y branch must return before {token}")

    return {
        "ok": not errors,
        "errors": errors,
        "runtime_executed": False,
        "model_loaded": False,
        "context_created": False,
        "draft_tokens_emitted": False,
    }


def _trace_from_log(path: pathlib.Path) -> str | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "draft-jetspec p5y_root_verify_mask_runtime" in line:
            return line
    return None


def probe_p5y_root_verify_mask_trace(trace_log: pathlib.Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    source_contract = _source_contract()
    if not source_contract.get("ok"):
        errors.extend(source_contract.get("errors") or [])

    self_test = validate_trace_line(SELF_TEST_TRACE)
    if not self_test.get("ok"):
        errors.extend(f"self-test trace invalid: {err}" for err in self_test.get("errors") or [])

    live_trace: dict[str, Any] | None = None
    live_runtime_executed = False
    if trace_log is not None:
        if not trace_log.exists():
            errors.append(f"missing trace log: {trace_log}")
        else:
            line = _trace_from_log(trace_log)
            if line is None:
                errors.append("trace log does not contain draft-jetspec p5y_root_verify_mask_runtime")
            else:
                live_runtime_executed = True
                live_trace = validate_trace_line(line)
                live_trace["line"] = line
                if not live_trace.get("ok"):
                    errors.extend(live_trace.get("errors") or [])

    return {
        "ok": not errors,
        "status": "p5y_root_verify_mask_trace_contract_verified" if not errors else "p5y_root_verify_mask_trace_contract_invalid",
        "errors": errors,
        "runtime_executed": live_runtime_executed,
        "model_loaded": live_runtime_executed,
        "context_created": live_runtime_executed,
        "draft_tokens_emitted": False,
        "source_contract": source_contract,
        "self_test_trace": self_test,
        "live_trace": live_trace,
        "limitations": [
            "default path is no-model and validates source plus trace parser contract only",
            "pass --trace-log to validate a separately captured live target+draft trace",
            "does not execute draft-head graph, accept, commit, KV mutation, publish, or draft tokens",
        ],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-log", type=pathlib.Path, help="optional captured live log to validate")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        out = probe_p5y_root_verify_mask_trace(args.trace_log)
    except P5YTraceProbeError as exc:
        out = {"ok": False, "status": "p5y_root_verify_mask_trace_probe_error", "errors": [str(exc)]}
    if args.json or not out.get("ok"):
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(f"P5Y trace contract probe passed status={out['status']} runtime_executed={str(out['runtime_executed']).lower()}")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
