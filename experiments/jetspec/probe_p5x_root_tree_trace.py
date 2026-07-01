#!/usr/bin/env python3
"""Fast no-model P5X root-tree trace contract probe.

This probe validates the trace contract for the gated P5X one-node runtime tree
without loading a target model or draft-head model. It can also validate a real
captured log via --trace-log when a live target+draft smoke is explicitly allowed.
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
    "draft-jetspec p5x_root_tree_runtime",
    "phase=tree_build_runtime_ready",
    "root_tree_runtime_ready=1",
    "actual_tree_nodes=1",
    "tree_parent_indices=[-1]",
    "tree_depth=[0]",
    "tree_rank=[-1]",
    "tree_cum_logprob=[0.0]",
    "root_parent=-1",
    "root_depth=0",
    "parent_before_child=1",
    "num_nodes_lte_budget=1",
    "no_draft_head_graph=1",
    "no_verify_mask=1",
    "no_accept=1",
    "no_token_commit=1",
    "no_hidden_kv_commit=1",
    "no_rejected_branch_discard=1",
    "no_publish=1",
    "no_visible_state_change=1",
    "no_draft_tokens=1",
]

FORBIDDEN_TRACE_TOKENS = [
    "verify_mask_descriptor_ready=1",
    "accept_path_descriptor_ready=1",
    "token_commit_descriptor_ready=1",
    "hidden_kv_survivor_commit_descriptor_ready=1",
    "rejected_branch_discard_descriptor_ready=1",
    "publish_gate_descriptor_ready=1",
    "actual_verify_mask_entries=1",
    "actual_accepted_nodes=1",
    "actual_committed_tokens=1",
    "actual_publish_visible_state=1",
    "draft token",
    "tree_accept",
]

SELF_TEST_TRACE = (
    "common_speculative_impl_draft_jetspec::process: draft-jetspec p5x_root_tree_runtime "
    "phase=tree_build_runtime_ready root_tree_runtime_ready=1 "
    "root_tree_runtime_hash=0123456789abcdef actual_tree_nodes=1 "
    "tree_token_ids=[42] tree_parent_indices=[-1] tree_depth=[0] tree_rank=[-1] "
    "tree_cum_logprob=[0.0] root_parent=-1 root_depth=0 parent_before_child=1 "
    "num_nodes_lte_budget=1 tree_build_node_budget=16 no_draft_head_graph=1 "
    "no_verify_mask=1 no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 "
    "no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_draft_tokens=1"
)


class P5XTraceProbeError(RuntimeError):
    """Raised when a probe input cannot be inspected."""


def _source() -> str:
    if not SOURCE.exists():
        raise P5XTraceProbeError(f"missing source: {SOURCE}")
    return SOURCE.read_text(encoding="utf-8", errors="replace")


def _slice(text: str, start_token: str, end_token: str) -> str:
    start = text.find(start_token)
    end = text.find(end_token, start + len(start_token))
    if start < 0 or end < 0 or end <= start:
        raise P5XTraceProbeError(f"cannot isolate slice {start_token!r}..{end_token!r}")
    return text[start:end]


def _parse_bracket_int(line: str, key: str) -> int | None:
    match = re.search(rf"{re.escape(key)}=\[(-?\d+)\]", line)
    return None if match is None else int(match.group(1))


def validate_trace_line(line: str) -> dict[str, Any]:
    errors: list[str] = []
    missing = [token for token in REQUIRED_TRACE_TOKENS if token not in line]
    forbidden = [token for token in FORBIDDEN_TRACE_TOKENS if token in line]
    errors.extend(f"missing P5X trace token: {token}" for token in missing)
    errors.extend(f"forbidden P5X trace token present: {token}" for token in forbidden)

    token_id = _parse_bracket_int(line, "tree_token_ids")
    parent = _parse_bracket_int(line, "tree_parent_indices")
    depth = _parse_bracket_int(line, "tree_depth")
    rank = _parse_bracket_int(line, "tree_rank")
    budget_match = re.search(r"tree_build_node_budget=(\d+)", line)
    actual_match = re.search(r"actual_tree_nodes=(\d+)", line)
    if token_id is None or token_id < 0:
        errors.append("P5X trace must expose one non-negative prompt-tail root token")
    if parent != -1:
        errors.append("P5X trace root parent must be -1")
    if depth != 0:
        errors.append("P5X trace root depth must be 0")
    if rank != -1:
        errors.append("P5X trace root rank must be -1")
    if actual_match is None or int(actual_match.group(1)) != 1:
        errors.append("P5X trace actual_tree_nodes must be exactly 1")
    if budget_match is None or int(budget_match.group(1)) < 1:
        errors.append("P5X trace tree_build_node_budget must be positive")

    return {
        "ok": not errors,
        "errors": errors,
        "missing_tokens": missing,
        "forbidden_hits": forbidden,
        "root_token": token_id,
        "root_parent": parent,
        "root_depth": depth,
        "root_rank": rank,
        "actual_tree_nodes": None if actual_match is None else int(actual_match.group(1)),
        "tree_build_node_budget": None if budget_match is None else int(budget_match.group(1)),
    }


def _source_contract() -> dict[str, Any]:
    text = _source()
    impl = _slice(text, "struct common_speculative_impl_draft_jetspec", "struct common_speculative_impl_draft_mtp")
    branch = _slice(
        impl,
        "if (p5x_root_tree_enabled) {\n            if (!build_root_only_runtime_tree())",
        "        if (!build_verify_mask_descriptor())",
    )
    build = _slice(impl, "bool build_root_only_runtime_tree()", "bool build_verify_mask_descriptor()")
    snapshot = _slice(impl, "bool build_pre_round_snapshot", "bool build_transaction_plan_scaffold")

    errors: list[str] = []
    required_source_tokens = [
        'common_speculative_env_enabled("LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY")',
        "p5x_root_tree_enabled && prompt.empty()",
        "pre_round_root_token_last = prompt.empty() ? -1 : prompt.back()",
        "if (!p5x_root_tree_enabled) {",
        "if (!tree_build_descriptor_ready || tree_build_descriptor_hash_last == 0) {",
        "if (pre_round_root_token_last < 0) {",
        "tree_build_actual_nodes_last = 1",
        "tree_build_actual_nodes_last > tree_build_node_budget_last",
        "runtime_phase = jetspec_runtime_phase::tree_build_runtime_ready",
        "return true;",
        "no_verify_mask=1",
        "no_accept=1",
        "no_publish=1",
        "no_draft_tokens=1",
    ]
    haystacks = [impl, build, branch, snapshot]
    for token in required_source_tokens:
        if not any(token in haystack for haystack in haystacks):
            errors.append(f"missing P5X source trace/negative token: {token}")

    forbidden_after_root = [
        "build_verify_mask_descriptor()",
        "build_accept_path_descriptor()",
        "build_token_commit_descriptor()",
        "build_hidden_kv_survivor_commit_descriptor()",
        "build_rejected_branch_discard_descriptor()",
        "build_publish_gate_descriptor()",
    ]
    for token in forbidden_after_root:
        if token in branch:
            errors.append(f"P5X branch must return before {token}")

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
        if "draft-jetspec p5x_root_tree_runtime" in line:
            return line
    return None


def probe_p5x_root_tree_trace(trace_log: pathlib.Path | None = None) -> dict[str, Any]:
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
                errors.append("trace log does not contain draft-jetspec p5x_root_tree_runtime")
            else:
                live_runtime_executed = True
                live_trace = validate_trace_line(line)
                live_trace["line"] = line
                if not live_trace.get("ok"):
                    errors.extend(live_trace.get("errors") or [])

    return {
        "ok": not errors,
        "status": "p5x_root_tree_trace_contract_verified" if not errors else "p5x_root_tree_trace_contract_invalid",
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
            "does not execute draft-head graph, verify, accept, commit, KV mutation, publish, or draft tokens",
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
        out = probe_p5x_root_tree_trace(args.trace_log)
    except P5XTraceProbeError as exc:
        out = {"ok": False, "status": "p5x_root_tree_trace_probe_error", "errors": [str(exc)]}
    if args.json or not out.get("ok"):
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(f"P5X trace contract probe passed status={out['status']} runtime_executed={str(out['runtime_executed']).lower()}")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
