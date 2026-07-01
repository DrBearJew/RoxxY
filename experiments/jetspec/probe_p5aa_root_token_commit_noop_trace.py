#!/usr/bin/env python3
"""Fast no-model P5AA root-token-commit no-op trace contract probe.

Default mode validates source plus a representative P5AA trace without loading a
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
    "draft-jetspec p5aa_root_token_commit_noop_runtime",
    "phase=token_commit_runtime_ready",
    "root_token_commit_noop_runtime_ready=1",
    "root_anchor_accept_path_runtime_ready=1",
    "root_verify_mask_runtime_ready=1",
    "root_tree_runtime_ready=1",
    "root_verified_anchor=1",
    "accept_path_len=0",
    "actual_tree_nodes=1",
    "actual_verify_mask_entries=1",
    "actual_accepted_nodes=0",
    "correction_token_present=0",
    "actual_committed_tokens=0",
    "no_real_token_commit=1",
    "no_visible_token_publish=1",
    "no_hidden_kv_commit=1",
    "no_rejected_branch_discard=1",
    "no_publish=1",
    "no_visible_state_change=1",
    "no_kv_mutation=1",
    "no_draft_head_graph=1",
    "no_draft_tokens=1",
]

FORBIDDEN_TRACE_TOKENS = [
    "token_commit_descriptor_ready=1",
    "hidden_kv_survivor_commit_descriptor_ready=1",
    "rejected_branch_discard_descriptor_ready=1",
    "publish_gate_descriptor_ready=1",
    "actual_committed_tokens=1",
    "actual_publish_visible_state=1",
    "draft token",
]

SELF_TEST_TRACE = (
    "common_speculative_impl_draft_jetspec::process: draft-jetspec p5aa_root_token_commit_noop_runtime "
    "phase=token_commit_runtime_ready root_token_commit_noop_runtime_ready=1 "
    "root_token_commit_noop_runtime_hash=aaaaaaaa55555555 root_anchor_accept_path_runtime_ready=1 "
    "root_anchor_accept_path_runtime_hash=0123456789abcdef root_verify_mask_runtime_ready=1 "
    "root_verify_mask_runtime_hash=fedcba9876543210 root_tree_runtime_ready=1 "
    "root_tree_runtime_hash=1111222233334444 root_verified_anchor=1 accept_path_len=0 "
    "actual_tree_nodes=1 actual_verify_mask_entries=1 actual_accepted_nodes=0 "
    "correction_token_present=0 actual_committed_tokens=0 no_real_token_commit=1 "
    "no_visible_token_publish=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 "
    "no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_head_graph=1 "
    "no_draft_tokens=1"
)


class P5AATraceProbeError(RuntimeError):
    """Raised when a probe input cannot be inspected."""


def _source() -> str:
    if not SOURCE.exists():
        raise P5AATraceProbeError(f"missing source: {SOURCE}")
    return SOURCE.read_text(encoding="utf-8", errors="replace")


def _slice(text: str, start_token: str, end_token: str) -> str:
    start = text.find(start_token)
    end = text.find(end_token, start + len(start_token))
    if start < 0 or end < 0 or end <= start:
        raise P5AATraceProbeError(f"cannot isolate slice {start_token!r}..{end_token!r}")
    return text[start:end]


def _parse_scalar(line: str, key: str) -> int | None:
    match = re.search(rf"{re.escape(key)}=(-?\d+)", line)
    return None if match is None else int(match.group(1))


def validate_trace_line(line: str) -> dict[str, Any]:
    errors: list[str] = []
    missing = [token for token in REQUIRED_TRACE_TOKENS if token not in line]
    forbidden = [token for token in FORBIDDEN_TRACE_TOKENS if token in line]
    errors.extend(f"missing P5AA trace token: {token}" for token in missing)
    errors.extend(f"forbidden P5AA trace token present: {token}" for token in forbidden)

    root_anchor = _parse_scalar(line, "root_verified_anchor")
    path_len = _parse_scalar(line, "accept_path_len")
    actual_nodes = _parse_scalar(line, "actual_tree_nodes")
    mask_entries = _parse_scalar(line, "actual_verify_mask_entries")
    accepted = _parse_scalar(line, "actual_accepted_nodes")
    correction = _parse_scalar(line, "correction_token_present")
    committed = _parse_scalar(line, "actual_committed_tokens")
    if root_anchor != 1:
        errors.append("P5AA trace root_verified_anchor must be 1")
    if path_len != 0:
        errors.append("P5AA trace accept_path_len must be 0")
    if actual_nodes != 1:
        errors.append("P5AA trace actual_tree_nodes must be exactly 1")
    if mask_entries != 1:
        errors.append("P5AA trace actual_verify_mask_entries must be exactly 1")
    if accepted != 0:
        errors.append("P5AA trace actual_accepted_nodes must stay 0")
    if correction != 0:
        errors.append("P5AA trace correction_token_present must stay 0")
    if committed != 0:
        errors.append("P5AA trace actual_committed_tokens must stay 0")

    return {
        "ok": not errors,
        "errors": errors,
        "missing_tokens": missing,
        "forbidden_hits": forbidden,
        "root_verified_anchor": root_anchor,
        "accept_path_len": path_len,
        "actual_tree_nodes": actual_nodes,
        "actual_verify_mask_entries": mask_entries,
        "actual_accepted_nodes": accepted,
        "correction_token_present": correction,
        "actual_committed_tokens": committed,
    }


def _source_contract() -> dict[str, Any]:
    text = _source()
    impl = _slice(text, "struct common_speculative_impl_draft_jetspec", "struct common_speculative_impl_draft_mtp")
    branch = _slice(
        impl,
        "if (p5aa_root_token_commit_noop_enabled) {\n                        if (!build_root_token_commit_noop_runtime())",
        "                    if (trace_taps) {\n                        LOG_INF(\"%s: draft-jetspec p5z_root_anchor_accept_path_runtime",
    )
    build = _slice(impl, "bool build_root_token_commit_noop_runtime()", "bool build_accept_path_descriptor()")

    errors: list[str] = []
    required_source_tokens = [
        'common_speculative_env_enabled("LLAMA_JETSPEC_ROOT_TOKEN_COMMIT_NOOP_ONLY")',
        "p5aa_root_token_commit_noop_enabled && (!p5x_root_tree_enabled || !p5y_root_verify_mask_enabled || !p5z_root_anchor_accept_path_enabled)",
        "if (!p5z_root_anchor_accept_path_enabled || !root_anchor_accept_path_runtime_ready || root_anchor_accept_path_runtime_hash_last == 0) {",
        "root_verified_anchor_last != 1 || accept_path_len_last != 0 || actual_accepted_nodes_last != 0 || correction_token_present_last != 0",
        "actual_committed_tokens_last = 0",
        "runtime_phase = jetspec_runtime_phase::token_commit_runtime_ready",
        "return true;",
        "no_real_token_commit=1",
        "no_visible_token_publish=1",
        "no_draft_tokens=1",
    ]
    haystacks = [impl, build, branch]
    for token in required_source_tokens:
        if not any(token in haystack for haystack in haystacks):
            errors.append(f"missing P5AA source trace/negative token: {token}")

    forbidden_after_commit = [
        "build_token_commit_descriptor()",
        "build_hidden_kv_survivor_commit_descriptor()",
        "build_rejected_branch_discard_descriptor()",
        "build_publish_gate_descriptor()",
        "token_commit_descriptor_ready = true",
        "hidden_kv_survivor_commit_descriptor_ready = true",
        "rejected_branch_discard_descriptor_ready = true",
        "publish_gate_descriptor_ready = true",
    ]
    for token in forbidden_after_commit:
        if token in branch or token in build:
            errors.append(f"P5AA branch/builder must return before {token}")

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
        if "draft-jetspec p5aa_root_token_commit_noop_runtime" in line:
            return line
    return None


def probe_p5aa_root_token_commit_noop_trace(trace_log: pathlib.Path | None = None) -> dict[str, Any]:
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
                errors.append("trace log does not contain draft-jetspec p5aa_root_token_commit_noop_runtime")
            else:
                live_runtime_executed = True
                live_trace = validate_trace_line(line)
                live_trace["line"] = line
                if not live_trace.get("ok"):
                    errors.extend(live_trace.get("errors") or [])

    return {
        "ok": not errors,
        "status": "p5aa_root_token_commit_noop_trace_contract_verified" if not errors else "p5aa_root_token_commit_noop_trace_contract_invalid",
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
            "does not execute draft-head graph, real token commit, KV mutation, publish, or draft tokens",
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
        out = probe_p5aa_root_token_commit_noop_trace(args.trace_log)
    except P5AATraceProbeError as exc:
        out = {"ok": False, "status": "p5aa_root_token_commit_noop_trace_probe_error", "errors": [str(exc)]}
    if args.json or not out.get("ok"):
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(f"P5AA trace contract probe passed status={out['status']} runtime_executed={str(out['runtime_executed']).lower()}")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
