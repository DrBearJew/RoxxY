#!/usr/bin/env python3
"""Fast no-model P5AV real draft-head top-k target-logits walk canary trace probe."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
CANDIDATE = HERE / "jetspec_p5av_real_draft_head_topk_target_logits_walk_canary_runtime_candidate.md"

REQUIRED_TRACE_TOKENS = [
    "draft-jetspec p5av_real_draft_head_topk_target_logits_walk_canary_runtime",
    "phase=real_draft_head_topk_target_logits_walk_canary_ready",
    "target_logits_walk_canary_ready=1",
    "p5av_target_logits_walk_canary_env=1",
    "real_topk_publish_gate_noop_runtime_ready=1",
    "planned_target_logits_rows=1",
    "actual_target_logits_rows_walked=1",
    "target_logits_source=target_model_full_vocab_logits",
    "target_logits_width=248320",
    "planned_parent_nodes=[0]",
    "planned_candidate_nodes=[1,2]",
    "row_semantics=parent_position_scores_candidate_children",
    "accept_decision_source=target_logits_canary_only_no_accept",
    "actual_target_accept_steps=0",
    "actual_accepted_nodes=0",
    "correction_token_present=0",
    "actual_committed_tokens=0",
    "actual_survivor_pages_committed=0",
    "actual_pages_discarded=0",
    "rejected_branch_pages_reachable_after_discard=0",
    "actual_publish_visible_state=0",
    "target_logits_walk_canary_only=1",
    "no_target_accept_walk=1",
    "no_accept=1",
    "no_real_token_commit=1",
    "no_visible_token_publish=1",
    "no_real_hidden_kv_commit=1",
    "no_hidden_kv_commit=1",
    "no_real_rejected_branch_discard=1",
    "no_rejected_branch_discard=1",
    "no_real_publish=1",
    "no_publish=1",
    "no_visible_state_change=1",
    "no_kv_mutation=1",
    "no_draft_tokens=1",
]

FORBIDDEN_TRACE_TOKENS = [
    "no_target_logits_walk=1",
    "target_logits_walk_canary_ready=0",
    "actual_target_logits_rows_walked=0",
    "actual_target_accept_steps=1",
    "actual_accepted_nodes=1",
    "correction_token_present=1",
    "actual_committed_tokens=1",
    "actual_survivor_pages_committed=1",
    "actual_pages_discarded=1",
    "actual_publish_visible_state=1",
    "token_commit_descriptor_ready=1",
    "hidden_kv_survivor_commit_descriptor_ready=1",
    "rejected_branch_discard_descriptor_ready=1",
    "publish_gate_descriptor_ready=1",
    "root_publish_gate_noop_runtime_ready=1",
    "#gen drafts = 1",
    "#gen tokens = 1",
]

ZERO_KEYS = [
    "actual_target_accept_steps",
    "actual_accepted_nodes",
    "correction_token_present",
    "actual_committed_tokens",
    "actual_survivor_pages_committed",
    "actual_pages_discarded",
    "rejected_branch_pages_reachable_after_discard",
    "actual_publish_visible_state",
]

SELF_TEST_TRACE = (
    "common_speculative_impl_draft_jetspec::process: draft-jetspec "
    "p5av_real_draft_head_topk_target_logits_walk_canary_runtime "
    "phase=real_draft_head_topk_target_logits_walk_canary_ready "
    "target_logits_walk_canary_ready=1 target_logits_walk_canary_hash=0123456789abcdef "
    "target_logits_walk_canary_seq_id=0 target_logits_walk_canary_builds=1 "
    "p5av_target_logits_walk_canary_env=1 real_topk_publish_gate_noop_runtime_ready=1 "
    "real_topk_publish_gate_noop_runtime_hash=abcdef0123456789 planned_target_logits_rows=1 "
    "actual_target_logits_rows_walked=1 target_logits_source=target_model_full_vocab_logits "
    "target_logits_width=248320 target_logits_batch_index=0 target_logits_pos=12 target_logits_seq_id=0 "
    "planned_parent_nodes=[0] planned_candidate_nodes=[1,2] candidate_ids=[92637,2054] "
    "target_candidate_logits=[12.5,11.25] row_semantics=parent_position_scores_candidate_children "
    "accept_decision_source=target_logits_canary_only_no_accept actual_target_accept_steps=0 "
    "actual_accepted_nodes=0 correction_token_present=0 actual_committed_tokens=0 "
    "actual_survivor_pages_committed=0 actual_pages_discarded=0 "
    "rejected_branch_pages_reachable_after_discard=0 actual_publish_visible_state=0 "
    "target_logits_walk_canary_only=1 no_target_accept_walk=1 no_accept=1 no_real_token_commit=1 "
    "no_visible_token_publish=1 no_real_hidden_kv_commit=1 no_hidden_kv_commit=1 "
    "no_real_rejected_branch_discard=1 no_rejected_branch_discard=1 no_real_publish=1 "
    "no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1"
)


def _extract_values(line: str, key: str) -> list[str]:
    return re.findall(rf"(?<![A-Za-z0-9_#]){re.escape(key)}=([^\s]+)", line)


def _parse_int_list(line: str, key: str) -> list[int] | None:
    values = _extract_values(line, key)
    if len(values) != 1:
        return None
    value = values[0]
    if not (value.startswith("[") and value.endswith("]")):
        return None
    body = value[1:-1]
    if not body:
        return []
    out: list[int] = []
    for item in body.split(","):
        try:
            out.append(int(item))
        except ValueError:
            return None
    return out


def _parse_float_list(line: str, key: str) -> list[float] | None:
    values = _extract_values(line, key)
    if len(values) != 1:
        return None
    value = values[0]
    if not (value.startswith("[") and value.endswith("]")):
        return None
    body = value[1:-1]
    if not body:
        return []
    out: list[float] = []
    for item in body.split(","):
        try:
            out.append(float(item))
        except ValueError:
            return None
    return out


def validate_trace_line(line: str) -> list[str]:
    errors: list[str] = []
    missing = [token for token in REQUIRED_TRACE_TOKENS if token not in line]
    forbidden = [token for token in FORBIDDEN_TRACE_TOKENS if token in line]
    errors.extend(f"missing P5AV trace token: {token}" for token in missing)
    errors.extend(f"forbidden P5AV trace token present: {token}" for token in forbidden)

    for key in ZERO_KEYS:
        values = _extract_values(line, key)
        if values != ["0"]:
            errors.append(f"P5AV trace {key} must appear exactly once with value 0, got {values}")

    if _extract_values(line, "actual_target_logits_rows_walked") != ["1"]:
        errors.append("P5AV trace actual_target_logits_rows_walked must be 1")
    if _extract_values(line, "planned_target_logits_rows") != ["1"]:
        errors.append("P5AV trace planned_target_logits_rows must be 1")

    candidate_ids = _parse_int_list(line, "candidate_ids")
    if candidate_ids is None or len(candidate_ids) != 2 or len(set(candidate_ids)) != 2 or any(v < 0 for v in candidate_ids):
        errors.append("P5AV trace candidate_ids must contain two distinct non-negative ids")
    scores = _parse_float_list(line, "target_candidate_logits")
    if scores is None or len(scores) != 2:
        errors.append("P5AV trace target_candidate_logits must contain two scores")
    if _parse_int_list(line, "planned_parent_nodes") != [0]:
        errors.append("P5AV trace planned_parent_nodes must be [0]")
    if _parse_int_list(line, "planned_candidate_nodes") != [1, 2]:
        errors.append("P5AV trace planned_candidate_nodes must be [1,2]")
    return errors


def _candidate_errors() -> list[str]:
    text = CANDIDATE.read_text(encoding="utf-8", errors="replace")
    required = [
        "P5AV real draft-head top-k target-logits walk canary",
        "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TARGET_LOGITS_WALK_CANARY=1",
        "p5av_real_draft_head_topk_target_logits_walk_canary_runtime",
        "target_logits_walk_canary_ready=1",
        "actual_target_logits_rows_walked=1",
        "target_logits_walk_canary_only=1",
        "no target accept walk",
        "no KV mutation",
        "no draft tokens",
    ]
    return [f"{CANDIDATE.name} missing P5AV contract token: {token}" for token in required if token not in text]


def _trace_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if "draft-jetspec p5av_real_draft_head_topk_target_logits_walk_canary_runtime" in line]


def probe_p5av_real_draft_head_topk_target_logits_walk_canary_trace(trace_log: pathlib.Path | None = None) -> dict[str, Any]:
    errors = _candidate_errors()
    if trace_log is None:
        trace_lines = [SELF_TEST_TRACE]
    else:
        trace_lines = _trace_lines(trace_log.read_text(encoding="utf-8", errors="replace"))
        if not trace_lines:
            errors.append("trace log does not contain draft-jetspec p5av_real_draft_head_topk_target_logits_walk_canary_runtime")
    for line in trace_lines:
        errors.extend(validate_trace_line(line))
    return {
        "ok": not errors,
        "status": "p5av_target_logits_walk_canary_trace_contract_verified" if not errors else "p5av_target_logits_walk_canary_trace_contract_invalid",
        "errors": errors,
        "trace_lines_checked": len(trace_lines),
        "runtime_executed": trace_log is not None,
        "accept_executed": False,
        "commit_executed": False,
        "publish_executed": False,
        "kv_mutated": False,
        "draft_tokens_emitted": False,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_log", nargs="?", type=pathlib.Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = probe_p5av_real_draft_head_topk_target_logits_walk_canary_trace(args.trace_log)
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
