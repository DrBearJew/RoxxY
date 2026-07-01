#!/usr/bin/env python3
"""Evaluate the inert P5AW JetSpec target-accept walk readiness packet."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

STATUS = "p5aw_target_accept_walk_readiness_verified_not_executed"
P5AV_TRACE_STATUS = "p5av_target_logits_walk_canary_trace_contract_verified"
P5AV_WIRING_STATUS = "p5av_real_draft_head_topk_target_logits_walk_canary_contract_wiring_validated"

REQUIRED_CHAIN = [
    "P5AK",
    "P5AL",
    "P5AM",
    "P5AN",
    "P5AO",
    "P5AP",
    "P5AQ",
    "P5AR",
    "P5AS",
    "P5AT",
    "P5AU",
    "P5AV",
]

APPROVAL_FLAGS = [
    "target_accept_walk_approved",
    "real_accept_approved",
    "real_token_commit_approved",
    "visible_token_publish_approved",
    "real_hidden_kv_commit_approved",
    "real_rejected_branch_discard_approved",
    "real_publish_approved",
    "product_runtime_hooks_approved",
    "draft_token_emission_approved",
    "performance_promotion_approved",
]

FORBIDDEN_RUNTIME_HOOKS = [
    "p5aw_target_accept_walk_runtime_hook",
    "p5t_token_commit_runtime_product_hook",
    "p5u_hidden_kv_survivor_commit_runtime_product_hook",
    "p5v_rejected_branch_discard_runtime_product_hook",
    "p5w_publish_gate_runtime_product_hook",
    "p5ad_root_publish_noop_product_hook_reuse",
    "common_speculative_hook",
    "server_draft_token_emission_hook",
    "public_api_route_hook",
    "ggml_kv_mutation_hook",
]

ZERO_COUNTERS = [
    "additional_target_logits_rows_walked",
    "actual_target_accept_steps",
    "actual_accepted_nodes",
    "correction_token_present",
    "actual_committed_tokens",
    "actual_survivor_pages_committed",
    "actual_pages_discarded",
    "rejected_branch_pages_reachable_after_discard",
    "actual_publish_visible_state",
    "actual_draft_tokens_emitted",
]

NO_RUNTIME_FLAGS = [
    "runtime_executed",
    "model_loaded",
    "context_created",
    "llama_decode_called",
    "target_accept_walk_executed",
    "accept_executed",
    "real_token_commit_executed",
    "visible_token_publish_executed",
    "real_hidden_kv_commit_executed",
    "real_rejected_branch_discard_executed",
    "real_publish_executed",
    "visible_state_changed",
    "kv_mutated",
    "draft_tokens_emitted",
]

EXPECTED_ALLOWED_EDGES = ["0:0", "1:0", "1:1", "2:0", "2:2"]
TARGET_VOCAB_SIZE = 248320


def _bool_map(data: dict[str, Any], key: str) -> dict[str, bool]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    out: dict[str, bool] = {}
    for name, raw in value.items():
        if not isinstance(raw, bool):
            raise ValueError(f"{key}.{name} must be boolean")
        out[name] = raw
    return out


def _int_map(data: dict[str, Any], key: str) -> dict[str, int]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    out: dict[str, int] = {}
    for name, raw in value.items():
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise ValueError(f"{key}.{name} must be integer")
        out[name] = raw
    return out


def _list_of_ints(value: Any, label: str, errors: list[str]) -> list[int]:
    if not isinstance(value, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        errors.append(f"{label} must be a list of integers")
        return []
    return value


def _list_of_numbers(value: Any, label: str, errors: list[str]) -> list[float]:
    if not isinstance(value, list) or any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        errors.append(f"{label} must be a list of numbers")
        return []
    return [float(item) for item in value]


def _validate_source_tree(data: dict[str, Any], errors: list[str]) -> tuple[list[int], list[int]]:
    tree = data.get("source_tree")
    if not isinstance(tree, dict):
        errors.append("source_tree must be an object")
        return [], []

    if tree.get("actual_tree_nodes") != 3:
        errors.append("source_tree.actual_tree_nodes must be 3")
    if tree.get("candidate_nodes") != 2:
        errors.append("source_tree.candidate_nodes must be 2")
    if tree.get("actual_verify_mask_entries") != 5:
        errors.append("source_tree.actual_verify_mask_entries must be 5")
    if tree.get("allowed_edges") != EXPECTED_ALLOWED_EDGES:
        errors.append(f"source_tree.allowed_edges must be {EXPECTED_ALLOWED_EDGES}")
    if tree.get("accept_path_descriptor_len") != 0:
        errors.append("source_tree.accept_path_descriptor_len must be 0")
    if tree.get("actual_accepted_nodes") != 0:
        errors.append("source_tree.actual_accepted_nodes must be 0")
    if tree.get("correction_token_present") != 0:
        errors.append("source_tree.correction_token_present must be 0")

    candidate_ids = _list_of_ints(tree.get("candidate_ids"), "source_tree.candidate_ids", errors)
    real_tree_token_ids = _list_of_ints(tree.get("real_tree_token_ids"), "source_tree.real_tree_token_ids", errors)
    if len(candidate_ids) != 2 or len(set(candidate_ids)) != 2 or any(token < 0 or token >= TARGET_VOCAB_SIZE for token in candidate_ids):
        errors.append("source_tree.candidate_ids must contain two distinct in-vocabulary token ids")
    if len(real_tree_token_ids) != 3:
        errors.append("source_tree.real_tree_token_ids must contain root plus two candidates")
    elif real_tree_token_ids[1:] != candidate_ids:
        errors.append("source_tree.real_tree_token_ids[1:] must equal candidate_ids")
    return candidate_ids, real_tree_token_ids


def _validate_p5av_evidence(data: dict[str, Any], candidate_ids: list[int], errors: list[str]) -> dict[str, Any]:
    evidence = data.get("p5av_target_logits_walk_evidence")
    if not isinstance(evidence, dict):
        errors.append("p5av_target_logits_walk_evidence must be an object")
        return {}

    if evidence.get("trace_status") != P5AV_TRACE_STATUS:
        errors.append(f"p5av_target_logits_walk_evidence.trace_status must be {P5AV_TRACE_STATUS}")
    if evidence.get("wiring_status") != P5AV_WIRING_STATUS:
        errors.append(f"p5av_target_logits_walk_evidence.wiring_status must be {P5AV_WIRING_STATUS}")
    if evidence.get("planned_target_logits_rows") != 1:
        errors.append("p5av_target_logits_walk_evidence.planned_target_logits_rows must be 1")
    if evidence.get("actual_target_logits_rows_walked") != 1:
        errors.append("p5av_target_logits_walk_evidence.actual_target_logits_rows_walked must be 1")
    if evidence.get("target_logits_source") != "target_model_full_vocab_logits":
        errors.append("p5av_target_logits_walk_evidence.target_logits_source must be target_model_full_vocab_logits")
    if evidence.get("target_logits_width") != TARGET_VOCAB_SIZE:
        errors.append(f"p5av_target_logits_walk_evidence.target_logits_width must be {TARGET_VOCAB_SIZE}")
    if evidence.get("target_logits_walk_canary_only") is not True:
        errors.append("p5av_target_logits_walk_evidence.target_logits_walk_canary_only must be true")
    if evidence.get("planned_parent_nodes") != [0]:
        errors.append("p5av_target_logits_walk_evidence.planned_parent_nodes must be [0]")
    if evidence.get("planned_candidate_nodes") != [1, 2]:
        errors.append("p5av_target_logits_walk_evidence.planned_candidate_nodes must be [1, 2]")
    if evidence.get("candidate_ids") != candidate_ids:
        errors.append("p5av_target_logits_walk_evidence.candidate_ids must equal source_tree.candidate_ids")
    if evidence.get("row_semantics") != "parent_position_scores_candidate_children":
        errors.append("p5av_target_logits_walk_evidence.row_semantics must be parent_position_scores_candidate_children")
    if evidence.get("accept_decision_source") != "target_logits_canary_only_no_accept":
        errors.append("p5av_target_logits_walk_evidence.accept_decision_source must be target_logits_canary_only_no_accept")

    logits = _list_of_numbers(evidence.get("target_candidate_logits"), "p5av_target_logits_walk_evidence.target_candidate_logits", errors)
    if len(logits) != 2:
        errors.append("p5av_target_logits_walk_evidence.target_candidate_logits must contain two scores")

    for name in ("target_logits_batch_index", "target_logits_pos"):
        raw = evidence.get(name)
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            errors.append(f"p5av_target_logits_walk_evidence.{name} must be a non-negative integer")
    seq_id = evidence.get("target_logits_seq_id")
    if not isinstance(seq_id, int) or isinstance(seq_id, bool) or seq_id < -1:
        errors.append("p5av_target_logits_walk_evidence.target_logits_seq_id must be an integer >= -1")

    for name in ("actual_target_accept_steps", "actual_accepted_nodes", "correction_token_present", "actual_committed_tokens", "actual_publish_visible_state"):
        if evidence.get(name) != 0:
            errors.append(f"p5av_target_logits_walk_evidence.{name} must be 0")
    return evidence


def _validate_accept_plan(data: dict[str, Any], candidate_ids: list[int], errors: list[str]) -> dict[str, Any]:
    plan = data.get("target_accept_plan")
    if not isinstance(plan, dict):
        errors.append("target_accept_plan must be an object")
        return {}

    if plan.get("planned_target_accept_steps") != 1:
        errors.append("target_accept_plan.planned_target_accept_steps must be 1")
    if plan.get("planned_accept_parent_nodes") != [0]:
        errors.append("target_accept_plan.planned_accept_parent_nodes must be [0]")
    if plan.get("planned_accept_candidate_nodes") != [1, 2]:
        errors.append("target_accept_plan.planned_accept_candidate_nodes must be [1, 2]")
    if plan.get("planned_accept_candidate_ids") != candidate_ids:
        errors.append("target_accept_plan.planned_accept_candidate_ids must equal source_tree.candidate_ids")
    if plan.get("planned_accept_score_source") != "target_candidate_logits_from_p5av_row":
        errors.append("target_accept_plan.planned_accept_score_source must be target_candidate_logits_from_p5av_row")
    if plan.get("planned_accept_rule") != "greedy_target_argmax_child_match_or_correction":
        errors.append("target_accept_plan.planned_accept_rule must be greedy_target_argmax_child_match_or_correction")
    if plan.get("planned_correction_token_source") != "target_full_vocab_argmax_when_no_child_match":
        errors.append("target_accept_plan.planned_correction_token_source must be target_full_vocab_argmax_when_no_child_match")
    if plan.get("planned_accept_output_visibility") != "metadata_only_until_explicit_runtime_approval":
        errors.append("target_accept_plan.planned_accept_output_visibility must be metadata_only_until_explicit_runtime_approval")
    for name in ("actual_target_accept_steps", "actual_accepted_nodes", "correction_token_present"):
        if plan.get(name) != 0:
            errors.append(f"target_accept_plan.{name} must be 0")
    return plan


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []

    if data.get("slice") != "P5AW":
        errors.append("slice must be P5AW")

    chain = data.get("completed_chain")
    if not isinstance(chain, list) or any(not isinstance(item, str) for item in chain):
        errors.append("completed_chain must be a list of slice names")
        chain = []
    missing_chain = [name for name in REQUIRED_CHAIN if name not in chain]
    if missing_chain:
        errors.append(f"missing required predecessor slices: {missing_chain}")

    candidate_ids, real_tree_token_ids = _validate_source_tree(data, errors)
    evidence = _validate_p5av_evidence(data, candidate_ids, errors)
    plan = _validate_accept_plan(data, candidate_ids, errors)

    try:
        approvals = _bool_map(data, "approvals")
    except ValueError as exc:
        errors.append(str(exc))
        approvals = {}
    for name in APPROVAL_FLAGS:
        if approvals.get(name) is not False:
            errors.append(f"{name} must remain false until explicit architecture approval")

    try:
        runtime_hooks = _bool_map(data, "runtime_hooks")
    except ValueError as exc:
        errors.append(str(exc))
        runtime_hooks = {}
    for name in FORBIDDEN_RUNTIME_HOOKS:
        if runtime_hooks.get(name) is not False:
            errors.append(f"{name} must remain false in P5AW")

    try:
        counters = _int_map(data, "side_effect_counters")
    except ValueError as exc:
        errors.append(str(exc))
        counters = {}
    for name in ZERO_COUNTERS:
        if counters.get(name) != 0:
            errors.append(f"{name} must be 0")

    try:
        runtime_boundary = _bool_map(data, "runtime_boundary")
    except ValueError as exc:
        errors.append(str(exc))
        runtime_boundary = {}
    for name in NO_RUNTIME_FLAGS:
        if runtime_boundary.get(name) is not False:
            errors.append(f"{name} must be false")

    if data.get("allowed_paths") != ["experiments/jetspec"]:
        errors.append("allowed_paths must be exactly ['experiments/jetspec']")

    if data.get("promotion_result") != "target_accept_walk_blocked_pending_explicit_approval":
        errors.append("promotion_result must be target_accept_walk_blocked_pending_explicit_approval")

    ok = not errors
    return {
        "ok": ok,
        "status": STATUS if ok else "p5aw_target_accept_walk_readiness_invalid",
        "errors": errors,
        "completed_chain": REQUIRED_CHAIN,
        "source_tree": {
            "actual_tree_nodes": 3,
            "candidate_nodes": 2,
            "candidate_ids": candidate_ids,
            "real_tree_token_ids": real_tree_token_ids,
            "actual_verify_mask_entries": 5,
            "allowed_edges": EXPECTED_ALLOWED_EDGES,
        },
        "p5av_target_logits_walk_evidence": {
            "trace_status": evidence.get("trace_status"),
            "wiring_status": evidence.get("wiring_status"),
            "planned_target_logits_rows": evidence.get("planned_target_logits_rows"),
            "actual_target_logits_rows_walked": evidence.get("actual_target_logits_rows_walked"),
            "target_logits_source": evidence.get("target_logits_source"),
            "target_logits_width": evidence.get("target_logits_width"),
            "target_logits_batch_index": evidence.get("target_logits_batch_index"),
            "target_logits_pos": evidence.get("target_logits_pos"),
            "target_logits_seq_id": evidence.get("target_logits_seq_id"),
            "planned_parent_nodes": evidence.get("planned_parent_nodes"),
            "planned_candidate_nodes": evidence.get("planned_candidate_nodes"),
            "candidate_ids": evidence.get("candidate_ids"),
            "target_candidate_logits": evidence.get("target_candidate_logits"),
            "row_semantics": evidence.get("row_semantics"),
            "accept_decision_source": evidence.get("accept_decision_source"),
            "target_logits_walk_canary_only": evidence.get("target_logits_walk_canary_only"),
            "actual_target_accept_steps": evidence.get("actual_target_accept_steps"),
            "actual_accepted_nodes": evidence.get("actual_accepted_nodes"),
            "correction_token_present": evidence.get("correction_token_present"),
        },
        "target_accept_plan": {
            "planned_target_accept_steps": plan.get("planned_target_accept_steps"),
            "planned_accept_parent_nodes": plan.get("planned_accept_parent_nodes"),
            "planned_accept_candidate_nodes": plan.get("planned_accept_candidate_nodes"),
            "planned_accept_candidate_ids": plan.get("planned_accept_candidate_ids"),
            "planned_accept_score_source": plan.get("planned_accept_score_source"),
            "planned_accept_rule": plan.get("planned_accept_rule"),
            "planned_correction_token_source": plan.get("planned_correction_token_source"),
            "planned_accept_output_visibility": plan.get("planned_accept_output_visibility"),
            "actual_target_accept_steps": plan.get("actual_target_accept_steps"),
            "actual_accepted_nodes": plan.get("actual_accepted_nodes"),
            "correction_token_present": plan.get("correction_token_present"),
        },
        "promotion_result": "target_accept_walk_blocked_pending_explicit_approval",
        "approvals": {name: approvals.get(name) is True for name in APPROVAL_FLAGS},
        "runtime_hooks": {name: runtime_hooks.get(name) is True for name in FORBIDDEN_RUNTIME_HOOKS},
        "side_effect_counters": {name: counters.get(name) for name in ZERO_COUNTERS},
        "runtime_boundary": {name: runtime_boundary.get(name) is True for name in NO_RUNTIME_FLAGS},
        "runtime_executed": False,
        "model_loaded": False,
        "context_created": False,
        "target_accept_walk_executed": False,
        "accept_executed": False,
        "kv_mutated": False,
        "draft_tokens_emitted": False,
    }


def load_fixture(path: pathlib.Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("fixture root must be an object")
    return data


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--json", action="store_true", help="Emit JSON output; accepted for consistency.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out = evaluate_fixture(load_fixture(args.fixture))
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
