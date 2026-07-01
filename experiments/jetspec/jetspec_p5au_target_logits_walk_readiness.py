#!/usr/bin/env python3
"""Evaluate the inert P5AU JetSpec target-logits walk readiness packet."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

STATUS = "p5au_target_logits_walk_readiness_verified_not_executed"
P5AT_STATUS = "p5at_real_draft_head_topk_promotion_blocker_verified_not_executed"

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
]

APPROVAL_FLAGS = [
    "target_logits_walk_approved",
    "target_accept_walk_approved",
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
    "p5au_target_logits_walk_runtime_hook",
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
    "actual_target_logits_rows_walked",
    "actual_target_accept_steps",
    "actual_accepted_nodes",
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
    "target_logits_walk_executed",
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
    if len(candidate_ids) != 2 or len(set(candidate_ids)) != 2 or any(token < 0 for token in candidate_ids):
        errors.append("source_tree.candidate_ids must contain two distinct non-negative token ids")
    if len(real_tree_token_ids) != 3:
        errors.append("source_tree.real_tree_token_ids must contain root plus two candidates")
    elif real_tree_token_ids[1:] != candidate_ids:
        errors.append("source_tree.real_tree_token_ids[1:] must equal candidate_ids")
    return candidate_ids, real_tree_token_ids


def _validate_target_plan(data: dict[str, Any], candidate_ids: list[int], errors: list[str]) -> dict[str, Any]:
    plan = data.get("target_logits_plan")
    if not isinstance(plan, dict):
        errors.append("target_logits_plan must be an object")
        return {}

    if plan.get("planned_target_logits_rows") != 1:
        errors.append("target_logits_plan.planned_target_logits_rows must be 1")
    if plan.get("planned_parent_nodes") != [0]:
        errors.append("target_logits_plan.planned_parent_nodes must be [0]")
    if plan.get("planned_candidate_nodes") != [1, 2]:
        errors.append("target_logits_plan.planned_candidate_nodes must be [1, 2]")
    if plan.get("planned_candidate_ids") != candidate_ids:
        errors.append("target_logits_plan.planned_candidate_ids must equal source_tree.candidate_ids")
    if plan.get("planned_row_semantics") != "parent_position_scores_candidate_children":
        errors.append("target_logits_plan.planned_row_semantics must be parent_position_scores_candidate_children")
    if plan.get("planned_target_logits_source") != "target_model_full_vocab_logits":
        errors.append("target_logits_plan.planned_target_logits_source must be target_model_full_vocab_logits")
    if plan.get("planned_target_position_source") != "target_cache_position_for_parent_node_0":
        errors.append("target_logits_plan.planned_target_position_source must be target_cache_position_for_parent_node_0")
    if plan.get("planned_accept_decision_source") != "target_logits_not_yet_executed":
        errors.append("target_logits_plan.planned_accept_decision_source must be target_logits_not_yet_executed")
    for name in ("actual_target_logits_rows_walked", "actual_target_accept_steps", "actual_accepted_nodes", "correction_token_present"):
        if plan.get(name) != 0:
            errors.append(f"target_logits_plan.{name} must be 0")
    return plan


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []

    if data.get("slice") != "P5AU":
        errors.append("slice must be P5AU")

    chain = data.get("completed_chain")
    if not isinstance(chain, list) or any(not isinstance(item, str) for item in chain):
        errors.append("completed_chain must be a list of slice names")
        chain = []
    missing_chain = [name for name in REQUIRED_CHAIN if name not in chain]
    if missing_chain:
        errors.append(f"missing required predecessor slices: {missing_chain}")

    if data.get("p5at_status") != P5AT_STATUS:
        errors.append(f"p5at_status must be {P5AT_STATUS}")

    candidate_ids, real_tree_token_ids = _validate_source_tree(data, errors)
    plan = _validate_target_plan(data, candidate_ids, errors)

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
            errors.append(f"{name} must remain false in P5AU")

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

    if data.get("promotion_result") != "target_logits_walk_blocked_pending_explicit_approval":
        errors.append("promotion_result must be target_logits_walk_blocked_pending_explicit_approval")

    ok = not errors
    return {
        "ok": ok,
        "status": STATUS if ok else "p5au_target_logits_walk_readiness_invalid",
        "errors": errors,
        "completed_chain": REQUIRED_CHAIN,
        "p5at_status": data.get("p5at_status"),
        "source_tree": {
            "actual_tree_nodes": 3,
            "candidate_nodes": 2,
            "candidate_ids": candidate_ids,
            "real_tree_token_ids": real_tree_token_ids,
            "actual_verify_mask_entries": 5,
            "allowed_edges": EXPECTED_ALLOWED_EDGES,
        },
        "target_logits_plan": {
            "planned_target_logits_rows": plan.get("planned_target_logits_rows"),
            "planned_parent_nodes": plan.get("planned_parent_nodes"),
            "planned_candidate_nodes": plan.get("planned_candidate_nodes"),
            "planned_candidate_ids": plan.get("planned_candidate_ids"),
            "planned_row_semantics": plan.get("planned_row_semantics"),
            "planned_target_logits_source": plan.get("planned_target_logits_source"),
            "planned_target_position_source": plan.get("planned_target_position_source"),
            "planned_accept_decision_source": plan.get("planned_accept_decision_source"),
            "actual_target_logits_rows_walked": plan.get("actual_target_logits_rows_walked"),
            "actual_target_accept_steps": plan.get("actual_target_accept_steps"),
            "actual_accepted_nodes": plan.get("actual_accepted_nodes"),
            "correction_token_present": plan.get("correction_token_present"),
        },
        "promotion_result": "target_logits_walk_blocked_pending_explicit_approval",
        "approvals": {name: approvals.get(name) is True for name in APPROVAL_FLAGS},
        "runtime_hooks": {name: runtime_hooks.get(name) is True for name in FORBIDDEN_RUNTIME_HOOKS},
        "side_effect_counters": {name: counters.get(name) for name in ZERO_COUNTERS},
        "runtime_boundary": {name: runtime_boundary.get(name) is True for name in NO_RUNTIME_FLAGS},
        "runtime_executed": False,
        "model_loaded": False,
        "context_created": False,
        "target_logits_walk_executed": False,
        "target_accept_walk_executed": False,
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
