#!/usr/bin/env python3
"""Evaluate the inert P5AT JetSpec real draft-head top-k promotion blocker audit."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

STATUS = "p5at_real_draft_head_topk_promotion_blocker_verified_not_executed"

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
]

REQUIRED_P5AS_STATUS = "p5as_real_draft_head_topk_publish_gate_noop_trace_contract_verified"

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
    "p5t_token_commit_runtime_product_hook",
    "p5u_hidden_kv_survivor_commit_runtime_product_hook",
    "p5v_rejected_branch_discard_runtime_product_hook",
    "p5w_publish_gate_runtime_product_hook",
    "p5ad_root_publish_noop_product_hook_reuse",
    "server_draft_token_emission_hook",
    "public_api_route_hook",
    "ggml_kv_mutation_hook",
]

ZERO_COUNTERS = [
    "actual_committed_tokens",
    "actual_survivor_pages_committed",
    "actual_pages_discarded",
    "rejected_branch_pages_reachable_after_discard",
    "actual_publish_visible_state",
    "actual_draft_tokens_emitted",
    "actual_target_logits_rows_walked",
    "actual_target_accept_steps",
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


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []

    if data.get("slice") != "P5AT":
        errors.append("slice must be P5AT")

    chain = data.get("completed_chain")
    if not isinstance(chain, list) or any(not isinstance(item, str) for item in chain):
        errors.append("completed_chain must be a list of slice names")
        chain = []
    missing_chain = [name for name in REQUIRED_CHAIN if name not in chain]
    if missing_chain:
        errors.append(f"missing required predecessor slices: {missing_chain}")

    if data.get("p5as_status") != REQUIRED_P5AS_STATUS:
        errors.append(f"p5as_status must be {REQUIRED_P5AS_STATUS}")

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
            errors.append(f"{name} must remain false in P5AT")

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

    allowed_paths = data.get("allowed_paths")
    if allowed_paths != ["experiments/jetspec"]:
        errors.append("allowed_paths must be exactly ['experiments/jetspec']")

    if data.get("promotion_result") != "blocked_pending_explicit_approval":
        errors.append("promotion_result must be blocked_pending_explicit_approval")

    ok = not errors
    return {
        "ok": ok,
        "status": STATUS if ok else "p5at_real_draft_head_topk_promotion_blocker_invalid",
        "errors": errors,
        "completed_chain": REQUIRED_CHAIN,
        "p5as_status": data.get("p5as_status"),
        "p5as_terminal_noop_chain_verified": ok,
        "promotion_result": "blocked_pending_explicit_approval",
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
