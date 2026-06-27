#!/usr/bin/env python3
"""P5I inert JetSpec tree-runtime approval matrix.

This converts P5G/P5H readiness contracts into a machine-checked approval packet
for a future production tree-runtime slice. It does not instantiate llama_context,
execute a draft-head graph, emit draft tokens, mutate real KV state, add server
behavior, or claim performance/promotion readiness.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


STATUS = "tree_runtime_approval_packet_verified_not_executed"

REQUIRED_ACTIONS = [
    "tree_build",
    "verify_mask",
    "accept_path",
    "token_commit",
    "hidden_kv_survivor_commit",
    "rejected_branch_discard",
    "cross_sequence_isolation",
    "rollback_fail_closed_disable",
]

ALLOWED_MAPPING_STATUS = {
    "validated_by_p5g",
    "validated_by_p5h",
    "missing_primitive",
    "blocked_pending_explicit_approval",
}

FORBIDDEN_PRODUCTION_PATHS = [
    "common/speculative.cpp",
    "common/common.h",
    "src/",
    "include/",
    "tools/server/",
    "tests/",
    "examples/",
    "pocs/",
    "ggml/src/",
    "CMakeLists.txt",
    ".cmake",
]

RUNTIME_CLAIM_KEYS = [
    "llama_context_created",
    "draft_head_graph_executed",
    "draft_tokens_emitted",
    "real_kv_mutated",
    "server_route_added",
    "performance_claimed",
    "promotion_claimed",
    "runtime_supported",
]


class ApprovalMatrixError(ValueError):
    """Raised when a P5I approval matrix fixture violates the contract."""


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_runtime_claims(claims: Mapping[str, Any]) -> dict[str, bool]:
    normalized: dict[str, bool] = {}
    for key in RUNTIME_CLAIM_KEYS:
        value = bool(claims.get(key, True))
        if value:
            raise ApprovalMatrixError(f"runtime claim {key}=true is forbidden for inert P5I")
        normalized[key] = False
    return normalized


def _validate_forbidden_paths(paths: Sequence[Any]) -> list[str]:
    touched = [str(path) for path in paths]
    if touched:
        raise ApprovalMatrixError(f"P5I must not list touched production paths: {touched}")
    return []


def _validate_action(action: Mapping[str, Any]) -> dict[str, Any]:
    action_id = str(action["id"])
    if action_id not in REQUIRED_ACTIONS:
        raise ApprovalMatrixError(f"unknown future runtime action: {action_id}")
    mapping_status = str(action["mapping_status"])
    if mapping_status not in ALLOWED_MAPPING_STATUS:
        raise ApprovalMatrixError(f"{action_id}: invalid mapping_status {mapping_status!r}")
    primitive = str(action.get("future_primitive", "")).strip()
    if not primitive:
        raise ApprovalMatrixError(f"{action_id}: future_primitive must be explicit")
    if primitive.lower() in {"tbd", "todo", "implicit", "seq_cp", "seq_rm", "seq_import_physical"}:
        raise ApprovalMatrixError(f"{action_id}: future_primitive is implicit or unsafe: {primitive!r}")
    if mapping_status == "missing_primitive" and primitive != "missing primitive":
        raise ApprovalMatrixError(f"{action_id}: missing_primitive mapping must state 'missing primitive'")
    if mapping_status != "missing_primitive" and primitive == "missing primitive":
        raise ApprovalMatrixError(f"{action_id}: 'missing primitive' requires missing_primitive mapping_status")
    if mapping_status == "blocked_pending_explicit_approval" and "explicit approval" not in str(action.get("blocked_reason", "")):
        raise ApprovalMatrixError(f"{action_id}: blocked action must cite explicit approval")
    if bool(action.get("touches_production_now", True)):
        raise ApprovalMatrixError(f"{action_id}: touches_production_now must be false")
    if bool(action.get("runtime_executed_now", True)):
        raise ApprovalMatrixError(f"{action_id}: runtime_executed_now must be false")
    if bool(action.get("claims_performance", True)):
        raise ApprovalMatrixError(f"{action_id}: claims_performance must be false")
    if bool(action.get("claims_promotion", True)):
        raise ApprovalMatrixError(f"{action_id}: claims_promotion must be false")
    if bool(action.get("approval_required", False)) is not True:
        raise ApprovalMatrixError(f"{action_id}: approval_required must be true")
    evidence = [str(item) for item in action.get("evidence", [])]
    if mapping_status == "validated_by_p5g" and not any("P5G" in item for item in evidence):
        raise ApprovalMatrixError(f"{action_id}: P5G-mapped action must cite P5G evidence")
    if mapping_status == "validated_by_p5h" and not any("P5H" in item for item in evidence):
        raise ApprovalMatrixError(f"{action_id}: P5H-mapped action must cite P5H evidence")
    return {
        "id": action_id,
        "future_primitive": primitive,
        "mapping_status": mapping_status,
        "blocked_reason": str(action.get("blocked_reason", "")),
        "approval_required": True,
        "touches_production_now": False,
        "runtime_executed_now": False,
        "claims_performance": False,
        "claims_promotion": False,
        "evidence": evidence,
    }


def _validate_actions(actions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [_validate_action(action) for action in actions]
    ids = [action["id"] for action in normalized]
    duplicate = sorted({item for item in ids if ids.count(item) > 1})
    if duplicate:
        raise ApprovalMatrixError(f"duplicate future runtime actions: {duplicate}")
    missing = [action_id for action_id in REQUIRED_ACTIONS if action_id not in ids]
    if missing:
        raise ApprovalMatrixError(f"missing required future runtime actions: {missing}")
    return sorted(normalized, key=lambda item: REQUIRED_ACTIONS.index(item["id"]))


def _validate_approval_gates(gates: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    required_gate_ids = {
        "aggregate_contracts",
        "default_build",
        "disabled_path",
        "existing_mtp_path",
        "jetspec_opt_in_fail_closed",
        "correctness_matrix",
        "baseline_benchmark",
    }
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for gate in gates:
        gate_id = str(gate["id"])
        seen.add(gate_id)
        command = str(gate.get("command", "")).strip()
        pass_condition = str(gate.get("pass_condition", "")).strip()
        if not command or not pass_condition:
            raise ApprovalMatrixError(f"approval gate {gate_id} needs command and pass_condition")
        if any(forbidden in pass_condition.lower() for forbidden in ["production-ready", "mature", "promoted by default"]):
            raise ApprovalMatrixError(f"approval gate {gate_id} contains forbidden promotion wording")
        normalized.append({"id": gate_id, "command": command, "pass_condition": pass_condition})
    missing = sorted(required_gate_ids - seen)
    if missing:
        raise ApprovalMatrixError(f"missing approval gates: {missing}")
    return normalized


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("status") != STATUS:
        raise ApprovalMatrixError(f"fixture status must be {STATUS!r}")
    if bool(data.get("production_tree_runtime_approved", True)):
        raise ApprovalMatrixError("production_tree_runtime_approved must be false")
    if str(data.get("scope")) != "experiments/jetspec only":
        raise ApprovalMatrixError("scope must be experiments/jetspec only")

    runtime_claims = _validate_runtime_claims(data["runtime_claims"])
    touched_paths = _validate_forbidden_paths(data.get("production_paths_touched", []))
    actions = _validate_actions(data["future_runtime_actions"])
    gates = _validate_approval_gates(data["approval_gates"])

    mapping_counts: dict[str, int] = {status: 0 for status in sorted(ALLOWED_MAPPING_STATUS)}
    for action in actions:
        mapping_counts[action["mapping_status"]] += 1

    return {
        "ok": True,
        "status": STATUS,
        "production_tree_runtime_approved": False,
        "scope": "experiments/jetspec only",
        "runtime_claims": {**runtime_claims, "no_runtime_execution": True},
        "production_paths_touched": touched_paths,
        "forbidden_production_paths": FORBIDDEN_PRODUCTION_PATHS,
        "future_runtime_actions": actions,
        "coverage": {
            "required_actions": REQUIRED_ACTIONS,
            "required_actions_present": True,
            "mapping_counts": mapping_counts,
            "all_actions_require_approval": all(action["approval_required"] for action in actions),
            "no_action_touches_production_now": True,
            "no_action_claims_performance_or_promotion": True,
        },
        "approval_gates": gates,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        out = evaluate_fixture(_load_json(args.fixture.resolve()))
    except (OSError, KeyError, TypeError, ValueError, ApprovalMatrixError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
