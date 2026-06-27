#!/usr/bin/env python3
"""P5M inert JetSpec transaction/failpoint plan oracle.

This models the ordered transaction semantics needed before any future JetSpec
runtime tree slice can be approved: snapshot, reserve, build, verify, accept,
commit, discard, publish, and rollback at failpoints. It is fixture validation
only. It does not instantiate llama_context, execute a draft-head graph, emit
draft tokens, mutate real KV state, add server behavior, or claim performance /
promotion readiness.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


STATUS = "transaction_plan_oracle_verified_not_executed"

REQUIRED_PHASES = [
    "snapshot_pre_round",
    "reserve_transient_tree_pages",
    "build_tree",
    "build_verify_mask",
    "accept_path",
    "commit_tokens",
    "commit_hidden_kv_survivors",
    "discard_rejected_branches",
    "publish_post_commit_state",
]

REQUIRED_ROLLBACK_POINTS = {
    "after_reserve": "reserve_transient_tree_pages",
    "after_build_tree": "build_tree",
    "after_verify_mask": "build_verify_mask",
    "after_accept": "accept_path",
    "after_token_commit": "commit_tokens",
    "after_hidden_kv_commit": "commit_hidden_kv_survivors",
    "after_rejected_discard": "discard_rejected_branches",
}

REQUIRED_VISIBILITY_RULES = [
    "no_committed_token_visibility_before_publish",
    "no_hidden_kv_visibility_before_publish",
    "no_page_map_visibility_before_publish",
    "publish_after_commit_and_discard_only",
    "rollback_clears_transient_state",
]

REQUIRED_CANDIDATE_PRIMITIVES = [
    "llama_kv_cache_jetspec_validate_page_ownership_oracle_candidate",
    "llama_kv_cache_jetspec_reserve_transient_tree_pages_candidate",
    "llama_kv_cache_jetspec_commit_page_survivor_path_candidate",
    "llama_kv_cache_jetspec_discard_rejected_tree_pages_candidate",
    "llama_kv_cache_jetspec_rollback_tree_transaction_candidate",
]

REQUIRED_SOURCE_BACKING = [
    "P5G tree-runtime readiness",
    "P5H KV/hidden commit readiness",
    "P5I tree-runtime approval packet",
    "P5K KV ownership primitive design",
    "P5L page-map ownership oracle",
]

FORBIDDEN_IMPLICIT_HELPERS = [
    "seq_cp",
    "seq_rm",
    "seq_import_physical",
]

RUNTIME_FALSE_KEYS = [
    "llama_context_created",
    "draft_head_graph_executed",
    "draft_tokens_emitted",
    "real_kv_mutated",
    "server_route_added",
    "runtime_supported",
    "performance_claimed",
    "promotion_claimed",
]


class TransactionPlanOracleError(ValueError):
    """Raised when a P5M transaction-plan fixture violates the contract."""


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TransactionPlanOracleError(f"{name} must be an integer")
    return value


def _as_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list):
        raise TransactionPlanOracleError(f"{name} must be a list")
    return [_as_int(item, f"{name} entry") for item in value]


def _validate_false_flags(item: Mapping[str, Any], prefix: str) -> None:
    for key in ["touches_production_now", "runtime_executed_now", "implementation_approved", "claims_performance", "claims_promotion"]:
        if bool(item.get(key, True)):
            raise TransactionPlanOracleError(f"{prefix}: {key} must be false")


def _validate_runtime_boundary(boundary: Mapping[str, Any]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for key in RUNTIME_FALSE_KEYS:
        value = bool(boundary.get(key, True))
        if value:
            raise TransactionPlanOracleError(f"runtime boundary requires {key}=false")
        out[key] = False
    out["transaction_plan_oracle_only"] = True
    out["no_runtime_execution"] = True
    return out


def _validate_production_paths_touched(paths: Sequence[Any]) -> list[str]:
    touched = [str(path) for path in paths]
    if touched:
        raise TransactionPlanOracleError(f"P5M must not touch production paths: {touched}")
    return []


def _validate_round_inputs(data: Mapping[str, Any]) -> dict[str, Any]:
    past_len = _as_int(data.get("past_len"), "past_len")
    if past_len < 0:
        raise TransactionPlanOracleError("past_len must be non-negative")
    root_node_id = _as_int(data.get("root_node_id"), "root_node_id")
    correction_token = _as_int(data.get("correction_token"), "correction_token")
    accepted_path = _as_int_list(data.get("accepted_path"), "accepted_path")
    if not accepted_path:
        raise TransactionPlanOracleError("accepted_path must be non-empty")
    if accepted_path[0] != root_node_id:
        raise TransactionPlanOracleError("accepted_path must begin with the root node")

    tree_nodes = data.get("tree_nodes", [])
    if not isinstance(tree_nodes, list) or not tree_nodes:
        raise TransactionPlanOracleError("tree_nodes must be a non-empty list")
    nodes: dict[int, dict[str, Any]] = {}
    for node in tree_nodes:
        node_id = _as_int(node.get("id"), "tree node id")
        if node_id in nodes:
            raise TransactionPlanOracleError(f"duplicate tree node id: {node_id}")
        parent_raw = node.get("parent")
        parent = None if parent_raw is None else _as_int(parent_raw, f"node {node_id} parent")
        role = str(node.get("role", ""))
        if role not in {"root", "accepted_survivor", "rejected_branch"}:
            raise TransactionPlanOracleError(f"node {node_id}: invalid role {role!r}")
        nodes[node_id] = {
            "id": node_id,
            "parent": parent,
            "token": _as_int(node.get("token"), f"node {node_id} token"),
            "physical_page": _as_int(node.get("physical_page"), f"node {node_id} physical_page"),
            "role": role,
            "seq_id": _as_int(node.get("seq_id", 0), f"node {node_id} seq_id"),
        }
    if root_node_id not in nodes or nodes[root_node_id]["role"] != "root":
        raise TransactionPlanOracleError("root_node_id must reference the root node")
    for node_id, node in nodes.items():
        if node_id == root_node_id:
            if node["parent"] is not None:
                raise TransactionPlanOracleError("root node parent must be null")
            continue
        if node["parent"] not in nodes:
            raise TransactionPlanOracleError(f"node {node_id}: parent is missing")
    for node_id in accepted_path:
        if node_id not in nodes:
            raise TransactionPlanOracleError(f"accepted_path references missing node: {node_id}")
        if nodes[node_id]["role"] not in {"root", "accepted_survivor"}:
            raise TransactionPlanOracleError(f"accepted_path includes non-survivor node: {node_id}")
    rejected_node_ids = [node_id for node_id, node in nodes.items() if node["role"] == "rejected_branch"]
    if any(node_id in accepted_path for node_id in rejected_node_ids):
        raise TransactionPlanOracleError("accepted_path must not include rejected nodes")
    accepted_tokens = [nodes[node_id]["token"] for node_id in accepted_path if node_id != root_node_id]
    accepted_pages = [nodes[node_id]["physical_page"] for node_id in accepted_path]
    rejected_pages = [nodes[node_id]["physical_page"] for node_id in rejected_node_ids]
    return {
        "past_len": past_len,
        "root_node_id": root_node_id,
        "accepted_path": accepted_path,
        "accepted_tokens_excluding_root": accepted_tokens,
        "correction_token": correction_token,
        "committed_token_suffix": accepted_tokens + [correction_token],
        "accepted_survivor_pages": accepted_pages,
        "rejected_node_ids": rejected_node_ids,
        "rejected_pages": rejected_pages,
        "tree_nodes": [nodes[node_id] for node_id in sorted(nodes)],
    }


def _validate_pre_round_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    tokens = _as_int_list(snapshot.get("committed_tokens", []), "pre_round_snapshot.committed_tokens")
    hidden_kv_pages = _as_int_list(snapshot.get("hidden_kv_pages", []), "pre_round_snapshot.hidden_kv_pages")
    other_sequence_pages = _as_int_list(snapshot.get("other_sequence_pages", []), "pre_round_snapshot.other_sequence_pages")
    return {
        "committed_tokens": tokens,
        "hidden_kv_pages": hidden_kv_pages,
        "other_sequence_pages": other_sequence_pages,
    }


def _validate_page_ownership(data: Mapping[str, Any]) -> dict[str, Any]:
    owners = data.get("page_owners", [])
    if not isinstance(owners, list) or not owners:
        raise TransactionPlanOracleError("page_owners must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    by_page: dict[int, list[dict[str, Any]]] = {}
    for owner in owners:
        item = {
            "physical_page": _as_int(owner.get("physical_page"), "owner physical_page"),
            "seq_id": _as_int(owner.get("seq_id"), "owner seq_id"),
            "role": str(owner.get("role", "")),
            "immutable_prefix_shared": bool(owner.get("immutable_prefix_shared", False)),
        }
        normalized.append(item)
        by_page.setdefault(item["physical_page"], []).append(item)
    duplicate_pages: list[int] = []
    for physical_page, page_owners in by_page.items():
        if len(page_owners) <= 1:
            continue
        immutable_shared = all(item["immutable_prefix_shared"] and item["role"] == "committed_prefix" for item in page_owners)
        if not immutable_shared:
            duplicate_pages.append(physical_page)
    if duplicate_pages:
        raise TransactionPlanOracleError(f"duplicate mutable physical page ownership: {duplicate_pages}")
    before = _as_int_list(data.get("other_sequence_pages_before", []), "other_sequence_pages_before")
    after = _as_int_list(data.get("other_sequence_pages_after", []), "other_sequence_pages_after")
    if before != after:
        raise TransactionPlanOracleError("other-sequence pages must remain unchanged")
    return {
        "page_owners": sorted(normalized, key=lambda item: (item["physical_page"], item["seq_id"], item["role"])),
        "duplicate_mutable_physical_page_ownership": False,
        "other_sequence_pages_before": before,
        "other_sequence_pages_after": after,
        "other_sequence_pages_unchanged": True,
    }


def _validate_phases(phases: Sequence[Mapping[str, Any]], round_inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    ids = [str(phase.get("id", "")) for phase in phases]
    if ids != REQUIRED_PHASES:
        raise TransactionPlanOracleError(f"transaction phases must be exactly ordered as {REQUIRED_PHASES}")
    for index, phase in enumerate(phases):
        phase_id = str(phase.get("id", ""))
        if _as_int(phase.get("order"), f"{phase_id}.order") != index:
            raise TransactionPlanOracleError(f"{phase_id}: order must be {index}")
        if str(phase.get("status", "")) != "planned_not_executed":
            raise TransactionPlanOracleError(f"{phase_id}: status must be planned_not_executed")
        _validate_false_flags(phase, phase_id)
        requires_after = [str(item) for item in phase.get("requires_after", [])]
        pre_publish_visible_state_unmodified = bool(phase.get("pre_publish_visible_state_unmodified", False))
        publishes_visible_state = bool(phase.get("publishes_visible_state", False))
        detail: dict[str, Any] = {}
        if phase_id != "publish_post_commit_state":
            if not pre_publish_visible_state_unmodified:
                raise TransactionPlanOracleError(f"{phase_id}: pre-publish visible state must remain unchanged")
            if publishes_visible_state:
                raise TransactionPlanOracleError(f"{phase_id}: only publish_post_commit_state may publish visible state")
        else:
            required_publish_deps = {"commit_tokens", "commit_hidden_kv_survivors", "discard_rejected_branches"}
            if not required_publish_deps.issubset(set(requires_after)):
                raise TransactionPlanOracleError("publish_post_commit_state must require token commit, hidden/KV commit, and rejected discard")
            if not publishes_visible_state:
                raise TransactionPlanOracleError("publish_post_commit_state must publish visible state")
            if pre_publish_visible_state_unmodified:
                raise TransactionPlanOracleError("publish_post_commit_state must be the only visible-state transition")
        if phase_id == "snapshot_pre_round" and bool(phase.get("captures_pre_round_snapshot")) is not True:
            raise TransactionPlanOracleError("snapshot_pre_round must capture the pre-round snapshot")
        if phase_id == "commit_tokens":
            if "accept_path" not in requires_after:
                raise TransactionPlanOracleError("commit_tokens must require accept_path")
            expected_suffix = list(round_inputs["committed_token_suffix"])
            token_suffix = _as_int_list(phase.get("committed_token_suffix", []), "commit_tokens.committed_token_suffix")
            if token_suffix != expected_suffix:
                raise TransactionPlanOracleError("commit_tokens suffix must be [accepted draft tokens | correction]")
            detail["committed_token_suffix"] = token_suffix
        if phase_id == "commit_hidden_kv_survivors":
            if "accept_path" not in requires_after or "reserve_transient_tree_pages" not in requires_after:
                raise TransactionPlanOracleError("commit_hidden_kv_survivors must require reserve and accept")
            if str(phase.get("survivor_pages_validated_by")) != "p5l_page_map_oracle":
                raise TransactionPlanOracleError("commit_hidden_kv_survivors must cite p5l_page_map_oracle")
            survivor_pages = _as_int_list(phase.get("survivor_pages", []), "commit_hidden_kv_survivors.survivor_pages")
            if survivor_pages != list(round_inputs["accepted_survivor_pages"]):
                raise TransactionPlanOracleError("hidden/KV survivor pages must match [root | accepted] pages")
            if bool(phase.get("correction_hidden_deferred")) is not True:
                raise TransactionPlanOracleError("commit_hidden_kv_survivors must defer correction hidden")
            detail["survivor_pages"] = survivor_pages
            detail["correction_hidden_deferred"] = True
        if phase_id == "discard_rejected_branches":
            if "commit_hidden_kv_survivors" not in requires_after:
                raise TransactionPlanOracleError("discard_rejected_branches must require hidden/KV survivor commit")
            rejected_pages = _as_int_list(phase.get("discarded_pages", []), "discard_rejected_branches.discarded_pages")
            if rejected_pages != list(round_inputs["rejected_pages"]):
                raise TransactionPlanOracleError("discarded pages must exactly match rejected branch pages")
            if bool(phase.get("rejected_pages_reachable_after_discard", True)):
                raise TransactionPlanOracleError("rejected branch pages must not be reachable after discard")
            detail["discarded_pages"] = rejected_pages
            detail["rejected_pages_reachable_after_discard"] = False
        normalized.append(
            {
                "id": phase_id,
                "order": index,
                "status": "planned_not_executed",
                "requires_after": requires_after,
                "pre_publish_visible_state_unmodified": pre_publish_visible_state_unmodified,
                "publishes_visible_state": publishes_visible_state,
                "touches_production_now": False,
                "runtime_executed_now": False,
                "implementation_approved": False,
                "claims_performance": False,
                "claims_promotion": False,
                **detail,
            }
        )
    return normalized


def _validate_rollback_points(points: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for point in points:
        point_id = str(point.get("id", ""))
        if point_id in by_id:
            raise TransactionPlanOracleError(f"duplicate rollback point: {point_id}")
        by_id[point_id] = point
    missing = [point_id for point_id in REQUIRED_ROLLBACK_POINTS if point_id not in by_id]
    if missing:
        raise TransactionPlanOracleError(f"missing rollback points: {missing}")
    for point_id, after_phase in REQUIRED_ROLLBACK_POINTS.items():
        point = by_id[point_id]
        if str(point.get("after_phase")) != after_phase:
            raise TransactionPlanOracleError(f"{point_id}: after_phase must be {after_phase}")
        if bool(point.get("restores_pre_round_snapshot")) is not True:
            raise TransactionPlanOracleError(f"{point_id}: must restore pre-round snapshot")
        tokens = _as_int_list(point.get("committed_tokens_after_rollback", []), f"{point_id}.committed_tokens_after_rollback")
        hidden_kv_pages = _as_int_list(point.get("hidden_kv_pages_after_rollback", []), f"{point_id}.hidden_kv_pages_after_rollback")
        other_pages = _as_int_list(point.get("other_sequence_pages_after_rollback", []), f"{point_id}.other_sequence_pages_after_rollback")
        if tokens != list(snapshot["committed_tokens"]):
            raise TransactionPlanOracleError(f"{point_id}: rollback must restore pre-round committed tokens")
        if hidden_kv_pages != list(snapshot["hidden_kv_pages"]):
            raise TransactionPlanOracleError(f"{point_id}: rollback must restore pre-round hidden/KV pages")
        if other_pages != list(snapshot["other_sequence_pages"]):
            raise TransactionPlanOracleError(f"{point_id}: rollback must preserve other-sequence pages")
        if bool(point.get("rejected_pages_reachable", True)):
            raise TransactionPlanOracleError(f"{point_id}: rejected pages must remain unreachable")
        if bool(point.get("partial_publish_visible", True)):
            raise TransactionPlanOracleError(f"{point_id}: rollback must not leave partial publish visible")
        normalized.append(
            {
                "id": point_id,
                "after_phase": after_phase,
                "restores_pre_round_snapshot": True,
                "committed_tokens_after_rollback": tokens,
                "hidden_kv_pages_after_rollback": hidden_kv_pages,
                "other_sequence_pages_after_rollback": other_pages,
                "rejected_pages_reachable": False,
                "partial_publish_visible": False,
            }
        )
    return normalized


def _validate_visibility_rules(rules: Mapping[str, Any]) -> dict[str, bool]:
    normalized: dict[str, bool] = {}
    for rule in REQUIRED_VISIBILITY_RULES:
        if bool(rules.get(rule)) is not True:
            raise TransactionPlanOracleError(f"visibility rule must be true: {rule}")
        normalized[rule] = True
    return normalized


def _validate_post_publish_state(state: Mapping[str, Any], snapshot: Mapping[str, Any], round_inputs: Mapping[str, Any]) -> dict[str, Any]:
    committed_tokens = _as_int_list(state.get("committed_tokens", []), "post_publish_state.committed_tokens")
    expected_tokens = list(snapshot["committed_tokens"]) + list(round_inputs["committed_token_suffix"])
    if committed_tokens != expected_tokens:
        raise TransactionPlanOracleError("post-publish committed tokens must be pre-round tokens plus [accepted draft tokens | correction]")
    hidden_kv_pages = _as_int_list(state.get("hidden_kv_survivor_pages", []), "post_publish_state.hidden_kv_survivor_pages")
    if hidden_kv_pages != list(round_inputs["accepted_survivor_pages"]):
        raise TransactionPlanOracleError("post-publish hidden/KV survivor pages must be [root | accepted]")
    if bool(state.get("correction_hidden_deferred")) is not True:
        raise TransactionPlanOracleError("post-publish correction hidden must be deferred")
    if bool(state.get("rejected_pages_reachable", True)):
        raise TransactionPlanOracleError("post-publish rejected pages must be unreachable")
    other_pages = _as_int_list(state.get("other_sequence_pages", []), "post_publish_state.other_sequence_pages")
    if other_pages != list(snapshot["other_sequence_pages"]):
        raise TransactionPlanOracleError("post-publish other-sequence pages must be unchanged")
    return {
        "committed_tokens": committed_tokens,
        "hidden_kv_survivor_pages": hidden_kv_pages,
        "correction_hidden_deferred": True,
        "rejected_pages_reachable": False,
        "other_sequence_pages": other_pages,
    }


def _validate_candidate_primitives(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        name = str(item.get("name", ""))
        if name not in REQUIRED_CANDIDATE_PRIMITIVES:
            raise TransactionPlanOracleError(f"unknown candidate primitive: {name}")
        if name in seen:
            raise TransactionPlanOracleError(f"duplicate candidate primitive: {name}")
        seen.add(name)
        if str(item.get("status")) != "design_only_missing_implementation":
            raise TransactionPlanOracleError(f"{name}: status must be design_only_missing_implementation")
        _validate_false_flags(item, name)
        source_backing = [str(backing) for backing in item.get("source_backing", [])]
        if not source_backing:
            raise TransactionPlanOracleError(f"{name}: source_backing is required")
        normalized.append(
            {
                "name": name,
                "status": "design_only_missing_implementation",
                "implementation_approved": False,
                "touches_production_now": False,
                "runtime_executed_now": False,
                "claims_performance": False,
                "claims_promotion": False,
                "source_backing": source_backing,
            }
        )
    missing = [name for name in REQUIRED_CANDIDATE_PRIMITIVES if name not in seen]
    if missing:
        raise TransactionPlanOracleError(f"missing candidate primitives: {missing}")
    return sorted(normalized, key=lambda item: REQUIRED_CANDIDATE_PRIMITIVES.index(item["name"]))


def _validate_helper_mappings(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        helper = str(item.get("helper", ""))
        if helper not in FORBIDDEN_IMPLICIT_HELPERS:
            raise TransactionPlanOracleError(f"unexpected helper mapping: {helper}")
        if helper in seen:
            raise TransactionPlanOracleError(f"duplicate helper mapping: {helper}")
        seen.add(helper)
        if str(item.get("verdict")) != "audited_non_exact_helper":
            raise TransactionPlanOracleError(f"{helper}: verdict must be audited_non_exact_helper")
        if bool(item.get("implicit_mapping_allowed", True)):
            raise TransactionPlanOracleError(f"{helper}: implicit mapping must be forbidden")
        if bool(item.get("exact_existing_primitive_proven", True)):
            raise TransactionPlanOracleError(f"{helper}: exact existing primitive must remain unproven")
        normalized.append(
            {
                "helper": helper,
                "verdict": "audited_non_exact_helper",
                "implicit_mapping_allowed": False,
                "exact_existing_primitive_proven": False,
                "reason": str(item.get("reason", "")),
            }
        )
    missing = [helper for helper in FORBIDDEN_IMPLICIT_HELPERS if helper not in seen]
    if missing:
        raise TransactionPlanOracleError(f"missing forbidden helper mappings: {missing}")
    return sorted(normalized, key=lambda item: FORBIDDEN_IMPLICIT_HELPERS.index(item["helper"]))


def _validate_source_backing(items: Sequence[Any]) -> list[str]:
    backing = [str(item) for item in items]
    missing = [item for item in REQUIRED_SOURCE_BACKING if item not in backing]
    if missing:
        raise TransactionPlanOracleError(f"missing source backing: {missing}")
    return backing


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("status") != STATUS:
        raise TransactionPlanOracleError(f"fixture status must be {STATUS!r}")
    if str(data.get("scope")) != "experiments/jetspec only":
        raise TransactionPlanOracleError("scope must be experiments/jetspec only")
    if str(data.get("approval_status")) != "blocked_pending_explicit_approval":
        raise TransactionPlanOracleError("approval_status must be blocked_pending_explicit_approval")

    runtime_boundary = _validate_runtime_boundary(data["runtime_boundary"])
    production_paths_touched = _validate_production_paths_touched(data.get("production_paths_touched", []))
    round_inputs = _validate_round_inputs(data["round_inputs"])
    snapshot = _validate_pre_round_snapshot(data["pre_round_snapshot"])
    page_ownership = _validate_page_ownership(data["page_ownership"])
    phases = _validate_phases(data["transaction_phases"], round_inputs)
    rollback_points = _validate_rollback_points(data["rollback_points"], snapshot)
    visibility_rules = _validate_visibility_rules(data["visibility_rules"])
    post_publish_state = _validate_post_publish_state(data["post_publish_state"], snapshot, round_inputs)
    candidate_primitives = _validate_candidate_primitives(data["candidate_primitives"])
    helper_mappings = _validate_helper_mappings(data["helper_mappings"])
    source_backing = _validate_source_backing(data["source_backing"])

    return {
        "ok": True,
        "status": STATUS,
        "scope": "experiments/jetspec only",
        "approval_status": "blocked_pending_explicit_approval",
        "runtime_boundary": runtime_boundary,
        "production_paths_touched": production_paths_touched,
        "round_inputs": round_inputs,
        "pre_round_snapshot": snapshot,
        "page_ownership": page_ownership,
        "transaction_phases": phases,
        "rollback_points": rollback_points,
        "visibility_rules": visibility_rules,
        "post_publish_state": post_publish_state,
        "candidate_primitives": candidate_primitives,
        "helper_mappings": helper_mappings,
        "source_backing": source_backing,
        "coverage": {
            "required_phases": REQUIRED_PHASES,
            "required_phases_ordered": True,
            "required_rollback_points": list(REQUIRED_ROLLBACK_POINTS),
            "required_rollback_points_present": True,
            "pre_publish_state_unmodified": True,
            "rollback_restores_pre_round_snapshot": True,
            "post_publish_tokens_are_accepted_plus_correction": True,
            "hidden_kv_survivors_are_root_plus_accepted": True,
            "rejected_branches_unreachable": True,
            "other_sequence_pages_unchanged": True,
            "duplicate_mutable_physical_page_ownership": False,
            "forbidden_implicit_helpers": FORBIDDEN_IMPLICIT_HELPERS,
            "forbidden_implicit_helpers_remain_non_exact": True,
            "required_candidate_primitives": REQUIRED_CANDIDATE_PRIMITIVES,
            "required_candidate_primitives_present": True,
            "no_runtime_or_production_touch": True,
            "no_performance_or_promotion_claim": True,
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        out = evaluate_fixture(_load_json(args.fixture))
    except (OSError, json.JSONDecodeError, KeyError, TransactionPlanOracleError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    text = json.dumps(out, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
