#!/usr/bin/env python3
"""P5L inert JetSpec page-map ownership oracle.

This converts the QBlock/PageAttention descriptor lessons into an executable
fixture oracle for future JetSpec KV ownership work. It models abstract page
maps and transient tree pages only. It does not instantiate llama_context,
execute a draft-head graph, emit draft tokens, mutate real KV state, add server
behavior, or claim performance/promotion readiness.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Mapping
from typing import Any


STATUS = "page_map_ownership_oracle_verified_not_executed"

REQUIRED_ORACLES = [
    "hidden_kv_survivor_page_ownership",
    "rejected_branch_page_unreachable",
    "cross_sequence_page_isolation",
]

REQUIRED_ORACLE_INVARIANTS = {
    "hidden_kv_survivor_page_ownership": [
        "accepted path pages map to [root | accepted] only",
        "correction hidden deferred",
        "accepted path physical gather/compact explicit",
    ],
    "rejected_branch_page_unreachable": [
        "rejected transient pages unreachable after commit",
        "accepted path cannot read rejected siblings or descendants",
        "rollback restores pre-round page snapshot",
    ],
    "cross_sequence_page_isolation": [
        "other-sequence pages unchanged",
        "no duplicate mutable physical page ownership",
        "rollback preserves other sequences",
    ],
}

REQUIRED_CANDIDATE_PRIMITIVES = [
    "llama_kv_cache_jetspec_validate_page_ownership_oracle_candidate",
    "llama_kv_cache_jetspec_reserve_transient_tree_pages_candidate",
    "llama_kv_cache_jetspec_commit_page_survivor_path_candidate",
    "llama_kv_cache_jetspec_discard_rejected_tree_pages_candidate",
]

FORBIDDEN_IMPLICIT_HELPERS = [
    "seq_cp",
    "seq_rm",
    "seq_import_physical",
]

REQUIRED_PAGE_FLAGS = [
    "transient_tree",
    "accepted_survivor",
    "rejected_branch",
    "cross_sequence_guard",
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

QBLOCK_LESSON_TRUE_KEYS = [
    "identity_maps_are_oracle_only",
    "visible_noncanonical_owned_overlays_fail_closed",
    "full_current_k_map_required",
    "canonical_write_through_required_until_proven_safe",
    "no_performance_or_promotion_claim",
]


class PageMapOwnershipOracleError(ValueError):
    """Raised when a P5L page-map oracle fixture violates the contract."""


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PageMapOwnershipOracleError(f"{name} must be an integer")
    return value


def _validate_runtime_boundary(boundary: Mapping[str, Any]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for key in RUNTIME_FALSE_KEYS:
        value = bool(boundary.get(key, True))
        if value:
            raise PageMapOwnershipOracleError(f"runtime boundary requires {key}=false")
        out[key] = False
    out["source_text_read_only"] = True
    out["page_map_oracle_only"] = True
    out["no_runtime_execution"] = True
    return out


def _validate_page_map(page_map: Mapping[str, Any]) -> dict[str, Any]:
    page_tokens = _as_int(page_map.get("page_tokens"), "page_tokens")
    logical_base_token = _as_int(page_map.get("logical_base_token"), "logical_base_token")
    valid_tail_tokens = _as_int(page_map.get("valid_tail_tokens"), "valid_tail_tokens")
    physical_pages = _as_int(page_map.get("physical_pages"), "physical_pages")
    block_table_pages = _as_int(page_map.get("block_table_pages"), "block_table_pages")
    generation = _as_int(page_map.get("generation"), "generation")
    if page_tokens <= 0 or valid_tail_tokens <= 0 or physical_pages <= 0 or block_table_pages <= 0 or generation <= 0:
        raise PageMapOwnershipOracleError("page map sizes and generation must be positive")
    if logical_base_token != 0:
        raise PageMapOwnershipOracleError("page-map oracle requires full current-K logical_base_token=0")
    block_table = [_as_int(item, "block_table entry") for item in page_map.get("block_table", [])]
    if len(block_table) != block_table_pages:
        raise PageMapOwnershipOracleError("block_table length must equal block_table_pages")
    required_pages = (valid_tail_tokens + page_tokens - 1) // page_tokens
    if required_pages != block_table_pages:
        raise PageMapOwnershipOracleError("page-map oracle requires a full current-K map")
    for physical_page in block_table:
        if physical_page < 0 or physical_page >= physical_pages:
            raise PageMapOwnershipOracleError("block_table physical page is out of range")
    flags = [str(item) for item in page_map.get("flags", [])]
    missing_flags = [flag for flag in REQUIRED_PAGE_FLAGS if flag not in flags]
    if missing_flags:
        raise PageMapOwnershipOracleError(f"page map missing required flags: {missing_flags}")
    return {
        "page_tokens": page_tokens,
        "logical_base_token": logical_base_token,
        "valid_tail_tokens": valid_tail_tokens,
        "physical_pages": physical_pages,
        "block_table_pages": block_table_pages,
        "block_table": block_table,
        "generation": generation,
        "flags": flags,
        "full_current_k_map": True,
    }


def _validate_tree_round(data: Mapping[str, Any], page_map: Mapping[str, Any]) -> dict[str, Any]:
    tree_nodes = data.get("tree_nodes", [])
    if not isinstance(tree_nodes, list) or not tree_nodes:
        raise PageMapOwnershipOracleError("tree_nodes must be a non-empty list")
    nodes: dict[int, dict[str, Any]] = {}
    for node in tree_nodes:
        node_id = _as_int(node.get("id"), "tree node id")
        if node_id in nodes:
            raise PageMapOwnershipOracleError(f"duplicate tree node id: {node_id}")
        physical_page = _as_int(node.get("physical_page"), f"node {node_id} physical_page")
        if physical_page < 0 or physical_page >= int(page_map["physical_pages"]):
            raise PageMapOwnershipOracleError(f"node {node_id} physical_page out of range")
        nodes[node_id] = {
            "id": node_id,
            "parent": node.get("parent"),
            "token": _as_int(node.get("token"), f"node {node_id} token"),
            "physical_page": physical_page,
            "role": str(node.get("role", "")),
            "seq_id": _as_int(node.get("seq_id", 0), f"node {node_id} seq_id"),
        }
    accepted_path = [_as_int(item, "accepted_path entry") for item in data.get("accepted_path", [])]
    if not accepted_path:
        raise PageMapOwnershipOracleError("accepted_path must be non-empty")
    root_node_id = _as_int(data.get("root_node_id"), "root_node_id")
    if accepted_path[0] != root_node_id:
        raise PageMapOwnershipOracleError("accepted_path must begin with the root node")
    for node_id in accepted_path:
        if node_id not in nodes:
            raise PageMapOwnershipOracleError(f"accepted_path references missing node: {node_id}")
        if nodes[node_id]["role"] not in {"root", "accepted_survivor"}:
            raise PageMapOwnershipOracleError(f"accepted_path includes non-survivor node: {node_id}")
    rejected_node_ids = [node_id for node_id, node in nodes.items() if node["role"] == "rejected_branch"]
    if any(node_id in accepted_path for node_id in rejected_node_ids):
        raise PageMapOwnershipOracleError("accepted_path must not include rejected nodes")
    survivor_pages = [nodes[node_id]["physical_page"] for node_id in accepted_path]
    rejected_pages = [nodes[node_id]["physical_page"] for node_id in rejected_node_ids]
    post_commit_pages = [_as_int(item, "post_commit_survivor_pages entry") for item in data.get("post_commit_survivor_pages", [])]
    if post_commit_pages != survivor_pages:
        raise PageMapOwnershipOracleError("post_commit_survivor_pages must exactly match accepted path pages")
    if any(page in post_commit_pages for page in rejected_pages):
        raise PageMapOwnershipOracleError("rejected transient pages must be unreachable after commit")
    if bool(data.get("accepted_path_reads_rejected_pages", True)):
        raise PageMapOwnershipOracleError("accepted path must not read rejected pages")
    if bool(data.get("rollback_restores_pre_round_state")) is not True:
        raise PageMapOwnershipOracleError("rollback must restore the pre-round page snapshot")
    if bool(data.get("correction_hidden_deferred")) is not True:
        raise PageMapOwnershipOracleError("correction hidden must be deferred")
    if str(data.get("committed_output_contract")) != "[root | accepted]":
        raise PageMapOwnershipOracleError("committed output contract must be [root | accepted]")
    return {
        "root_node_id": root_node_id,
        "accepted_path": accepted_path,
        "survivor_pages": survivor_pages,
        "rejected_node_ids": rejected_node_ids,
        "rejected_pages": rejected_pages,
        "post_commit_survivor_pages": post_commit_pages,
        "accepted_path_reads_rejected_pages": False,
        "rollback_restores_pre_round_state": True,
        "correction_hidden_deferred": True,
        "committed_output_contract": "[root | accepted]",
        "tree_nodes": [nodes[node_id] for node_id in sorted(nodes)],
    }


def _validate_page_ownership(data: Mapping[str, Any], page_map: Mapping[str, Any]) -> dict[str, Any]:
    owners = data.get("page_owners", [])
    if not isinstance(owners, list) or not owners:
        raise PageMapOwnershipOracleError("page_owners must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    by_page: dict[int, list[dict[str, Any]]] = {}
    for owner in owners:
        physical_page = _as_int(owner.get("physical_page"), "owner physical_page")
        if physical_page < 0 or physical_page >= int(page_map["physical_pages"]):
            raise PageMapOwnershipOracleError("page owner physical_page out of range")
        item = {
            "physical_page": physical_page,
            "seq_id": _as_int(owner.get("seq_id"), "owner seq_id"),
            "role": str(owner.get("role", "")),
            "immutable_prefix_shared": bool(owner.get("immutable_prefix_shared", False)),
        }
        normalized.append(item)
        by_page.setdefault(physical_page, []).append(item)
    duplicate_pages: list[int] = []
    for physical_page, page_owners in by_page.items():
        if len(page_owners) <= 1:
            continue
        immutable_shared = all(item["immutable_prefix_shared"] and item["role"] == "committed_prefix" for item in page_owners)
        if not immutable_shared:
            duplicate_pages.append(physical_page)
    if duplicate_pages:
        raise PageMapOwnershipOracleError(f"duplicate mutable physical page ownership: {duplicate_pages}")
    before = data.get("other_sequence_pages_before", [])
    after = data.get("other_sequence_pages_after", [])
    if before != after:
        raise PageMapOwnershipOracleError("other-sequence pages must remain unchanged")
    return {
        "page_owners": sorted(normalized, key=lambda item: (item["physical_page"], item["seq_id"], item["role"])),
        "duplicate_mutable_physical_page_ownership": False,
        "other_sequence_pages_before": before,
        "other_sequence_pages_after": after,
        "other_sequence_pages_unchanged": True,
    }


def _validate_helper_mappings(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise PageMapOwnershipOracleError("helper_mappings must be a list")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        helper = str(item.get("helper", ""))
        seen.add(helper)
        if helper not in FORBIDDEN_IMPLICIT_HELPERS:
            raise PageMapOwnershipOracleError(f"unexpected helper mapping: {helper}")
        if str(item.get("verdict", "")) != "audited_non_exact_helper":
            raise PageMapOwnershipOracleError(f"{helper}: verdict must remain audited_non_exact_helper")
        if bool(item.get("implicit_mapping_allowed", True)):
            raise PageMapOwnershipOracleError(f"{helper}: implicit mapping must be forbidden")
        if bool(item.get("exact_existing_primitive_proven", True)):
            raise PageMapOwnershipOracleError(f"{helper}: exact existing primitive must not be proven")
        out.append({
            "helper": helper,
            "verdict": "audited_non_exact_helper",
            "implicit_mapping_allowed": False,
            "exact_existing_primitive_proven": False,
            "reason": str(item.get("reason", "")),
        })
    missing = [helper for helper in FORBIDDEN_IMPLICIT_HELPERS if helper not in seen]
    if missing:
        raise PageMapOwnershipOracleError(f"missing forbidden helper mappings: {missing}")
    return sorted(out, key=lambda item: FORBIDDEN_IMPLICIT_HELPERS.index(item["helper"]))


def _validate_candidate_primitives(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise PageMapOwnershipOracleError("candidate_primitives must be a list")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        name = str(item.get("name", ""))
        seen.add(name)
        if name not in REQUIRED_CANDIDATE_PRIMITIVES:
            raise PageMapOwnershipOracleError(f"unexpected candidate primitive: {name}")
        if not name.startswith("llama_kv_cache_jetspec_"):
            raise PageMapOwnershipOracleError("candidate primitives must use llama_kv_cache_jetspec_ prefix")
        if str(item.get("status", "")) != "design_only_missing_implementation":
            raise PageMapOwnershipOracleError(f"{name}: status must be design_only_missing_implementation")
        for key in ["implementation_approved", "touches_production_now", "runtime_executed_now", "claims_performance", "claims_promotion"]:
            if bool(item.get(key, True)):
                raise PageMapOwnershipOracleError(f"{name}: {key} must be false")
        out.append({
            "name": name,
            "status": "design_only_missing_implementation",
            "implementation_approved": False,
            "touches_production_now": False,
            "runtime_executed_now": False,
            "claims_performance": False,
            "claims_promotion": False,
            "source_backing": [str(entry) for entry in item.get("source_backing", [])],
        })
    missing = [name for name in REQUIRED_CANDIDATE_PRIMITIVES if name not in seen]
    if missing:
        raise PageMapOwnershipOracleError(f"missing candidate primitives: {missing}")
    return sorted(out, key=lambda item: REQUIRED_CANDIDATE_PRIMITIVES.index(item["name"]))


def _validate_oracle_case(case: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("id", ""))
    if case_id not in REQUIRED_ORACLES:
        raise PageMapOwnershipOracleError(f"unexpected oracle case: {case_id}")
    if str(case.get("status", "")) != "oracle_case_verified":
        raise PageMapOwnershipOracleError(f"{case_id}: status must be oracle_case_verified")
    for key in ["touches_production_now", "runtime_executed_now", "claims_performance", "claims_promotion"]:
        if bool(case.get(key, True)):
            raise PageMapOwnershipOracleError(f"{case_id}: {key} must be false")
    invariants = [str(item) for item in case.get("invariants", [])]
    missing_invariants = [item for item in REQUIRED_ORACLE_INVARIANTS[case_id] if item not in invariants]
    if missing_invariants:
        raise PageMapOwnershipOracleError(f"{case_id}: missing invariants {missing_invariants}")
    evidence = [str(item) for item in case.get("source_backing", [])]
    if not any("P5K" in item for item in evidence):
        raise PageMapOwnershipOracleError(f"{case_id}: source_backing must cite P5K")
    if not any(("P5H" in item or "P5J" in item or "QBlock" in item or "PageAttention" in item) for item in evidence):
        raise PageMapOwnershipOracleError(f"{case_id}: source_backing must cite P5H/P5J or QBlock evidence")
    checks = case.get("checks", {})
    if case_id == "hidden_kv_survivor_page_ownership":
        if bool(checks.get("accepted_path_physical_gather_compact_explicit")) is not True:
            raise PageMapOwnershipOracleError(f"{case_id}: accepted path gather/compact must be explicit")
        if str(checks.get("committed_output_contract")) != "[root | accepted]":
            raise PageMapOwnershipOracleError(f"{case_id}: committed output must be [root | accepted]")
        if bool(checks.get("correction_hidden_deferred")) is not True:
            raise PageMapOwnershipOracleError(f"{case_id}: correction hidden must be deferred")
    elif case_id == "rejected_branch_page_unreachable":
        if bool(checks.get("rejected_transient_pages_unreachable_after_commit")) is not True:
            raise PageMapOwnershipOracleError(f"{case_id}: rejected pages must be unreachable")
        if bool(checks.get("accepted_path_reads_rejected_pages", True)):
            raise PageMapOwnershipOracleError(f"{case_id}: accepted path must not read rejected pages")
        if bool(checks.get("rollback_restores_pre_round_state")) is not True:
            raise PageMapOwnershipOracleError(f"{case_id}: rollback must restore pre-round state")
    elif case_id == "cross_sequence_page_isolation":
        if bool(checks.get("other_sequence_pages_unchanged")) is not True:
            raise PageMapOwnershipOracleError(f"{case_id}: other-sequence pages must be unchanged")
        if bool(checks.get("duplicate_mutable_physical_page_ownership")):
            raise PageMapOwnershipOracleError(f"{case_id}: duplicate mutable page ownership must be false")
        if bool(checks.get("rollback_preserves_other_sequences")) is not True:
            raise PageMapOwnershipOracleError(f"{case_id}: rollback must preserve other sequences")
    return {
        "id": case_id,
        "status": "oracle_case_verified",
        "touches_production_now": False,
        "runtime_executed_now": False,
        "claims_performance": False,
        "claims_promotion": False,
        "invariants": invariants,
        "source_backing": evidence,
        "checks": checks,
    }


def _validate_oracle_cases(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise PageMapOwnershipOracleError("oracle_cases must be a list")
    out = [_validate_oracle_case(item) for item in items]
    ids = [item["id"] for item in out]
    missing = [case_id for case_id in REQUIRED_ORACLES if case_id not in ids]
    if missing:
        raise PageMapOwnershipOracleError(f"missing required oracle cases: {missing}")
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise PageMapOwnershipOracleError(f"duplicate oracle cases: {duplicates}")
    return sorted(out, key=lambda item: REQUIRED_ORACLES.index(item["id"]))


def _validate_qblock_lessons(lessons: Mapping[str, Any]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for key in QBLOCK_LESSON_TRUE_KEYS:
        if bool(lessons.get(key)) is not True:
            raise PageMapOwnershipOracleError(f"QBlock/PageAttention lesson must hold: {key}")
        out[key] = True
    if bool(lessons.get("identity_map_claims_speed_feature", True)):
        raise PageMapOwnershipOracleError("identity maps must not claim to be a speed feature")
    out["identity_map_claims_speed_feature"] = False
    return out


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("status") != STATUS:
        raise PageMapOwnershipOracleError(f"fixture status must be {STATUS!r}")
    if str(data.get("scope")) != "experiments/jetspec only":
        raise PageMapOwnershipOracleError("scope must be experiments/jetspec only")
    if data.get("production_paths_touched", []) != []:
        raise PageMapOwnershipOracleError("P5L oracle must not touch production paths")

    boundary = _validate_runtime_boundary(data["runtime_boundary"])
    page_map = _validate_page_map(data["page_map"])
    tree_round = _validate_tree_round(data["tree_round"], page_map)
    ownership = _validate_page_ownership(data["page_ownership"], page_map)
    helper_mappings = _validate_helper_mappings(data["helper_mappings"])
    candidate_primitives = _validate_candidate_primitives(data["candidate_primitives"])
    oracle_cases = _validate_oracle_cases(data["oracle_cases"])
    qblock_lessons = _validate_qblock_lessons(data["qblock_page_attention_lessons"])

    return {
        "ok": True,
        "status": STATUS,
        "scope": "experiments/jetspec only",
        "production_paths_touched": [],
        "runtime_boundary": boundary,
        "page_map": page_map,
        "tree_round": tree_round,
        "page_ownership": ownership,
        "helper_mappings": helper_mappings,
        "candidate_primitives": candidate_primitives,
        "oracle_cases": oracle_cases,
        "qblock_page_attention_lessons": qblock_lessons,
        "coverage": {
            "required_oracles": REQUIRED_ORACLES,
            "required_oracles_present": True,
            "required_candidate_primitives": REQUIRED_CANDIDATE_PRIMITIVES,
            "required_candidate_primitives_present": True,
            "forbidden_implicit_helpers": FORBIDDEN_IMPLICIT_HELPERS,
            "forbidden_implicit_helpers_remain_non_exact": True,
            "accepted_survivor_mapping_exact": True,
            "rejected_branches_unreachable": True,
            "cross_sequence_isolation_holds": True,
            "page_map_oracle_only": True,
            "no_runtime_or_production_touch": True,
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
        out = evaluate_fixture(_load_json(args.fixture.resolve()))
    except (OSError, KeyError, TypeError, ValueError, PageMapOwnershipOracleError) as exc:
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
