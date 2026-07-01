#!/usr/bin/env python3
"""No-runtime JetSpec P5N-W descriptor-chain contract.

This fixture models only descriptor readiness and fail-closed ordering. It does
not build trees, masks, accept paths, mutate KV, publish state, or run a model.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Mapping
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
STATUS = "descriptor_chain_verified_not_executed"

REQUIRED_CHAIN = [
    {
        "id": "p5o_pre_round_snapshot",
        "phase": "snapshot_pre_round",
        "requires": [],
        "ready_flag": "pre_round_snapshot_ready",
        "hash_key": "pre_round_snapshot_hash_last",
        "zero_actuals": {},
    },
    {
        "id": "p5n_transaction_plan",
        "phase": "transaction_plan_scaffold",
        "requires": ["p5o_pre_round_snapshot"],
        "ready_flag": "transaction_plan_ready",
        "hash_key": "transaction_plan_hash_last",
        "zero_actuals": {},
    },
    {
        "id": "p5p_transient_reservation",
        "phase": "reserve_transient_tree_pages",
        "requires": ["p5n_transaction_plan"],
        "rollback_point": "after_reserve",
        "ready_flag": "transient_reservation_ready",
        "hash_key": "transient_reservation_hash_last",
        "zero_actuals": {"actual_pages_reserved": 0},
    },
    {
        "id": "p5q_tree_build",
        "phase": "build_tree",
        "requires": ["p5p_transient_reservation"],
        "rollback_point": "after_build_tree",
        "ready_flag": "tree_build_descriptor_ready",
        "hash_key": "tree_build_descriptor_hash_last",
        "zero_actuals": {"actual_tree_nodes": 0},
    },
    {
        "id": "p5r_verify_mask",
        "phase": "build_verify_mask",
        "requires": ["p5q_tree_build"],
        "rollback_point": "after_verify_mask",
        "ready_flag": "verify_mask_descriptor_ready",
        "hash_key": "verify_mask_descriptor_hash_last",
        "zero_actuals": {"actual_verify_mask_entries": 0},
    },
    {
        "id": "p5s_accept_path",
        "phase": "accept_path",
        "requires": ["p5r_verify_mask"],
        "rollback_point": "after_accept",
        "ready_flag": "accept_path_descriptor_ready",
        "hash_key": "accept_path_descriptor_hash_last",
        "zero_actuals": {"actual_accepted_nodes": 0, "correction_token_present": 0},
    },
    {
        "id": "p5t_token_commit",
        "phase": "commit_tokens",
        "requires": ["p5s_accept_path"],
        "rollback_point": "after_token_commit",
        "ready_flag": "token_commit_descriptor_ready",
        "hash_key": "token_commit_descriptor_hash_last",
        "zero_actuals": {"actual_committed_tokens": 0},
    },
    {
        "id": "p5u_hidden_kv_survivor_commit",
        "phase": "commit_hidden_kv_survivors",
        "requires": ["p5t_token_commit"],
        "rollback_point": "after_hidden_kv_commit",
        "ready_flag": "hidden_kv_survivor_commit_descriptor_ready",
        "hash_key": "hidden_kv_survivor_commit_descriptor_hash_last",
        "zero_actuals": {"actual_survivor_pages_committed": 0},
    },
    {
        "id": "p5v_rejected_branch_discard",
        "phase": "discard_rejected_branches",
        "requires": ["p5u_hidden_kv_survivor_commit"],
        "rollback_point": "after_rejected_discard",
        "ready_flag": "rejected_branch_discard_descriptor_ready",
        "hash_key": "rejected_branch_discard_descriptor_hash_last",
        "zero_actuals": {"actual_pages_discarded": 0, "rejected_branch_pages_reachable_after_discard": 0},
    },
    {
        "id": "p5w_publish_gate",
        "phase": "publish_post_commit_state",
        "requires": ["p5v_rejected_branch_discard"],
        "ready_flag": "publish_gate_descriptor_ready",
        "hash_key": "publish_gate_descriptor_hash_last",
        "zero_actuals": {"actual_publish_visible_state": 0},
    },
]

RUNTIME_FALSE_KEYS = [
    "draft_head_graph_executed",
    "draft_tokens_emitted",
    "real_tree_built",
    "real_verify_mask_built",
    "real_accept_executed",
    "real_token_commit_executed",
    "real_kv_mutated",
    "real_publish_executed",
    "server_route_added",
    "public_api_added",
]


class DescriptorChainError(ValueError):
    """Raised when the descriptor-chain fixture violates the no-runtime contract."""


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _descriptor_by_id(data: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    descriptors = data.get("descriptors")
    if not isinstance(descriptors, list):
        raise DescriptorChainError("descriptors must be a list")
    out: dict[str, Mapping[str, Any]] = {}
    for item in descriptors:
        if not isinstance(item, Mapping):
            raise DescriptorChainError("descriptor entries must be objects")
        descriptor_id = str(item.get("id", ""))
        if not descriptor_id:
            raise DescriptorChainError("descriptor id is required")
        if descriptor_id in out:
            raise DescriptorChainError(f"duplicate descriptor id: {descriptor_id}")
        out[descriptor_id] = item
    return out


def evaluate_descriptor_chain(data: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    try:
        by_id = _descriptor_by_id(data)
    except DescriptorChainError as exc:
        errors.append(str(exc))

    observed_order = [str(item.get("id", "")) for item in data.get("descriptors", []) if isinstance(item, Mapping)]
    expected_order = [item["id"] for item in REQUIRED_CHAIN]
    if observed_order != expected_order:
        errors.append(f"descriptor order mismatch: observed={observed_order} expected={expected_order}")

    ready_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    prerequisite_edges: list[tuple[str, str]] = []
    zero_actuals_verified: list[str] = []
    for spec in REQUIRED_CHAIN:
        descriptor_id = spec["id"]
        item = by_id.get(descriptor_id)
        if item is None:
            errors.append(f"missing descriptor: {descriptor_id}")
            continue
        if item.get("phase") != spec["phase"]:
            errors.append(f"{descriptor_id}: phase mismatch {item.get('phase')!r} != {spec['phase']!r}")
        if item.get("ready") is not True:
            errors.append(f"{descriptor_id}: ready must be true")
        if int(item.get("hash", 0)) <= 0:
            errors.append(f"{descriptor_id}: hash must be positive")
        for parent in spec.get("requires", []):
            prerequisite_edges.append((parent, descriptor_id))
            if parent not in ready_ids:
                errors.append(f"{descriptor_id}: prerequisite {parent} was not ready")
        if "rollback_point" in spec and item.get("rollback_point") != spec["rollback_point"]:
            errors.append(f"{descriptor_id}: rollback mismatch {item.get('rollback_point')!r} != {spec['rollback_point']!r}")
        if "rollback_point" not in spec and "rollback_point" in item:
            errors.append(f"{descriptor_id}: publish gate must not invent a post-publish rollback point")
        actuals = item.get("actuals", {})
        if not isinstance(actuals, Mapping):
            errors.append(f"{descriptor_id}: actuals must be an object")
            actuals = {}
        for key, expected in spec["zero_actuals"].items():
            if actuals.get(key) != expected:
                errors.append(f"{descriptor_id}: {key} must be {expected}, got {actuals.get(key)!r}")
            else:
                zero_actuals_verified.append(f"{descriptor_id}.{key}")
        ready_ids.add(descriptor_id)
        normalized.append({
            "id": descriptor_id,
            "phase": spec["phase"],
            "ready_flag": spec["ready_flag"],
            "hash_key": spec["hash_key"],
            "zero_actuals": spec["zero_actuals"],
        })

    runtime_boundary = data.get("runtime_boundary", {})
    if not isinstance(runtime_boundary, Mapping):
        errors.append("runtime_boundary must be an object")
        runtime_boundary = {}
    for key in RUNTIME_FALSE_KEYS:
        if runtime_boundary.get(key) is not False:
            errors.append(f"runtime boundary requires {key}=false")

    return {
        "ok": not errors,
        "errors": errors,
        "status": STATUS if not errors else "descriptor_chain_invalid",
        "descriptor_count": len(normalized),
        "descriptor_order": observed_order,
        "prerequisite_edges": [list(edge) for edge in prerequisite_edges],
        "zero_actuals_verified": zero_actuals_verified,
        "runtime_boundary": {key: False for key in RUNTIME_FALSE_KEYS},
        "runtime_executed": False,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=pathlib.Path, default=HERE / "fixtures/jetspec_descriptor_chain_smoke.json")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out = evaluate_descriptor_chain(_load_json(args.fixture.resolve()))
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
