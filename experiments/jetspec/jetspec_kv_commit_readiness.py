#!/usr/bin/env python3
"""P5H inert JetSpec KV/hidden commit ownership readiness contract.

This models ownership and gather/commit bookkeeping only. It does not instantiate
llama_context, execute a draft-head graph, emit draft tokens, mutate a real KV
cache, call llama.cpp KV primitives, or add server behavior.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import committed_hidden_cache
from tree_semantics import DraftTree


STATUS = "kv_commit_readiness_verified_not_executed"
MISSING_PRIMITIVE = "missing primitive"
REQUIRED_MAPPING_KEYS = [
    "reserve_transient_tree_slots",
    "gather_accepted_path",
    "discard_rejected_tree_slots",
    "compact_survivors_to_committed_tail",
    "preserve_cross_sequence_slots",
]


class KVCommitReadinessError(ValueError):
    """Raised when a P5H KV/hidden commit fixture violates the contract."""


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_from_data(data: Mapping[str, Any]) -> DraftTree:
    tree_obj = data["tree"]
    return DraftTree(
        token_ids=[int(x) for x in tree_obj["token_ids"]],
        parent_indices=[int(x) for x in tree_obj["parent_indices"]],
        depth=[int(x) for x in tree_obj["depth"]],
        cum_logprob=[float(x) for x in tree_obj.get("cum_logprob", [0.0] * len(tree_obj["token_ids"]))],
        rank=[int(x) for x in tree_obj.get("rank", [-1] * len(tree_obj["token_ids"]))],
    )


def _validate_path(path: Sequence[Any], num_nodes: int) -> list[int]:
    accepted_path = [int(x) for x in path]
    if not accepted_path:
        raise KVCommitReadinessError("accepted_path must be non-empty")
    if accepted_path[0] != 0:
        raise KVCommitReadinessError(f"accepted_path must be root-inclusive and start at 0, got {accepted_path}")
    if len(set(accepted_path)) != len(accepted_path):
        raise KVCommitReadinessError(f"accepted_path must not contain duplicate nodes: {accepted_path}")
    bad = [idx for idx in accepted_path if idx < 0 or idx >= num_nodes]
    if bad:
        raise KVCommitReadinessError(f"accepted_path contains out-of-range nodes for num_nodes={num_nodes}: {bad}")
    return accepted_path


def _validate_mapping(mapping: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in REQUIRED_MAPPING_KEYS:
        value = str(mapping.get(key, "")).strip()
        if not value:
            raise KVCommitReadinessError(f"ownership mapping for {key} must name an exact primitive or state missing primitive")
        if value != MISSING_PRIMITIVE and not value.startswith("llama_kv_cache_"):
            raise KVCommitReadinessError(
                f"ownership mapping for {key} must be {MISSING_PRIMITIVE!r} or an exact llama_kv_cache_* primitive, got {value!r}"
            )
        out[key] = value
    return out


def _validate_boundary(boundary: Mapping[str, Any]) -> dict[str, Any]:
    required_false = [
        "llama_context_instantiated",
        "draft_head_graph_executed",
        "draft_tokens_emitted",
        "real_kv_cache_mutated",
        "server_route_added",
        "runtime_supported",
    ]
    for key in required_false:
        if bool(boundary.get(key, True)):
            raise KVCommitReadinessError(f"readiness boundary requires {key}=false")
    return {
        "llama_context_instantiated": False,
        "draft_head_graph_executed": False,
        "draft_tokens_emitted": False,
        "real_kv_cache_mutated": False,
        "server_route_added": False,
        "runtime_supported": False,
        "no_production_kv_mutation": True,
    }


def _normalize_pre_slots(slots: Sequence[Mapping[str, Any]], target_seq_id: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for slot in slots:
        logical_slot = int(slot["logical_slot"])
        if logical_slot in seen:
            raise KVCommitReadinessError(f"duplicate pre-committed KV slot: {logical_slot}")
        seen.add(logical_slot)
        seq_id = int(slot["seq_id"])
        if seq_id != target_seq_id:
            raise KVCommitReadinessError(f"pre-committed KV slot {logical_slot} belongs to seq {seq_id}, expected {target_seq_id}")
        refcount = int(slot.get("refcount", 1))
        if refcount != 1:
            raise KVCommitReadinessError(f"pre-committed KV slot {logical_slot} refcount must be 1, got {refcount}")
        out.append({
            "logical_slot": logical_slot,
            "seq_id": seq_id,
            "token": int(slot["token"]),
            "refcount": refcount,
            "source": "pre_committed",
        })
    out.sort(key=lambda item: item["logical_slot"])
    if [slot["logical_slot"] for slot in out] != list(range(len(out))):
        raise KVCommitReadinessError("pre-committed KV slots must be compact from 0")
    return out


def _normalize_tree_slots(tree: DraftTree, slots: Sequence[Mapping[str, Any]], *, past_len: int, target_seq_id: int) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    used_temp_slots: set[int] = set()
    for slot in slots:
        node_index = int(slot["node_index"])
        if node_index < 0 or node_index >= tree.num_nodes:
            raise KVCommitReadinessError(f"tree slot node_index out of range: {node_index}")
        if node_index in out:
            raise KVCommitReadinessError(f"duplicate tree KV slot for node {node_index}")
        temp_slot = int(slot["temp_slot"])
        expected_temp_slot = past_len + node_index
        if temp_slot != expected_temp_slot:
            raise KVCommitReadinessError(
                f"tree temp slot must equal past_len + node_index for node {node_index}: {temp_slot} != {expected_temp_slot}"
            )
        if temp_slot in used_temp_slots:
            raise KVCommitReadinessError(f"duplicate transient tree KV slot: {temp_slot}")
        used_temp_slots.add(temp_slot)
        seq_id = int(slot["seq_id"])
        if seq_id != target_seq_id:
            raise KVCommitReadinessError(f"tree KV slot {temp_slot} belongs to seq {seq_id}, expected {target_seq_id}")
        token = int(slot["token"])
        if token != int(tree.token_ids[node_index]):
            raise KVCommitReadinessError(
                f"tree KV slot token mismatch for node {node_index}: {token} != {tree.token_ids[node_index]}"
            )
        refcount = int(slot.get("refcount", 1))
        if refcount != 1:
            raise KVCommitReadinessError(f"tree KV slot {temp_slot} refcount must be 1 before gather, got {refcount}")
        out[node_index] = {
            "node_index": node_index,
            "temp_slot": temp_slot,
            "seq_id": seq_id,
            "token": token,
            "refcount": refcount,
            "sentinel": slot.get("sentinel"),
        }
    missing = [idx for idx in range(tree.num_nodes) if idx not in out]
    if missing:
        raise KVCommitReadinessError(f"missing transient tree KV slots for nodes: {missing}")
    return out


def _validate_other_sequence_slots(slots: Sequence[Mapping[str, Any]], *, target_seq_id: int, touched_slots: set[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for slot in slots:
        logical_slot = int(slot["logical_slot"])
        seq_id = int(slot["seq_id"])
        if seq_id == target_seq_id:
            raise KVCommitReadinessError(f"other-sequence slot {logical_slot} unexpectedly uses target seq_id {target_seq_id}")
        if logical_slot in touched_slots:
            raise KVCommitReadinessError(f"cross-sequence isolation violation: slot {logical_slot} would be touched")
        out.append({
            "logical_slot": logical_slot,
            "seq_id": seq_id,
            "token": int(slot["token"]),
            "refcount": int(slot.get("refcount", 1)),
            "unchanged": True,
        })
    return out


def _ensure_no_rejected_sentinels(rows: Sequence[Mapping[str, Any]], rejected_sentinels: Sequence[Any]) -> bool:
    sentinels = {str(value) for value in rejected_sentinels}
    leaked = [row.get("sentinel") for row in rows if row.get("sentinel") is not None and str(row.get("sentinel")) in sentinels]
    if leaked:
        raise KVCommitReadinessError(f"rejected tree sentinel leaked into committed KV rows: {leaked}")
    return True


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    tree = _tree_from_data(data)
    target_seq_id = int(data.get("target_seq_id", 0))
    past_len = int(data["past_len"])
    pre_committed_token_ids = [int(x) for x in data["pre_committed_token_ids"]]
    if past_len != len(pre_committed_token_ids) - 1:
        raise KVCommitReadinessError(
            f"past_len must equal hidden/KV rows that trail committed tokens by one: {past_len} != {len(pre_committed_token_ids) - 1}"
        )

    accepted_path = _validate_path(data["accepted_path"], tree.num_nodes)
    correction_token = int(data["correction_token"])
    accepted_set = set(accepted_path)
    rejected_node_indices = [idx for idx in range(tree.num_nodes) if idx not in accepted_set]

    pre_slots = _normalize_pre_slots(data["pre_kv_slots"], target_seq_id)
    if len(pre_slots) != past_len:
        raise KVCommitReadinessError(f"pre_kv_slots length must equal past_len: {len(pre_slots)} != {past_len}")
    if [slot["token"] for slot in pre_slots] != pre_committed_token_ids[:-1]:
        raise KVCommitReadinessError("pre_kv_slots must cover committed tokens except the current anchor")

    tree_slots = _normalize_tree_slots(tree, data["tree_kv_slots"], past_len=past_len, target_seq_id=target_seq_id)
    gather_positions = [past_len + idx for idx in accepted_path]
    expected_gather_positions = [int(x) for x in data.get("expected_gather_positions", gather_positions)]
    if gather_positions != expected_gather_positions:
        raise KVCommitReadinessError(f"gather_positions mismatch: {gather_positions} != {expected_gather_positions}")

    discarded_temp_slots = [past_len + idx for idx in rejected_node_indices]
    if set(gather_positions) & set(discarded_temp_slots):
        raise KVCommitReadinessError("gather positions include rejected tree slots")

    touched_slots = set(gather_positions) | set(discarded_temp_slots)
    other_sequence_slots = _validate_other_sequence_slots(data.get("other_sequence_slots", []), target_seq_id=target_seq_id, touched_slots=touched_slots)

    survivor_rows: list[dict[str, Any]] = []
    post_slots = [dict(slot) for slot in pre_slots]
    next_slot = len(post_slots)
    for node_index in accepted_path:
        tree_slot = tree_slots[node_index]
        row = {
            "logical_slot": next_slot,
            "source_temp_slot": tree_slot["temp_slot"],
            "node_index": node_index,
            "seq_id": target_seq_id,
            "token": tree_slot["token"],
            "refcount": 1,
            "source": "accepted_tree_node",
            "sentinel": tree_slot.get("sentinel"),
        }
        survivor_rows.append(row)
        post_slots.append(row)
        next_slot += 1

    _ensure_no_rejected_sentinels(survivor_rows, data.get("rejected_sentinels", []))

    accepted_draft_tokens = [tree.token_ids[idx] for idx in accepted_path[1:]]
    committed_append_tokens = accepted_draft_tokens + [correction_token]
    post_committed_token_ids = pre_committed_token_ids + committed_append_tokens
    expected_post_committed = [int(x) for x in data.get("expected_post_committed_token_ids", post_committed_token_ids)]
    if post_committed_token_ids != expected_post_committed:
        raise KVCommitReadinessError(f"post_committed_token_ids mismatch: {post_committed_token_ids} != {expected_post_committed}")

    expected_post_kv_tokens = [int(x) for x in data.get("expected_post_kv_tokens", [slot["token"] for slot in post_slots])]
    if [slot["token"] for slot in post_slots] != expected_post_kv_tokens:
        raise KVCommitReadinessError(f"post KV token sequence mismatch: {[slot['token'] for slot in post_slots]} != {expected_post_kv_tokens}")
    if len(post_slots) != len(post_committed_token_ids) - 1:
        raise KVCommitReadinessError("post KV/hidden rows must trail committed tokens by one correction anchor")

    hidden_commit = committed_hidden_cache.commit_tree_hidden_round(
        pre_hidden_cache=data["pre_hidden_cache"],
        tree_token_ids=tree.token_ids,
        node_hidden_rows=data["node_hidden_rows"],
        accepted_path=accepted_path,
        correction_token=correction_token,
        width=int(data["hidden_width"]),
        pre_committed_token_ids=pre_committed_token_ids,
    ).as_dict()
    if hidden_commit["committed_append_tokens"] != committed_append_tokens:
        raise KVCommitReadinessError("hidden commit tokens differ from KV commit tokens")
    if hidden_commit["appended_node_indices"] != accepted_path:
        raise KVCommitReadinessError("hidden commit survivors differ from accepted path")
    if hidden_commit["correction_hidden_appended"]:
        raise KVCommitReadinessError("correction hidden must not be appended in the same round")

    mapping = _validate_mapping(data["llama_cpp_ownership_mapping"])
    boundary = _validate_boundary(data["readiness_boundary"])

    return {
        "ok": True,
        "status": STATUS,
        "slot_ownership": {
            "target_seq_id": target_seq_id,
            "past_len": past_len,
            "pre_kv_slots": pre_slots,
            "tree_temp_slots": [tree_slots[idx] for idx in range(tree.num_nodes)],
            "accepted_path": accepted_path,
            "survivor_node_indices": accepted_path,
            "rejected_node_indices": rejected_node_indices,
            "gather_positions": gather_positions,
            "gather_contract": "past_len + accepted_path",
            "discarded_temp_slots": discarded_temp_slots,
            "post_kv_slots": post_slots,
            "post_kv_tokens": [slot["token"] for slot in post_slots],
            "refcounts_all_one_after_commit": all(int(slot["refcount"]) == 1 for slot in post_slots),
            "rejected_tree_nodes_unreachable_after_commit": True,
            "cross_sequence_slots": other_sequence_slots,
            "cross_sequence_isolated": all(slot["unchanged"] for slot in other_sequence_slots),
        },
        "commit": {
            "pre_committed_token_ids": pre_committed_token_ids,
            "committed_append_tokens": committed_append_tokens,
            "post_committed_token_ids": post_committed_token_ids,
            "token_commit_contract": "[accepted draft tokens | correction]",
            "hidden_kv_survivor_contract": "[root | accepted] only",
            "correction_hidden_appended_same_round": False,
            "model_hidden_cache_trails_committed_tokens_by_one": len(post_slots) == len(post_committed_token_ids) - 1,
            "hidden_commit": hidden_commit,
        },
        "llama_cpp_ownership_mapping": mapping,
        "readiness_boundary": boundary,
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
    except (OSError, KeyError, TypeError, ValueError, KVCommitReadinessError) as exc:
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
