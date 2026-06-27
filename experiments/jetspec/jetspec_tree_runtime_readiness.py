#!/usr/bin/env python3
"""P5G inert JetSpec tree-runtime readiness contract.

This file translates upstream JetSpec tree/verify/gather semantics into a
llama.cpp-facing fixture contract before any production tree runtime exists. It
is intentionally staged under experiments/jetspec/ only. It does not instantiate
llama_context, run the draft-head graph, mutate KV cache, emit draft tokens, or
call server/CUDA code.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import committed_hidden_cache
import tree_verify_mask
from tree_semantics import DraftCandidate, DraftTree, build_accum_logp_tree, build_child_maps, tree_accept


STATUS = "tree_runtime_readiness_verified_not_executed"
FULL_VOCAB_LOGPROB_SOURCE = "full_vocab_softmax"
EPS = 1.0e-6


class TreeRuntimeReadinessError(ValueError):
    """Raised when a P5G readiness fixture violates the staged contract."""


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_rows(raw_topk: Sequence[Sequence[Any]]) -> list[list[DraftCandidate]]:
    rows: list[list[DraftCandidate]] = []
    for row in raw_topk:
        rows.append([item if isinstance(item, DraftCandidate) else DraftCandidate.from_obj(item) for item in row])
    return rows


def _validate_full_vocab_topk(data: Mapping[str, Any]) -> dict[str, Any]:
    source = data.get("topk_logprob_source")
    if source != FULL_VOCAB_LOGPROB_SOURCE:
        raise TreeRuntimeReadinessError(
            f"topk_logprob_source must be {FULL_VOCAB_LOGPROB_SOURCE!r}; got {source!r}"
        )
    if data.get("renormalize_topk", False):
        raise TreeRuntimeReadinessError("top-k-only renormalization is forbidden")

    raw_topk = data["topk_by_depth"]
    rows = _candidate_rows(raw_topk)
    row_summaries: list[dict[str, Any]] = []
    for depth, row in enumerate(rows):
        if not row:
            raise TreeRuntimeReadinessError(f"empty top-k row at depth {depth}")
        probs: list[float] = []
        for rank, cand in enumerate(row):
            if cand.p is None:
                prob = math.exp(cand.logprob)
            else:
                prob = float(cand.p)
                expected_logprob = math.log(prob)
                if abs(expected_logprob - cand.logprob) > EPS:
                    raise TreeRuntimeReadinessError(
                        f"candidate depth={depth} rank={rank} logprob does not match full-vocab p: "
                        f"{cand.logprob} vs log({prob})={expected_logprob}"
                    )
            if not (0.0 < prob < 1.0):
                raise TreeRuntimeReadinessError(f"candidate depth={depth} rank={rank} invalid probability {prob}")
            probs.append(prob)
        prob_sum = sum(probs)
        if prob_sum >= 1.0 - EPS:
            raise TreeRuntimeReadinessError(
                f"top-k row {depth} appears renormalized over top-k only; probability sum={prob_sum}"
            )
        row_summaries.append({
            "depth": depth + 1,
            "candidate_count": len(row),
            "prob_sum": prob_sum,
            "prob_sum_lt_one": True,
            "source": FULL_VOCAB_LOGPROB_SOURCE,
        })
    return {
        "source": FULL_VOCAB_LOGPROB_SOURCE,
        "renormalize_topk": False,
        "rows": row_summaries,
    }


def _compare_float_lists(label: str, actual: Sequence[float], expected: Sequence[float]) -> None:
    if len(actual) != len(expected):
        raise TreeRuntimeReadinessError(f"{label} length mismatch: {len(actual)} != {len(expected)}")
    for i, (a, e) in enumerate(zip(actual, expected)):
        if abs(float(a) - float(e)) > EPS:
            raise TreeRuntimeReadinessError(f"{label}[{i}] mismatch: {a} != {e}")


def _compare_int_lists(label: str, actual: Sequence[int], expected: Sequence[int]) -> None:
    a = [int(x) for x in actual]
    e = [int(x) for x in expected]
    if a != e:
        raise TreeRuntimeReadinessError(f"{label} mismatch: {a} != {e}")


def _validate_tree_abi(tree: DraftTree, data: Mapping[str, Any]) -> dict[str, Any]:
    budget = int(data["budget"])
    if tree.num_nodes > budget:
        raise TreeRuntimeReadinessError(f"num_nodes exceeds budget: {tree.num_nodes} > {budget}")
    if tree.parent_indices[0] != -1 or tree.depth[0] != 0:
        raise TreeRuntimeReadinessError("root parent/depth ABI violated")
    for idx in range(1, tree.num_nodes):
        parent = tree.parent_indices[idx]
        if parent < 0 or parent >= idx:
            raise TreeRuntimeReadinessError(f"parent index must precede child: node={idx} parent={parent}")
        if tree.depth[idx] != tree.depth[parent] + 1:
            raise TreeRuntimeReadinessError(f"depth mismatch for node={idx}")

    expected = data.get("expected_tree")
    if expected:
        _compare_int_lists("expected tree token_ids", tree.token_ids, expected["token_ids"])
        _compare_int_lists("expected tree parent_indices", tree.parent_indices, expected["parent_indices"])
        _compare_int_lists("expected tree depth", tree.depth, expected["depth"])
        if "rank" in expected:
            _compare_int_lists("expected tree rank", tree.rank, expected["rank"])
        if "cum_logprob" in expected:
            _compare_float_lists("expected tree cum_logprob", tree.cum_logprob, expected["cum_logprob"])

    return {
        "token_ids": list(tree.token_ids),
        "parent_indices": list(tree.parent_indices),
        "depth": list(tree.depth),
        "rank": list(tree.rank),
        "cum_logprob": [float(x) for x in tree.cum_logprob],
        "num_nodes": tree.num_nodes,
        "max_depth": tree.max_depth,
        "budget": budget,
        "parent_before_child": True,
        "root_parent": -1,
        "root_depth": 0,
        "num_nodes_lte_budget": True,
    }


def _duplicate_child_policy(tree: DraftTree) -> dict[str, Any]:
    children_by_parent_token: dict[tuple[int, int], list[int]] = {}
    for child_idx in range(1, tree.num_nodes):
        key = (tree.parent_indices[child_idx], tree.token_ids[child_idx])
        children_by_parent_token.setdefault(key, []).append(child_idx)

    child_maps = build_child_maps(tree)
    duplicates: list[dict[str, Any]] = []
    for (parent, token), child_indices in sorted(children_by_parent_token.items()):
        if len(child_indices) <= 1:
            continue
        expected_child = max(child_indices)
        actual_child = child_maps[parent][token]
        if actual_child != expected_child:
            raise TreeRuntimeReadinessError(
                f"duplicate child overwrite policy mismatch: parent={parent} token={token} "
                f"actual={actual_child} expected={expected_child}"
            )
        duplicates.append({
            "parent": parent,
            "token": token,
            "child_indices": child_indices,
            "selected_child": actual_child,
            "policy": "later_child_overwrites_earlier",
        })
    return {
        "policy": "later_child_overwrites_earlier",
        "deterministic": True,
        "duplicates": duplicates,
    }


def _validate_mask_abi(tree: DraftTree, data: Mapping[str, Any]) -> dict[str, Any]:
    past_len = int(data.get("past_len", 0))
    mask = tree_verify_mask.validate_mask_contract(tree, past_len)
    if not mask["ok"]:
        raise TreeRuntimeReadinessError("tree verify mask validation failed: " + "; ".join(mask["errors"]))

    tokens = {entry for row in mask["qq_bias"] for entry in row}
    if tokens != {tree_verify_mask.ZERO, tree_verify_mask.NEG_INF}:
        raise TreeRuntimeReadinessError(f"qq_bias must use only 0/-inf tokens, got {tokens}")

    bucket_size = int(data.get("bucket_size", tree.num_nodes))
    bucketed = tree_verify_mask.pad_qq_bias_to_bucket(mask["qq_bias"], bucket_size)
    bucket_tokens = {entry for row in bucketed for entry in row}
    if bucket_tokens != {tree_verify_mask.ZERO, tree_verify_mask.NEG_INF}:
        raise TreeRuntimeReadinessError(f"bucketed qq_bias must use only 0/-inf tokens, got {bucket_tokens}")

    other_seq_cols = int(data.get("other_sequence_tree_cols", 0))
    other_seq_bias = [[tree_verify_mask.NEG_INF for _ in range(other_seq_cols)] for _ in range(tree.num_nodes)]
    return {
        "past_len": past_len,
        "ancestor": mask["ancestor"],
        "verify_allowed_mask": mask["verify_allowed_mask"],
        "qq_bias": mask["qq_bias"],
        "bucket_size": bucket_size,
        "bucketed_qq_bias": bucketed,
        "prefix_visible_to_all": all(row[:past_len] == [True] * past_len for row in mask["verify_allowed_mask"]),
        "tree_visibility": "ancestor_only_self_included",
        "siblings_descendants_rejected_hidden": True,
        "additive_mask_tokens": sorted(tokens),
        "other_sequence_tree_cols": other_seq_cols,
        "other_sequence_tree_bias": other_seq_bias,
        "other_sequence_tree_cols_hidden": all(v == tree_verify_mask.NEG_INF for row in other_seq_bias for v in row),
    }


def _validate_accept_abi(tree: DraftTree, data: Mapping[str, Any]) -> dict[str, Any]:
    target_argmax = [int(x) for x in data["target_argmax_by_node"]]
    accepted_path, acceptance_length, correction_token = tree_accept(tree, target_argmax)
    if not accepted_path or accepted_path[0] != 0:
        raise TreeRuntimeReadinessError(f"accepted_path must be root-inclusive, got {accepted_path}")
    if acceptance_length != len(accepted_path) - 1:
        raise TreeRuntimeReadinessError("acceptance_length must exclude root")
    if correction_token != target_argmax[accepted_path[-1]]:
        raise TreeRuntimeReadinessError("correction token must equal target greedy token at last accepted node")

    expected = data.get("expected_accept")
    if expected:
        _compare_int_lists("accepted_path", accepted_path, expected["accepted_path"])
        if acceptance_length != int(expected["acceptance_length"]):
            raise TreeRuntimeReadinessError(
                f"acceptance_length mismatch: {acceptance_length} != {expected['acceptance_length']}"
            )
        if correction_token != int(expected["correction_token"]):
            raise TreeRuntimeReadinessError(
                f"correction_token mismatch: {correction_token} != {expected['correction_token']}"
            )

    return {
        "target_argmax_by_node": target_argmax,
        "accepted_path": accepted_path,
        "accepted_path_root_inclusive": True,
        "acceptance_length": acceptance_length,
        "acceptance_length_excludes_root": True,
        "accepted_draft_tokens": [tree.token_ids[idx] for idx in accepted_path[1:]],
        "correction_token": correction_token,
        "correction_token_source": "target_greedy_at_last_accepted_node",
        "duplicate_child_policy": _duplicate_child_policy(tree),
    }


def _ensure_sentinels_absent(post_hidden_cache: Sequence[Sequence[Any]], sentinels: Sequence[Any]) -> bool:
    flat = {float(x) for row in post_hidden_cache for x in row}
    leaked = [float(x) for x in sentinels if float(x) in flat]
    if leaked:
        raise TreeRuntimeReadinessError(f"rejected branch hidden sentinels leaked after commit: {leaked}")
    return True


def _validate_commit_and_gather(tree: DraftTree, accept: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    accepted_path = [int(x) for x in accept["accepted_path"]]
    correction_token = int(accept["correction_token"])
    commit = committed_hidden_cache.commit_tree_hidden_round(
        pre_hidden_cache=data["pre_hidden_cache"],
        tree_token_ids=tree.token_ids,
        node_hidden_rows=data["node_hidden_rows"],
        accepted_path=accepted_path,
        correction_token=correction_token,
        width=int(data["hidden_width"]),
        pre_committed_token_ids=data.get("pre_committed_token_ids"),
    ).as_dict()

    expected_append_tokens = [tree.token_ids[idx] for idx in accepted_path[1:]] + [correction_token]
    _compare_int_lists("committed_append_tokens", commit["committed_append_tokens"], expected_append_tokens)
    _compare_int_lists("appended_node_indices", commit["appended_node_indices"], accepted_path)
    if commit["correction_hidden_appended"]:
        raise TreeRuntimeReadinessError("correction hidden must not be appended in the same round")

    max_len = int(data["kv_max_len"])
    gather_positions = [max_len + idx for idx in accepted_path]
    expected_gather_positions = [int(x) for x in data.get("expected_gather_positions", gather_positions)]
    _compare_int_lists("gather_positions", gather_positions, expected_gather_positions)
    sentinels_absent = _ensure_sentinels_absent(commit["post_hidden_cache"], data.get("rejected_sentinels", []))

    return {
        "commit": commit,
        "token_commit_contract": "[accepted draft tokens | correction]",
        "hidden_kv_commit_contract": "[root | accepted] only",
        "correction_hidden_appended_same_round": False,
        "rejected_tree_nodes_unreachable_after_commit": sentinels_absent,
        "gather": {
            "max_len": max_len,
            "accepted_path": accepted_path,
            "positions": gather_positions,
            "contract": "max_len + accepted_path",
        },
    }


def _validate_boundary(data: Mapping[str, Any]) -> dict[str, Any]:
    boundary = data.get("readiness_boundary")
    if not isinstance(boundary, Mapping):
        raise TreeRuntimeReadinessError("readiness_boundary object is required")
    required_false = [
        "draft_head_graph_executed",
        "llama_context_runtime_instantiated",
        "draft_tokens_emitted",
        "kv_cache_mutated",
        "server_route_added",
    ]
    for key in required_false:
        if bool(boundary.get(key, True)):
            raise TreeRuntimeReadinessError(f"readiness boundary requires {key}=false")
    if bool(boundary.get("runtime_supported", True)):
        raise TreeRuntimeReadinessError("preview GGUF must remain runtime_supported=false")
    return {
        "runtime_supported": False,
        "preview_gguf_runtime_supported": False,
        "draft_head_graph_executed": False,
        "llama_context_runtime_instantiated": False,
        "draft_tokens_emitted": False,
        "kv_cache_mutated": False,
        "server_route_added": False,
        "no_draft_runtime_execution": True,
    }


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    topk = _validate_full_vocab_topk(data)
    tree = build_accum_logp_tree(int(data["root_token"]), data["topk_by_depth"], int(data["budget"]))
    tree_abi = _validate_tree_abi(tree, data)
    mask = _validate_mask_abi(tree, data)
    accept = _validate_accept_abi(tree, data)
    commit_gather = _validate_commit_and_gather(tree, accept, data)
    boundary = _validate_boundary(data)

    return {
        "ok": True,
        "status": STATUS,
        "topk": topk,
        "tree": tree_abi,
        "mask": mask,
        "accept": accept,
        "commit": commit_gather["commit"],
        "gather": commit_gather["gather"],
        "runtime_readiness_contract": {
            "draft_tree_abi": "token_ids,parent_indices,depth,num_nodes,parent_before_child",
            "topk_build_abi": "full_vocab_softmax_logprobs_no_topk_renormalization",
            "mask_abi": "prefix_visible_and_tree_ancestor_only",
            "accept_abi": "root_inclusive_path_length_excludes_root",
            "commit_abi": commit_gather["token_commit_contract"],
            "hidden_kv_commit_abi": commit_gather["hidden_kv_commit_contract"],
            "gather_abi": "max_len + accepted_path",
            "rejected_tree_nodes_unreachable_after_commit": commit_gather["rejected_tree_nodes_unreachable_after_commit"],
        },
        "readiness_boundary": boundary,
        "upstream_contract_refs": [
            "DraftTree",
            "build_from_topk/AccumLogP",
            "build_ancestor_matrix",
            "tree_accept",
            "PagedKVCache.gather max_len + accepted_path",
            "paged_tree_attn qq_bias",
        ],
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
    except (OSError, KeyError, TypeError, ValueError, TreeRuntimeReadinessError) as exc:
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
