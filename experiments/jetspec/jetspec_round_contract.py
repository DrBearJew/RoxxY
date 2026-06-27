#!/usr/bin/env python3
"""Inert end-to-end JetSpec round contract fixture.

This composes existing pure-Python staging contracts:

1. build an accum-logp draft tree;
2. build/validate the tree-causal verify mask;
3. run greedy tree_accept over target argmax rows;
4. commit accepted target-hidden rows and discard rejected rows.

It does not call llama.cpp, PyTorch, CUDA, kernels, model weights, or server code.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

import committed_hidden_cache
import tree_verify_mask
from tree_semantics import DraftTree, build_accum_logp_tree, tree_accept


class RoundContractError(ValueError):
    """Raised when an end-to-end round fixture violates the staged contract."""


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_summary(tree: DraftTree) -> dict[str, Any]:
    return {
        "token_ids": list(tree.token_ids),
        "parent_indices": list(tree.parent_indices),
        "depth": list(tree.depth),
        "cum_logprob": list(tree.cum_logprob),
        "rank": list(tree.rank),
        "num_nodes": tree.num_nodes,
        "max_depth": tree.max_depth,
    }


def _ensure_no_rejected_sentinels(post_hidden_cache: Sequence[Sequence[float]], sentinels: Sequence[float]) -> None:
    flat = {float(x) for row in post_hidden_cache for x in row}
    leaked = [float(x) for x in sentinels if float(x) in flat]
    if leaked:
        raise RoundContractError(f"rejected hidden sentinel leaked into committed cache: {leaked}")


def _ensure_accepted_path_cannot_attend_rejected(ancestor: Sequence[Sequence[bool]], accepted_path: Sequence[int]) -> bool:
    accepted = {int(x) for x in accepted_path}
    leaks: list[tuple[int, int]] = []
    for query in accepted:
        for key in range(len(ancestor)):
            if key not in accepted and bool(ancestor[query][key]):
                leaks.append((query, key))
    if leaks:
        raise RoundContractError(f"accepted path can attend rejected tree nodes: {leaks}")
    return True


def _baseline_parity(data: dict[str, Any], committed_append_tokens: Sequence[int]) -> dict[str, Any] | None:
    if "baseline_greedy_token_ids" not in data:
        return None
    baseline = [int(x) for x in data["baseline_greedy_token_ids"]]
    jetspec = [int(x) for x in committed_append_tokens]
    if baseline != jetspec:
        raise RoundContractError(f"baseline greedy output mismatch: baseline={baseline}, jetspec={jetspec}")
    return {
        "baseline_greedy_token_ids": baseline,
        "jetspec_committed_append_tokens": jetspec,
        "greedy_output_matches_baseline": True,
    }


def _build_tree(data: dict[str, Any]) -> DraftTree:
    return build_accum_logp_tree(
        int(data["root_token"]),
        data["topk_by_depth"],
        int(data["budget"]),
    )


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    tree = _build_tree(data)
    past_len = int(data.get("past_len", 0))
    verify = tree_verify_mask.validate_mask_contract(tree, past_len)
    if not verify["ok"]:
        raise RoundContractError("tree verify mask validation failed: " + "; ".join(verify["errors"]))

    target_argmax_by_node = [int(x) for x in data["target_argmax_by_node"]]
    accepted_path, acceptance_length, correction_token = tree_accept(tree, target_argmax_by_node)

    expected_accept = data.get("expected_accept")
    if expected_accept is not None:
        expected_path = [int(x) for x in expected_accept.get("accepted_path", [])]
        expected_length = int(expected_accept.get("acceptance_length", -1))
        expected_correction = int(expected_accept.get("correction_token", -1))
        if accepted_path != expected_path:
            raise RoundContractError(f"accepted_path mismatch: {accepted_path} != {expected_path}")
        if acceptance_length != expected_length:
            raise RoundContractError(f"acceptance_length mismatch: {acceptance_length} != {expected_length}")
        if correction_token != expected_correction:
            raise RoundContractError(f"correction_token mismatch: {correction_token} != {expected_correction}")

    commit = committed_hidden_cache.commit_tree_hidden_round(
        pre_hidden_cache=data["pre_hidden_cache"],
        tree_token_ids=tree.token_ids,
        node_hidden_rows=data["node_hidden_rows"],
        accepted_path=accepted_path,
        correction_token=correction_token,
        width=int(data["hidden_width"]),
        pre_committed_token_ids=data.get("pre_committed_token_ids"),
    ).as_dict()

    _ensure_no_rejected_sentinels(commit["post_hidden_cache"], data.get("rejected_sentinels", []))
    accepted_path_isolated = _ensure_accepted_path_cannot_attend_rejected(verify["ancestor"], accepted_path)

    committed_before = [int(x) for x in data.get("pre_committed_token_ids", [])]
    committed_append_tokens = list(commit["committed_append_tokens"])
    committed_after = committed_before + committed_append_tokens
    baseline_parity = _baseline_parity(data, committed_append_tokens)
    accepted_tokens = [tree.token_ids[idx] for idx in accepted_path[1:]]
    rejected_node_indices = [idx for idx in range(tree.num_nodes) if idx not in set(accepted_path)]

    result = {
        "ok": True,
        "round_contract": {
            "pre_hidden_trails_committed_by_one": commit["invariants"]["pre_hidden_trails_committed_by_one"],
            "post_hidden_trails_committed_by_one": commit["invariants"]["post_hidden_trails_committed_by_one"],
            "correction_hidden_appended": commit["correction_hidden_appended"],
            "rejected_sentinels_absent": True,
            "accepted_path_isolated_from_rejected_nodes": accepted_path_isolated,
            "prefix_visible_to_all": all(row[:past_len] == [True] * past_len for row in verify["verify_allowed_mask"]),
        },
        "tree": _tree_summary(tree),
        "verify": {
            "past_len": past_len,
            "ancestor": verify["ancestor"],
            "verify_allowed_mask": verify["verify_allowed_mask"],
            "qq_bias": verify["qq_bias"],
            "sibling_pairs": verify["sibling_pairs"],
        },
        "accept": {
            "target_argmax_by_node": target_argmax_by_node,
            "accepted_path": accepted_path,
            "acceptance_length": acceptance_length,
            "accepted_tokens": accepted_tokens,
            "correction_token": correction_token,
            "rejected_node_indices": rejected_node_indices,
        },
        "commit": {
            "pre_committed_token_ids": committed_before,
            "post_committed_token_ids": committed_after,
            "accepted_draft_tokens": commit["accepted_draft_tokens"],
            "committed_append_tokens": committed_append_tokens,
            "appended_node_indices": commit["appended_node_indices"],
            "discarded_node_indices": commit["discarded_node_indices"],
            "pre_hidden_len": commit["pre_hidden_len"],
            "post_hidden_len": commit["post_hidden_len"],
            "post_hidden_cache": commit["post_hidden_cache"],
        },
    }
    if baseline_parity is not None:
        result["parity"] = baseline_parity
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        out = evaluate_fixture(_load_json(args.fixture.resolve()))
    except (OSError, KeyError, TypeError, ValueError, RoundContractError) as exc:
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
