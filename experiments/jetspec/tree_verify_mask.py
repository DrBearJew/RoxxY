#!/usr/bin/env python3
"""Inert JetSpec tree-causal verify mask helpers.

This models the verify-time attention contract only. It does not call llama.cpp,
PyTorch, CUDA, or JetSpec kernels.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

from tree_semantics import DraftTree, build_ancestor_matrix

ZERO = "0"
NEG_INF = "-inf"


class TreeVerifyMaskError(ValueError):
    """Raised when a tree verify mask fixture violates the contract."""


def tree_from_obj(obj: dict[str, Any]) -> DraftTree:
    tree_obj = obj.get("tree", obj)
    return DraftTree(
        token_ids=[int(x) for x in tree_obj["token_ids"]],
        parent_indices=[int(x) for x in tree_obj["parent_indices"]],
        depth=[int(x) for x in tree_obj["depth"]],
        cum_logprob=[float(x) for x in tree_obj.get("cum_logprob", [0.0] * len(tree_obj["token_ids"]))],
        rank=[int(x) for x in tree_obj.get("rank", [-1] * len(tree_obj["token_ids"]))],
    )


def build_verify_allowed_mask(tree: DraftTree, past_len: int) -> list[list[bool]]:
    """Return dense verify mask for query tree rows against prefix + tree keys.

    Shape: `(N, past_len + N)`. Prefix keys are visible to every query node.
    Tree-key visibility is exactly the ancestor relation, including self.
    """

    if past_len < 0:
        raise TreeVerifyMaskError(f"past_len must be non-negative: {past_len}")
    ancestor = build_ancestor_matrix(tree)
    return [[True] * past_len + list(row) for row in ancestor]


def build_qq_bias(tree: DraftTree) -> list[list[str]]:
    """Return tree-node additive bias block as JSON-safe `0` / `-inf` tokens."""

    return bool_mask_to_bias(build_ancestor_matrix(tree))


def bool_mask_to_bias(mask: Sequence[Sequence[bool]]) -> list[list[str]]:
    return [[ZERO if bool(v) else NEG_INF for v in row] for row in mask]


def pad_qq_bias_to_bucket(qq_bias: Sequence[Sequence[str]], bucket_size: int) -> list[list[str]]:
    """Pad an `(N,N)` tree bias to `(B,B)` with real rows isolated from pad rows.

    Mirrors upstream bucket semantics for compiled verify: real block unchanged;
    every real-to-pad, pad-to-real, and pad-to-pad entry is `-inf` except each pad
    row's self edge, which is `0` to avoid all-masked softmax rows.
    """

    n = len(qq_bias)
    if bucket_size < n:
        raise TreeVerifyMaskError(f"bucket_size={bucket_size} must be >= N={n}")
    for row in qq_bias:
        if len(row) != n:
            raise TreeVerifyMaskError("qq_bias must be square")
        bad = [v for v in row if v not in (ZERO, NEG_INF)]
        if bad:
            raise TreeVerifyMaskError(f"qq_bias entries must be {ZERO!r}/{NEG_INF!r}, got {bad[:3]}")
    out = [[NEG_INF for _ in range(bucket_size)] for _ in range(bucket_size)]
    for i in range(n):
        for j in range(n):
            out[i][j] = str(qq_bias[i][j])
    for i in range(n, bucket_size):
        out[i][i] = ZERO
    return out


def can_attend_tree_node(tree: DraftTree, query_node: int, key_node: int) -> bool:
    n = len(tree.token_ids)
    if query_node < 0 or query_node >= n or key_node < 0 or key_node >= n:
        raise TreeVerifyMaskError(f"query/key out of range for N={n}: {query_node}, {key_node}")
    return build_ancestor_matrix(tree)[query_node][key_node]


def sibling_pairs(tree: DraftTree) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    by_parent: dict[int, list[int]] = {}
    for idx, parent in enumerate(tree.parent_indices):
        if idx == 0:
            continue
        by_parent.setdefault(parent, []).append(idx)
    for siblings in by_parent.values():
        for i in siblings:
            for j in siblings:
                if i != j:
                    pairs.append((i, j))
    return pairs


def assert_sibling_isolation(tree: DraftTree) -> None:
    leaks = [(q, k) for q, k in sibling_pairs(tree) if can_attend_tree_node(tree, q, k)]
    if leaks:
        raise TreeVerifyMaskError(f"sibling attention leak: {leaks}")


def validate_mask_contract(tree: DraftTree, past_len: int) -> dict[str, Any]:
    errors: list[str] = []
    allowed = build_verify_allowed_mask(tree, past_len)
    ancestor = build_ancestor_matrix(tree)
    n = len(tree.token_ids)

    if len(allowed) != n:
        errors.append(f"allowed row count mismatch: {len(allowed)} != {n}")
    for i, row in enumerate(allowed):
        if len(row) != past_len + n:
            errors.append(f"allowed row {i} width mismatch: {len(row)} != {past_len + n}")
        if row[:past_len] != [True] * past_len:
            errors.append(f"prefix is not fully visible for row {i}")
        if row[past_len:] != ancestor[i]:
            errors.append(f"tree block row {i} differs from ancestor matrix")

    for q, k in sibling_pairs(tree):
        if ancestor[q][k]:
            errors.append(f"sibling node {q} can attend sibling {k}")

    for q in range(n):
        for k in range(n):
            # If q is ancestor of k but not equal, parent/query must not attend descendant/key.
            if q != k and ancestor[k][q] and ancestor[q][k]:
                errors.append(f"descendant leak: ancestor query {q} can attend descendant key {k}")

    return {
        "ok": not errors,
        "errors": errors,
        "past_len": past_len,
        "num_nodes": n,
        "ancestor": ancestor,
        "verify_allowed_mask": allowed,
        "qq_bias": bool_mask_to_bias(ancestor),
        "sibling_pairs": [list(pair) for pair in sibling_pairs(tree)],
    }


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    tree = tree_from_obj(data)
    past_len = int(data.get("past_len", 0))
    out = validate_mask_contract(tree, past_len)
    if not out["ok"]:
        raise TreeVerifyMaskError("; ".join(out["errors"]))
    bucket_size = data.get("bucket_size")
    if bucket_size is not None:
        out["bucket_size"] = int(bucket_size)
        out["bucketed_qq_bias"] = pad_qq_bias_to_bucket(out["qq_bias"], int(bucket_size))
    return out


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        out = evaluate_fixture(_load_json(args.fixture.resolve()))
    except (OSError, KeyError, TypeError, ValueError, TreeVerifyMaskError) as exc:
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
