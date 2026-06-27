#!/usr/bin/env python3
"""Pure Python JetSpec tree semantics prototype.

This module is intentionally staged under experiments/jetspec/ and is not used by
llama.cpp runtime builds. It mirrors the upstream JetSpec tree contract closely
enough to create deterministic fixtures before any C++/GGUF/runtime wiring:

- flat DraftTree nodes in parent-before-child order
- accum_logp best-first tree construction from per-depth top-k candidates
- dense ancestor matrix for tree-causal attention
- greedy tree_accept walk over target argmax tokens
"""

from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import heapq
import json
import math
import pathlib
import sys
from typing import Any, Iterable


@dataclass(frozen=True)
class DraftCandidate:
    token: int
    logprob: float
    rank: int = -1
    logit: float | None = None
    p: float | None = None

    @classmethod
    def from_obj(cls, obj: dict[str, Any] | list[Any] | tuple[Any, ...]) -> "DraftCandidate":
        if isinstance(obj, dict):
            return cls(
                token=int(obj["token"]),
                logprob=float(obj["logprob"]),
                rank=int(obj.get("rank", -1)),
                logit=None if obj.get("logit") is None else float(obj["logit"]),
                p=None if obj.get("p") is None else float(obj["p"]),
            )
        if len(obj) < 2:
            raise ValueError(f"candidate tuple needs at least token/logprob: {obj!r}")
        return cls(token=int(obj[0]), logprob=float(obj[1]), rank=int(obj[2]) if len(obj) > 2 else -1)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "token": self.token,
            "logprob": self.logprob,
            "rank": self.rank,
        }
        if self.logit is not None:
            out["logit"] = self.logit
        if self.p is not None:
            out["p"] = self.p
        return out


@dataclass
class DraftTree:
    token_ids: list[int]
    parent_indices: list[int]
    depth: list[int]
    cum_logprob: list[float]
    rank: list[int] = field(default_factory=list)
    ancestor: list[list[bool]] | None = None
    child_maps: list[dict[int, int]] | None = None

    def __post_init__(self) -> None:
        n = len(self.token_ids)
        if len(self.parent_indices) != n or len(self.depth) != n or len(self.cum_logprob) != n:
            raise ValueError("DraftTree fields must have equal length")
        if not self.rank:
            self.rank = [-1] * n
        if len(self.rank) != n:
            raise ValueError("rank length must match token_ids")
        if n == 0:
            raise ValueError("DraftTree must contain a root node")
        if self.parent_indices[0] != -1 or self.depth[0] != 0:
            raise ValueError("root must have parent=-1 and depth=0")
        for i in range(1, n):
            parent = self.parent_indices[i]
            if parent < 0 or parent >= i:
                raise ValueError(f"node {i} parent must be in [0,{i}); got {parent}")
            expected_depth = self.depth[parent] + 1
            if self.depth[i] != expected_depth:
                raise ValueError(f"node {i} depth {self.depth[i]} != parent depth + 1 ({expected_depth})")

    @property
    def num_nodes(self) -> int:
        return len(self.token_ids)

    @property
    def max_depth(self) -> int:
        return max(self.depth) if self.depth else 0

    def to_json(self) -> dict[str, Any]:
        return {
            "token_ids": self.token_ids,
            "parent_indices": self.parent_indices,
            "depth": self.depth,
            "rank": self.rank,
            "cum_logprob": self.cum_logprob,
            "ancestor": self.ancestor if self.ancestor is not None else build_ancestor_matrix(self),
            "num_nodes": self.num_nodes,
            "max_depth": self.max_depth,
        }


def normalize_topk(topk_by_depth: Iterable[Iterable[DraftCandidate | dict[str, Any] | list[Any] | tuple[Any, ...]]]) -> list[list[DraftCandidate]]:
    out: list[list[DraftCandidate]] = []
    for depth, row in enumerate(topk_by_depth):
        norm_row: list[DraftCandidate] = []
        for i, item in enumerate(row):
            cand = item if isinstance(item, DraftCandidate) else DraftCandidate.from_obj(item)
            if cand.rank < 0:
                cand = DraftCandidate(cand.token, cand.logprob, i, cand.logit, cand.p)
            if not math.isfinite(cand.logprob):
                raise ValueError(f"non-finite logprob at depth {depth}, rank {i}: {cand.logprob}")
            norm_row.append(cand)
        if not norm_row:
            raise ValueError(f"empty top-k row at depth {depth}")
        out.append(norm_row)
    return out


def build_accum_logp_tree(root_token: int, topk_by_depth: Iterable[Iterable[DraftCandidate | dict[str, Any] | list[Any] | tuple[Any, ...]]], budget: int) -> DraftTree:
    """Build JetSpec upstream-style accum_logp tree from per-depth top-k.

    `topk_by_depth[d]` contains candidates for draft depth `d + 1` and is shared
    by every parent at that depth. The heap key is cumulative log-probability.
    The output is parent-before-child, with root at node 0.
    """

    if budget < 1:
        raise ValueError(f"budget must be >= 1; got {budget}")
    topk = normalize_topk(topk_by_depth)

    token_ids = [int(root_token)]
    parent_indices = [-1]
    depth = [0]
    cum_logprob = [0.0]
    rank = [-1]

    # heap item: (negative cumulative logprob, insertion order, node index)
    counter = 0
    heap: list[tuple[float, int, int]] = [(0.0, counter, 0)]

    while heap and len(token_ids) < budget:
        neg_cum_lp, _, node_idx = heapq.heappop(heap)
        d = depth[node_idx]
        if d >= len(topk):
            continue
        children_to_add = min(len(topk[d]), budget - len(token_ids))
        parent_cum_lp = -neg_cum_lp
        for j in range(children_to_add):
            cand = topk[d][j]
            child_idx = len(token_ids)
            child_cum_lp = parent_cum_lp + cand.logprob
            token_ids.append(cand.token)
            parent_indices.append(node_idx)
            depth.append(d + 1)
            cum_logprob.append(child_cum_lp)
            rank.append(cand.rank)
            counter += 1
            heapq.heappush(heap, (-child_cum_lp, counter, child_idx))

    tree = DraftTree(
        token_ids=token_ids,
        parent_indices=parent_indices,
        depth=depth,
        rank=rank,
        cum_logprob=cum_logprob,
    )
    tree.ancestor = build_ancestor_matrix(tree)
    tree.child_maps = build_child_maps(tree)
    return tree


def build_ancestor_matrix(tree: DraftTree) -> list[list[bool]]:
    """Return dense bool ancestor matrix in parent-before-child order."""

    n = tree.num_nodes
    ancestor = [[False for _ in range(n)] for _ in range(n)]
    for i in range(n):
        ancestor[i][i] = True
    for i in range(1, n):
        parent = tree.parent_indices[i]
        if 0 <= parent < n:
            # Match upstream: row i inherits all ancestors of parent, plus self.
            ancestor[i] = ancestor[parent].copy()
            ancestor[i][i] = True
    return ancestor


def build_child_maps(tree: DraftTree) -> list[dict[int, int]]:
    """Build parent-token -> child-node lookup.

    Duplicate child tokens under the same parent follow upstream CPU semantics:
    later nodes overwrite earlier ones because dict assignment is used while
    scanning child indices in order.
    """

    child_maps: list[dict[int, int]] = [dict() for _ in range(tree.num_nodes)]
    for child_idx in range(1, tree.num_nodes):
        parent = tree.parent_indices[child_idx]
        if 0 <= parent < tree.num_nodes:
            child_maps[parent][tree.token_ids[child_idx]] = child_idx
    return child_maps


def tree_accept(tree: DraftTree, target_argmax_by_node: list[int]) -> tuple[list[int], int, int]:
    """Find longest target-greedy matching root-to-leaf path.

    Returns `(accepted_path, acceptance_length, correction_token)`, where
    `accepted_path` is root-inclusive and `acceptance_length` excludes root.
    """

    if len(target_argmax_by_node) < tree.num_nodes:
        raise ValueError(f"target_argmax_by_node has {len(target_argmax_by_node)} rows, expected {tree.num_nodes}")
    child_maps = tree.child_maps if tree.child_maps is not None else build_child_maps(tree)
    accepted_path = [0]
    current = 0
    while True:
        next_token = int(target_argmax_by_node[current])
        child_idx = child_maps[current].get(next_token)
        if child_idx is None:
            break
        accepted_path.append(child_idx)
        current = child_idx
    return accepted_path, len(accepted_path) - 1, int(target_argmax_by_node[current])


def _load_topk(path: pathlib.Path) -> list[list[DraftCandidate]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("topk_by_depth")
    if not isinstance(raw, list):
        raise ValueError("top-k JSON must be a list or object with topk_by_depth")
    return normalize_topk(raw)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an experimental JetSpec accum_logp tree from a top-k JSON fixture")
    parser.add_argument("--root-token", type=int, default=0)
    parser.add_argument("--topk-json", type=pathlib.Path, help="JSON list of per-depth candidates")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--target-argmax-json", type=pathlib.Path, help="optional JSON list of target argmax tokens by node")
    parser.add_argument("--output", type=pathlib.Path, help="optional JSON output path")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.topk_json is None:
        raise SystemExit("--topk-json is required")
    tree = build_accum_logp_tree(args.root_token, _load_topk(args.topk_json), args.budget)
    out: dict[str, Any] = {"tree": tree.to_json()}
    if args.target_argmax_json is not None:
        targets = [int(x) for x in json.loads(args.target_argmax_json.read_text(encoding="utf-8"))]
        path, acc, correction = tree_accept(tree, targets)
        out["accept"] = {
            "path": path,
            "acceptance_length": acc,
            "correction_token": correction,
        }
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
