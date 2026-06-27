#!/usr/bin/env python3
"""Inert committed target-hidden cache contract for JetSpec tree verify.

This models the bookkeeping contract only. It does not touch llama.cpp runtime,
KV cache storage, CUDA kernels, or target model code.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_WIDTH = 10240


class HiddenCacheContractError(ValueError):
    """Raised when a hidden-cache commit fixture violates the contract."""


@dataclasses.dataclass(frozen=True)
class HiddenCacheCommitResult:
    pre_hidden_len: int
    post_hidden_len: int
    pre_committed_len: int | None
    post_committed_len: int | None
    accepted_path: tuple[int, ...]
    accepted_draft_tokens: tuple[int, ...]
    correction_token: int
    committed_append_tokens: tuple[int, ...]
    appended_node_indices: tuple[int, ...]
    discarded_node_indices: tuple[int, ...]
    correction_hidden_appended: bool
    post_hidden_cache: tuple[tuple[float, ...], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pre_hidden_len": self.pre_hidden_len,
            "post_hidden_len": self.post_hidden_len,
            "pre_committed_len": self.pre_committed_len,
            "post_committed_len": self.post_committed_len,
            "accepted_path": list(self.accepted_path),
            "accepted_draft_tokens": list(self.accepted_draft_tokens),
            "correction_token": self.correction_token,
            "committed_append_tokens": list(self.committed_append_tokens),
            "appended_node_indices": list(self.appended_node_indices),
            "discarded_node_indices": list(self.discarded_node_indices),
            "correction_hidden_appended": self.correction_hidden_appended,
            "post_hidden_cache": [list(row) for row in self.post_hidden_cache],
            "invariants": {
                "pre_hidden_trails_committed_by_one": None
                if self.pre_committed_len is None
                else self.pre_hidden_len == self.pre_committed_len - 1,
                "post_hidden_trails_committed_by_one": None
                if self.post_committed_len is None
                else self.post_hidden_len == self.post_committed_len - 1,
                "appended_rows_are_root_plus_accepted_nodes": len(self.appended_node_indices) == len(self.accepted_path),
                "correction_has_no_hidden_until_next_round": not self.correction_hidden_appended,
            },
        }


def _coerce_row(row: Sequence[Any], width: int, *, label: str) -> tuple[float, ...]:
    if isinstance(row, (str, bytes, bytearray)):
        raise HiddenCacheContractError(f"{label} must be a numeric sequence, got text")
    values = tuple(float(x) for x in row)
    if len(values) != width:
        raise HiddenCacheContractError(f"{label} width mismatch: expected {width}, got {len(values)}")
    return values


def _coerce_rows(rows: Sequence[Sequence[Any]], width: int, *, label: str) -> tuple[tuple[float, ...], ...]:
    return tuple(_coerce_row(row, width, label=f"{label}[{i}]") for i, row in enumerate(rows))


def _normalize_hidden_by_node(hidden_by_node: Mapping[int | str, Sequence[Any]], width: int) -> dict[int, tuple[float, ...]]:
    normalized: dict[int, tuple[float, ...]] = {}
    for key, row in hidden_by_node.items():
        node_idx = int(key)
        if node_idx in normalized:
            raise HiddenCacheContractError(f"duplicate hidden row for node {node_idx}")
        if node_idx < 0:
            raise HiddenCacheContractError(f"node index must be non-negative: {node_idx}")
        normalized[node_idx] = _coerce_row(row, width, label=f"node_hidden[{node_idx}]")
    return normalized


def _validate_pre_round_invariant(pre_hidden_len: int, pre_committed_len: int | None) -> None:
    if pre_committed_len is not None and pre_hidden_len != pre_committed_len - 1:
        raise HiddenCacheContractError(
            "pre-round hidden cache must trail committed tokens by one anchor: "
            f"hidden_len={pre_hidden_len}, committed_len={pre_committed_len}"
        )


def _validate_path(accepted_path: Sequence[int], num_nodes: int) -> tuple[int, ...]:
    path = tuple(int(x) for x in accepted_path)
    if not path:
        raise HiddenCacheContractError("accepted_path must be non-empty")
    if path[0] != 0:
        raise HiddenCacheContractError(f"accepted_path must be root-inclusive and start at 0, got {path}")
    if len(set(path)) != len(path):
        raise HiddenCacheContractError(f"accepted_path must not contain duplicate nodes: {path}")
    bad = [idx for idx in path if idx < 0 or idx >= num_nodes]
    if bad:
        raise HiddenCacheContractError(f"accepted_path contains out-of-range nodes for num_nodes={num_nodes}: {bad}")
    return path


def commit_tree_hidden_round(
    *,
    pre_hidden_cache: Sequence[Sequence[Any]],
    tree_token_ids: Sequence[int],
    node_hidden_rows: Mapping[int | str, Sequence[Any]],
    accepted_path: Sequence[int],
    correction_token: int,
    width: int = DEFAULT_WIDTH,
    pre_committed_token_ids: Sequence[int] | None = None,
) -> HiddenCacheCommitResult:
    """Append accepted tree hidden rows and discard rejected branch rows.

    Contract mirrors upstream JetSpec tree verify:

    - before a round, target hidden/KV trail committed tokens by one anchor;
    - verify produces hidden rows for every tree node, including root;
    - commit appends hidden rows for `[root | accepted nodes]` only;
    - rejected branch rows are discarded;
    - the correction token is appended to committed tokens but has no hidden row
      until it becomes the next round's root/anchor.
    """

    if width <= 0:
        raise HiddenCacheContractError(f"width must be positive: {width}")
    tokens = tuple(int(x) for x in tree_token_ids)
    if not tokens:
        raise HiddenCacheContractError("tree_token_ids must contain a root token")
    pre_rows = _coerce_rows(pre_hidden_cache, width, label="pre_hidden_cache")
    pre_committed = tuple(int(x) for x in pre_committed_token_ids) if pre_committed_token_ids is not None else None
    pre_committed_len = len(pre_committed) if pre_committed is not None else None
    _validate_pre_round_invariant(len(pre_rows), pre_committed_len)
    if pre_committed is not None and tokens[0] != pre_committed[-1]:
        raise HiddenCacheContractError(
            f"tree root token must equal committed anchor token: root={tokens[0]}, anchor={pre_committed[-1]}"
        )

    hidden_by_node = _normalize_hidden_by_node(node_hidden_rows, width)
    path = _validate_path(accepted_path, len(tokens))
    missing = [idx for idx in path if idx not in hidden_by_node]
    if missing:
        raise HiddenCacheContractError(f"missing hidden rows for accepted nodes: {missing}")

    appended_rows = tuple(hidden_by_node[idx] for idx in path)
    post_rows = pre_rows + appended_rows
    accepted_draft_tokens = tuple(tokens[idx] for idx in path[1:])
    correction = int(correction_token)
    committed_append_tokens = accepted_draft_tokens + (correction,)
    post_committed_len = None if pre_committed_len is None else pre_committed_len + len(committed_append_tokens)
    if post_committed_len is not None and len(post_rows) != post_committed_len - 1:
        raise HiddenCacheContractError(
            "post-round hidden cache must trail committed tokens by one correction anchor: "
            f"hidden_len={len(post_rows)}, committed_len={post_committed_len}"
        )

    discarded = tuple(sorted(idx for idx in hidden_by_node if idx not in set(path)))
    return HiddenCacheCommitResult(
        pre_hidden_len=len(pre_rows),
        post_hidden_len=len(post_rows),
        pre_committed_len=pre_committed_len,
        post_committed_len=post_committed_len,
        accepted_path=path,
        accepted_draft_tokens=accepted_draft_tokens,
        correction_token=correction,
        committed_append_tokens=committed_append_tokens,
        appended_node_indices=path,
        discarded_node_indices=discarded,
        correction_hidden_appended=False,
        post_hidden_cache=post_rows,
    )


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    result = commit_tree_hidden_round(
        pre_hidden_cache=data["pre_hidden_cache"],
        tree_token_ids=data["tree_token_ids"],
        node_hidden_rows=data["node_hidden_rows"],
        accepted_path=data["accepted_path"],
        correction_token=int(data["correction_token"]),
        width=int(data.get("width", DEFAULT_WIDTH)),
        pre_committed_token_ids=data.get("pre_committed_token_ids"),
    )
    return result.as_dict()


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
    except (OSError, KeyError, TypeError, ValueError, HiddenCacheContractError) as exc:
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
