#!/usr/bin/env python3
"""Validate JetSpec P5AE synthetic top-k tree ABI runtime source slice."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

REQUIRED: dict[pathlib.Path, list[str]] = {
    pathlib.Path("common/speculative.cpp"): [
        "LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY",
        "JETSPEC_TOPK_TREE_RUNTIME_PHASE",
        "JETSPEC_SYNTHETIC_FULL_VOCAB_SOFTMAX",
        "JETSPEC_TOPK_ABI_WIDTH = 2",
        "JETSPEC_TOPK_ABI_DEPTH = 1",
        "JETSPEC_TOPK_ABI_NODES = 3",
        "JETSPEC_TOPK_ABI_NON_ROOT_NODES = 2",
        "p5ae_topk_tree_enabled",
        "topk_tree_runtime_ready",
        "topk_tree_runtime_hash_last",
        "topk_tree_runtime_seq_id_last",
        "n_topk_tree_runtime_builds",
        "invalid_topk_tree_runtime",
        "topk_abi_root_tail_conflict()",
        "std::array<llama_token, JETSPEC_QWEN36_DRAFT_BLOCK_SIZE>",
        "build_topk_tree_runtime",
        "tree_build_actual_nodes_last = JETSPEC_TOPK_ABI_NODES",
        "tree_token_ids[1] = (pre_round_root_token_last + 1) % JETSPEC_QWEN36_VOCAB_SIZE",
        "tree_token_ids[2] = (pre_round_root_token_last + 2) % JETSPEC_QWEN36_VOCAB_SIZE",
        "tree_parent_indices[1] = 0",
        "tree_parent_indices[2] = 0",
        "tree_depth[1] = 1",
        "tree_depth[2] = 1",
        "tree_rank[1] = 0",
        "tree_rank[2] = 1",
        "tree_cum_logprob[1] = -0.1f",
        "tree_cum_logprob[2] = -0.3f",
        "tree_build_node_budget_last = std::max(tree_build_node_budget_last, JETSPEC_TOPK_ABI_NODES)",
        "tree_build_node_budget_last < JETSPEC_TOPK_ABI_NODES",
        "p5ae_topk_tree_runtime",
        "topk_logprob_source=%s",
        "topk_width=%d",
        "topk_depth=%d",
        "actual_tree_nodes=%d",
        "tree_token_ids=[%d,%d,%d]",
        "tree_parent_indices=[%d,%d,%d]",
        "parent_before_child=1",
        "num_nodes_lte_budget=1",
        "non_root_nodes=%d",
        "no_draft_logits=1",
        "no_verify_mask=1",
        "no_accept=1",
        "no_token_commit=1",
        "no_kv_mutation=1",
        "no_publish=1",
        "no_draft_tokens=1",
    ],
    pathlib.Path("docs/speculative.md"): [
        "P5AE top-k tree ABI materialization",
        "LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1",
        "synthetic_full_vocab_softmax",
        "actual_tree_nodes=3",
        "tree_parent_indices=[-1,0,0]",
        "tree_depth=[0,1,1]",
        "tree_rank=[-1,0,1]",
        "tree_cum_logprob=[0.0,-0.1,-0.3]",
        "non_root_nodes=2",
        "no draft logits",
        "no verify mask",
        "no KV mutation",
        "no draft tokens",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5ae_topk_tree_runtime_candidate.md"): [
        "JetSpec P5AE top-k tree ABI runtime candidate",
        "LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1",
        "topk_tree_runtime_ready=1",
        "synthetic_full_vocab_softmax",
        "actual_tree_nodes=3",
        "tree_parent_indices=[-1,0,0]",
        "returns before verify-mask",
        "no draft logits",
        "no KV mutation",
        "no draft tokens",
    ],
}

FORBIDDEN_IN_BRANCH = [
    "llama_decode",
    "llama_graph",
    "tree_accept",
    "llama_kv_cache",
    "result->push_back",
    "build_root_only_verify_mask_runtime()",
    "build_root_anchor_accept_path_runtime()",
    "build_root_token_commit_noop_runtime()",
    "build_root_hidden_kv_commit_noop_runtime()",
    "build_root_rejected_branch_discard_noop_runtime()",
    "build_root_publish_gate_noop_runtime()",
    "build_verify_mask_descriptor()",
    "build_accept_path_descriptor()",
    "build_token_commit_descriptor()",
]

CMAKE_TOKENS = [
    "p5ae_topk_tree_runtime",
    "validate_p5ae_topk_tree_runtime",
    "LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY",
    "JETSPEC_TOPK_TREE_RUNTIME_PHASE",
]


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(str(rel))
    return path.read_text(encoding="utf-8", errors="replace")


def _function_slice(source: str, name: str) -> str:
    start = source.find(f"bool {name}()")
    if start < 0:
        return ""
    next_fn = source.find("\n    bool ", start + 1)
    if next_fn < 0:
        next_fn = source.find("\n    void ", start + 1)
    return source[start: next_fn if next_fn > start else len(source)]


def _cmake_violations() -> list[str]:
    out: list[str] = []
    for path in list(REPO_ROOT.rglob("CMakeLists.txt")) + list(REPO_ROOT.rglob("*.cmake")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in CMAKE_TOKENS:
            if token in text:
                out.append(f"{path.relative_to(REPO_ROOT)} contains {token}")
    return out


def validate_p5ae_topk_tree_runtime() -> dict[str, Any]:
    errors: list[str] = []
    for rel, tokens in REQUIRED.items():
        try:
            text = _read(rel)
        except FileNotFoundError:
            errors.append(f"missing required file: {rel}")
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel}: missing token {token!r}")
    source = _read(pathlib.Path("common/speculative.cpp"))
    branch = _function_slice(source, "build_topk_tree_runtime")
    if not branch:
        errors.append("cannot isolate build_topk_tree_runtime")
    else:
        for token in FORBIDDEN_IN_BRANCH:
            if token in branch:
                errors.append(f"P5AE builder must not contain {token}")
    if "if (p5ae_topk_tree_enabled && (!p5x_root_tree_enabled || topk_abi_root_tail_conflict()))" not in source:
        errors.append("P5AE must fail closed without P5X or when root-tail gates conflict")
    if "if (p5ae_topk_tree_enabled) {" not in source:
        errors.append("P5AE process branch missing")
    errors.extend(_cmake_violations())
    return {
        "ok": not errors,
        "status": "p5ae_topk_tree_runtime_validated" if not errors else "p5ae_topk_tree_runtime_invalid",
        "errors": errors,
        "runtime_executed": False,
        "draft_tokens_emitted": False,
        "kv_mutated": False,
        "published_visible_state": False,
        "files_checked": [str(p) for p in REQUIRED],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = validate_p5ae_topk_tree_runtime()
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
