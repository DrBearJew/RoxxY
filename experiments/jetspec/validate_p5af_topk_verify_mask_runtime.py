#!/usr/bin/env python3
"""Validate JetSpec P5AF synthetic top-k verify-mask ABI runtime source slice."""

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
        "LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY",
        "JETSPEC_TOPK_VERIFY_MASK_RUNTIME_PHASE",
        "JETSPEC_TOPK_ABI_MASK_ENTRIES = 5",
        "p5af_topk_verify_mask_enabled",
        "topk_verify_mask_runtime_ready",
        "topk_verify_mask_runtime_hash_last",
        "topk_verify_mask_runtime_seq_id_last",
        "n_topk_verify_mask_runtime_builds",
        "invalid_topk_verify_mask_runtime",
        "build_topk_verify_mask_runtime",
        "!p5ae_topk_tree_enabled || !topk_tree_runtime_ready",
        "actual_verify_mask_entries_last = JETSPEC_TOPK_ABI_MASK_ENTRIES",
        "root_verify_mask_rows[0] = 0",
        "root_verify_mask_cols[0] = 0",
        "root_verify_mask_rows[1] = 1",
        "root_verify_mask_cols[1] = 0",
        "root_verify_mask_rows[2] = 1",
        "root_verify_mask_cols[2] = 1",
        "root_verify_mask_rows[3] = 2",
        "root_verify_mask_cols[3] = 0",
        "root_verify_mask_rows[4] = 2",
        "root_verify_mask_cols[4] = 2",
        "allowed_edges=[0:0,1:0,1:1,2:0,2:2]",
        "prefix_visible=1",
        "ancestor_only=1",
        "root_attends_self=1",
        "child_attends_root=1",
        "child_attends_self=1",
        "sibling_visible=0",
        "descendant_visible=0",
        "no_mask_tensor=1",
        "no_accept=1",
        "no_token_commit=1",
        "no_kv_mutation=1",
        "no_publish=1",
        "no_draft_tokens=1",
    ],
    pathlib.Path("docs/speculative.md"): [
        "P5AF top-k verify-mask ABI materialization",
        "LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1",
        "actual_verify_mask_entries=5",
        "allowed_edges=[0:0,1:0,1:1,2:0,2:2]",
        "ancestor-only",
        "no mask tensor",
        "no accept",
        "no KV mutation",
        "no draft tokens",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5af_topk_verify_mask_runtime_candidate.md"): [
        "JetSpec P5AF top-k verify-mask ABI runtime candidate",
        "LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1",
        "topk_verify_mask_runtime_ready=1",
        "actual_verify_mask_entries=5",
        "allowed_edges=[0:0,1:0,1:1,2:0,2:2]",
        "no mask tensor",
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
    "build_root_anchor_accept_path_runtime()",
    "build_root_token_commit_noop_runtime()",
    "build_root_hidden_kv_commit_noop_runtime()",
    "build_root_rejected_branch_discard_noop_runtime()",
    "build_root_publish_gate_noop_runtime()",
    "build_accept_path_descriptor()",
    "build_token_commit_descriptor()",
]

CMAKE_TOKENS = [
    "p5af_topk_verify_mask_runtime",
    "validate_p5af_topk_verify_mask_runtime",
    "LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY",
    "JETSPEC_TOPK_VERIFY_MASK_RUNTIME_PHASE",
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


def validate_p5af_topk_verify_mask_runtime() -> dict[str, Any]:
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
    branch = _function_slice(source, "build_topk_verify_mask_runtime")
    if not branch:
        errors.append("cannot isolate build_topk_verify_mask_runtime")
    else:
        for token in FORBIDDEN_IN_BRANCH:
            if token in branch:
                errors.append(f"P5AF builder must not contain {token}")
    if "if (p5af_topk_verify_mask_enabled && (!p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict()))" not in source:
        errors.append("P5AF must fail closed without P5AE/P5X or when root-tail gates conflict")
    if "if (p5af_topk_verify_mask_enabled) {" not in source:
        errors.append("P5AF process branch missing")
    errors.extend(_cmake_violations())
    return {
        "ok": not errors,
        "status": "p5af_topk_verify_mask_runtime_validated" if not errors else "p5af_topk_verify_mask_runtime_invalid",
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
    out = validate_p5af_topk_verify_mask_runtime()
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
