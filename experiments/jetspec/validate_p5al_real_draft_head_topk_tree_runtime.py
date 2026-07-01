#!/usr/bin/env python3
"""Validate JetSpec P5AL real draft-head top-k shadow-tree ABI runtime slice."""

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
        "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_ABI_ONLY",
        "JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_RUNTIME_PHASE",
        "real_draft_head_topk_tree_ready",
        "invalid_real_draft_head_topk_tree_runtime",
        "p5al_real_draft_head_topk_tree_enabled",
        "real_draft_head_topk_tree_runtime_ready",
        "real_draft_head_topk_tree_hash_last",
        "real_draft_head_topk_tree_seq_id_last",
        "n_real_draft_head_topk_tree_runtime_builds",
        "real_draft_head_topk_tree_token_ids",
        "real_draft_head_topk_tree_parent_indices",
        "real_draft_head_topk_tree_depth",
        "real_draft_head_topk_tree_rank",
        "real_draft_head_topk_tree_cum_logit",
        "reset_real_draft_head_topk_tree_arrays",
        "build_real_draft_head_topk_tree_runtime",
        "!p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled",
        "real_draft_head_topk_candidate_runtime_ready",
        "real_draft_head_topk_candidate_hash_last == 0",
        "real_draft_head_canary_logits_rows_last != 1",
        "real_draft_head_canary_output_rows_last != 1",
        "real_draft_head_topk_verified_logits_rows_last != 1",
        "topk_accept_boundary_runtime_ready",
        "topk_verify_mask_runtime_ready",
        "topk_tree_runtime_ready",
        "real_draft_head_topk_tree_token_ids[1] = real_draft_head_topk_candidate_ids[0]",
        "real_draft_head_topk_tree_token_ids[2] = real_draft_head_topk_candidate_ids[1]",
        "real_draft_head_topk_tree_cum_logit[1] = real_draft_head_topk_candidate_logits[0]",
        "real_draft_head_topk_tree_cum_logit[2] = real_draft_head_topk_candidate_logits[1]",
        "real_draft_head_topk_tree_runtime_ready = true",
        "p5al_real_draft_head_topk_tree_runtime",
        "real_topk_tree_runtime_ready=%d",
        "real_tree_token_ids=[%d,%d,%d]",
        "real_tree_parent_indices=[%d,%d,%d]",
        "real_tree_depth=[%d,%d,%d]",
        "real_tree_rank=[%d,%d,%d]",
        "real_tree_logits=[%.6g,%.6g,%.6g]",
        "no_synthetic_token_ids=1",
        "no_target_logits_walk=1",
        "no_target_accept_walk=1",
        "no_accept=1",
        "no_token_commit=1",
        "no_hidden_kv_commit=1",
        "no_rejected_branch_discard=1",
        "no_publish=1",
        "no_visible_state_change=1",
        "no_kv_mutation=1",
        "no_draft_tokens=1",
    ],
    pathlib.Path("docs/speculative.md"): [
        "P5AL real draft-head top-k tree ABI",
        "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_ABI_ONLY=1",
        "shadow real-tree ABI",
        "real_tree_token_ids=[root_token,top1,top2]",
        "real_tree_logits",
        "does not rewrite the existing P5AE/P5AF/P5AG synthetic tree hashes",
        "no accept",
        "no token commit",
        "no KV mutation",
        "no draft tokens",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5al_real_draft_head_topk_tree_runtime_candidate.md"): [
        "JetSpec P5AL real draft-head top-k tree ABI runtime candidate",
        "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_ABI_ONLY=1",
        "shadow real-tree ABI",
        "real_topk_tree_runtime_ready=1",
        "real_tree_token_ids=[root_token,top1,top2]",
        "real_tree_logits=[0,top1_logit,top2_logit]",
        "does not mutate the canonical P5AE synthetic tree arrays",
        "actual_accepted_nodes=0",
        "correction_token_present=0",
        "no target logits walk",
        "no token commit",
        "no KV mutation",
        "no draft tokens",
    ],
    pathlib.Path("experiments/jetspec/probe_p5al_real_draft_head_topk_tree_trace.py"): [
        "p5al_real_draft_head_topk_tree_trace_contract_verified",
        "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_ABI_ONLY",
        "real_tree_token_ids=[10240,46746,128519]",
        "candidate_ids=[46746,128519]",
        "real_tree_logits=[0,5.72088,5.48563]",
        "no_synthetic_token_ids=1",
        "no_draft_tokens=1",
    ],
}

FORBIDDEN_IN_BUILDER = [
    "llama_decode",
    "llama_graph",
    "llama_kv_cache",
    "result->push_back",
    "common_sampler_sample",
    "llama_sampler",
    "tree_accept",
    "build_token_commit_descriptor",
    "build_hidden_kv_survivor_commit_descriptor",
    "build_rejected_branch_discard_descriptor",
    "build_publish_gate_descriptor",
]

FORBIDDEN_CANONICAL_MUTATION_IN_BUILDER = [
    "\n        tree_token_ids[1] =",
    "\n        tree_token_ids[2] =",
    "\n        tree_cum_logprob[1] =",
    "\n        tree_cum_logprob[2] =",
]

CMAKE_TOKENS = [
    "jetspec_p5al_real_draft_head_topk_tree_runtime",
    "validate_p5al_real_draft_head_topk_tree_runtime",
    "probe_p5al_real_draft_head_topk_tree_trace",
    "test_p5al_real_draft_head_topk_tree_trace_probe",
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_ABI_ONLY",
    "JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_RUNTIME_PHASE",
    "real_tree_token_ids",
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


def validate_p5al_real_draft_head_topk_tree_runtime() -> dict[str, Any]:
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
    builder = _function_slice(source, "build_real_draft_head_topk_tree_runtime")
    if not builder:
        errors.append("cannot isolate build_real_draft_head_topk_tree_runtime")
    else:
        for token in FORBIDDEN_IN_BUILDER:
            if token in builder:
                errors.append(f"P5AL builder must not contain {token}")
        for token in FORBIDDEN_CANONICAL_MUTATION_IN_BUILDER:
            if token in builder:
                errors.append(f"P5AL builder must not mutate canonical synthetic state: {token}")
    if "if (p5al_real_draft_head_topk_tree_enabled && (!p5ak_real_draft_head_topk_candidate_enabled" not in source:
        errors.append("P5AL fail-closed gate must require P5AK/P5AJ/P5AG/P5AF/P5AE/P5X chain")
    if "if (!build_real_draft_head_topk_tree_runtime())" not in source:
        errors.append("P5AL process branch must fail closed through its builder")
    errors.extend(_cmake_violations())

    return {
        "ok": not errors,
        "status": "p5al_real_draft_head_topk_tree_runtime_validated" if not errors else "p5al_real_draft_head_topk_tree_runtime_invalid",
        "errors": errors,
        "runtime_executed": False,
        "model_loaded": False,
        "draft_tokens_emitted": False,
        "kv_mutated": False,
        "published_visible_state": False,
        "files_checked": [str(path) for path in REQUIRED],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = validate_p5al_real_draft_head_topk_tree_runtime()
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
