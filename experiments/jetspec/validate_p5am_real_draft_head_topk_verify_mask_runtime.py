#!/usr/bin/env python3
"""Validate the P5AM real draft-head top-k verify-mask ABI runtime candidate without loading a model."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SOURCE = REPO_ROOT / "common/speculative.cpp"
DOC = REPO_ROOT / "docs/speculative.md"
CANDIDATE = HERE / "jetspec_p5am_real_draft_head_topk_verify_mask_runtime_candidate.md"


def _function_slice(source: str, name: str) -> str:
    start = source.find(f"bool {name}()")
    if start < 0:
        return ""
    next_fn = source.find("\n    bool ", start + 1)
    if next_fn < 0:
        next_fn = source.find("\n    void ", start + 1)
    return source[start: next_fn if next_fn > start else len(source)]


def validate_p5am_real_draft_head_topk_verify_mask_runtime() -> dict[str, Any]:
    source = SOURCE.read_text(encoding="utf-8", errors="replace")
    doc = DOC.read_text(encoding="utf-8", errors="replace") if DOC.exists() else ""
    candidate = CANDIDATE.read_text(encoding="utf-8", errors="replace") if CANDIDATE.exists() else ""
    errors: list[str] = []

    required_source = [
        'common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_VERIFY_MASK_ABI_ONLY")',
        "JETSPEC_REAL_DRAFT_HEAD_TOPK_VERIFY_MASK_RUNTIME_PHASE",
        "real_draft_head_topk_verify_mask_ready",
        "invalid_real_draft_head_topk_verify_mask_runtime",
        "p5am_real_draft_head_topk_verify_mask_enabled",
        "real_draft_head_topk_verify_mask_runtime_ready",
        "real_draft_head_topk_verify_mask_hash_last",
        "real_draft_head_topk_verify_mask_seq_id_last",
        "real_draft_head_topk_verify_mask_entries_last",
        "real_draft_head_topk_verify_mask_rows",
        "real_draft_head_topk_verify_mask_cols",
        "real_draft_head_topk_verify_mask_values",
        "build_real_draft_head_topk_verify_mask_runtime",
        "p5am_real_draft_head_topk_verify_mask_enabled && (!p5al_real_draft_head_topk_tree_enabled",
        "if (!build_real_draft_head_topk_verify_mask_runtime())",
        "draft-jetspec p5am_real_draft_head_topk_verify_mask_runtime",
        "allowed_edges=[0:0,1:0,1:1,2:0,2:2]",
        "no_synthetic_mask_mutation=1",
        "no_mask_tensor=1",
    ]
    for token in required_source:
        if token not in source:
            errors.append(f"missing source token: {token}")

    builder = _function_slice(source, "build_real_draft_head_topk_verify_mask_runtime")
    if not builder:
        errors.append("cannot isolate build_real_draft_head_topk_verify_mask_runtime")
    else:
        required_builder = [
            "!p5al_real_draft_head_topk_tree_enabled",
            "!real_draft_head_topk_tree_runtime_ready || real_draft_head_topk_tree_hash_last == 0",
            "real_draft_head_canary_logits_rows_last != 1",
            "real_draft_head_topk_tree_nodes_last != JETSPEC_TOPK_ABI_NODES",
            "real_draft_head_topk_tree_token_ids[1] != real_draft_head_topk_candidate_ids[0]",
            "real_draft_head_topk_verify_mask_entries_last = JETSPEC_TOPK_ABI_MASK_ENTRIES",
            "real_draft_head_topk_verify_mask_rows[0] = 0",
            "real_draft_head_topk_verify_mask_cols[0] = 0",
            "real_draft_head_topk_verify_mask_rows[1] = 1",
            "real_draft_head_topk_verify_mask_cols[1] = 0",
            "real_draft_head_topk_verify_mask_rows[2] = 1",
            "real_draft_head_topk_verify_mask_cols[2] = 1",
            "real_draft_head_topk_verify_mask_rows[3] = 2",
            "real_draft_head_topk_verify_mask_cols[3] = 0",
            "real_draft_head_topk_verify_mask_rows[4] = 2",
            "real_draft_head_topk_verify_mask_cols[4] = 2",
            "actual_committed_tokens_last != 0",
            "actual_publish_visible_state_last != 0",
            "JETSPEC_VERIFY_MASK_PHASE",
        ]
        for token in required_builder:
            if token not in builder:
                errors.append(f"P5AM builder missing invariant: {token}")
        for forbidden in [
            "llama_decode", "llama_graph", "llama_kv_cache", "result->push_back",
            "common_sampler_sample", "llama_sampler", "tree_accept",
            "actual_committed_tokens_last = 1", "actual_publish_visible_state_last = 1",
            "\n        root_verify_mask_rows[", "\n        root_verify_mask_cols[", "\n        root_verify_mask_values[",
        ]:
            if forbidden in builder:
                errors.append(f"forbidden P5AM builder token present: {forbidden}")

    for text, name in [(doc, "docs/speculative.md"), (candidate, CANDIDATE.name)]:
        for token in [
            "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_VERIFY_MASK_ABI_ONLY=1",
            "shadow real-tree verify-mask ABI",
            "allowed_edges=[0:0,1:0,1:1,2:0,2:2]",
            "no mask tensor",
            "no accept",
            "no token commit",
            "no KV mutation",
            "no draft tokens",
        ]:
            if token not in text:
                errors.append(f"{name} missing token: {token}")

    return {
        "ok": not errors,
        "status": "p5am_real_draft_head_topk_verify_mask_runtime_validated" if not errors else "p5am_real_draft_head_topk_verify_mask_runtime_invalid",
        "errors": errors,
        "runtime_executed": False,
        "model_loaded": False,
        "context_created": False,
        "mask_tensor_allocated": False,
        "accepted_nodes": 0,
        "committed_tokens": 0,
        "kv_mutated": False,
        "published_visible_state": False,
        "draft_tokens_emitted": False,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = validate_p5am_real_draft_head_topk_verify_mask_runtime()
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
