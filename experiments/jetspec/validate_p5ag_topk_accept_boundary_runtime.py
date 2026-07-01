#!/usr/bin/env python3
"""Validate JetSpec P5AG synthetic top-k accept-boundary ABI runtime source slice."""

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
        "LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY",
        "JETSPEC_TOPK_ACCEPT_BOUNDARY_RUNTIME_PHASE",
        "JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_LOGITS",
        "invalid_topk_accept_boundary_runtime",
        "p5ag_topk_accept_boundary_enabled",
        "topk_accept_boundary_runtime_ready",
        "topk_accept_boundary_runtime_hash_last",
        "topk_accept_boundary_runtime_seq_id_last",
        "n_topk_accept_boundary_runtime_builds",
        "topk_accept_candidate_nodes_last",
        "topk_accept_boundary_verified_edges_last",
        "topk_actual_verified_logits_rows_last",
        "build_topk_accept_boundary_runtime",
        "!p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled",
        "topk_abi_root_tail_conflict()",
        "topk_verify_mask_runtime_ready",
        "topk_tree_runtime_ready",
        "tree_build_actual_nodes_last != JETSPEC_TOPK_ABI_NODES",
        "actual_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES",
        "topk_accept_candidate_nodes_last = JETSPEC_TOPK_ABI_NON_ROOT_NODES",
        "topk_accept_boundary_verified_edges_last = actual_verify_mask_entries_last",
        "topk_actual_verified_logits_rows_last = 0",
        "accept_path_len_last = 0",
        "actual_accepted_nodes_last = 0",
        "correction_token_present_last = 0",
        "topk_accept_candidate_nodes_last != 2",
        "topk_accept_boundary_verified_edges_last != 5",
        "topk_actual_verified_logits_rows_last != 0",
        "p5ag_topk_accept_boundary_runtime",
        "topk_accept_boundary_runtime_ready=%d",
        "accept_boundary_candidate_nodes=%d",
        "accept_boundary_verified_edges=%d",
        "actual_verified_logits_rows=%d",
        "accept_decision_source=%s",
        "accept_path_len=%d",
        "actual_accepted_nodes=%d",
        "correction_token_present=%d",
        "no_target_logits_walk=1",
        "no_target_accept_walk=1",
        "no_token_commit=1",
        "no_hidden_kv_commit=1",
        "no_rejected_branch_discard=1",
        "no_publish=1",
        "no_visible_state_change=1",
        "no_kv_mutation=1",
        "no_draft_head_graph=1",
        "no_draft_tokens=1",
    ],
    pathlib.Path("docs/speculative.md"): [
        "P5AG top-k accept-boundary ABI materialization",
        "LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1",
        "topk_accept_boundary_runtime_ready=1",
        "accept_boundary_candidate_nodes=2",
        "accept_boundary_verified_edges=5",
        "actual_verified_logits_rows=0",
        "accept_decision_source=none_no_logits",
        "actual_accepted_nodes=0",
        "correction_token_present=0",
        "no target logits walk",
        "no target accept walk",
        "no KV mutation",
        "no draft tokens",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5ag_topk_accept_boundary_runtime_candidate.md"): [
        "JetSpec P5AG top-k accept-boundary ABI runtime candidate",
        "LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1",
        "topk_accept_boundary_runtime_ready=1",
        "actual_tree_nodes=3",
        "actual_verify_mask_entries=5",
        "accept_boundary_candidate_nodes=2",
        "accept_boundary_verified_edges=5",
        "actual_verified_logits_rows=0",
        "accept_decision_source=none_no_logits",
        "accept_path_len=0",
        "actual_accepted_nodes=0",
        "correction_token_present=0",
        "no target logits walk",
        "no target accept walk",
        "no token commit",
        "no KV mutation",
        "no draft tokens",
    ],
}

FORBIDDEN_IN_BRANCH = [
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

FORBIDDEN_PATH_TOKENS = [
    "include/llama.h",
    "src/CMakeLists.txt",
    "ggml/src/",
    "tools/server/",
]


def _read(path: pathlib.Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8", errors="replace")


def validate_p5ag_topk_accept_boundary_runtime() -> dict[str, Any]:
    errors: list[str] = []
    for rel, tokens in REQUIRED.items():
        try:
            text = _read(rel)
        except OSError as exc:
            errors.append(f"cannot read {rel}: {exc}")
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"missing {token!r} in {rel}")

    source = _read(pathlib.Path("common/speculative.cpp"))
    branch_start = source.find("bool build_topk_accept_boundary_runtime()")
    branch_end = source.find("bool build_root_only_verify_mask_runtime()", branch_start)
    if branch_start < 0 or branch_end < 0 or branch_end <= branch_start:
        errors.append("cannot isolate P5AG top-k accept-boundary branch")
        branch = ""
    else:
        branch = source[branch_start:branch_end]
    for token in FORBIDDEN_IN_BRANCH:
        if token in branch:
            errors.append(f"P5AG branch must not contain {token}")
    for token in FORBIDDEN_PATH_TOKENS:
        if token in branch:
            errors.append(f"P5AG branch must not reference forbidden path/surface token {token}")
    if "if (p5ag_topk_accept_boundary_enabled)" not in source:
        errors.append("P5AG process branch must be gated")
    if "if (!build_topk_accept_boundary_runtime())" not in source:
        errors.append("P5AG process branch must fail closed through its builder")
    if source.find("if (p5ag_topk_accept_boundary_enabled)") < source.find("if (trace_taps) {\n                        LOG_INF(\"%s: draft-jetspec p5af_topk_verify_mask_runtime"):
        pass
    else:
        errors.append("P5AG process branch must run before the P5AF trace/return")

    return {
        "ok": not errors,
        "status": "p5ag_topk_accept_boundary_runtime_validated" if not errors else "p5ag_topk_accept_boundary_runtime_invalid",
        "errors": errors,
        "files_checked": [str(path) for path in REQUIRED],
        "runtime_executed": False,
        "draft_tokens_emitted": False,
        "kv_mutated": False,
        "published_visible_state": False,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = validate_p5ag_topk_accept_boundary_runtime()
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
