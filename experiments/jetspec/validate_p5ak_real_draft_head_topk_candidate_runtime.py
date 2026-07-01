#!/usr/bin/env python3
"""Validate JetSpec P5AK real draft-head top-k candidate ABI runtime source slice."""

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
        "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY",
        "JETSPEC_REAL_DRAFT_HEAD_TOPK_CANDIDATE_RUNTIME_PHASE",
        "JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE",
        "JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS",
        "real_draft_head_topk_candidate_ready",
        "invalid_real_draft_head_topk_candidate_runtime",
        "p5ak_real_draft_head_topk_candidate_enabled",
        "real_draft_head_topk_candidate_runtime_ready",
        "real_draft_head_topk_candidate_hash_last",
        "real_draft_head_topk_candidate_seq_id_last",
        "n_real_draft_head_topk_candidate_runtime_builds",
        "real_draft_head_topk_candidate_ids",
        "real_draft_head_topk_candidate_logits",
        "build_real_draft_head_topk_candidate_runtime",
        "!p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict()",
        "!p5aj_real_draft_head_logits_canary_enabled || !real_draft_head_logits_canary_ready",
        "!p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled",
        "topk_accept_boundary_runtime_ready",
        "topk_verify_mask_runtime_ready",
        "topk_tree_runtime_ready",
        "real_draft_head_canary_ctx_present_last != 1",
        "real_draft_head_canary_decode_rc_last != 0",
        "real_draft_head_canary_logits_rows_last <= 0",
        "real_draft_head_canary_output_width_last != JETSPEC_QWEN36_VOCAB_SIZE",
        "real_draft_head_canary_topk_k_last != JETSPEC_TOPK_ABI_WIDTH",
        "real_draft_head_topk_candidate_ids[0] = real_draft_head_canary_top1_id_last",
        "real_draft_head_topk_candidate_ids[1] = real_draft_head_canary_top2_id_last",
        "real_draft_head_topk_candidate_logits[0] = real_draft_head_canary_top1_logit_last",
        "real_draft_head_topk_candidate_logits[1] = real_draft_head_canary_top2_logit_last",
        "real_draft_head_topk_parent_node_last = 0",
        "real_draft_head_topk_candidate_nodes_last = JETSPEC_TOPK_ABI_NON_ROOT_NODES",
        "real_draft_head_topk_verified_logits_rows_last = real_draft_head_canary_logits_rows_last",
        "p5ak_real_draft_head_topk_candidate_runtime",
        "real_topk_candidate_runtime_ready=%d",
        "logits_source=%s",
        "ctx_dft_present=%d",
        "decode_rc=%d",
        "logits_rows=%d",
        "logits_width=%d",
        "actual_verified_logits_rows=%d",
        "topk_k=%d",
        "parent_node=%d",
        "candidate_nodes=%d",
        "candidate_ids=[%d,%d]",
        "candidate_logits=[%.6g,%.6g]",
        "rank_semantics=%s",
        "accept_path_len=%d",
        "actual_accepted_nodes=%d",
        "correction_token_present=%d",
        "no_external_logits_walk=1",
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
        "P5AK real draft-head top-k candidate ABI metadata",
        "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY=1",
        "LLAMA_JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY=1",
        "LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1",
        "real_topk_candidate_runtime_ready=1",
        "logits_source=draft_head_full_vocab_logits",
        "actual_verified_logits_rows=1",
        "candidate_nodes=2",
        "rank_semantics=rank_stable_descending_logit",
        "P5AJ top1/top2",
        "P5AG accept-boundary",
        "no sampler",
        "no target logits walk",
        "no target accept walk",
        "no KV mutation",
        "no draft tokens",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5ak_real_draft_head_topk_candidate_runtime_candidate.md"): [
        "JetSpec P5AK real draft-head top-k candidate ABI runtime candidate",
        "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY=1",
        "LLAMA_JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY=1",
        "LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1",
        "real_topk_candidate_runtime_ready=1",
        "logits_source=draft_head_full_vocab_logits",
        "ctx_dft_present=1",
        "decode_rc=0",
        "actual_verified_logits_rows=1",
        "topk_k=2",
        "parent_node=0",
        "candidate_nodes=2",
        "candidate_ids=[top1,top2]",
        "candidate_logits=[top1_logit,top2_logit]",
        "rank_semantics=rank_stable_descending_logit",
        "actual_accepted_nodes=0",
        "correction_token_present=0",
        "P5AJ top1/top2",
        "P5AG readiness boundary",
        "no sampler",
        "no target logits walk",
        "no target accept walk",
        "no token commit",
        "no KV mutation",
        "no draft tokens",
    ],
    pathlib.Path("experiments/jetspec/probe_p5ak_real_draft_head_topk_candidate_trace.py"): [
        "p5ak_real_draft_head_topk_candidate_trace_contract_verified",
        "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY",
        "candidate_ids=[46746,128519]",
        "candidate_logits=[5.72088,5.48563]",
        "no_external_logits_walk=1",
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
    "build_accept_path_descriptor",
    "build_token_commit_descriptor",
    "build_hidden_kv_survivor_commit_descriptor",
    "build_rejected_branch_discard_descriptor",
    "build_publish_gate_descriptor",
]

CMAKE_TOKENS = [
    "p5ak_real_draft_head_topk_candidate_runtime",
    "probe_p5ak_real_draft_head_topk_candidate_trace",
    "test_p5ak_real_draft_head_topk_candidate_trace_probe",
    "validate_p5ak_real_draft_head_topk_candidate_runtime",
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY",
    "JETSPEC_REAL_DRAFT_HEAD_TOPK_CANDIDATE_RUNTIME_PHASE",
    "p5ak_real_draft_head_topk_candidate_trace_contract_verified",
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


def validate_p5ak_real_draft_head_topk_candidate_runtime() -> dict[str, Any]:
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
    builder = _function_slice(source, "build_real_draft_head_topk_candidate_runtime")
    if not builder:
        errors.append("cannot isolate build_real_draft_head_topk_candidate_runtime")
    else:
        for token in FORBIDDEN_IN_BUILDER:
            if token in builder:
                errors.append(f"P5AK builder must not contain {token}")
    if "if (p5ak_real_draft_head_topk_candidate_enabled) {" not in source:
        errors.append("P5AK process branch missing")
    if "if (!build_real_draft_head_topk_candidate_runtime())" not in source:
        errors.append("P5AK process branch must fail closed through its builder")
    if source.find("if (p5ak_real_draft_head_topk_candidate_enabled)") <= source.find("if (p5ag_topk_accept_boundary_enabled)"):
        errors.append("P5AK must be nested after the P5AG accept-boundary build")
    if "common_sampler_sample" in builder or "llama_sampler" in builder:
        errors.append("P5AK builder must not sample")
    errors.extend(_cmake_violations())

    return {
        "ok": not errors,
        "status": "p5ak_real_draft_head_topk_candidate_runtime_validated" if not errors else "p5ak_real_draft_head_topk_candidate_runtime_invalid",
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
    out = validate_p5ak_real_draft_head_topk_candidate_runtime()
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
