#!/usr/bin/env python3
"""Validate the P5AN real draft-head top-k accept-boundary no-model contract wiring."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DOC = REPO_ROOT / "docs/speculative.md"
README = HERE / "README.md"
CANDIDATE = HERE / "jetspec_p5an_real_draft_head_topk_accept_boundary_runtime_candidate.md"
PROBE = HERE / "probe_p5an_real_draft_head_topk_accept_boundary_trace.py"
TEST = HERE / "test_p5an_real_draft_head_topk_accept_boundary_trace_probe.py"
AGGREGATE = HERE / "run_all_jetspec_contracts.py"
SOURCE = REPO_ROOT / "common/speculative.cpp"

REQUIRED_DOC_TOKENS = [
    "P5AN real draft-head top-k accept-boundary ABI",
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_BOUNDARY_ABI_ONLY=1",
    "p5an_real_draft_head_topk_accept_boundary_runtime",
    "phase=real_draft_head_topk_accept_boundary_ready",
    "real_topk_accept_boundary_runtime_ready=1",
    "P5AM",
    "P5AL",
    "P5AK",
    "P5AJ",
    "P5AG",
    "P5AF",
    "P5AE",
    "P5X",
    "actual_verified_logits_rows=1",
    "accept_boundary_candidate_nodes=2",
    "accept_boundary_verified_edges=5",
    "allowed_edges=[0:0,1:0,1:1,2:0,2:2]",
    "accept_path_len=0",
    "actual_accepted_nodes=0",
    "correction_token_present=0",
    "no target logits walk",
    "no target accept walk",
    "no accept",
    "no token commit",
    "no KV mutation",
    "no publish",
    "no draft tokens",
]

REQUIRED_CANDIDATE_TOKENS = REQUIRED_DOC_TOKENS + [
    "real_tree_token_ids[1:] == candidate_ids",
    "real_verify_mask_rows=[0,1,1,2,2]",
    "real_verify_mask_cols=[0,0,1,0,2]",
    "real_verify_mask_values=[1,1,1,1,1]",
    "accept_decision_source=none_no_target_logits",
    "no mask tensor",
]

REQUIRED_PROBE_TOKENS = [
    "REQUIRED_TRACE_TOKENS",
    "FORBIDDEN_TRACE_TOKENS",
    "SELF_TEST_TRACE",
    "validate_trace_line",
    "probe_p5an_real_draft_head_topk_accept_boundary_trace",
    "p5an_real_draft_head_topk_accept_boundary_trace_contract_verified",
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_BOUNDARY_ABI_ONLY=1",
    "phase=real_draft_head_topk_accept_boundary_ready",
    "actual_verified_logits_rows=1",
    "allowed_edges=[0:0,1:0,1:1,2:0,2:2]",
    "actual_accepted_nodes=0",
]

REQUIRED_TEST_TOKENS = [
    "P5ANRealDraftHeadTopKAcceptBoundaryTraceProbeTests",
    "test_probe_passes_without_live_log",
    "test_trace_line_requires_real_tree_mask_and_accept_boundary_metadata",
    "test_validator_passes",
]

REQUIRED_AGGREGATE_TOKENS = [
    "test_p5an_real_draft_head_topk_accept_boundary_trace_probe.py",
    "validate_p5an_real_draft_head_topk_accept_boundary_runtime.py",
    "probe_p5an_real_draft_head_topk_accept_boundary_trace.py",
    "p5an_real_draft_head_topk_accept_boundary_trace_contract_verified",
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_BOUNDARY_ABI_ONLY",
]

FORBIDDEN_CONTRACT_TOKENS = [
    "llama_decode(",
    "llama_graph",
    "llama_kv_cache",
    "common_sampler_sample",
    "llama_sampler",
    "result->push_back",
]


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _require_tokens(text: str, tokens: list[str], label: str, errors: list[str]) -> None:
    for token in tokens:
        if token not in text:
            errors.append(f"{label} missing token: {token}")


def validate_p5an_real_draft_head_topk_accept_boundary_runtime() -> dict[str, Any]:
    errors: list[str] = []
    doc = _read(DOC)
    readme = _read(README)
    candidate = _read(CANDIDATE)
    probe = _read(PROBE)
    test = _read(TEST)
    aggregate = _read(AGGREGATE)
    source = _read(SOURCE)

    _require_tokens(doc, REQUIRED_DOC_TOKENS, "docs/speculative.md", errors)
    _require_tokens(readme, REQUIRED_DOC_TOKENS, "experiments/jetspec/README.md", errors)
    _require_tokens(candidate, REQUIRED_CANDIDATE_TOKENS, CANDIDATE.name, errors)
    _require_tokens(probe, REQUIRED_PROBE_TOKENS, PROBE.name, errors)
    _require_tokens(test, REQUIRED_TEST_TOKENS, TEST.name, errors)
    _require_tokens(aggregate, REQUIRED_AGGREGATE_TOKENS, AGGREGATE.name, errors)

    contract_text = "\n".join([candidate, probe, test])
    for token in FORBIDDEN_CONTRACT_TOKENS:
        if token in contract_text:
            errors.append(f"P5AN no-model contract must not contain runtime side-effect token: {token}")

    for path_token in ["tools/server/", "ggml/src/", "src/CMakeLists.txt", "include/llama.h"]:
        if path_token in contract_text:
            errors.append(f"P5AN no-model contract must not reference forbidden production path: {path_token}")

    return {
        "ok": not errors,
        "status": "p5an_real_draft_head_topk_accept_boundary_contract_wiring_validated" if not errors else "p5an_real_draft_head_topk_accept_boundary_contract_wiring_invalid",
        "errors": errors,
        "runtime_executed": False,
        "model_loaded": False,
        "context_created": False,
        "source_implementation_required_in_this_lane": False,
        "source_implementation_present": "p5an_real_draft_head_topk_accept_boundary_runtime" in source,
        "actual_verified_logits_rows": 1,
        "accept_path_len": 0,
        "actual_accepted_nodes": 0,
        "correction_token_present": 0,
        "committed_tokens": 0,
        "kv_mutated": False,
        "published_visible_state": False,
        "draft_tokens_emitted": False,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = validate_p5an_real_draft_head_topk_accept_boundary_runtime()
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
