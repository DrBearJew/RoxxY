#!/usr/bin/env python3
"""Validate the P5AR real draft-head top-k rejected-branch discard no-op no-model contract wiring."""

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
CANDIDATE = HERE / "jetspec_p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime_candidate.md"
PROBE = HERE / "probe_p5ar_real_draft_head_topk_rejected_branch_discard_noop_trace.py"
TEST = HERE / "test_p5ar_real_draft_head_topk_rejected_branch_discard_noop_trace_probe.py"
AGGREGATE = HERE / "run_all_jetspec_contracts.py"
PRODUCTION_SOURCE = REPO_ROOT / "common/speculative.cpp"

P5AR_TOKENS = [
    "P5AR real draft-head top-k rejected-branch discard no-op ABI",
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_REJECTED_BRANCH_DISCARD_NOOP_ABI_ONLY",
    "p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime",
    "real_draft_head_topk_rejected_branch_discard_noop_ready",
    "invalid_real_draft_head_topk_rejected_branch_discard_noop_runtime",
    "real_topk_rejected_branch_discard_noop_runtime_ready",
]

REQUIRED_SOURCE_TOKENS = [
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_REJECTED_BRANCH_DISCARD_NOOP_ABI_ONLY",
    "p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime",
    "real_draft_head_topk_rejected_branch_discard_noop_ready",
    "invalid_real_draft_head_topk_rejected_branch_discard_noop_runtime",
    "real_topk_rejected_branch_discard_noop_runtime_ready",
    "real_topk_hidden_kv_commit_noop_runtime_ready",
]

REQUIRED_DOC_TOKENS = [
    "P5AR real draft-head top-k rejected-branch discard no-op ABI",
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_REJECTED_BRANCH_DISCARD_NOOP_ABI_ONLY=1",
    "p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime",
    "phase=real_draft_head_topk_rejected_branch_discard_noop_ready",
    "invalid_real_draft_head_topk_rejected_branch_discard_noop_runtime",
    "real_topk_rejected_branch_discard_noop_runtime_ready=1",
    "real_topk_hidden_kv_commit_noop_runtime_ready=1",
    "real_topk_token_commit_noop_runtime_ready=1",
    "P5AQ",
    "P5AP",
    "P5AO",
    "P5AN",
    "P5AM",
    "P5AL",
    "P5AK",
    "P5AJ",
    "P5AG",
    "P5AF",
    "P5AE",
    "P5X",
    "root-tail conflict disabled",
    "actual_verified_logits_rows=1",
    "topk_k=2",
    "actual_tree_nodes=3",
    "candidate_nodes=2",
    "actual_verify_mask_entries=5",
    "accept_boundary_candidate_nodes=2",
    "accept_boundary_verified_edges=5",
    "accept_path_descriptor_len=0",
    "actual_accepted_nodes=0",
    "correction_token_present=0",
    "token_commit_noop=1",
    "hidden_kv_commit_noop=1",
    "reuse_p5aq_hidden_kv_commit_noop=1",
    "rejected_branch_discard_noop=1",
    "actual_committed_tokens=0",
    "actual_survivor_pages_committed=0",
    "actual_pages_discarded=0",
    "rejected_branch_pages_reachable_after_discard=0",
    "actual_publish_visible_state=0",
    "no target logits walk",
    "no target accept walk",
    "no accept",
    "no real token commit",
    "no visible token publish",
    "no real hidden/KV commit",
    "no hidden/KV commit",
    "no real rejected-branch discard",
    "no rejected-branch discard",
    "no publish",
    "no KV mutation",
    "no draft tokens",
    "not the generic P5V",
    "not the root P5AC",
    "returns before downstream publish",
]

REQUIRED_CANDIDATE_TOKENS = REQUIRED_DOC_TOKENS + [
    "LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1",
    "LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1",
    "LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1",
    "LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1",
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY=1",
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_HIDDEN_KV_COMMIT_NOOP_ABI_ONLY=1",
    "real_tree_token_ids[1:] == candidate_ids",
    "source hook is limited to `common/speculative.cpp`",
    "root-tail conflict disabled",
    "rejected_branch_discard_descriptor_ready=1",
    "rejected_branch_discard_runtime_ready=1",
    "root_rejected_branch_discard_noop_runtime_ready=1",
    "publish_gate_descriptor_ready=1",
    "root_publish_gate_noop_runtime_ready=1",
    "hidden_kv_survivor_commit_descriptor_ready=1",
    "root_hidden_kv_commit_noop_runtime_ready=1",
    "token_commit_descriptor_ready=1",
    "root_token_commit_noop_runtime_ready=1",
    "#gen drafts = 1",
    "#gen tokens = 1",
]

REQUIRED_PROBE_TOKENS = [
    "REQUIRED_TRACE_TOKENS",
    "FORBIDDEN_TRACE_TOKENS",
    "SELF_TEST_TRACE",
    "validate_trace_line",
    "probe_p5ar_real_draft_head_topk_rejected_branch_discard_noop_trace",
    "p5ar_real_draft_head_topk_rejected_branch_discard_noop_trace_contract_verified",
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_REJECTED_BRANCH_DISCARD_NOOP_ABI_ONLY=1",
    "phase=real_draft_head_topk_rejected_branch_discard_noop_ready",
    "real_topk_rejected_branch_discard_noop_runtime_ready=1",
    "real_topk_hidden_kv_commit_noop_runtime_ready=1",
    "accept_path_descriptor_len=0",
    "token_commit_noop=1",
    "hidden_kv_commit_noop=1",
    "rejected_branch_discard_noop=1",
    "no_real_rejected_branch_discard=1",
    "no_rejected_branch_discard=1",
    "rejected_branch_discard_descriptor_ready=1",
    "rejected_branch_discard_runtime_ready=1",
    "root_rejected_branch_discard_noop_runtime_ready=1",
    "publish_gate_descriptor_ready=1",
    "root_publish_gate_noop_runtime_ready=1",
    "#gen drafts = 1",
    "#gen tokens = 1",
]

REQUIRED_TEST_TOKENS = [
    "P5ARRealDraftHeadTopKRejectedBranchDiscardNoopTraceProbeTests",
    "test_probe_passes_without_live_log",
    "test_trace_line_requires_discard_noop_and_zero_side_effects",
    "test_parse_helpers_require_key_boundaries",
    "test_trace_line_rejects_missing_required_zero_counters",
    "test_trace_line_rejects_duplicated_nonzero_side_effects",
    "test_trace_line_rejects_duplicate_nonzero_side_effect_counters_above_one",
    "test_trace_line_rejects_forbidden_readiness_and_generation_stats",
    "test_trace_line_rejects_candidate_tree_mismatch",
    "test_validator_passes",
]

REQUIRED_AGGREGATE_TOKENS = [
    "test_p5ar_real_draft_head_topk_rejected_branch_discard_noop_trace_probe.py",
    "validate_p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime.py",
    "probe_p5ar_real_draft_head_topk_rejected_branch_discard_noop_trace.py",
    "p5ar_real_draft_head_topk_rejected_branch_discard_noop_trace_contract_verified",
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_REJECTED_BRANCH_DISCARD_NOOP_ABI_ONLY",
    "invalid_real_draft_head_topk_rejected_branch_discard_noop_runtime",
]

FORBIDDEN_CONTRACT_TOKENS = [
    "llama_decode(",
    "llama_graph",
    "llama_kv_cache",
    "common_sampler_sample",
    "llama_sampler",
    "result->push_back",
]

FORBIDDEN_WIRING_ROOTS = [
    pathlib.Path("src"),
    pathlib.Path("include"),
    pathlib.Path("tools/server"),
    pathlib.Path("ggml/src"),
    pathlib.Path("tests"),
    pathlib.Path("examples"),
    pathlib.Path("pocs"),
]

CMAKE_FORBIDDEN_TOKENS = P5AR_TOKENS + [
    "jetspec_p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime_candidate",
    "validate_p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime",
    "probe_p5ar_real_draft_head_topk_rejected_branch_discard_noop_trace",
    "test_p5ar_real_draft_head_topk_rejected_branch_discard_noop_trace_probe",
]


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _require_tokens(text: str, tokens: list[str], label: str, errors: list[str]) -> None:
    for token in tokens:
        if token not in text:
            errors.append(f"{label} missing token: {token}")


def _cmake_files() -> list[pathlib.Path]:
    files = list(REPO_ROOT.rglob("CMakeLists.txt"))
    files.extend(REPO_ROOT.rglob("*.cmake"))
    return sorted(path for path in files if path.is_file())


def _scan_forbidden_wiring() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    suffixes = {".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".md"}
    for root in FORBIDDEN_WIRING_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            text = _read(path)
            matched = [token for token in P5AR_TOKENS if token in text]
            if matched:
                hits.append({"path": str(path.relative_to(REPO_ROOT)), "tokens": matched})
    return hits


def _scan_common_hook_paths() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    base = REPO_ROOT / "common"
    if not base.exists():
        return hits
    suffixes = {".c", ".cc", ".cpp", ".h", ".hpp"}
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        text = _read(path)
        matched = [token for token in P5AR_TOKENS if token in text]
        if matched:
            hits.append({"path": str(path.relative_to(REPO_ROOT)), "tokens": matched})
    return hits


def _scan_cmake_wiring() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path in _cmake_files():
        text = _read(path)
        matched = [token for token in CMAKE_FORBIDDEN_TOKENS if token in text]
        if matched:
            hits.append({"path": str(path.relative_to(REPO_ROOT)), "tokens": matched})
    return hits


def _source_hook_errors(production_source: str, common_hits: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    bad_common = [hit for hit in common_hits if hit["path"] != "common/speculative.cpp"]
    for hit in bad_common:
        errors.append(f"P5AR source hook must be limited to common/speculative.cpp: {hit}")
    if "p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime" in production_source:
        _require_tokens(production_source, REQUIRED_SOURCE_TOKENS, "common/speculative.cpp", errors)
        if "rejected_branch_discard_descriptor_ready=1" in production_source:
            errors.append("P5AR source hook must not use generic P5V rejected-branch discard readiness")
        if "root_rejected_branch_discard_noop_runtime_ready=1" in production_source:
            errors.append("P5AR source hook must not use root P5AC rejected-branch discard no-op readiness")
        if "publish_gate_descriptor_ready=1" in production_source:
            errors.append("P5AR source hook must return before downstream publish readiness")
        if "root_publish_gate_noop_runtime_ready=1" in production_source:
            errors.append("P5AR source hook must not use root P5AD publish readiness")
    return errors


def validate_p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime() -> dict[str, Any]:
    errors: list[str] = []
    doc = _read(DOC)
    readme = _read(README)
    candidate = _read(CANDIDATE)
    probe = _read(PROBE)
    test = _read(TEST)
    aggregate = _read(AGGREGATE)
    production_source = _read(PRODUCTION_SOURCE)

    _require_tokens(doc, REQUIRED_DOC_TOKENS, "docs/speculative.md", errors)
    _require_tokens(readme, REQUIRED_DOC_TOKENS, "experiments/jetspec/README.md", errors)
    _require_tokens(candidate, REQUIRED_CANDIDATE_TOKENS, CANDIDATE.name, errors)
    _require_tokens(probe, REQUIRED_PROBE_TOKENS, PROBE.name, errors)
    _require_tokens(test, REQUIRED_TEST_TOKENS, TEST.name, errors)
    _require_tokens(aggregate, REQUIRED_AGGREGATE_TOKENS, AGGREGATE.name, errors)

    contract_text = "\n".join([candidate, probe, test])
    for token in FORBIDDEN_CONTRACT_TOKENS:
        if token in contract_text:
            errors.append(f"P5AR no-model contract must not contain runtime side-effect token: {token}")

    cmake_hits = _scan_cmake_wiring()
    wiring_hits = _scan_forbidden_wiring()
    common_hits = _scan_common_hook_paths()
    errors.extend(_source_hook_errors(production_source, common_hits))
    for hit in cmake_hits:
        errors.append(f"P5AR contract must not add CMake wiring: {hit}")
    for hit in wiring_hits:
        errors.append(f"P5AR contract must not add server/public API/ggml wiring: {hit}")

    return {
        "ok": not errors,
        "status": "p5ar_real_draft_head_topk_rejected_branch_discard_noop_contract_wiring_validated" if not errors else "p5ar_real_draft_head_topk_rejected_branch_discard_noop_contract_wiring_invalid",
        "errors": errors,
        "runtime_executed": False,
        "model_loaded": False,
        "context_created": False,
        "source_hook_limited_to_common_speculative_cpp": not any(hit["path"] != "common/speculative.cpp" for hit in common_hits),
        "source_implementation_present": "p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime" in production_source,
        "source_implementation_required_in_this_lane": False,
        "cmake_hits": cmake_hits,
        "forbidden_wiring_hits": wiring_hits,
        "common_hook_hits": common_hits,
        "actual_verified_logits_rows": 1,
        "accept_path_descriptor_len": 0,
        "actual_accepted_nodes": 0,
        "correction_token_present": 0,
        "committed_tokens": 0,
        "survivor_pages_committed": 0,
        "pages_discarded": 0,
        "rejected_branch_pages_reachable_after_discard": 0,
        "published_visible_state": False,
        "kv_mutated": False,
        "draft_tokens_emitted": False,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = validate_p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime()
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
