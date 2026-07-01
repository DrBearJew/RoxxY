#!/usr/bin/env python3
"""Validate the P5AV real draft-head top-k target-logits walk canary contract wiring."""

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
CANDIDATE = HERE / "jetspec_p5av_real_draft_head_topk_target_logits_walk_canary_runtime_candidate.md"
PROBE = HERE / "probe_p5av_real_draft_head_topk_target_logits_walk_canary_trace.py"
TEST = HERE / "test_p5av_real_draft_head_topk_target_logits_walk_canary_trace_probe.py"
AGGREGATE = HERE / "run_all_jetspec_contracts.py"
PRODUCTION_SOURCE = REPO_ROOT / "common/speculative.cpp"

P5AV_TOKENS = [
    "P5AV real draft-head top-k target-logits walk canary",
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TARGET_LOGITS_WALK_CANARY",
    "p5av_real_draft_head_topk_target_logits_walk_canary_runtime",
    "real_draft_head_topk_target_logits_walk_canary_ready",
    "invalid_real_draft_head_topk_target_logits_walk_canary_runtime",
    "target_logits_walk_canary_ready",
]

REQUIRED_SOURCE_TOKENS = [
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TARGET_LOGITS_WALK_CANARY",
    "p5av_real_draft_head_topk_target_logits_walk_canary_runtime",
    "real_draft_head_topk_target_logits_walk_canary_ready",
    "invalid_real_draft_head_topk_target_logits_walk_canary_runtime",
    "llama_get_logits_ith(params.ctx_tgt, batch_index)",
    "target_logits_walk_canary_only=1",
    "actual_target_logits_rows_walked=%d",
    "actual_target_accept_steps=0",
    "no_target_accept_walk=1",
    "no_kv_mutation=1",
    "no_draft_tokens=1",
]

REQUIRED_DOC_TOKENS = [
    "P5AV real draft-head top-k target-logits walk canary",
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TARGET_LOGITS_WALK_CANARY=1",
    "p5av_real_draft_head_topk_target_logits_walk_canary_runtime",
    "phase=real_draft_head_topk_target_logits_walk_canary_ready",
    "invalid_real_draft_head_topk_target_logits_walk_canary_runtime",
    "target_logits_walk_canary_ready=1",
    "real_topk_publish_gate_noop_runtime_ready=1",
    "planned_target_logits_rows=1",
    "actual_target_logits_rows_walked=1",
    "target_logits_source=target_model_full_vocab_logits",
    "target_logits_width=248320",
    "planned_parent_nodes=[0]",
    "planned_candidate_nodes=[1,2]",
    "row_semantics=parent_position_scores_candidate_children",
    "accept_decision_source=target_logits_canary_only_no_accept",
    "actual_target_accept_steps=0",
    "actual_accepted_nodes=0",
    "correction_token_present=0",
    "actual_committed_tokens=0",
    "actual_survivor_pages_committed=0",
    "actual_pages_discarded=0",
    "rejected_branch_pages_reachable_after_discard=0",
    "actual_publish_visible_state=0",
    "no target accept walk",
    "no accept",
    "no token commit",
    "no hidden/KV commit",
    "no rejected-branch discard",
    "no publish",
    "no KV mutation",
    "no draft tokens",
    "not a P5T",
    "not a P5W",
]

REQUIRED_CANDIDATE_TOKENS = REQUIRED_DOC_TOKENS + [
    "reuses the existing target batch output row; no new target decode is issued",
    "target_logits_batch_index",
    "target_logits_pos",
    "target_logits_seq_id",
    "target_candidate_logits=[target_logit_top1,target_logit_top2]",
    "target_logits_walk_canary_only=1",
    "rejects `no_target_logits_walk=1`",
    "source hook is limited to `common/speculative.cpp`",
]

REQUIRED_PROBE_TOKENS = [
    "REQUIRED_TRACE_TOKENS",
    "FORBIDDEN_TRACE_TOKENS",
    "SELF_TEST_TRACE",
    "validate_trace_line",
    "probe_p5av_real_draft_head_topk_target_logits_walk_canary_trace",
    "p5av_target_logits_walk_canary_trace_contract_verified",
    "actual_target_logits_rows_walked=1",
    "target_logits_source=target_model_full_vocab_logits",
    "target_candidate_logits",
    "no_target_logits_walk=1",
    "actual_target_accept_steps=1",
    "#gen drafts = 1",
]

REQUIRED_TEST_TOKENS = [
    "P5AVRealDraftHeadTopKTargetLogitsWalkCanaryTraceProbeTests",
    "test_probe_passes_without_live_log",
    "test_trace_line_requires_target_logits_walk_and_zero_side_effects",
    "test_trace_line_rejects_no_target_logits_walk_marker",
    "test_trace_line_rejects_missing_or_zero_target_walk",
    "test_trace_line_rejects_accept_commit_publish_and_generation",
    "test_trace_line_rejects_bad_candidate_lists",
    "test_validator_passes",
]

REQUIRED_AGGREGATE_TOKENS = [
    "test_p5av_real_draft_head_topk_target_logits_walk_canary_trace_probe.py",
    "validate_p5av_real_draft_head_topk_target_logits_walk_canary_runtime.py",
    "probe_p5av_real_draft_head_topk_target_logits_walk_canary_trace.py",
    "p5av_target_logits_walk_canary_trace_contract_verified",
    "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TARGET_LOGITS_WALK_CANARY",
    "invalid_real_draft_head_topk_target_logits_walk_canary_runtime",
]

FORBIDDEN_CONTRACT_TOKENS = [
    "llama_decode(",
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

CMAKE_FORBIDDEN_TOKENS = P5AV_TOKENS + [
    "jetspec_p5av_real_draft_head_topk_target_logits_walk_canary_runtime_candidate",
    "validate_p5av_real_draft_head_topk_target_logits_walk_canary_runtime",
    "probe_p5av_real_draft_head_topk_target_logits_walk_canary_trace",
    "test_p5av_real_draft_head_topk_target_logits_walk_canary_trace_probe",
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
            matched = [token for token in P5AV_TOKENS if token in text]
            if matched:
                hits.append({"path": str(path.relative_to(REPO_ROOT)), "tokens": matched})
    return hits


def _scan_common_hook_paths() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    base = REPO_ROOT / "common"
    suffixes = {".c", ".cc", ".cpp", ".h", ".hpp"}
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        text = _read(path)
        matched = [token for token in P5AV_TOKENS if token in text]
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
    for hit in common_hits:
        if hit["path"] != "common/speculative.cpp":
            errors.append(f"P5AV source hook must be limited to common/speculative.cpp: {hit}")
    if "p5av_real_draft_head_topk_target_logits_walk_canary_runtime" in production_source:
        _require_tokens(production_source, REQUIRED_SOURCE_TOKENS, "common/speculative.cpp", errors)
        if "actual_committed_tokens_last = 1" in production_source:
            errors.append("P5AV source hook must not commit tokens")
        if "actual_publish_visible_state_last = 1" in production_source:
            errors.append("P5AV source hook must not publish visible state")
        if "common_speculative_jetspec_real_draft_head_canary_eval(params.ctx_tgt" in production_source:
            errors.append("P5AV source hook must not issue a new target decode")
    return errors


def validate_p5av_real_draft_head_topk_target_logits_walk_canary_runtime() -> dict[str, Any]:
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
            errors.append(f"P5AV no-model contract must not contain runtime side-effect token: {token}")

    cmake_hits = _scan_cmake_wiring()
    wiring_hits = _scan_forbidden_wiring()
    common_hits = _scan_common_hook_paths()
    errors.extend(_source_hook_errors(production_source, common_hits))
    for hit in cmake_hits:
        errors.append(f"P5AV contract must not add CMake wiring: {hit}")
    for hit in wiring_hits:
        errors.append(f"P5AV contract must not add server/public API/ggml wiring: {hit}")

    return {
        "ok": not errors,
        "status": "p5av_real_draft_head_topk_target_logits_walk_canary_contract_wiring_validated" if not errors else "p5av_real_draft_head_topk_target_logits_walk_canary_contract_wiring_invalid",
        "errors": errors,
        "runtime_executed": False,
        "model_loaded": False,
        "context_created": False,
        "source_hook_limited_to_common_speculative_cpp": not any(hit["path"] != "common/speculative.cpp" for hit in common_hits),
        "source_implementation_present": "p5av_real_draft_head_topk_target_logits_walk_canary_runtime" in production_source,
        "cmake_hits": cmake_hits,
        "forbidden_wiring_hits": wiring_hits,
        "common_hook_hits": common_hits,
        "planned_target_logits_rows": 1,
        "actual_target_accept_steps": 0,
        "actual_accepted_nodes": 0,
        "committed_tokens": 0,
        "published_visible_state": False,
        "kv_mutated": False,
        "draft_tokens_emitted": False,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = validate_p5av_real_draft_head_topk_target_logits_walk_canary_runtime()
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
