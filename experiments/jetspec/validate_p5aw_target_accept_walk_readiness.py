#!/usr/bin/env python3
"""Validate the inert P5AW target-accept walk readiness packet."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import jetspec_p5aw_target_accept_walk_readiness as readiness

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
README = HERE / "README.md"
DOC = REPO_ROOT / "docs/speculative.md"
AGGREGATE = HERE / "run_all_jetspec_contracts.py"
FIXTURE = HERE / "fixtures" / "jetspec_p5aw_target_accept_walk_readiness_smoke.json"
EXPECTED_OUT = HERE / "fixtures" / "jetspec_p5aw_target_accept_walk_readiness_smoke.out.json"

REQUIRED_FILES = [
    pathlib.Path("experiments/jetspec/jetspec_p5aw_target_accept_walk_readiness.md"),
    pathlib.Path("experiments/jetspec/jetspec_p5aw_target_accept_walk_readiness.py"),
    pathlib.Path("experiments/jetspec/validate_p5aw_target_accept_walk_readiness.py"),
    pathlib.Path("experiments/jetspec/test_p5aw_target_accept_walk_readiness.py"),
    pathlib.Path("experiments/jetspec/fixtures/jetspec_p5aw_target_accept_walk_readiness_smoke.json"),
    pathlib.Path("experiments/jetspec/fixtures/jetspec_p5aw_target_accept_walk_readiness_smoke.out.json"),
]

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("experiments/jetspec/jetspec_p5aw_target_accept_walk_readiness.md"): [
        "P5AW target-accept walk readiness",
        "inert experiments-only readiness packet",
        "not production source approval",
        "P5AV real draft-head top-k target-logits walk canary",
        "p5aw_target_accept_walk_readiness_verified_not_executed",
        "actual_target_logits_rows_walked=1",
        "target_logits_source=target_model_full_vocab_logits",
        "target_logits_width=248320",
        "target_candidate_logits=[target_logit_top1,target_logit_top2]",
        "planned_target_accept_steps=1",
        "planned_accept_parent_nodes=[0]",
        "planned_accept_candidate_nodes=[1,2]",
        "planned_accept_score_source=target_candidate_logits_from_p5av_row",
        "planned_accept_rule=greedy_target_argmax_child_match_or_correction",
        "planned_correction_token_source=target_full_vocab_argmax_when_no_child_match",
        "actual_target_accept_steps=0",
        "actual_accepted_nodes=0",
        "correction_token_present=0",
        "target_accept_walk_approved=false",
        "real_accept_approved=false",
        "product_runtime_hooks_approved=false",
        "draft_token_emission_approved=false",
        "P5AW target-accept walk runtime hook",
        "P5T token-commit runtime/product hook",
        "P5W publish-gate runtime/product hook",
        "P5AD root publish/no-op product hook reuse",
        "no `common/speculative.cpp` hook",
        "no `tools/server` hook",
        "no public API hook",
        "no `ggml` hook",
        "no CMake wiring",
        "no additional target logits walk",
        "no target accept walk",
        "no KV mutation",
        "no draft tokens",
        "default-off target-accept walk canary",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5aw_target_accept_walk_readiness.py"): [
        "p5aw_target_accept_walk_readiness_verified_not_executed",
        "P5AV_TRACE_STATUS",
        "p5av_target_logits_walk_canary_trace_contract_verified",
        "P5AV_WIRING_STATUS",
        "p5av_real_draft_head_topk_target_logits_walk_canary_contract_wiring_validated",
        "REQUIRED_CHAIN",
        "P5AV",
        "APPROVAL_FLAGS",
        "target_accept_walk_approved",
        "FORBIDDEN_RUNTIME_HOOKS",
        "p5aw_target_accept_walk_runtime_hook",
        "ZERO_COUNTERS",
        "actual_target_accept_steps",
        "NO_RUNTIME_FLAGS",
        "target_accept_walk_executed",
        "target_candidate_logits_from_p5av_row",
        "greedy_target_argmax_child_match_or_correction",
        "target_accept_walk_blocked_pending_explicit_approval",
    ],
    pathlib.Path("experiments/jetspec/test_p5aw_target_accept_walk_readiness.py"): [
        "P5AWTargetAcceptWalkReadinessTests",
        "test_smoke_fixture_matches_expected_output",
        "test_rejects_missing_p5av_predecessor",
        "test_rejects_bad_p5av_evidence",
        "test_rejects_candidate_tree_mismatch",
        "test_rejects_wrong_planned_accept_steps",
        "test_rejects_runtime_hook_and_approval",
        "test_rejects_actual_accept_commit_publish_or_draft_tokens",
        "test_validator_passes",
    ],
}

README_TOKENS = [
    "P5AW target-accept walk readiness",
    "p5aw_target_accept_walk_readiness_verified_not_executed",
    "actual_target_logits_rows_walked=1",
    "planned_target_accept_steps=1",
    "target_candidate_logits_from_p5av_row",
    "greedy_target_argmax_child_match_or_correction",
    "P5AW target-accept walk runtime hook",
    "default-off target-accept walk canary",
]

DOC_TOKENS = [
    "P5AW target-accept walk readiness",
    "p5aw_target_accept_walk_readiness_verified_not_executed",
    "actual_target_logits_rows_walked=1",
    "planned_target_accept_steps=1",
    "planned_accept_rule=greedy_target_argmax_child_match_or_correction",
    "actual_target_accept_steps=0",
    "no additional target logits walk",
    "no target accept walk",
    "default-off target-accept walk canary",
]

AGGREGATE_TOKENS = [
    "test_p5aw_target_accept_walk_readiness.py",
    "jetspec_p5aw_target_accept_walk_readiness.py",
    "validate_p5aw_target_accept_walk_readiness.py",
    "p5aw_target_accept_walk_readiness_verified_not_executed",
    "jetspec_p5aw_target_accept_walk_readiness_smoke.json",
]

P5AW_TOKENS = [
    "P5AW target-accept walk readiness",
    "p5aw_target_accept_walk_readiness_verified_not_executed",
    "jetspec_p5aw_target_accept_walk_readiness",
    "validate_p5aw_target_accept_walk_readiness",
    "test_p5aw_target_accept_walk_readiness",
]

FORBIDDEN_ROOTS = [
    pathlib.Path("common"),
    pathlib.Path("src"),
    pathlib.Path("include"),
    pathlib.Path("tools/server"),
    pathlib.Path("tests"),
    pathlib.Path("examples"),
    pathlib.Path("pocs"),
    pathlib.Path("ggml/src"),
]

CMAKE_TOKENS = P5AW_TOKENS + [
    "jetspec_p5aw_target_accept_walk_readiness_smoke",
]

FORBIDDEN_CONTRACT_TOKENS = [
    "llama_decode(",
    "llama_graph",
    "llama_kv_cache",
    "common_sampler_sample",
    "llama_sampler",
    "result->push_back",
]


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _require_tokens(text: str, tokens: list[str], label: str, errors: list[str]) -> None:
    for token in tokens:
        if token not in text:
            errors.append(f"{label} missing token: {token}")


def _cmake_files() -> list[pathlib.Path]:
    files = list(REPO_ROOT.rglob("CMakeLists.txt"))
    files.extend(REPO_ROOT.rglob("*.cmake"))
    return sorted(path.relative_to(REPO_ROOT) for path in files if path.is_file())


def _scan_forbidden_roots() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    suffixes = {".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".md"}
    for root in FORBIDDEN_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            rel = path.relative_to(REPO_ROOT)
            text = path.read_text(encoding="utf-8", errors="replace")
            matched = [token for token in P5AW_TOKENS if token in text]
            if matched:
                hits.append({"path": str(rel), "tokens": matched})
    return hits


def validate_p5aw_target_accept_walk_readiness() -> dict[str, Any]:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"missing required P5AW file: {rel}")

    for rel, tokens in REQUIRED_TOKENS.items():
        text = _read(rel)
        _require_tokens(text, tokens, str(rel), errors)

    _require_tokens(README.read_text(encoding="utf-8", errors="replace"), README_TOKENS, "experiments/jetspec/README.md", errors)
    _require_tokens(DOC.read_text(encoding="utf-8", errors="replace"), DOC_TOKENS, "docs/speculative.md", errors)
    _require_tokens(AGGREGATE.read_text(encoding="utf-8", errors="replace"), AGGREGATE_TOKENS, "run_all_jetspec_contracts.py", errors)

    contract_rel_paths = [
        pathlib.Path("experiments/jetspec/jetspec_p5aw_target_accept_walk_readiness.md"),
        pathlib.Path("experiments/jetspec/jetspec_p5aw_target_accept_walk_readiness.py"),
        pathlib.Path("experiments/jetspec/test_p5aw_target_accept_walk_readiness.py"),
    ]
    contract_text = "\n".join(_read(rel) for rel in contract_rel_paths)
    for token in FORBIDDEN_CONTRACT_TOKENS:
        if token in contract_text:
            errors.append(f"P5AW no-runtime contract must not contain runtime side-effect token: {token}")

    try:
        fixture_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        actual = readiness.evaluate_fixture(fixture_data)
        expected = json.loads(EXPECTED_OUT.read_text(encoding="utf-8"))
        if actual != expected:
            errors.append("P5AW smoke output differs from expected fixture output")
        if actual.get("status") != readiness.STATUS:
            errors.append(f"P5AW smoke status mismatch: {actual.get('status')}")
        p5av = actual.get("p5av_target_logits_walk_evidence", {})
        if p5av.get("actual_target_logits_rows_walked") != 1:
            errors.append("P5AW must consume one walked P5AV target logits row")
        plan = actual.get("target_accept_plan", {})
        if plan.get("planned_target_accept_steps") != 1:
            errors.append("P5AW must plan one future target accept step")
        if plan.get("actual_target_accept_steps") != 0:
            errors.append("P5AW must not execute target accept steps")
        if actual.get("promotion_result") != "target_accept_walk_blocked_pending_explicit_approval":
            errors.append("P5AW must block target accept walk pending explicit approval")
        for key, value in actual.get("approvals", {}).items():
            if value is not False:
                errors.append(f"P5AW approval flag unexpectedly true: {key}")
        for key, value in actual.get("runtime_hooks", {}).items():
            if value is not False:
                errors.append(f"P5AW runtime hook unexpectedly true: {key}")
        if actual.get("runtime_executed") is not False or actual.get("target_accept_walk_executed") is not False:
            errors.append("P5AW must not execute runtime or target accept walk")
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        errors.append(f"P5AW smoke fixture failed: {exc}")

    forbidden_hits = _scan_forbidden_roots()
    for hit in forbidden_hits:
        errors.append(f"P5AW token appears in forbidden production path: {hit}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5AW must not add explicit CMake wiring in {rel}: {matched}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5aw_target_accept_walk_readiness_validated" if not errors else "p5aw_target_accept_walk_readiness_invalid",
        "files_checked": [str(path) for path in REQUIRED_FILES],
        "runtime_executed": False,
        "model_loaded": False,
        "context_created": False,
        "production_source_hook_present": False,
        "target_accept_walk_executed": False,
        "accept_executed": False,
        "forbidden_root_hits": forbidden_hits,
        "cmake_hits": cmake_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON output; accepted for consistency.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    parse_args(argv)
    out = validate_p5aw_target_accept_walk_readiness()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
