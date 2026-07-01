#!/usr/bin/env python3
"""Validate the inert P5AT real draft-head top-k promotion blocker audit."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import jetspec_p5at_real_draft_head_topk_promotion_blocker as blocker

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
README = HERE / "README.md"
AGGREGATE = HERE / "run_all_jetspec_contracts.py"
FIXTURE = HERE / "fixtures" / "jetspec_p5at_real_draft_head_topk_promotion_blocker_smoke.json"
EXPECTED_OUT = HERE / "fixtures" / "jetspec_p5at_real_draft_head_topk_promotion_blocker_smoke.out.json"

REQUIRED_FILES = [
    pathlib.Path("experiments/jetspec/jetspec_p5at_real_draft_head_topk_promotion_blocker.md"),
    pathlib.Path("experiments/jetspec/jetspec_p5at_real_draft_head_topk_promotion_blocker.py"),
    pathlib.Path("experiments/jetspec/validate_p5at_real_draft_head_topk_promotion_blocker.py"),
    pathlib.Path("experiments/jetspec/test_p5at_real_draft_head_topk_promotion_blocker.py"),
    pathlib.Path("experiments/jetspec/fixtures/jetspec_p5at_real_draft_head_topk_promotion_blocker_smoke.json"),
    pathlib.Path("experiments/jetspec/fixtures/jetspec_p5at_real_draft_head_topk_promotion_blocker_smoke.out.json"),
]

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("experiments/jetspec/jetspec_p5at_real_draft_head_topk_promotion_blocker.md"): [
        "P5AT real draft-head top-k promotion blocker audit",
        "inert experiments-only audit packet",
        "not production source approval",
        "P5AS real draft-head top-k publish-gate no-op ABI",
        "p5as_real_draft_head_topk_publish_gate_noop_trace_contract_verified",
        "p5at_real_draft_head_topk_promotion_blocker_verified_not_executed",
        "target_logits_walk_approved=false",
        "target_accept_walk_approved=false",
        "real_token_commit_approved=false",
        "real_hidden_kv_commit_approved=false",
        "real_publish_approved=false",
        "product_runtime_hooks_approved=false",
        "draft_token_emission_approved=false",
        "P5T token-commit runtime/product hook",
        "P5W publish-gate runtime/product hook",
        "P5AD root publish/no-op product hook reuse",
        "no `common/speculative.cpp` hook",
        "no `tools/server` hook",
        "no public API hook",
        "no `ggml` hook",
        "no CMake wiring",
        "no target logits walk",
        "no target accept walk",
        "no real token commit",
        "no real publish",
        "no KV mutation",
        "no draft tokens",
        "separate explicit architecture approval",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5at_real_draft_head_topk_promotion_blocker.py"): [
        "p5at_real_draft_head_topk_promotion_blocker_verified_not_executed",
        "REQUIRED_CHAIN",
        "P5AS",
        "REQUIRED_P5AS_STATUS",
        "APPROVAL_FLAGS",
        "target_logits_walk_approved",
        "product_runtime_hooks_approved",
        "FORBIDDEN_RUNTIME_HOOKS",
        "p5t_token_commit_runtime_product_hook",
        "p5w_publish_gate_runtime_product_hook",
        "p5ad_root_publish_noop_product_hook_reuse",
        "ZERO_COUNTERS",
        "actual_committed_tokens",
        "actual_publish_visible_state",
        "NO_RUNTIME_FLAGS",
        "target_logits_walk_executed",
        "draft_tokens_emitted",
        "blocked_pending_explicit_approval",
    ],
    pathlib.Path("experiments/jetspec/test_p5at_real_draft_head_topk_promotion_blocker.py"): [
        "P5ATRealDraftHeadTopKPromotionBlockerTests",
        "test_smoke_fixture_matches_expected_output",
        "test_rejects_missing_p5as_predecessor",
        "test_rejects_product_runtime_hook_approval",
        "test_rejects_p5t_p5w_p5ad_runtime_hooks",
        "test_rejects_side_effects_and_draft_tokens",
        "test_validator_passes",
    ],
}

README_TOKENS = [
    "P5AT real draft-head top-k promotion blocker audit",
    "p5at_real_draft_head_topk_promotion_blocker_verified_not_executed",
    "P5AS",
    "P5T/P5U/P5V/P5W/P5AD runtime/product hooks",
    "separate explicit architecture approval",
]

AGGREGATE_TOKENS = [
    "test_p5at_real_draft_head_topk_promotion_blocker.py",
    "jetspec_p5at_real_draft_head_topk_promotion_blocker.py",
    "validate_p5at_real_draft_head_topk_promotion_blocker.py",
    "p5at_real_draft_head_topk_promotion_blocker_verified_not_executed",
    "jetspec_p5at_real_draft_head_topk_promotion_blocker_smoke.json",
]

P5AT_TOKENS = [
    "P5AT real draft-head top-k promotion blocker audit",
    "p5at_real_draft_head_topk_promotion_blocker_verified_not_executed",
    "jetspec_p5at_real_draft_head_topk_promotion_blocker",
    "validate_p5at_real_draft_head_topk_promotion_blocker",
    "test_p5at_real_draft_head_topk_promotion_blocker",
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
    pathlib.Path("docs"),
]

CMAKE_TOKENS = P5AT_TOKENS + [
    "jetspec_p5at_real_draft_head_topk_promotion_blocker_smoke",
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
            matched = [token for token in P5AT_TOKENS if token in text]
            if matched:
                hits.append({"path": str(rel), "tokens": matched})
    return hits


def validate_p5at_real_draft_head_topk_promotion_blocker() -> dict[str, Any]:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"missing required P5AT file: {rel}")

    for rel, tokens in REQUIRED_TOKENS.items():
        text = _read(rel)
        _require_tokens(text, tokens, str(rel), errors)

    _require_tokens(README.read_text(encoding="utf-8", errors="replace"), README_TOKENS, "experiments/jetspec/README.md", errors)
    _require_tokens(AGGREGATE.read_text(encoding="utf-8", errors="replace"), AGGREGATE_TOKENS, "run_all_jetspec_contracts.py", errors)

    contract_rel_paths = [
        pathlib.Path("experiments/jetspec/jetspec_p5at_real_draft_head_topk_promotion_blocker.md"),
        pathlib.Path("experiments/jetspec/jetspec_p5at_real_draft_head_topk_promotion_blocker.py"),
        pathlib.Path("experiments/jetspec/test_p5at_real_draft_head_topk_promotion_blocker.py"),
    ]
    contract_text = "\n".join(_read(rel) for rel in contract_rel_paths)
    for token in FORBIDDEN_CONTRACT_TOKENS:
        if token in contract_text:
            errors.append(f"P5AT no-runtime contract must not contain runtime side-effect token: {token}")

    try:
        fixture_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        actual = blocker.evaluate_fixture(fixture_data)
        expected = json.loads(EXPECTED_OUT.read_text(encoding="utf-8"))
        if actual != expected:
            errors.append("P5AT smoke output differs from expected fixture output")
        if actual.get("status") != blocker.STATUS:
            errors.append(f"P5AT smoke status mismatch: {actual.get('status')}")
        if actual.get("promotion_result") != "blocked_pending_explicit_approval":
            errors.append("P5AT must block promotion pending explicit approval")
        for key, value in actual.get("approvals", {}).items():
            if value is not False:
                errors.append(f"P5AT approval flag unexpectedly true: {key}")
        for key, value in actual.get("runtime_hooks", {}).items():
            if value is not False:
                errors.append(f"P5AT runtime hook unexpectedly true: {key}")
        for key, value in actual.get("side_effect_counters", {}).items():
            if value != 0:
                errors.append(f"P5AT side-effect counter must stay zero: {key}={value}")
        if actual.get("runtime_executed") is not False or actual.get("draft_tokens_emitted") is not False:
            errors.append("P5AT must not execute runtime or emit draft tokens")
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        errors.append(f"P5AT smoke fixture failed: {exc}")

    forbidden_hits = _scan_forbidden_roots()
    for hit in forbidden_hits:
        errors.append(f"P5AT token appears in forbidden production path: {hit}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5AT must not add explicit CMake wiring in {rel}: {matched}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5at_real_draft_head_topk_promotion_blocker_validated" if not errors else "p5at_real_draft_head_topk_promotion_blocker_invalid",
        "files_checked": [str(path) for path in REQUIRED_FILES],
        "runtime_executed": False,
        "model_loaded": False,
        "context_created": False,
        "production_source_hook_present": False,
        "forbidden_root_hits": forbidden_hits,
        "cmake_hits": cmake_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON output; accepted for consistency.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    parse_args(argv)
    out = validate_p5at_real_draft_head_topk_promotion_blocker()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
