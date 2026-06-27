#!/usr/bin/env python3
"""Validate the inert P5M JetSpec transaction/failpoint plan oracle."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import transaction_plan_oracle as oracle


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
FIXTURE = HERE / "fixtures" / "transaction_plan_oracle_smoke.json"
EXPECTED_OUT = HERE / "fixtures" / "transaction_plan_oracle_smoke.out.json"

REQUIRED_FILES = [
    pathlib.Path("experiments/jetspec/jetspec_p5m_transaction_plan_oracle.md"),
    pathlib.Path("experiments/jetspec/transaction_plan_oracle.py"),
    pathlib.Path("experiments/jetspec/validate_p5m_transaction_plan_oracle.py"),
    pathlib.Path("experiments/jetspec/test_p5m_transaction_plan_oracle.py"),
    pathlib.Path("experiments/jetspec/fixtures/transaction_plan_oracle_smoke.json"),
    pathlib.Path("experiments/jetspec/fixtures/transaction_plan_oracle_smoke.out.json"),
]

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("experiments/jetspec/jetspec_p5m_transaction_plan_oracle.md"): [
        "P5M JetSpec transaction/failpoint plan oracle",
        "inert transaction/failpoint plan oracle only",
        "transaction_plan_oracle_verified_not_executed",
        "transaction_plan_oracle_only",
        "snapshot_pre_round",
        "reserve_transient_tree_pages",
        "build_tree",
        "build_verify_mask",
        "accept_path",
        "commit_tokens",
        "commit_hidden_kv_survivors",
        "discard_rejected_branches",
        "publish_post_commit_state",
        "after_reserve",
        "after_build_tree",
        "after_verify_mask",
        "after_accept",
        "after_token_commit",
        "after_hidden_kv_commit",
        "after_rejected_discard",
        "pre_publish_visible_state_unmodified=true",
        "publishes_visible_state=true",
        "no committed token visibility before publish",
        "no hidden/KV visibility before publish",
        "no page-map visibility before publish",
        "publish after commit and discard only",
        "rollback clears transient state",
        "[accepted draft tokens | correction]",
        "[root | accepted]",
        "correction hidden deferred",
        "rejected branches unreachable",
        "other-sequence pages unchanged",
        "no duplicate mutable physical page ownership",
        "llama_kv_cache_jetspec_validate_page_ownership_oracle_candidate",
        "llama_kv_cache_jetspec_reserve_transient_tree_pages_candidate",
        "llama_kv_cache_jetspec_commit_page_survivor_path_candidate",
        "llama_kv_cache_jetspec_discard_rejected_tree_pages_candidate",
        "llama_kv_cache_jetspec_rollback_tree_transaction_candidate",
        "design_only_missing_implementation",
        "seq_cp",
        "seq_rm",
        "seq_import_physical",
        "P5G tree-runtime readiness",
        "P5H KV/hidden commit readiness",
        "P5I tree-runtime approval packet",
        "P5K KV ownership primitive design",
        "P5L page-map ownership oracle",
        "runtime_supported=false",
        "no llama_context creation",
        "no draft-head graph execution",
        "no draft tokens emitted",
        "no real KV mutation",
        "no server route",
        "no performance claim",
        "no promotion claim",
    ],
    pathlib.Path("experiments/jetspec/transaction_plan_oracle.py"): [
        "transaction_plan_oracle_verified_not_executed",
        "REQUIRED_PHASES",
        "snapshot_pre_round",
        "reserve_transient_tree_pages",
        "publish_post_commit_state",
        "REQUIRED_ROLLBACK_POINTS",
        "after_hidden_kv_commit",
        "REQUIRED_VISIBILITY_RULES",
        "REQUIRED_CANDIDATE_PRIMITIVES",
        "llama_kv_cache_jetspec_rollback_tree_transaction_candidate",
        "FORBIDDEN_IMPLICIT_HELPERS",
        "transaction_plan_oracle_only",
        "design_only_missing_implementation",
    ],
    pathlib.Path("experiments/jetspec/test_p5m_transaction_plan_oracle.py"): [
        "test_smoke_fixture_matches_expected_output",
        "test_rejects_missing_rollback_point",
        "test_rejects_publish_before_all_commits_and_discards",
        "test_rejects_token_commit_before_accept_path",
        "test_rejects_hidden_kv_commit_before_survivor_pages_validated",
        "test_rejects_rejected_branch_reachable_after_discard",
        "test_rejects_rollback_that_mutates_other_sequence_pages",
        "test_rejects_duplicate_mutable_page_ownership",
        "test_rejects_runtime_boundary_crossing",
        "test_rejects_production_path_touch",
        "test_rejects_implementation_approval",
        "test_rejects_performance_or_promotion_claim",
        "test_rejects_implicit_helper_mapping",
        "test_validator_passes",
    ],
}

P5M_TOKENS = [
    "P5M JetSpec transaction/failpoint plan oracle",
    "transaction_plan_oracle",
    "transaction_plan_oracle_verified_not_executed",
    "validate_p5m_transaction_plan_oracle",
    "test_p5m_transaction_plan_oracle",
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

CMAKE_TOKENS = [
    "transaction_plan_oracle",
    "validate_p5m_transaction_plan_oracle",
    "test_p5m_transaction_plan_oracle",
    "transaction_plan_oracle_verified_not_executed",
]


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {rel}")
    return path.read_text(encoding="utf-8", errors="replace")


def _cmake_files() -> list[pathlib.Path]:
    files = list(REPO_ROOT.rglob("CMakeLists.txt"))
    files.extend(REPO_ROOT.rglob("*.cmake"))
    return sorted(path.relative_to(REPO_ROOT) for path in files if path.is_file())


def _scan_forbidden_roots() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    suffixes = {".c", ".cc", ".cpp", ".h", ".hpp", ".md"}
    for root in FORBIDDEN_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            rel = path.relative_to(REPO_ROOT)
            text = path.read_text(encoding="utf-8", errors="replace")
            matched = [token for token in P5M_TOKENS if token in text]
            if matched:
                hits.append({"path": str(rel), "tokens": matched})
    return hits


def validate_p5m_transaction_plan_oracle() -> dict[str, Any]:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"missing required P5M file: {rel}")

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except OSError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    try:
        fixture_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        actual = oracle.evaluate_fixture(fixture_data)
        expected = json.loads(EXPECTED_OUT.read_text(encoding="utf-8"))
        if actual != expected:
            errors.append("P5M smoke output differs from expected fixture output")
        if actual.get("status") != oracle.STATUS:
            errors.append(f"P5M smoke status mismatch: {actual.get('status')}")
        if actual.get("coverage", {}).get("required_phases_ordered") is not True:
            errors.append("P5M smoke must contain ordered required phases")
        if actual.get("coverage", {}).get("required_rollback_points_present") is not True:
            errors.append("P5M smoke must contain all rollback points")
        if actual.get("coverage", {}).get("rollback_restores_pre_round_snapshot") is not True:
            errors.append("P5M smoke must restore pre-round snapshot")
        if actual.get("runtime_boundary", {}).get("no_runtime_execution") is not True:
            errors.append("P5M runtime boundary must report no_runtime_execution=true")
        if actual.get("runtime_boundary", {}).get("transaction_plan_oracle_only") is not True:
            errors.append("P5M runtime boundary must report transaction_plan_oracle_only=true")
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        errors.append(f"P5M smoke fixture failed: {exc}")

    forbidden_hits = _scan_forbidden_roots()
    for hit in forbidden_hits:
        errors.append(f"P5M token appears in forbidden production path: {hit}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5M must not add explicit CMake wiring in {rel}: {matched}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5m_transaction_plan_oracle_validated" if not errors else "p5m_transaction_plan_oracle_invalid",
        "files_checked": [str(path) for path in REQUIRED_FILES],
        "forbidden_root_hits": forbidden_hits,
        "cmake_hits": cmake_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    parse_args(argv)
    out = validate_p5m_transaction_plan_oracle()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main([]))
