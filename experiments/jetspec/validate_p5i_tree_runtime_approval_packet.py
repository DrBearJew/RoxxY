#!/usr/bin/env python3
"""Validate the inert P5I JetSpec tree-runtime approval packet."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import tree_runtime_approval_matrix as matrix


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
FIXTURE = HERE / "fixtures" / "tree_runtime_approval_matrix_smoke.json"
EXPECTED_OUT = HERE / "fixtures" / "tree_runtime_approval_matrix_smoke.out.json"

REQUIRED_FILES = [
    pathlib.Path("experiments/jetspec/jetspec_p5i_tree_runtime_approval_packet.md"),
    pathlib.Path("experiments/jetspec/tree_runtime_approval_matrix.py"),
    pathlib.Path("experiments/jetspec/validate_p5i_tree_runtime_approval_packet.py"),
    pathlib.Path("experiments/jetspec/test_p5i_tree_runtime_approval_packet.py"),
    pathlib.Path("experiments/jetspec/fixtures/tree_runtime_approval_matrix_smoke.json"),
    pathlib.Path("experiments/jetspec/fixtures/tree_runtime_approval_matrix_smoke.out.json"),
]

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("experiments/jetspec/jetspec_p5i_tree_runtime_approval_packet.md"): [
        "P5I tree-runtime approval packet",
        "inert approval packet only",
        "not production tree-runtime approval",
        "not runtime execution",
        "not a performance claim",
        "experiments/jetspec/",
        "no `llama_context` creation",
        "no draft-head graph execution",
        "no draft tokens emitted",
        "no real KV mutation",
        "no server route",
        "no performance claim",
        "no promotion claim",
        "runtime_supported=false",
        "validated_by_p5g",
        "validated_by_p5h",
        "missing_primitive",
        "blocked_pending_explicit_approval",
        "tree build",
        "verify mask",
        "accept path",
        "token commit",
        "hidden/KV survivor commit",
        "rejected branch discard",
        "cross-sequence isolation",
        "rollback/fail-closed disable",
        "tree_runtime_approval_packet_verified_not_executed",
    ],
    pathlib.Path("experiments/jetspec/tree_runtime_approval_matrix.py"): [
        "tree_runtime_approval_packet_verified_not_executed",
        "REQUIRED_ACTIONS",
        "tree_build",
        "verify_mask",
        "accept_path",
        "token_commit",
        "hidden_kv_survivor_commit",
        "rejected_branch_discard",
        "cross_sequence_isolation",
        "rollback_fail_closed_disable",
        "validated_by_p5g",
        "validated_by_p5h",
        "missing_primitive",
        "blocked_pending_explicit_approval",
        "production_paths_touched",
        "runtime claim",
        "future_primitive is implicit or unsafe",
    ],
    pathlib.Path("experiments/jetspec/test_p5i_tree_runtime_approval_packet.py"): [
        "test_smoke_fixture_matches_expected_output",
        "test_rejects_missing_required_action",
        "test_rejects_production_path_touch",
        "test_rejects_runtime_execution_claim",
        "test_rejects_implicit_primitive",
        "test_rejects_promotion_claim",
        "test_validator_passes",
    ],
}

P5I_TOKENS = [
    "P5I tree-runtime approval packet",
    "tree_runtime_approval_packet",
    "tree_runtime_approval_packet_verified_not_executed",
    "tree_runtime_approval_matrix",
    "validate_p5i_tree_runtime_approval_packet",
    "test_p5i_tree_runtime_approval_packet",
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
    "tree_runtime_approval_matrix",
    "validate_p5i_tree_runtime_approval_packet",
    "test_p5i_tree_runtime_approval_packet",
    "tree_runtime_approval_packet_verified_not_executed",
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
            matched = [token for token in P5I_TOKENS if token in text]
            if matched:
                hits.append({"path": str(rel), "tokens": matched})
    return hits


def validate_p5i_tree_runtime_approval_packet() -> dict[str, Any]:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"missing required P5I file: {rel}")

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
        actual = matrix.evaluate_fixture(fixture_data)
        expected = json.loads(EXPECTED_OUT.read_text(encoding="utf-8"))
        if actual != expected:
            errors.append("P5I smoke output differs from expected fixture output")
        if actual.get("status") != matrix.STATUS:
            errors.append(f"P5I smoke status mismatch: {actual.get('status')}")
        if actual.get("coverage", {}).get("required_actions_present") is not True:
            errors.append("P5I required action coverage must pass")
        if actual.get("runtime_claims", {}).get("no_runtime_execution") is not True:
            errors.append("P5I runtime claims must report no_runtime_execution=true")
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        errors.append(f"P5I smoke fixture failed: {exc}")

    forbidden_hits = _scan_forbidden_roots()
    for hit in forbidden_hits:
        errors.append(f"P5I token appears in forbidden production path: {hit}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5I must not add explicit CMake wiring in {rel}: {matched}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5i_tree_runtime_approval_packet_validated" if not errors else "p5i_tree_runtime_approval_packet_invalid",
        "files_checked": [str(path) for path in REQUIRED_FILES],
        "forbidden_root_hits": forbidden_hits,
        "cmake_hits": cmake_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    parse_args(argv)
    out = validate_p5i_tree_runtime_approval_packet()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main([]))
