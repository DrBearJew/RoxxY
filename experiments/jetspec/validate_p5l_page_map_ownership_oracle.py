#!/usr/bin/env python3
"""Validate the inert P5L JetSpec page-map ownership oracle."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import page_map_ownership_oracle as oracle


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
FIXTURE = HERE / "fixtures" / "page_map_ownership_oracle_smoke.json"
EXPECTED_OUT = HERE / "fixtures" / "page_map_ownership_oracle_smoke.out.json"

REQUIRED_FILES = [
    pathlib.Path("experiments/jetspec/jetspec_p5l_page_map_ownership_oracle.md"),
    pathlib.Path("experiments/jetspec/page_map_ownership_oracle.py"),
    pathlib.Path("experiments/jetspec/validate_p5l_page_map_ownership_oracle.py"),
    pathlib.Path("experiments/jetspec/test_p5l_page_map_ownership_oracle.py"),
    pathlib.Path("experiments/jetspec/fixtures/page_map_ownership_oracle_smoke.json"),
    pathlib.Path("experiments/jetspec/fixtures/page_map_ownership_oracle_smoke.out.json"),
]

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("experiments/jetspec/jetspec_p5l_page_map_ownership_oracle.md"): [
        "P5L JetSpec page-map ownership oracle",
        "inert page-map ownership oracle only",
        "page_map_ownership_oracle_verified_not_executed",
        "page_map_oracle_only",
        "hidden_kv_survivor_page_ownership",
        "rejected_branch_page_unreachable",
        "cross_sequence_page_isolation",
        "accepted path pages map to [root | accepted] only",
        "correction hidden deferred",
        "accepted path physical gather/compact explicit",
        "rejected transient pages unreachable after commit",
        "accepted path cannot read rejected siblings or descendants",
        "rollback restores pre-round page snapshot",
        "other-sequence pages unchanged",
        "no duplicate mutable physical page ownership",
        "rollback preserves other sequences",
        "identity maps are only parity/oracle cases",
        "visible noncanonical owned overlays fail closed",
        "full current-K map",
        "canonical write-through remains required",
        "seq_cp",
        "seq_rm",
        "seq_import_physical",
        "llama_kv_cache_jetspec_validate_page_ownership_oracle_candidate",
        "llama_kv_cache_jetspec_reserve_transient_tree_pages_candidate",
        "llama_kv_cache_jetspec_commit_page_survivor_path_candidate",
        "llama_kv_cache_jetspec_discard_rejected_tree_pages_candidate",
        "design_only_missing_implementation",
        "runtime_supported=false",
        "no llama_context creation",
        "no draft-head graph execution",
        "no draft tokens emitted",
        "no real KV mutation",
        "no server route",
        "no performance claim",
        "no promotion claim",
    ],
    pathlib.Path("experiments/jetspec/page_map_ownership_oracle.py"): [
        "page_map_ownership_oracle_verified_not_executed",
        "REQUIRED_ORACLES",
        "hidden_kv_survivor_page_ownership",
        "rejected_branch_page_unreachable",
        "cross_sequence_page_isolation",
        "REQUIRED_CANDIDATE_PRIMITIVES",
        "FORBIDDEN_IMPLICIT_HELPERS",
        "QBLOCK_LESSON_TRUE_KEYS",
        "page_map_oracle_only",
        "design_only_missing_implementation",
    ],
    pathlib.Path("experiments/jetspec/test_p5l_page_map_ownership_oracle.py"): [
        "test_smoke_fixture_matches_expected_output",
        "test_rejects_missing_required_oracle",
        "test_rejects_duplicate_page_owner",
        "test_rejects_rejected_page_reachable",
        "test_rejects_accepted_path_without_root",
        "test_rejects_implicit_helper_mapping",
        "test_rejects_runtime_boundary_crossing",
        "test_rejects_production_path_touch",
        "test_validator_passes",
    ],
}

P5L_TOKENS = [
    "P5L JetSpec page-map ownership oracle",
    "page_map_ownership_oracle",
    "page_map_ownership_oracle_verified_not_executed",
    "validate_p5l_page_map_ownership_oracle",
    "test_p5l_page_map_ownership_oracle",
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
    "page_map_ownership_oracle",
    "validate_p5l_page_map_ownership_oracle",
    "test_p5l_page_map_ownership_oracle",
    "page_map_ownership_oracle_verified_not_executed",
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
            matched = [token for token in P5L_TOKENS if token in text]
            if matched:
                hits.append({"path": str(rel), "tokens": matched})
    return hits


def validate_p5l_page_map_ownership_oracle() -> dict[str, Any]:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"missing required P5L file: {rel}")

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
            errors.append("P5L smoke output differs from expected fixture output")
        if actual.get("status") != oracle.STATUS:
            errors.append(f"P5L smoke status mismatch: {actual.get('status')}")
        if actual.get("coverage", {}).get("required_oracles_present") is not True:
            errors.append("P5L smoke must contain all required oracle cases")
        if actual.get("runtime_boundary", {}).get("no_runtime_execution") is not True:
            errors.append("P5L runtime boundary must report no_runtime_execution=true")
        if actual.get("runtime_boundary", {}).get("page_map_oracle_only") is not True:
            errors.append("P5L runtime boundary must report page_map_oracle_only=true")
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        errors.append(f"P5L smoke fixture failed: {exc}")

    forbidden_hits = _scan_forbidden_roots()
    for hit in forbidden_hits:
        errors.append(f"P5L token appears in forbidden production path: {hit}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5L must not add explicit CMake wiring in {rel}: {matched}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5l_page_map_ownership_oracle_validated" if not errors else "p5l_page_map_ownership_oracle_invalid",
        "files_checked": [str(path) for path in REQUIRED_FILES],
        "forbidden_root_hits": forbidden_hits,
        "cmake_hits": cmake_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    parse_args(argv)
    out = validate_p5l_page_map_ownership_oracle()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main([]))
