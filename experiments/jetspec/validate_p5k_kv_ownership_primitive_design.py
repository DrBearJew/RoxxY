#!/usr/bin/env python3
"""Validate the inert P5K JetSpec KV ownership primitive design packet."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import kv_ownership_primitive_design as design


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
FIXTURE = HERE / "fixtures" / "kv_ownership_primitive_design_smoke.json"
EXPECTED_OUT = HERE / "fixtures" / "kv_ownership_primitive_design_smoke.out.json"

REQUIRED_FILES = [
    pathlib.Path("experiments/jetspec/jetspec_p5k_kv_ownership_primitive_design.md"),
    pathlib.Path("experiments/jetspec/kv_ownership_primitive_design.py"),
    pathlib.Path("experiments/jetspec/validate_p5k_kv_ownership_primitive_design.py"),
    pathlib.Path("experiments/jetspec/test_p5k_kv_ownership_primitive_design.py"),
    pathlib.Path("experiments/jetspec/fixtures/kv_ownership_primitive_design_smoke.json"),
    pathlib.Path("experiments/jetspec/fixtures/kv_ownership_primitive_design_smoke.out.json"),
]

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("experiments/jetspec/jetspec_p5k_kv_ownership_primitive_design.md"): [
        "P5K KV ownership primitive design packet",
        "inert design packet only",
        "exact_missing_primitive",
        "kv_ownership_primitive_design_verified_not_executed",
        "design_only_missing_implementation",
        "llama_kv_cache_jetspec_commit_survivor_path_candidate",
        "llama_kv_cache_jetspec_discard_rejected_tree_candidate",
        "llama_kv_cache_jetspec_assert_cross_sequence_isolation_candidate",
        "accepted_path physical gather/compact",
        "[root | accepted] only",
        "correction hidden deferred",
        "committed tail compact",
        "rejected transient tree slots unreachable",
        "not range removal only",
        "rollback restores pre-round state",
        "other-sequence slots unchanged",
        "seq_to_stream",
        "no shared-slot corruption",
        "rollback preserves other sequences",
        "seq_cp",
        "seq_rm",
        "seq_import_physical",
        "no `llama_context` creation",
        "no draft-head graph execution",
        "no draft tokens emitted",
        "no real KV mutation",
        "runtime_supported=false",
    ],
    pathlib.Path("experiments/jetspec/kv_ownership_primitive_design.py"): [
        "kv_ownership_primitive_design_verified_not_executed",
        "REQUIRED_HELPERS",
        "REQUIRED_DESIGNS",
        "hidden_kv_survivor_commit",
        "rejected_branch_discard",
        "cross_sequence_isolation",
        "design_only_missing_implementation",
        "must_not_use_as_implicit_mapping",
        "llama_kv_cache_jetspec_",
    ],
    pathlib.Path("experiments/jetspec/test_p5k_kv_ownership_primitive_design.py"): [
        "test_smoke_fixture_matches_expected_output",
        "test_rejects_missing_required_design",
        "test_rejects_helper_marked_exact",
        "test_rejects_missing_forbidden_implicit_mapping",
        "test_rejects_implementation_approval",
        "test_rejects_runtime_boundary_crossing",
        "test_validator_passes",
    ],
}

P5K_TOKENS = [
    "P5K KV ownership primitive design packet",
    "kv_ownership_primitive_design",
    "kv_ownership_primitive_design_verified_not_executed",
    "validate_p5k_kv_ownership_primitive_design",
    "test_p5k_kv_ownership_primitive_design",
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
    "kv_ownership_primitive_design",
    "validate_p5k_kv_ownership_primitive_design",
    "test_p5k_kv_ownership_primitive_design",
    "kv_ownership_primitive_design_verified_not_executed",
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
            matched = [token for token in P5K_TOKENS if token in text]
            if matched:
                hits.append({"path": str(rel), "tokens": matched})
    return hits


def validate_p5k_kv_ownership_primitive_design() -> dict[str, Any]:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"missing required P5K file: {rel}")

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
        actual = design.evaluate_fixture(fixture_data)
        expected = json.loads(EXPECTED_OUT.read_text(encoding="utf-8"))
        if actual != expected:
            errors.append("P5K smoke output differs from expected fixture output")
        if actual.get("status") != design.STATUS:
            errors.append(f"P5K smoke status mismatch: {actual.get('status')}")
        if actual.get("coverage", {}).get("design_count") != 3:
            errors.append("P5K smoke must contain three primitive designs")
        if actual.get("runtime_boundary", {}).get("no_runtime_execution") is not True:
            errors.append("P5K runtime boundary must report no_runtime_execution=true")
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        errors.append(f"P5K smoke fixture failed: {exc}")

    forbidden_hits = _scan_forbidden_roots()
    for hit in forbidden_hits:
        errors.append(f"P5K token appears in forbidden production path: {hit}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5K must not add explicit CMake wiring in {rel}: {matched}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5k_kv_ownership_primitive_design_validated" if not errors else "p5k_kv_ownership_primitive_design_invalid",
        "files_checked": [str(path) for path in REQUIRED_FILES],
        "forbidden_root_hits": forbidden_hits,
        "cmake_hits": cmake_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    parse_args(argv)
    out = validate_p5k_kv_ownership_primitive_design()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main([]))
