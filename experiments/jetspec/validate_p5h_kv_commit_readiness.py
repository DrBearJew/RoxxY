#!/usr/bin/env python3
"""Validate the inert P5H JetSpec KV/hidden commit ownership readiness contract."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import jetspec_kv_commit_readiness as readiness


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
FIXTURE = HERE / "fixtures" / "jetspec_kv_commit_readiness_smoke.json"
EXPECTED_OUT = HERE / "fixtures" / "jetspec_kv_commit_readiness_smoke.out.json"

REQUIRED_FILES = [
    pathlib.Path("experiments/jetspec/jetspec_p5h_kv_commit_readiness.md"),
    pathlib.Path("experiments/jetspec/jetspec_kv_commit_readiness.py"),
    pathlib.Path("experiments/jetspec/validate_p5h_kv_commit_readiness.py"),
    pathlib.Path("experiments/jetspec/test_p5h_kv_commit_readiness.py"),
    pathlib.Path("experiments/jetspec/fixtures/jetspec_kv_commit_readiness_smoke.json"),
    pathlib.Path("experiments/jetspec/fixtures/jetspec_kv_commit_readiness_smoke.out.json"),
]

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("experiments/jetspec/jetspec_p5h_kv_commit_readiness.md"): [
        "P5H KV/hidden commit ownership readiness",
        "inert readiness contract only",
        "not production tree-runtime approval",
        "not a llama.cpp KV mutation",
        "experiments/jetspec/",
        "reserve_tree_slots()",
        "gather()",
        "no production runtime execution",
        "no `llama_context` draft runtime instantiation",
        "no draft-head graph execution",
        "no draft tokens emitted",
        "no real KV cache mutation",
        "runtime_supported=false",
        "model hidden/KV rows trail committed tokens by one",
        "[accepted draft tokens | correction]",
        "[root | accepted]",
        "correction hidden is not appended",
        "rejected tree nodes are unreachable after commit",
        "past_len + accepted_path",
        "cross-sequence slots remain isolated",
        "missing primitive",
        "kv_commit_readiness_verified_not_executed",
    ],
    pathlib.Path("experiments/jetspec/jetspec_kv_commit_readiness.py"): [
        "kv_commit_readiness_verified_not_executed",
        "REQUIRED_MAPPING_KEYS",
        "reserve_transient_tree_slots",
        "gather_accepted_path",
        "discard_rejected_tree_slots",
        "compact_survivors_to_committed_tail",
        "preserve_cross_sequence_slots",
        "accepted_path must not contain duplicate nodes",
        "accepted_path contains out-of-range nodes",
        "past_len + accepted_path",
        "cross-sequence isolation violation",
        "ownership mapping",
        "missing primitive",
        "real_kv_cache_mutated",
        "model_hidden_cache_trails_committed_tokens_by_one",
    ],
    pathlib.Path("experiments/jetspec/test_p5h_kv_commit_readiness.py"): [
        "test_smoke_fixture_matches_expected_output",
        "test_rejects_duplicate_accepted_path",
        "test_rejects_out_of_range_accepted_path",
        "test_rejects_gather_position_mismatch",
        "test_rejects_cross_sequence_slot_touch",
        "test_rejects_silent_ownership_mapping",
        "test_rejects_runtime_boundary_crossing",
        "test_validator_passes",
    ],
}

P5H_TOKENS = [
    "P5H KV/hidden commit ownership readiness",
    "kv_commit_readiness",
    "kv_commit_readiness_verified_not_executed",
    "jetspec_kv_commit_readiness",
    "validate_p5h_kv_commit_readiness",
    "test_p5h_kv_commit_readiness",
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
    "experiments/jetspec/jetspec_kv_commit_readiness",
    "jetspec_kv_commit_readiness",
    "validate_p5h_kv_commit_readiness",
    "test_p5h_kv_commit_readiness",
    "kv_commit_readiness_verified_not_executed",
]


class P5HKVCommitReadinessError(ValueError):
    """Raised when the P5H readiness contract is invalid."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5HKVCommitReadinessError(f"missing required file: {rel}")
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
            matched = [token for token in P5H_TOKENS if token in text]
            if matched:
                hits.append({"path": str(rel), "tokens": matched})
    return hits


def validate_p5h_kv_commit_readiness() -> dict[str, Any]:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"missing required P5H file: {rel}")

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5HKVCommitReadinessError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    try:
        fixture_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        actual = readiness.evaluate_fixture(fixture_data)
        expected = json.loads(EXPECTED_OUT.read_text(encoding="utf-8"))
        if actual != expected:
            errors.append("P5H smoke output differs from expected fixture output")
        if actual.get("status") != readiness.STATUS:
            errors.append(f"P5H smoke status mismatch: {actual.get('status')}")
        if actual.get("slot_ownership", {}).get("gather_positions") != [2, 3, 6]:
            errors.append("P5H smoke gather positions must be past_len + accepted_path")
        if actual.get("readiness_boundary", {}).get("no_production_kv_mutation") is not True:
            errors.append("P5H readiness boundary must report no_production_kv_mutation=true")
        mapping = actual.get("llama_cpp_ownership_mapping", {})
        missing = [key for key, value in mapping.items() if value != readiness.MISSING_PRIMITIVE and not str(value).startswith("llama_kv_cache_")]
        if missing:
            errors.append(f"P5H ownership mapping has invalid values: {missing}")
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        errors.append(f"P5H smoke fixture failed: {exc}")

    forbidden_hits = _scan_forbidden_roots()
    for hit in forbidden_hits:
        errors.append(f"P5H token appears in forbidden production path: {hit}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5H must not add explicit CMake wiring in {rel}: {matched}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5h_kv_commit_readiness_validated" if not errors else "p5h_kv_commit_readiness_invalid",
        "files_checked": [str(path) for path in REQUIRED_FILES],
        "forbidden_root_hits": forbidden_hits,
        "cmake_hits": cmake_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    parse_args(argv)
    out = validate_p5h_kv_commit_readiness()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
