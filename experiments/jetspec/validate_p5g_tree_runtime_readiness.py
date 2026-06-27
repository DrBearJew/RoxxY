#!/usr/bin/env python3
"""Validate the inert P5G JetSpec tree-runtime readiness contract."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import jetspec_tree_runtime_readiness as readiness


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
FIXTURE = HERE / "fixtures" / "jetspec_tree_runtime_readiness_smoke.json"
EXPECTED_OUT = HERE / "fixtures" / "jetspec_tree_runtime_readiness_smoke.out.json"

REQUIRED_FILES = [
    pathlib.Path("experiments/jetspec/jetspec_p5g_tree_runtime_readiness.md"),
    pathlib.Path("experiments/jetspec/jetspec_tree_runtime_readiness.py"),
    pathlib.Path("experiments/jetspec/validate_p5g_tree_runtime_readiness.py"),
    pathlib.Path("experiments/jetspec/test_p5g_tree_runtime_readiness.py"),
    pathlib.Path("experiments/jetspec/fixtures/jetspec_tree_runtime_readiness_smoke.json"),
    pathlib.Path("experiments/jetspec/fixtures/jetspec_tree_runtime_readiness_smoke.out.json"),
]

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("experiments/jetspec/jetspec_p5g_tree_runtime_readiness.md"): [
        "P5G tree-runtime readiness",
        "inert readiness contract only",
        "not production tree-runtime approval",
        "experiments/jetspec/",
        "full-vocab softmax",
        "not top-k-only renormalization",
        "parent-before-child",
        "ancestor-only",
        "siblings, descendants, rejected branches",
        "other-sequence tree columns are hidden",
        "accepted_path is root-inclusive",
        "acceptance_length excludes the root",
        "later child overwrites the earlier one",
        "[accepted draft tokens | correction]",
        "[root | accepted] only",
        "correction hidden is not appended",
        "max_len + accepted_path",
        "tree_runtime_readiness_verified_not_executed",
        "no target/draft `llama_context` runtime instantiation",
        "no draft-head graph execution",
        "no draft tokens",
        "no KV cache mutation",
        "no server route",
    ],
    pathlib.Path("experiments/jetspec/jetspec_tree_runtime_readiness.py"): [
        "tree_runtime_readiness_verified_not_executed",
        "FULL_VOCAB_LOGPROB_SOURCE",
        "top-k-only renormalization is forbidden",
        "parent index must precede child",
        "prefix_visible_to_all",
        "ancestor_only_self_included",
        "other_sequence_tree_cols_hidden",
        "accepted_path_root_inclusive",
        "acceptance_length_excludes_root",
        "later_child_overwrites_earlier",
        "[accepted draft tokens | correction]",
        "[root | accepted] only",
        "max_len + accepted_path",
        "draft_head_graph_executed",
        "llama_context_runtime_instantiated",
        "draft_tokens_emitted",
        "kv_cache_mutated",
        "server_route_added",
    ],
    pathlib.Path("experiments/jetspec/test_p5g_tree_runtime_readiness.py"): [
        "test_smoke_fixture_matches_expected_output",
        "test_rejects_topk_only_renormalization",
        "test_rejects_gather_position_mismatch",
        "test_duplicate_child_token_overwrite_is_deterministic",
        "test_rejects_runtime_execution_boundary_crossing",
        "test_validator_passes",
    ],
}

P5G_TOKENS = [
    "P5G tree-runtime readiness",
    "tree_runtime_readiness",
    "tree_runtime_readiness_verified_not_executed",
    "jetspec_tree_runtime_readiness",
    "validate_p5g_tree_runtime_readiness",
    "test_p5g_tree_runtime_readiness",
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
    "experiments/jetspec/jetspec_tree_runtime_readiness",
    "jetspec_tree_runtime_readiness",
    "validate_p5g_tree_runtime_readiness",
    "test_p5g_tree_runtime_readiness",
    "tree_runtime_readiness_verified_not_executed",
]


class P5GTreeRuntimeReadinessError(ValueError):
    """Raised when the P5G readiness contract is invalid."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5GTreeRuntimeReadinessError(f"missing required file: {rel}")
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
            matched = [token for token in P5G_TOKENS if token in text]
            if matched:
                hits.append({"path": str(rel), "tokens": matched})
    return hits


def validate_p5g_tree_runtime_readiness() -> dict[str, Any]:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"missing required P5G file: {rel}")

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5GTreeRuntimeReadinessError as exc:
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
            errors.append("P5G smoke output differs from expected fixture output")
        if actual.get("status") != readiness.STATUS:
            errors.append(f"P5G smoke status mismatch: {actual.get('status')}")
        boundary = actual.get("readiness_boundary", {})
        if boundary.get("no_draft_runtime_execution") is not True:
            errors.append("P5G readiness boundary must report no_draft_runtime_execution=true")
        if actual.get("gather", {}).get("positions") != [100, 101, 104]:
            errors.append("P5G smoke gather positions must be max_len + accepted_path")
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        errors.append(f"P5G smoke fixture failed: {exc}")

    forbidden_hits = _scan_forbidden_roots()
    for hit in forbidden_hits:
        errors.append(f"P5G token appears in forbidden production path: {hit}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5G must not add explicit CMake wiring in {rel}: {matched}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5g_tree_runtime_readiness_validated" if not errors else "p5g_tree_runtime_readiness_invalid",
        "files_checked": [str(path) for path in REQUIRED_FILES],
        "forbidden_root_hits": forbidden_hits,
        "cmake_hits": cmake_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    parse_args(argv)
    out = validate_p5g_tree_runtime_readiness()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
