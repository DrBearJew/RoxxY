#!/usr/bin/env python3
"""Validate the P5N JetSpec transaction-plan scaffold source slice."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

P5N_ALLOWED_FILES = {
    pathlib.Path("common/speculative.cpp"),
    pathlib.Path("docs/speculative.md"),
}

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("common/speculative.cpp"): [
        "JETSPEC_TRANSACTION_PHASE_COUNT  = 9",
        "JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT = 7",
        "JETSPEC_TRANSACTION_PHASE_ORDER",
        "JETSPEC_TRANSACTION_ROLLBACK_POINTS",
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
        "transaction_plan_scaffold_ready",
        "invalid_transaction_plan",
        "build_transaction_plan_scaffold",
        "transaction_plan_ready",
        "transaction_plan_hash_last",
        "transaction_plan_phase_count_last",
        "transaction_plan_rollback_count_last",
        "transaction_plan_ready=%d",
        "transaction_plan_hash=%016",
        "no_kv_mutation=1",
        "no_publish=1",
        "no_draft_tokens=1",
        "do not emit draft tokens",
    ],
    pathlib.Path("docs/speculative.md"): [
        "P5N transaction-plan scaffold",
        "snapshot_pre_round",
        "reserve_transient_tree_pages",
        "build_tree",
        "build_verify_mask",
        "accept_path",
        "commit_tokens",
        "commit_hidden_kv_survivors",
        "discard_rejected_branches",
        "publish_post_commit_state",
        "rollback failpoints",
        "does not publish",
        "mutate KV",
        "dispatch CUDA",
        "emits no draft",
        "transaction plan readiness",
        "transaction plan hash",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5n_transaction_scaffold_candidate.md"): [
        "JetSpec P5N transaction-plan scaffold candidate",
        "approved bounded production-source slice",
        "default-off and non-drafting",
        "common/speculative.cpp",
        "docs/speculative.md",
        "JETSPEC_TRANSACTION_PHASE_COUNT = 9",
        "JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT = 7",
        "build_transaction_plan_scaffold",
        "transaction_plan_scaffold_ready",
        "invalid_transaction_plan",
        "no draft-head graph execution",
        "no draft tokens emitted",
        "no real KV mutation",
        "no CUDA dispatch",
        "no server route",
        "no public API",
        "no CMake wiring",
    ],
}

P5N_TOKENS = [
    "P5N transaction-plan scaffold",
    "jetspec_p5n_transaction_scaffold",
    "validate_p5n_transaction_scaffold",
    "test_p5n_transaction_scaffold",
    "JETSPEC_TRANSACTION_PHASE_ORDER",
    "JETSPEC_TRANSACTION_ROLLBACK_POINTS",
    "transaction_plan_scaffold_ready",
    "invalid_transaction_plan",
    "transaction_plan_hash_last",
]

FORBIDDEN_SOURCE_ROOTS = [
    pathlib.Path("src"),
    pathlib.Path("include"),
    pathlib.Path("tools/server"),
    pathlib.Path("tests"),
    pathlib.Path("examples"),
    pathlib.Path("pocs"),
    pathlib.Path("ggml/src"),
]

CMAKE_TOKENS = [
    "jetspec_p5n_transaction_scaffold",
    "validate_p5n_transaction_scaffold",
    "test_p5n_transaction_scaffold",
    "JETSPEC_TRANSACTION_PHASE_ORDER",
    "transaction_plan_scaffold_ready",
]

FORBIDDEN_IMPL_TOKENS = [
    "llama_decode",
    "llama_graph",
    "tree_accept",
    "llama_kv_cache",
    "result->push_back",
]


class P5NTransactionScaffoldError(ValueError):
    """Raised when P5N source-slice constraints are violated."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5NTransactionScaffoldError(f"missing required file: {rel}")
    return path.read_text(encoding="utf-8", errors="replace")


def _iter_source_files() -> list[pathlib.Path]:
    suffixes = {".c", ".cc", ".cpp", ".h", ".hpp", ".md"}
    roots = [pathlib.Path("common"), pathlib.Path("docs"), *FORBIDDEN_SOURCE_ROOTS]
    files: list[pathlib.Path] = []
    for root in roots:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in suffixes:
                files.append(path.relative_to(REPO_ROOT))
    return sorted(set(files))


def _cmake_files() -> list[pathlib.Path]:
    files = list(REPO_ROOT.rglob("CMakeLists.txt"))
    files.extend(REPO_ROOT.rglob("*.cmake"))
    return sorted(path.relative_to(REPO_ROOT) for path in files if path.is_file())


def _impl_slice(source: str) -> str:
    start = source.find("struct common_speculative_impl_draft_jetspec")
    end = source.find("struct common_speculative_impl_draft_mtp")
    if start < 0 or end < 0 or end <= start:
        raise P5NTransactionScaffoldError("cannot isolate draft-jetspec implementation slice")
    return source[start:end]


def validate_p5n_transaction_scaffold() -> dict[str, Any]:
    errors: list[str] = []

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5NTransactionScaffoldError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    token_hits: list[dict[str, Any]] = []
    for rel in _iter_source_files():
        text = _read(rel)
        for token in P5N_TOKENS:
            if token in text:
                token_hits.append({"path": str(rel), "token": token})
                if rel not in P5N_ALLOWED_FILES:
                    errors.append(f"P5N token {token!r} appears outside allowlist: {rel}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5N must not add explicit CMake wiring in {rel}: {matched}")

    try:
        source = _read(pathlib.Path("common/speculative.cpp"))
        impl = _impl_slice(source)
    except P5NTransactionScaffoldError as exc:
        errors.append(str(exc))
        impl = ""

    for token in FORBIDDEN_IMPL_TOKENS:
        if token in impl:
            errors.append(f"draft-jetspec P5N implementation must not contain {token!r}")
    if "push_back(JETSPEC_TRANSACTION_PHASE_COUNT)" not in impl:
        errors.append("P5N transaction hash must include phase count")
    if "push_back(JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT)" not in impl:
        errors.append("P5N transaction hash must include rollback point count")
    if "runtime_phase = jetspec_runtime_phase::transaction_plan_scaffold_ready" not in impl:
        errors.append("P5N must mark transaction_plan_scaffold_ready only after scaffold build")
    if "disable_runtime_state(jetspec_runtime_failure::invalid_transaction_plan" not in impl:
        errors.append("P5N must fail closed with invalid_transaction_plan")
    if "// fail closed: do not emit draft tokens" not in impl:
        errors.append("P5N must keep the no-draft fail-closed comment")

    docs = _read(pathlib.Path("docs/speculative.md"))
    for token in ["does not publish", "mutate KV", "dispatch CUDA", "emits no draft tokens"]:
        if token not in docs:
            errors.append(f"docs/speculative.md missing P5N boundary token: {token}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5n_transaction_scaffold_validated" if not errors else "p5n_transaction_scaffold_invalid",
        "allowed_files": sorted(str(path) for path in P5N_ALLOWED_FILES),
        "token_hits": token_hits,
        "cmake_hits": cmake_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    parse_args(argv)
    out = validate_p5n_transaction_scaffold()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main([]))
