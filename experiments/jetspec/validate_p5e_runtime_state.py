#!/usr/bin/env python3
"""Validate the P5E JetSpec runtime-state bookkeeping candidate.

P5E may track private JetSpec runtime state around target-tap ingestion inside the
explicit `draft-jetspec` route, but it must not emit drafts, execute a draft-head
graph, or add server/public/CMake/kernel routing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

P5E_ALLOWED_FILES = {
    pathlib.Path("common/speculative.cpp"),
    pathlib.Path("docs/speculative.md"),
}

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("common/speculative.cpp"): [
        "struct common_speculative_impl_draft_jetspec",
        "enum class jetspec_runtime_phase",
        "waiting_for_target_taps",
        "target_taps_captured",
        "enum class jetspec_runtime_failure",
        "missing_target_context",
        "invalid_target_taps",
        "struct jetspec_target_tap_row_state",
        "batch_index",
        "llama_pos    pos",
        "llama_seq_id seq_id",
        "target_tap_row_state",
        "n_target_tap_rows_cached",
        "n_runtime_state_resets",
        "n_runtime_draft_calls",
        "LLAMA_JETSPEC_STATE_TRACE",
        "disable_runtime_state(jetspec_runtime_failure::invalid_target_taps, tap_count, tap_width, taps)",
        "runtime_phase = jetspec_runtime_phase::target_taps_captured",
        "runtime_failure = jetspec_runtime_failure::none",
        "n_row_state != n_rows",
        "no_draft=1",
        "do not emit draft tokens",
    ],
    pathlib.Path("docs/speculative.md"): [
        "draft-jetspec",
        "runtime-state",
        "failure state",
        "LLAMA_JETSPEC_STATE_TRACE=1",
        "no draft tokens",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5e_runtime_state_candidate.md"): [
        "P5E runtime-state bookkeeping",
        "waiting_for_target_taps",
        "target_taps_captured",
        "invalid_target_taps",
        "LLAMA_JETSPEC_STATE_TRACE=1",
        "no draft tokens",
    ],
}

P5E_TOKENS = [
    "jetspec_runtime_phase",
    "jetspec_runtime_failure",
    "jetspec_target_tap_row_state",
    "LLAMA_JETSPEC_STATE_TRACE",
    "n_runtime_state_resets",
    "n_runtime_draft_calls",
]

FORBIDDEN_PATH_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("include/llama.h"): P5E_TOKENS,
    pathlib.Path("tools/server/server-context.cpp"): P5E_TOKENS,
    pathlib.Path("common/speculative.h"): P5E_TOKENS,
    pathlib.Path("common/arg.cpp"): P5E_TOKENS,
    pathlib.Path("src/models/jetspec_qwen3_draft_head.cpp"): [
        "LLAMA_JETSPEC_STATE_TRACE",
        "jetspec_runtime_phase",
        "jetspec_runtime_failure",
    ],
}

CMAKE_TOKENS = [
    "LLAMA_JETSPEC_STATE_TRACE",
    "jetspec_p5e_runtime_state",
    "validate_p5e_runtime_state",
    "test_p5e_runtime_state",
]

FORBIDDEN_ROOTS = [
    pathlib.Path("ggml/src"),
    pathlib.Path("examples"),
    pathlib.Path("pocs"),
    pathlib.Path("tests"),
]


class P5ERuntimeStateError(ValueError):
    """Raised when P5E runtime-state constraints are violated."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5ERuntimeStateError(f"missing required file: {rel}")
    return path.read_text(encoding="utf-8", errors="replace")


def _iter_source_files() -> list[pathlib.Path]:
    suffixes = {".c", ".cc", ".cpp", ".h", ".hpp", ".md"}
    roots = [pathlib.Path("src"), pathlib.Path("include"), pathlib.Path("common"), pathlib.Path("tools/server"), pathlib.Path("docs")]
    files: list[pathlib.Path] = []
    for root in roots:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in suffixes:
                files.append(path.relative_to(REPO_ROOT))
    return sorted(files)


def _cmake_files() -> list[pathlib.Path]:
    files = list(REPO_ROOT.rglob("CMakeLists.txt"))
    files.extend(REPO_ROOT.rglob("*.cmake"))
    return sorted(path.relative_to(REPO_ROOT) for path in files if path.is_file())


def _impl_slice(source: str) -> str:
    start = source.find("struct common_speculative_impl_draft_jetspec")
    end = source.find("struct common_speculative_impl_draft_mtp")
    if start < 0 or end < 0 or end <= start:
        raise P5ERuntimeStateError("cannot isolate draft-jetspec implementation slice")
    return source[start:end]


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
            matched = [token for token in P5E_TOKENS if token in text]
            if matched:
                hits.append({"path": str(rel), "tokens": matched})
    return hits


def validate_p5e_runtime_state() -> dict[str, Any]:
    errors: list[str] = []

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5ERuntimeStateError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    token_hits: list[dict[str, Any]] = []
    for rel in _iter_source_files():
        text = _read(rel)
        for token in P5E_TOKENS:
            if token in text:
                token_hits.append({"path": str(rel), "token": token})
                if rel not in P5E_ALLOWED_FILES:
                    errors.append(f"P5E token {token!r} appears outside allowlist: {rel}")

    for rel, tokens in FORBIDDEN_PATH_TOKENS.items():
        if not (REPO_ROOT / rel).exists():
            continue
        text = _read(rel)
        for token in tokens:
            if token in text:
                errors.append(f"forbidden P5E token {token!r} in {rel}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5E must not add explicit CMake wiring in {rel}: {matched}")

    forbidden_root_hits = _scan_forbidden_roots()
    for hit in forbidden_root_hits:
        errors.append(f"P5E token appears in forbidden root: {hit}")

    try:
        impl = _impl_slice(_read(pathlib.Path("common/speculative.cpp")))
    except P5ERuntimeStateError as exc:
        errors.append(str(exc))
        impl = ""
    if "result->push_back" in impl or "dp.result->push_back" in impl:
        errors.append("P5E must not emit draft tokens")
    for forbidden in ["llama_decode", "llama_graph", "tree_accept", "build_ancestor_matrix"]:
        if forbidden in impl:
            errors.append(f"P5E must not execute or wire tree/draft runtime token: {forbidden}")

    loader = _read(pathlib.Path("src/models/jetspec_qwen3_draft_head.cpp"))
    for token in [
        "runtime_supported=false",
        "preview_not_allowed",
        "unsupported_runtime",
        "build_arch_graph",
        "throw std::runtime_error(\"unsupported_runtime: JetSpec P5A has no graph execution path\")",
    ]:
        if token not in loader:
            errors.append(f"JetSpec draft-head loader no longer preserves fail-closed token: {token}")

    return {
        "ok": not errors,
        "errors": errors,
        "allowed_files": [str(path) for path in sorted(P5E_ALLOWED_FILES)],
        "token_hits": token_hits,
        "cmake_hits": cmake_hits,
        "forbidden_root_hits": forbidden_root_hits,
        "checks": {
            "required_file_count": len(REQUIRED_TOKENS),
            "source_files_scanned": len(_iter_source_files()),
            "cmake_files_scanned": len(_cmake_files()),
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable validation result")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = validate_p5e_runtime_state()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print("P5E runtime-state validation passed")
    else:
        print("P5E runtime-state validation failed", file=sys.stderr)
        for error in result["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
