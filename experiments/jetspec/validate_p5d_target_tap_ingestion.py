#!/usr/bin/env python3
"""Validate the P5D JetSpec target-tap ingestion candidate.

P5D may ingest target-side P5B tap rows inside the explicit `draft-jetspec` route,
but it must not emit drafts or add server/public/CMake/kernel routing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

P5D_ALLOWED_FILES = {
    pathlib.Path("common/speculative.cpp"),
    pathlib.Path("docs/speculative.md"),
}

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("common/speculative.cpp"): [
        "struct common_speculative_impl_draft_jetspec",
        "target_tap_rows",
        "n_target_tap_rows_total",
        "n_target_tap_process",
        "target_tap_hash_last",
        "target_taps_active",
        "LLAMA_JETSPEC_TRACE",
        "LLAMA_JETSPEC_TAP_TRACE",
        "llama_set_jetspec_target_hidden_taps(this->params.ctx_tgt, true, true)",
        "llama_set_jetspec_target_hidden_taps(params.ctx_tgt, false, true)",
        "batch.logits",
        "llama_get_jetspec_target_hidden_taps(params.ctx_tgt)",
        "tap_count != 5 || tap_width != 10240 || taps == nullptr",
        "target_taps_active = false",
        "target_tap_rows.assign(taps, taps + n_values)",
        "common_speculative_fnv1a64(target_tap_rows.data(), n_values * sizeof(float))",
        "no draft tokens will be generated",
        "do not emit draft tokens",
    ],
    pathlib.Path("docs/speculative.md"): [
        "draft-jetspec",
        "target tap",
        "LLAMA_JETSPEC_TRACE=1",
        "no draft tokens",
    ],
}

P5D_TOKENS = [
    "LLAMA_JETSPEC_TAP_TRACE",
    "target_tap_rows",
    "target_tap_hash_last",
]

FORBIDDEN_PATH_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("include/llama.h"): P5D_TOKENS,
    pathlib.Path("tools/server/server-context.cpp"): P5D_TOKENS,
    pathlib.Path("common/speculative.h"): P5D_TOKENS,
    pathlib.Path("common/arg.cpp"): P5D_TOKENS,
}

CMAKE_TOKENS = [
    "LLAMA_JETSPEC_TAP_TRACE",
    "target_tap_rows",
    "jetspec_p5d_target_tap_ingestion",
]

FORBIDDEN_ROOTS = [
    pathlib.Path("ggml/src"),
    pathlib.Path("examples"),
    pathlib.Path("pocs"),
    pathlib.Path("tests"),
]


class P5DTargetTapIngestionError(ValueError):
    """Raised when P5D target-tap ingestion constraints are violated."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5DTargetTapIngestionError(f"missing required file: {rel}")
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
        raise P5DTargetTapIngestionError("cannot isolate draft-jetspec implementation slice")
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
            matched = [token for token in P5D_TOKENS if token in text]
            if matched:
                hits.append({"path": str(rel), "tokens": matched})
    return hits


def validate_p5d_target_tap_ingestion() -> dict[str, Any]:
    errors: list[str] = []

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5DTargetTapIngestionError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    token_hits: list[dict[str, Any]] = []
    for rel in _iter_source_files():
        text = _read(rel)
        for token in P5D_TOKENS:
            if token in text:
                token_hits.append({"path": str(rel), "token": token})
                if rel not in P5D_ALLOWED_FILES:
                    errors.append(f"P5D token {token!r} appears outside allowlist: {rel}")

    for rel, tokens in FORBIDDEN_PATH_TOKENS.items():
        if not (REPO_ROOT / rel).exists():
            continue
        text = _read(rel)
        for token in tokens:
            if token in text:
                errors.append(f"forbidden P5D token {token!r} in {rel}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5D must not add explicit CMake wiring in {rel}: {matched}")

    forbidden_root_hits = _scan_forbidden_roots()
    for hit in forbidden_root_hits:
        errors.append(f"P5D token appears in forbidden root: {hit}")

    try:
        impl = _impl_slice(_read(pathlib.Path("common/speculative.cpp")))
    except P5DTargetTapIngestionError as exc:
        errors.append(str(exc))
        impl = ""
    if "result->push_back" in impl or "dp.result->push_back" in impl:
        errors.append("P5D must not emit draft tokens")
    if "llama_decode" in impl or "llama_graph" in impl:
        errors.append("P5D must not execute draft-head graphs")

    return {
        "ok": not errors,
        "errors": errors,
        "allowed_files": [str(path) for path in sorted(P5D_ALLOWED_FILES)],
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
    result = validate_p5d_target_tap_ingestion()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print("P5D target tap ingestion validation passed")
    else:
        print("P5D target tap ingestion validation failed", file=sys.stderr)
        for error in result["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
