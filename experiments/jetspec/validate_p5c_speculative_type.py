#!/usr/bin/env python3
"""Validate the P5C JetSpec speculative type candidate.

P5C may add only an explicit, default-off, fail-closed `draft-jetspec` route. A
later approved runtime-loader slice may reference the type in server model-only
binding, but must not add a public API, CMake wiring, kernels, or a runtime tree
implementation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

P5C_ALLOWED_FILES = {
    pathlib.Path("common/common.h"),
    pathlib.Path("common/speculative.cpp"),
    pathlib.Path("docs/speculative.md"),
    pathlib.Path("tools/server/server-context.cpp"),
}

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("common/common.h"): [
        "COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC",
        "JetSpec draft-head speculative decoding",
    ],
    pathlib.Path("common/speculative.cpp"): [
        '{"draft-jetspec", COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC}',
        '{"jetspec",       COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC}',
        "struct common_speculative_impl_draft_jetspec",
        "COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC",
        "return \"draft-jetspec\"",
        "LLAMA_JETSPEC_EXPERIMENTAL",
        "runtime_supported=false",
        "no draft tokens will be generated",
        "!has_draft_jetspec",
        "static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 10)",
        "llama_set_jetspec_target_hidden_taps(params.draft.ctx_tgt, true, true)",
        "llama_set_jetspec_target_hidden_taps(params.draft.ctx_tgt, false, true)",
    ],
    pathlib.Path("docs/speculative.md"): [
        "draft-jetspec",
        "LLAMA_JETSPEC_EXPERIMENTAL=1",
        "runtime_supported=false",
        "fail-closed",
    ],
    pathlib.Path("tools/server/server-context.cpp"): [
        "COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC",
        "spec_jetspec",
        "llama_model_link_shared_tensors(model_dft.get(), model_tgt)",
        "params_base.speculative.draft.ctx_dft = nullptr",
        "loaded JetSpec draft-head model-only binding; draft context and graph execution remain disabled",
    ],
}

P5C_TOKENS = [
    "COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC",
    "draft-jetspec",
]

FORBIDDEN_PATH_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("include/llama.h"): P5C_TOKENS,
    pathlib.Path("common/speculative.h"): P5C_TOKENS,
    pathlib.Path("common/arg.cpp"): ["COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC"],
}

CMAKE_TOKENS = [
    "draft-jetspec",
    "COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC",
    "jetspec_p5c_speculative_type",
]

FORBIDDEN_ROOTS = [
    pathlib.Path("ggml/src"),
    pathlib.Path("examples"),
    pathlib.Path("pocs"),
    pathlib.Path("tests"),
]


class P5CSpeculativeTypeError(ValueError):
    """Raised when P5C speculative type constraints are violated."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5CSpeculativeTypeError(f"missing required file: {rel}")
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
            matched = [token for token in P5C_TOKENS if token in text]
            if matched:
                hits.append({"path": str(rel), "tokens": matched})
    return hits


def validate_p5c_speculative_type() -> dict[str, Any]:
    errors: list[str] = []

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5CSpeculativeTypeError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    token_hits: list[dict[str, Any]] = []
    for rel in _iter_source_files():
        text = _read(rel)
        for token in P5C_TOKENS:
            if token in text:
                token_hits.append({"path": str(rel), "token": token})
                if rel not in P5C_ALLOWED_FILES:
                    errors.append(f"P5C token {token!r} appears outside allowlist: {rel}")

    for rel, tokens in FORBIDDEN_PATH_TOKENS.items():
        if not (REPO_ROOT / rel).exists():
            continue
        text = _read(rel)
        for token in tokens:
            if token in text:
                errors.append(f"forbidden P5C token {token!r} in {rel}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5C must not add explicit CMake wiring in {rel}: {matched}")

    forbidden_root_hits = _scan_forbidden_roots()
    for hit in forbidden_root_hits:
        errors.append(f"P5C token appears in forbidden root: {hit}")

    source = _read(pathlib.Path("common/speculative.cpp"))
    if "result->push_back" in source[source.find("common_speculative_impl_draft_jetspec"):source.find("struct common_speculative_impl_draft_mtp")]:
        errors.append("P5C placeholder must not emit draft tokens")

    return {
        "ok": not errors,
        "errors": errors,
        "allowed_files": [str(path) for path in sorted(P5C_ALLOWED_FILES)],
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
    result = validate_p5c_speculative_type()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print("P5C speculative type validation passed")
    else:
        print("P5C speculative type validation failed", file=sys.stderr)
        for error in result["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
