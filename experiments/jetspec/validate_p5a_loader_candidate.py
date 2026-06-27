#!/usr/bin/env python3
"""Validate the P5A JetSpec production loader candidate stays default-off.

This validator lives under experiments/jetspec, but it intentionally inspects the
small approved P5A production hook set. It does not execute llama.cpp; runtime
fail-closed behavior is verified separately by loading a metadata-only preview.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

P5A_ALLOWED_PRODUCTION_FILES = {
    pathlib.Path("src/llama-arch.h"),
    pathlib.Path("src/llama-arch.cpp"),
    pathlib.Path("src/llama-model.cpp"),
    pathlib.Path("src/models/models.h"),
    pathlib.Path("src/models/jetspec_qwen3_draft_head.cpp"),
}

# Later approved P5F binding preflight validates the draft architecture string in
# common/speculative.cpp without adding a loader API or executable graph.
P5A_APPROVED_DOWNSTREAM_TOKEN_HITS = {
    (pathlib.Path("common/speculative.cpp"), "jetspec_qwen3_draft_head"),
}

REQUIRED_FILE_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("src/llama-arch.h"): [
        "LLM_ARCH_JETSPEC_QWEN3_DRAFT_HEAD",
    ],
    pathlib.Path("src/llama-arch.cpp"): [
        "LLM_ARCH_JETSPEC_QWEN3_DRAFT_HEAD",
        '"jetspec_qwen3_draft_head"',
    ],
    pathlib.Path("src/llama-model.cpp"): [
        "case LLM_ARCH_JETSPEC_QWEN3_DRAFT_HEAD:",
        "new llama_model_jetspec_qwen3_draft_head(params)",
        "case LLM_ARCH_JETSPEC_QWEN3_DRAFT_HEAD:",
        "LLAMA_ROPE_TYPE_NORM",
    ],
    pathlib.Path("src/models/models.h"): [
        "struct llama_model_jetspec_qwen3_draft_head",
        "void load_hparams(llama_model_loader & ml) override;",
        "void load_arch_tensors(llama_model_loader & ml) override;",
        "build_arch_graph",
    ],
    pathlib.Path("src/models/jetspec_qwen3_draft_head.cpp"): [
        "LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD",
        "LLAMA_JETSPEC_EXPERIMENTAL",
        "preview_not_allowed",
        "unsupported_runtime",
        "jetspec.experimental.runtime_supported",
        "jetspec.experimental.metadata_only",
        "n_tensors == 0 || n_tensors == 91",
        "GGML_TYPE_BF16",
        "draft.fc.weight",
        "draft.layers.",
        "build_arch_graph",
        "has no graph execution path",
    ],
}

PRODUCTION_SCAN_ROOTS = [
    pathlib.Path("src"),
    pathlib.Path("common"),
    pathlib.Path("tools/server"),
    pathlib.Path("include"),
    pathlib.Path("docs"),
]

PRODUCTION_TOKENS = [
    "LLM_ARCH_JETSPEC",
    "llama_model_jetspec_qwen3_draft_head",
    "jetspec_qwen3_draft_head",
    "preview_not_allowed",
    "unsupported_runtime",
]

CMAKE_TOKENS = [
    "jetspec_qwen3_draft_head",
    "LLM_ARCH_JETSPEC_QWEN3_DRAFT_HEAD",
]

FORBIDDEN_PRODUCTION_TOKENS = [
    "LLAMA_CONTEXT_TYPE_JETSPEC",
]


class P5ALoaderCandidateError(ValueError):
    """Raised when P5A production candidate constraints are violated."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5ALoaderCandidateError(f"required P5A file is missing: {rel}")
    return path.read_text(encoding="utf-8", errors="replace")


def _iter_scan_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    suffixes = {".c", ".cc", ".cpp", ".h", ".hpp", ".md"}
    for root in PRODUCTION_SCAN_ROOTS:
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


def validate_p5a_loader_candidate() -> dict[str, Any]:
    errors: list[str] = []

    for rel, tokens in REQUIRED_FILE_TOKENS.items():
        try:
            text = _read(rel)
        except P5ALoaderCandidateError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    p5a_source = _read(pathlib.Path("src/models/jetspec_qwen3_draft_head.cpp"))
    if "return std::make_unique" in p5a_source or "ggml_build_forward_expand" in p5a_source:
        errors.append("P5A loader source must not construct an executable graph")
    if "runtime_supported=false" not in p5a_source and "runtime_supported=false" not in _read(pathlib.Path("experiments/jetspec/jetspec_p5_default_off_plan.md")):
        errors.append("P5A evidence must preserve runtime_supported=false language")

    token_hits: list[dict[str, Any]] = []
    for rel in _iter_scan_files():
        text = _read(rel)
        for token in PRODUCTION_TOKENS:
            if token in text:
                token_hits.append({"path": str(rel), "token": token})
                if rel not in P5A_ALLOWED_PRODUCTION_FILES and (rel, token) not in P5A_APPROVED_DOWNSTREAM_TOKEN_HITS:
                    errors.append(f"production token {token!r} appears outside P5A allowlist/downstream approvals: {rel}")

    for rel in [pathlib.Path("common/common.h"), pathlib.Path("common/speculative.cpp"), pathlib.Path("common/speculative.h"), pathlib.Path("common/arg.cpp"), pathlib.Path("tools/server/server-context.cpp"), pathlib.Path("include/llama.h")]:
        if (REPO_ROOT / rel).exists():
            text = _read(rel)
            for token in FORBIDDEN_PRODUCTION_TOKENS:
                if token in text:
                    errors.append(f"P5A must not wire P5C token {token!r} in {rel}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5A must rely on existing model glob, not explicit CMake token in {rel}: {matched}")

    result = {
        "ok": not errors,
        "errors": errors,
        "allowed_files": [str(path) for path in sorted(P5A_ALLOWED_PRODUCTION_FILES)],
        "token_hits": token_hits,
        "cmake_hits": cmake_hits,
        "checks": {
            "required_file_count": len(REQUIRED_FILE_TOKENS),
            "production_files_scanned": len(_iter_scan_files()),
            "cmake_files_scanned": len(_cmake_files()),
        },
    }
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable validation result")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = validate_p5a_loader_candidate()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print("P5A loader candidate validation passed")
    else:
        print("P5A loader candidate validation failed", file=sys.stderr)
        for error in result["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
