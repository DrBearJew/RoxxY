#!/usr/bin/env python3
"""Validate the P5B JetSpec target-hidden tap capture candidate.

P5B is a private, default-off, side-channel-only target capture path. It must not
add P5C speculative/server routing or public API surface.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

P5B_ALLOWED_FILES = {
    pathlib.Path("src/llama-cparams.h"),
    pathlib.Path("src/llama-graph.h"),
    pathlib.Path("src/llama-graph.cpp"),
    pathlib.Path("src/llama-context.h"),
    pathlib.Path("src/llama-context.cpp"),
    pathlib.Path("src/llama-ext.h"),
    pathlib.Path("src/models/qwen35.cpp"),
    pathlib.Path("src/models/qwen35moe.cpp"),
}

P5B_DOWNSTREAM_ALLOWED_FILES = {
    pathlib.Path("common/speculative.cpp"),
}

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("src/llama-cparams.h"): [
        "jetspec_target_hidden_taps",
        "jetspec_target_hidden_taps_masked",
    ],
    pathlib.Path("src/llama-graph.h"): [
        "t_jetspec_target_hidden_taps",
        "get_jetspec_target_hidden_taps",
        "cparams.jetspec_target_hidden_taps",
        "cparams.jetspec_target_hidden_taps_masked",
    ],
    pathlib.Path("src/llama-graph.cpp"): [
        "t_jetspec_target_hidden_taps = nullptr",
        "ggml_set_output(t_jetspec_target_hidden_taps)",
    ],
    pathlib.Path("src/llama-context.h"): [
        "set_jetspec_target_hidden_taps",
        "get_jetspec_target_hidden_taps",
        "get_jetspec_target_hidden_tap_count",
        "get_jetspec_target_hidden_tap_width",
        "buffer_view<float> jetspec_target_hidden_taps",
    ],
    pathlib.Path("src/llama-context.cpp"): [
        "cparams.jetspec_target_hidden_taps = false",
        "set_jetspec_target_hidden_taps",
        "LLM_ARCH_QWEN35",
        "LLM_ARCH_QWEN35MOE",
        "target hidden width 2048",
        "5 * hparams.n_embd",
        "res->get_jetspec_target_hidden_taps()",
        "jetspec_target_hidden_taps.data",
    ],
    pathlib.Path("src/llama-ext.h"): [
        "llama_set_jetspec_target_hidden_taps",
        "llama_get_jetspec_target_hidden_taps",
        "llama_get_jetspec_target_hidden_tap_count",
        "llama_get_jetspec_target_hidden_tap_width",
        "private staging API",
    ],
    pathlib.Path("src/models/qwen35.cpp"): [
        "qwen35_jetspec_target_hidden_tap_layer",
        "il == 1 || il == 10 || il == 19 || il == 28 || il == 37",
        "cparams.jetspec_target_hidden_taps",
        "cparams.jetspec_target_hidden_taps_masked",
        "ggml_concat(ctx0, jetspec_target_hidden_taps, tap, 0)",
        "res->t_jetspec_target_hidden_taps",
        "concat width 10240",
    ],
    pathlib.Path("src/models/qwen35moe.cpp"): [
        "qwen35moe_jetspec_target_hidden_tap_layer",
        "il == 1 || il == 10 || il == 19 || il == 28 || il == 37",
        "cparams.jetspec_target_hidden_taps",
        "cparams.jetspec_target_hidden_taps_masked",
        "ggml_concat(ctx0, jetspec_target_hidden_taps, tap, 0)",
        "res->t_jetspec_target_hidden_taps",
        "concat width 10240",
    ],
}

P5B_TOKENS = [
    "jetspec_target_hidden_taps",
    "llama_set_jetspec_target_hidden_taps",
    "llama_get_jetspec_target_hidden_taps",
    "t_jetspec_target_hidden_taps",
    "jetspec_target_hidden_tap_layer",
]

FORBIDDEN_PUBLIC_OR_P5C = {
    pathlib.Path("include/llama.h"): ["jetspec_target_hidden_taps", "llama_set_jetspec_target_hidden_taps"],
    pathlib.Path("common/common.h"): ["jetspec_target_hidden_taps"],
    pathlib.Path("common/speculative.h"): ["jetspec_target_hidden_taps", "llama_set_jetspec_target_hidden_taps"],
    pathlib.Path("common/arg.cpp"): ["jetspec_target_hidden_taps", "llama_set_jetspec_target_hidden_taps"],
    pathlib.Path("tools/server/server-context.cpp"): ["jetspec_target_hidden_taps", "llama_set_jetspec_target_hidden_taps"],
}

CMAKE_TOKENS = [
    "jetspec_target_hidden_taps",
    "llama_set_jetspec_target_hidden_taps",
    "t_jetspec_target_hidden_taps",
]


class P5BTargetTapsError(ValueError):
    """Raised when P5B target tap constraints are violated."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5BTargetTapsError(f"missing required file: {rel}")
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


def validate_p5b_target_taps() -> dict[str, Any]:
    errors: list[str] = []

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5BTargetTapsError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    token_hits: list[dict[str, Any]] = []
    for rel in _iter_source_files():
        text = _read(rel)
        for token in P5B_TOKENS:
            if token in text:
                token_hits.append({"path": str(rel), "token": token})
                if rel not in P5B_ALLOWED_FILES and rel not in P5B_DOWNSTREAM_ALLOWED_FILES:
                    errors.append(f"P5B token {token!r} appears outside allowlist: {rel}")

    for rel, tokens in FORBIDDEN_PUBLIC_OR_P5C.items():
        if not (REPO_ROOT / rel).exists():
            continue
        text = _read(rel)
        for token in tokens:
            if token in text:
                errors.append(f"forbidden public/P5C token {token!r} in {rel}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5B must not add explicit CMake wiring in {rel}: {matched}")

    return {
        "ok": not errors,
        "errors": errors,
        "allowed_files": [str(path) for path in sorted(P5B_ALLOWED_FILES)],
        "token_hits": token_hits,
        "cmake_hits": cmake_hits,
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
    result = validate_p5b_target_taps()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print("P5B target hidden tap validation passed")
    else:
        print("P5B target hidden tap validation failed", file=sys.stderr)
        for error in result["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
