#!/usr/bin/env python3
"""Validate the P5F JetSpec draft-head/target binding preflight candidate.

P5F may validate explicit `draft-jetspec` target/draft bindings before the private
JetSpec implementation is instantiated, but it must not emit drafts, execute a
draft-head graph, or add server/public/CMake/kernel routing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

P5F_ALLOWED_FILES = {
    pathlib.Path("common/speculative.cpp"),
    pathlib.Path("docs/speculative.md"),
    pathlib.Path("src/models/jetspec_qwen3_draft_head.cpp"),
}

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("common/speculative.cpp"): [
        "common_speculative_jetspec_preflight",
        "common_speculative_jetspec_expect_i32",
        "common_speculative_jetspec_expect_meta_str",
        "llama_model_meta_val_str(model, key, value, sizeof(value))",
        "JETSPEC_QWEN36_DRAFT_BLOCK_SIZE  = 16",
        "JETSPEC_QWEN36_TARGET_TAP_COUNT  = 5",
        "JETSPEC_QWEN36_TARGET_HIDDEN     = 2048",
        "JETSPEC_QWEN36_TARGET_LAYERS     = 40",
        "JETSPEC_QWEN36_TARGET_TAP_WIDTH  = JETSPEC_QWEN36_TARGET_TAP_COUNT * JETSPEC_QWEN36_TARGET_HIDDEN",
        "JETSPEC_QWEN36_DRAFT_LAYERS      = 8",
        "JETSPEC_QWEN36_DRAFT_HEADS       = 32",
        "JETSPEC_QWEN36_DRAFT_HEADS_KV    = 4",
        "JETSPEC_QWEN36_VOCAB_SIZE        = 248320",
        "missing target context",
        "missing target or draft model",
        "missing target vocab",
        "general.architecture",
        "jetspec_qwen3_draft_head",
        "jetspec.architecture",
        "qwen3_draft_head",
        "jetspec.source_architecture",
        "DFlashDraftModel",
        "jetspec.tensor_data_dtype",
        "bfloat16",
        "target.n_embd",
        "target.effective_n_layer",
        "qwen35moe.nextn_predict_layers",
        "target.vocab",
        "draft.n_ctx_train",
        "draft.n_embd",
        "draft.n_layer",
        "draft.n_head",
        "draft.n_head_kv",
        "draft.vocab",
        "target_tap_count",
        "target_tap_width",
        "target_tap_width_vs_target",
        "tap_count * target_hidden",
        "common_speculative_jetspec_preflight(params.draft, tap_count, tap_width, preflight_reason)",
        "draft-jetspec preflight failed",
        "llama_set_jetspec_target_hidden_taps(params.draft.ctx_tgt, false, true)",
        "invalid_binding",
        "no draft tokens will be generated",
    ],
    pathlib.Path("src/models/jetspec_qwen3_draft_head.cpp"): [
        "jetspec_require_target_tensor",
        "missing required target tensor",
        "target tensor",
        "must have shape",
        "jetspec_require_target_tensor(main_model, \"token_embd.weight\", { 2048, 248320 })",
        "jetspec_require_target_tensor(main_model, \"output.weight\", { 2048, 248320 })",
        "jetspec_require_target_tensor(main_model, \"output_norm.weight\", { 2048 })",
        "target-owned JetSpec draft-head artifact",
        "throw std::runtime_error(\"unsupported_runtime: JetSpec draft-head graph execution is not implemented\")",
    ],
    pathlib.Path("docs/speculative.md"): [
        "draft-jetspec",
        "binding preflight",
        "metadata/shape",
        "target tap count/width",
        "target tensor presence/shape",
        "emits no draft",
        "tokens and does not execute the draft head",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5f_binding_preflight_candidate.md"): [
        "P5F binding preflight",
        "general.architecture=jetspec_qwen3_draft_head",
        "jetspec.architecture=qwen3_draft_head",
        "target shape: hidden size 2048, layer count 40, vocab size 248320",
        "draft shape: context train size 16, hidden size 2048, layer count 8, heads 32",
        "target tap shape: count 5, width 10240",
        "tap_count *",
        "no draft tokens",
    ],
}

P5F_TOKENS = [
    "common_speculative_jetspec_preflight",
    "JETSPEC_QWEN36_TARGET_TAP_WIDTH",
    "target_tap_width_vs_target",
    "draft-jetspec preflight failed",
    "jetspec_require_target_tensor",
    "jetspec_p5f_binding_preflight",
]

FORBIDDEN_PATH_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("include/llama.h"): P5F_TOKENS,
    pathlib.Path("tools/server/server-context.cpp"): P5F_TOKENS,
    pathlib.Path("common/speculative.h"): P5F_TOKENS,
    pathlib.Path("common/arg.cpp"): P5F_TOKENS,
    pathlib.Path("src/models/jetspec_qwen3_draft_head.cpp"): [
        "common_speculative_jetspec_preflight",
        "draft-jetspec preflight failed",
        "target_tap_width_vs_target",
    ],
}

CMAKE_TOKENS = [
    "draft-jetspec preflight failed",
    "jetspec_p5f_binding_preflight",
    "validate_p5f_binding_preflight",
    "test_p5f_binding_preflight",
]

FORBIDDEN_ROOTS = [
    pathlib.Path("ggml/src"),
    pathlib.Path("examples"),
    pathlib.Path("pocs"),
    pathlib.Path("tests"),
]


class P5FBindingPreflightError(ValueError):
    """Raised when P5F binding-preflight constraints are violated."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5FBindingPreflightError(f"missing required file: {rel}")
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
        raise P5FBindingPreflightError("cannot isolate draft-jetspec implementation slice")
    return source[start:end]


def _preflight_slice(source: str) -> str:
    start = source.find("static bool common_speculative_jetspec_preflight")
    end = source.find("static bool common_speculative_are_compatible")
    if start < 0 or end < 0 or end <= start:
        raise P5FBindingPreflightError("cannot isolate draft-jetspec preflight slice")
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
            matched = [token for token in P5F_TOKENS if token in text]
            if matched:
                hits.append({"path": str(rel), "tokens": matched})
    return hits


def validate_p5f_binding_preflight() -> dict[str, Any]:
    errors: list[str] = []

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5FBindingPreflightError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    token_hits: list[dict[str, Any]] = []
    for rel in _iter_source_files():
        text = _read(rel)
        for token in P5F_TOKENS:
            if token in text:
                token_hits.append({"path": str(rel), "token": token})
                if rel not in P5F_ALLOWED_FILES:
                    errors.append(f"P5F token {token!r} appears outside allowlist: {rel}")

    for rel, tokens in FORBIDDEN_PATH_TOKENS.items():
        if not (REPO_ROOT / rel).exists():
            continue
        text = _read(rel)
        for token in tokens:
            if token in text:
                errors.append(f"forbidden P5F token {token!r} in {rel}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5F must not add explicit CMake wiring in {rel}: {matched}")

    forbidden_root_hits = _scan_forbidden_roots()
    for hit in forbidden_root_hits:
        errors.append(f"P5F token appears in forbidden root: {hit}")

    try:
        source = _read(pathlib.Path("common/speculative.cpp"))
        impl = _impl_slice(source)
        preflight = _preflight_slice(source)
    except P5FBindingPreflightError as exc:
        errors.append(str(exc))
        impl = ""
        preflight = ""

    if "result->push_back" in impl or "dp.result->push_back" in impl:
        errors.append("P5F must not emit draft tokens")
    for forbidden in ["llama_decode", "llama_graph", "tree_accept", "build_ancestor_matrix"]:
        if forbidden in impl or forbidden in preflight:
            errors.append(f"P5F must not execute or wire tree/draft runtime token: {forbidden}")

    for required in [
        "llama_get_model(params.ctx_tgt)",
        "params.ctx_dft != nullptr ? llama_get_model(params.ctx_dft) : params.model",
        "llama_model_get_vocab(model_tgt)",
        "common_speculative_jetspec_meta_i32",
        "qwen35moe.nextn_predict_layers",
        "llama_model_n_ctx_train(model_dft)",
        "llama_model_n_embd(model_tgt)",
        "llama_model_n_layer(model_tgt)",
        "llama_model_n_embd(model_dft)",
        "llama_model_n_layer(model_dft)",
        "llama_model_n_head(model_dft)",
        "llama_model_n_head_kv(model_dft)",
        "llama_vocab_n_tokens(vocab_tgt)",
        "draft_vocab",
    ]:
        if required not in preflight:
            errors.append(f"preflight missing accessor or check: {required}")

    loader = _read(pathlib.Path("src/models/jetspec_qwen3_draft_head.cpp"))
    for token in [
        "runtime_supported=false",
        "preview_not_allowed",
        "unsupported_runtime",
        "jetspec_expect(!meta.runtime_supported",
        "jetspec.experimental.runtime_supported must remain false until JetSpec draft-head graph execution is implemented",
        "LLAMA_JETSPEC_DRAFT_HEAD_LOAD",
        "target-owned JetSpec draft-head artifact",
        "jetspec_require_target_tensor(main_model, \"token_embd.weight\", { 2048, 248320 })",
        "jetspec_require_target_tensor(main_model, \"output.weight\", { 2048, 248320 })",
        "jetspec_require_target_tensor(main_model, \"output_norm.weight\", { 2048 })",
        "missing required target tensor",
        "must have shape",
        "build_arch_graph",
        "throw std::runtime_error(\"unsupported_runtime: JetSpec draft-head graph execution is not implemented\")",
    ]:
        if token not in loader:
            errors.append(f"JetSpec draft-head loader no longer preserves fail-closed token: {token}")

    return {
        "ok": not errors,
        "errors": errors,
        "allowed_files": [str(path) for path in sorted(P5F_ALLOWED_FILES)],
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
    result = validate_p5f_binding_preflight()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print("P5F binding-preflight validation passed")
    else:
        print("P5F binding-preflight validation failed", file=sys.stderr)
        for error in result["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
