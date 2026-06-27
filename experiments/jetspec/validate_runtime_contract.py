#!/usr/bin/env python3
"""Validate the inert JetSpec runtime C++ contract header by text inspection.

This intentionally avoids compiling llama.cpp. It checks that the staged header
has the expected constants, enums, structs, static invariants, and no production
include/reference leaks.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any


DEFAULT_HEADER = "jetspec_runtime_contract.hpp"

EXPECTED_CONSTANTS = {
    "jetspec_qwen36_block_size": 16,
    "jetspec_qwen36_draft_depth": 15,
    "jetspec_qwen36_mask_token_id": 248070,
    "jetspec_qwen36_target_layers": 40,
    "jetspec_qwen36_target_tap_count": 5,
    "jetspec_qwen36_hidden_size": 2048,
    "jetspec_qwen36_concat_width": 10240,
    "jetspec_qwen36_draft_layers": 8,
    "jetspec_qwen36_attention_heads": 32,
    "jetspec_qwen36_attention_heads_kv": 4,
    "jetspec_qwen36_head_dim": 128,
    "jetspec_qwen36_ffn_size": 6144,
    "jetspec_qwen36_vocab_size": 248320,
    "jetspec_qwen36_tensor_count": 91,
}

REQUIRED_ENUMS = [
    "tensor_payload_mode",
    "runtime_phase",
    "runtime_failure",
]

REQUIRED_STRUCTS = [
    "draft_head_metadata",
    "draft_head_tensor_info",
    "draft_head_loader_plan",
    "target_model_bindings",
    "target_hidden_cache_state",
    "tree_verify_plan",
    "round_commit_plan",
    "round_state",
]

REQUIRED_TOKENS = [
    "jetspec_qwen3_draft_head",
    "qwen3_draft_head",
    "DFlashDraftModel",
    "{1, 10, 19, 28, 37}",
    "runtime_supported = false",
    "requires_target_embeddings = true",
    "requires_target_lm_head = true",
    "correction_hidden_appended = false",
    "accepted_path accepted",
    "draft_tree tree",
]

ALLOWED_INCLUDES = {
    "jetspec_tree_contract.hpp",
    "cstdint",
    "string",
    "vector",
}

FORBIDDEN_PATTERNS = [
    r"#\s*include\s*[<\"]llama\.h[>\"]",
    r"#\s*include\s*[<\"]ggml[^>\"]*[>\"]",
    r"#\s*include\s*[<\"]common/",
    r"#\s*include\s*[<\"]tools/",
    r"#\s*include\s*[<\"]src/",
    r"#\s*include\s*[<\"]ggml/",
    r"llama_context",
    r"llama_model",
    r"ggml_context",
    r"ggml_tensor",
]


class RuntimeContractError(ValueError):
    """Raised when the inert runtime header contract is invalid."""


def _extract_includes(text: str) -> list[str]:
    includes: list[str] = []
    for match in re.finditer(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", text, flags=re.MULTILINE):
        includes.append(match.group(1))
    return includes


def _extract_constants(text: str) -> dict[str, int]:
    found: dict[str, int] = {}
    pattern = re.compile(r"constexpr\s+int32_t\s+(jetspec_qwen36_[A-Za-z0-9_]+)\s*=\s*([0-9]+)\s*;")
    for name, value in pattern.findall(text):
        found[name] = int(value)
    return found


def validate_header(path: pathlib.Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []

    if "namespace llama_jetspec_experiment" not in text:
        errors.append("missing llama_jetspec_experiment namespace")
    if "#pragma once" not in text:
        errors.append("missing #pragma once")
    if "not included by any production llama.cpp source file" not in text:
        errors.append("missing inert production-isolation warning")

    includes = _extract_includes(text)
    bad_includes = [inc for inc in includes if inc not in ALLOWED_INCLUDES]
    if bad_includes:
        errors.append(f"unexpected includes: {bad_includes}")
    if "jetspec_tree_contract.hpp" not in includes:
        errors.append("runtime contract must include local jetspec_tree_contract.hpp")

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text):
            errors.append(f"forbidden production reference matched: {pattern}")

    constants = _extract_constants(text)
    for name, expected in EXPECTED_CONSTANTS.items():
        actual = constants.get(name)
        if actual != expected:
            errors.append(f"constant {name}: expected {expected}, got {actual}")

    if constants.get("jetspec_qwen36_draft_depth") != constants.get("jetspec_qwen36_block_size", 0) - 1:
        errors.append("draft_depth constant does not equal block_size - 1")
    if constants.get("jetspec_qwen36_concat_width") != constants.get("jetspec_qwen36_target_tap_count", 0) * constants.get("jetspec_qwen36_hidden_size", 0):
        errors.append("concat_width constant does not equal tap_count * hidden_size")

    for enum_name in REQUIRED_ENUMS:
        if not re.search(rf"enum\s+class\s+{re.escape(enum_name)}\b", text):
            errors.append(f"missing enum class {enum_name}")

    for struct_name in REQUIRED_STRUCTS:
        if not re.search(rf"struct\s+{re.escape(struct_name)}\b", text):
            errors.append(f"missing struct {struct_name}")

    for token in REQUIRED_TOKENS:
        if token not in text:
            errors.append(f"missing required token: {token}")

    if "static_assert(jetspec_qwen36_draft_depth == jetspec_qwen36_block_size - 1" not in text:
        errors.append("missing draft_depth static_assert")
    if "static_assert(jetspec_qwen36_concat_width == jetspec_qwen36_target_tap_count * jetspec_qwen36_hidden_size" not in text:
        errors.append("missing concat_width static_assert")

    return {
        "ok": not errors,
        "errors": errors,
        "path": str(path),
        "includes": includes,
        "constants": constants,
        "enums": REQUIRED_ENUMS,
        "structs": REQUIRED_STRUCTS,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=pathlib.Path, default=here / DEFAULT_HEADER)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        out = validate_header(args.header.resolve())
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
