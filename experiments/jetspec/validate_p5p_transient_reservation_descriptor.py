#!/usr/bin/env python3
"""Validate the P5P JetSpec transient-reservation descriptor source slice."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

P5P_ALLOWED_FILES = {
    pathlib.Path("common/speculative.cpp"),
    pathlib.Path("docs/speculative.md"),
}

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("common/speculative.cpp"): [
        "JETSPEC_TRANSIENT_RESERVATION_PHASE",
        "JETSPEC_TRANSIENT_RESERVATION_ROLLBACK_POINT",
        "JETSPEC_TRANSIENT_RESERVATION_DESCRIPTOR",
        "reserve_transient_tree_pages",
        "after_reserve",
        "transient_reservation_descriptor_only",
        "transient_reservation_descriptor_ready",
        "invalid_transient_reservation_descriptor",
        "n_transient_reservation_descriptors",
        "transient_reservation_ready",
        "transient_reservation_hash_last",
        "transient_reservation_node_budget_last",
        "transient_reservation_actual_pages_last",
        "build_transient_reservation_descriptor",
        "!transaction_plan_ready || transaction_plan_hash_last == 0",
        "!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0",
        "n_target_tap_rows_cached == 0 || target_tap_row_state.size() != n_target_tap_rows_cached",
        "transient_reservation_node_budget_last <= 0 || transient_reservation_node_budget_last > JETSPEC_QWEN36_DRAFT_BLOCK_SIZE",
        "transient_reservation_actual_pages_last = 0",
        "reservation_words.push_back((int64_t) transaction_plan_hash_last)",
        "reservation_words.push_back((int64_t) pre_round_snapshot_hash_last)",
        "reservation_words.push_back((int64_t) pre_round_seq_id_last)",
        "reservation_words.push_back((int64_t) pre_round_prompt_tokens_last)",
        "reservation_words.push_back((int64_t) target_tap_hash_last)",
        "reservation_words.push_back(JETSPEC_QWEN36_DRAFT_BLOCK_SIZE)",
        "common_speculative_fnv1a64(JETSPEC_TRANSIENT_RESERVATION_PHASE",
        "common_speculative_fnv1a64(JETSPEC_TRANSIENT_RESERVATION_ROLLBACK_POINT",
        "disable_runtime_state(jetspec_runtime_failure::invalid_transient_reservation_descriptor",
        "transient_reservation_ready=%d",
        "transient_reservation_hash=%016",
        "transient_reservation_phase=reserve_transient_tree_pages",
        "rollback_point=after_reserve",
        "transient_tree_node_budget=%d",
        "actual_pages_reserved=0",
        "pre_publish_visible_state_unmodified=1",
        "no_real_reserve=1",
        "no_page_map_write=1",
        "no_tree_build=1",
        "no_verify_mask=1",
        "no_kv_mutation=1",
        "no_publish=1",
        "no_draft_tokens=1",
    ],
    pathlib.Path("docs/speculative.md"): [
        "P5P transient-reservation descriptor",
        "reserve_transient_tree_pages",
        "reservation descriptor",
        "actual_pages_reserved=0",
        "no real page reservation",
        "no llama_kv_cache primitive",
        "no tree build",
        "no verify mask",
        "no draft tokens",
        "no CUDA",
        "server",
        "public API",
        "CMake",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5p_transient_reservation_descriptor_candidate.md"): [
        "JetSpec P5P transient-reservation descriptor candidate",
        "approved bounded production-source slice",
        "default-off and non-drafting",
        "common/speculative.cpp",
        "docs/speculative.md",
        "transient_reservation_descriptor_ready",
        "invalid_transient_reservation_descriptor",
        "actual_pages_reserved=0",
        "no real page reservation",
        "no llama_kv_cache primitive",
        "no draft-head graph execution",
        "no draft tokens emitted",
        "no real KV mutation",
        "no CUDA dispatch",
        "no server route",
        "no public API",
        "no CMake wiring",
    ],
}

P5P_TOKENS = [
    "P5P transient-reservation descriptor",
    "jetspec_p5p_transient_reservation_descriptor",
    "validate_p5p_transient_reservation_descriptor",
    "test_p5p_transient_reservation_descriptor",
    "JETSPEC_TRANSIENT_RESERVATION_PHASE",
    "JETSPEC_TRANSIENT_RESERVATION_ROLLBACK_POINT",
    "transient_reservation_descriptor_ready",
    "invalid_transient_reservation_descriptor",
    "transient_reservation_hash_last",
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
    "jetspec_p5p_transient_reservation_descriptor",
    "validate_p5p_transient_reservation_descriptor",
    "test_p5p_transient_reservation_descriptor",
    "JETSPEC_TRANSIENT_RESERVATION_PHASE",
    "transient_reservation_descriptor_ready",
]

FORBIDDEN_IMPL_TOKENS = [
    "llama_decode",
    "llama_graph",
    "tree_accept",
    "llama_kv_cache",
    "result->push_back",
    "seq_cp",
    "seq_rm",
    "seq_import_physical",
]


class P5PTransientReservationDescriptorError(ValueError):
    """Raised when P5P source-slice constraints are violated."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5PTransientReservationDescriptorError(f"missing required file: {rel}")
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
        raise P5PTransientReservationDescriptorError("cannot isolate draft-jetspec implementation slice")
    return source[start:end]


def validate_p5p_transient_reservation_descriptor() -> dict[str, Any]:
    errors: list[str] = []

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5PTransientReservationDescriptorError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    token_hits: list[dict[str, Any]] = []
    for rel in _iter_source_files():
        text = _read(rel)
        for token in P5P_TOKENS:
            if token in text:
                token_hits.append({"path": str(rel), "token": token})
                if rel not in P5P_ALLOWED_FILES:
                    errors.append(f"P5P token {token!r} appears outside allowlist: {rel}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5P must not add explicit CMake wiring in {rel}: {matched}")

    try:
        source = _read(pathlib.Path("common/speculative.cpp"))
        impl = _impl_slice(source)
    except P5PTransientReservationDescriptorError as exc:
        errors.append(str(exc))
        impl = ""

    for token in FORBIDDEN_IMPL_TOKENS:
        if token in impl:
            errors.append(f"draft-jetspec P5P implementation must not contain {token!r}")
    if "build_transient_reservation_descriptor()" not in impl:
        errors.append("P5P must build a transient reservation descriptor")
    if "transient_reservation_actual_pages_last = 0" not in impl:
        errors.append("P5P must keep actual pages reserved at zero")
    if "disable_runtime_state(jetspec_runtime_failure::invalid_transient_reservation_descriptor" not in impl:
        errors.append("P5P must fail closed with invalid_transient_reservation_descriptor")
    if "// fail closed: do not emit draft tokens" not in impl:
        errors.append("P5P must keep the no-draft fail-closed comment")

    docs = _read(pathlib.Path("docs/speculative.md"))
    for token in ["actual_pages_reserved=0", "no real page reservation", "no llama_kv_cache primitive", "no tree build", "no verify mask", "no draft tokens"]:
        if token not in docs:
            errors.append(f"docs/speculative.md missing P5P boundary token: {token}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5p_transient_reservation_descriptor_validated" if not errors else "p5p_transient_reservation_descriptor_invalid",
        "allowed_files": sorted(str(path) for path in P5P_ALLOWED_FILES),
        "token_hits": token_hits,
        "cmake_hits": cmake_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    parse_args(argv)
    out = validate_p5p_transient_reservation_descriptor()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main([]))
