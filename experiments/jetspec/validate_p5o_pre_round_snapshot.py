#!/usr/bin/env python3
"""Validate the P5O JetSpec pre-round snapshot source slice."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

P5O_ALLOWED_FILES = {
    pathlib.Path("common/speculative.cpp"),
    pathlib.Path("docs/speculative.md"),
}

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("common/speculative.cpp"): [
        "JETSPEC_PRE_ROUND_SNAPSHOT_PHASE",
        "snapshot_pre_round",
        "waiting_for_pre_round_snapshot",
        "pre_round_snapshot_ready",
        "invalid_pre_round_snapshot",
        "n_pre_round_snapshots",
        "pre_round_snapshot_hash_last",
        "pre_round_prompt_hash_last",
        "pre_round_seq_id_last",
        "pre_round_prompt_tokens_last",
        "build_pre_round_snapshot",
        "seq_id < 0 || (uint32_t) seq_id >= n_seq",
        "common_speculative_fnv1a64(prompt.data(), prompt.size() * sizeof(prompt[0]))",
        "pre_round_snapshot_hash_last ^= common_speculative_fnv1a64(JETSPEC_PRE_ROUND_SNAPSHOT_PHASE",
        "pre_round_snapshot_hash_last == 0",
        "plan_words.push_back((int64_t) pre_round_snapshot_hash_last)",
        "plan_words.push_back((int64_t) pre_round_seq_id_last)",
        "plan_words.push_back((int64_t) pre_round_prompt_tokens_last)",
        "plan_words.push_back((int64_t) pre_round_prompt_hash_last)",
        "disable_runtime_state(jetspec_runtime_failure::invalid_pre_round_snapshot",
        "pre_round_snapshot_ready=%d",
        "pre_round_snapshot_hash=%016",
        "pre_round_seq_id=%d",
        "pre_round_prompt_tokens=%zu",
        "pre_round_prompt_hash=%016",
        "transaction_phase=%s",
        "no_reserve=1",
        "no_tree_build=1",
        "no_verify_mask=1",
        "no_kv_mutation=1",
        "no_publish=1",
        "no_draft_tokens=1",
    ],
    pathlib.Path("docs/speculative.md"): [
        "P5O pre-round snapshot descriptor",
        "sequence id",
        "prompt token count",
        "prompt hash",
        "snapshot_pre_round",
        "no reserve",
        "no tree build",
        "no verify mask",
        "does not publish",
        "mutate KV",
        "dispatch CUDA",
        "emits no draft",
        "pre-round snapshot readiness",
        "pre-round snapshot hash",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5o_pre_round_snapshot_candidate.md"): [
        "JetSpec P5O pre-round snapshot descriptor candidate",
        "approved bounded production-source slice",
        "default-off and non-drafting",
        "common/speculative.cpp",
        "docs/speculative.md",
        "pre_round_snapshot_ready",
        "pre_round_snapshot_hash_last",
        "pre_round_seq_id_last",
        "pre_round_prompt_tokens_last",
        "pre_round_prompt_hash_last",
        "invalid_pre_round_snapshot",
        "no reserve_transient_tree_pages",
        "no build_tree",
        "no build_verify_mask",
        "no draft-head graph execution",
        "no draft tokens emitted",
        "no real KV mutation",
        "no CUDA dispatch",
        "no server route",
        "no public API",
        "no CMake wiring",
    ],
}

P5O_TOKENS = [
    "P5O pre-round snapshot",
    "jetspec_p5o_pre_round_snapshot",
    "validate_p5o_pre_round_snapshot",
    "test_p5o_pre_round_snapshot",
    "JETSPEC_PRE_ROUND_SNAPSHOT_PHASE",
    "pre_round_snapshot_ready",
    "pre_round_snapshot_hash_last",
    "invalid_pre_round_snapshot",
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
    "jetspec_p5o_pre_round_snapshot",
    "validate_p5o_pre_round_snapshot",
    "test_p5o_pre_round_snapshot",
    "JETSPEC_PRE_ROUND_SNAPSHOT_PHASE",
    "pre_round_snapshot_ready",
]

FORBIDDEN_IMPL_TOKENS = [
    "llama_decode",
    "llama_graph",
    "tree_accept",
    "llama_kv_cache",
    "result->push_back",
]


class P5OPreRoundSnapshotError(ValueError):
    """Raised when P5O source-slice constraints are violated."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5OPreRoundSnapshotError(f"missing required file: {rel}")
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
        raise P5OPreRoundSnapshotError("cannot isolate draft-jetspec implementation slice")
    return source[start:end]


def validate_p5o_pre_round_snapshot() -> dict[str, Any]:
    errors: list[str] = []

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5OPreRoundSnapshotError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    token_hits: list[dict[str, Any]] = []
    for rel in _iter_source_files():
        text = _read(rel)
        for token in P5O_TOKENS:
            if token in text:
                token_hits.append({"path": str(rel), "token": token})
                if rel not in P5O_ALLOWED_FILES:
                    errors.append(f"P5O token {token!r} appears outside allowlist: {rel}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5O must not add explicit CMake wiring in {rel}: {matched}")

    try:
        source = _read(pathlib.Path("common/speculative.cpp"))
        impl = _impl_slice(source)
    except P5OPreRoundSnapshotError as exc:
        errors.append(str(exc))
        impl = ""

    for token in FORBIDDEN_IMPL_TOKENS:
        if token in impl:
            errors.append(f"draft-jetspec P5O implementation must not contain {token!r}")
    if "if (!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0)" not in impl:
        errors.append("P5O transaction scaffold must require pre_round_snapshot_ready")
    if "build_pre_round_snapshot(seq_id, prompt)" not in impl:
        errors.append("P5O begin() must build the pre-round snapshot")
    if "disable_runtime_state(jetspec_runtime_failure::invalid_pre_round_snapshot" not in impl:
        errors.append("P5O must fail closed with invalid_pre_round_snapshot")
    if "plan_words.push_back((int64_t) pre_round_snapshot_hash_last)" not in impl:
        errors.append("P5O snapshot hash must be included in transaction hash input")
    if "// fail closed: do not emit draft tokens" not in impl:
        errors.append("P5O must keep the no-draft fail-closed comment")

    docs = _read(pathlib.Path("docs/speculative.md"))
    for token in ["no reserve", "no tree build", "no verify mask", "does not publish", "mutate KV", "dispatch CUDA", "emits no draft tokens"]:
        if token not in docs:
            errors.append(f"docs/speculative.md missing P5O boundary token: {token}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5o_pre_round_snapshot_validated" if not errors else "p5o_pre_round_snapshot_invalid",
        "allowed_files": sorted(str(path) for path in P5O_ALLOWED_FILES),
        "token_hits": token_hits,
        "cmake_hits": cmake_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    parse_args(argv)
    out = validate_p5o_pre_round_snapshot()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main([]))
