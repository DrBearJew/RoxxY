#!/usr/bin/env python3
"""Validate the P5Q JetSpec tree-build descriptor source slice."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

P5Q_ALLOWED_FILES = {
    pathlib.Path("common/speculative.cpp"),
    pathlib.Path("docs/speculative.md"),
}

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("common/speculative.cpp"): [
        "JETSPEC_TREE_BUILD_PHASE",
        "JETSPEC_TREE_BUILD_ROLLBACK_POINT",
        "JETSPEC_TREE_BUILD_DESCRIPTOR",
        "JETSPEC_TREE_ROOT_PARENT",
        "JETSPEC_TREE_ROOT_DEPTH",
        "build_tree",
        "after_build_tree",
        "tree_build_descriptor_only",
        "tree_build_descriptor_ready",
        "invalid_tree_build_descriptor",
        "n_tree_build_descriptors",
        "tree_build_descriptor_hash_last",
        "tree_build_node_budget_last",
        "tree_build_root_parent_last",
        "tree_build_root_depth_last",
        "tree_build_actual_nodes_last",
        "build_tree_build_descriptor",
        "!transient_reservation_ready || transient_reservation_hash_last == 0",
        "!transaction_plan_ready || transaction_plan_hash_last == 0",
        "!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0",
        "transient_reservation_actual_pages_last != 0",
        "transient_reservation_node_budget_last <= 0 || transient_reservation_node_budget_last > JETSPEC_QWEN36_DRAFT_BLOCK_SIZE",
        "n_target_tap_rows_cached == 0 || target_tap_row_state.size() != n_target_tap_rows_cached",
        "tree_build_actual_nodes_last = 0",
        "tree_words.push_back((int64_t) transaction_plan_hash_last)",
        "tree_words.push_back((int64_t) pre_round_snapshot_hash_last)",
        "tree_words.push_back((int64_t) transient_reservation_hash_last)",
        "tree_words.push_back((int64_t) pre_round_prompt_tokens_last)",
        "tree_words.push_back((int64_t) pre_round_prompt_hash_last)",
        "tree_words.push_back((int64_t) target_tap_hash_last)",
        "tree_words.push_back((int64_t) tree_build_actual_nodes_last)",
        "common_speculative_fnv1a64(JETSPEC_TREE_BUILD_PHASE",
        "common_speculative_fnv1a64(JETSPEC_TREE_BUILD_ROLLBACK_POINT",
        "disable_runtime_state(jetspec_runtime_failure::invalid_tree_build_descriptor",
        "tree_build_descriptor_ready=%d",
        "tree_build_descriptor_hash=%016",
        "tree_build_phase=build_tree",
        "rollback_point=after_build_tree",
        "planned_tree_node_budget=%d",
        "actual_tree_nodes=0",
        "tree_build_descriptor_only=1",
        "root_parent=%d",
        "root_depth=%d",
        "pre_publish_visible_state_unmodified=1",
        "no_real_tree_build=1",
        "no_tree_arrays=1",
        "no_verify_mask=1",
        "no_accept=1",
        "no_kv_mutation=1",
        "no_publish=1",
        "no_draft_tokens=1",
    ],
    pathlib.Path("docs/speculative.md"): [
        "P5Q tree-build descriptor",
        "build_tree",
        "tree-build descriptor",
        "actual_tree_nodes=0",
        "no real tree build",
        "no tree arrays",
        "no verify mask",
        "no accept",
        "no draft tokens",
        "no CUDA",
        "server",
        "public API",
        "CMake",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5q_tree_build_descriptor_candidate.md"): [
        "JetSpec P5Q tree-build descriptor candidate",
        "approved bounded production-source slice",
        "default-off and non-drafting",
        "common/speculative.cpp",
        "docs/speculative.md",
        "tree_build_descriptor_ready",
        "invalid_tree_build_descriptor",
        "actual_tree_nodes=0",
        "no real tree build",
        "no tree arrays",
        "no draft-head graph execution",
        "no draft tokens emitted",
        "no real KV mutation",
        "no CUDA dispatch",
        "no server route",
        "no public API",
        "no CMake wiring",
    ],
}

P5Q_TOKENS = [
    "P5Q tree-build descriptor",
    "jetspec_p5q_tree_build_descriptor",
    "validate_p5q_tree_build_descriptor",
    "test_p5q_tree_build_descriptor",
    "JETSPEC_TREE_BUILD_PHASE",
    "JETSPEC_TREE_BUILD_ROLLBACK_POINT",
    "tree_build_descriptor_ready",
    "invalid_tree_build_descriptor",
    "tree_build_descriptor_hash_last",
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
    "jetspec_p5q_tree_build_descriptor",
    "validate_p5q_tree_build_descriptor",
    "test_p5q_tree_build_descriptor",
    "JETSPEC_TREE_BUILD_PHASE",
    "tree_build_descriptor_ready",
]

FORBIDDEN_IMPL_TOKENS = [
    "llama_decode",
    "llama_graph",
    "tree_accept",
    "llama_kv_cache",
    "result->push_back",
]

FORBIDDEN_TREE_ARRAY_TOKENS = [
    "std::vector<llama_token> token_ids",
    "std::vector<int32_t> parent_indices",
    "std::vector<int32_t> depth",
    "std::vector<float> cum_logprob",
    "token_ids.resize",
    "parent_indices.resize",
    "cum_logprob.resize",
]

FORBIDDEN_IMPL_REGEXES = [
    r"std::vector\s*<\s*llama_token\s*>",
    r"std::vector\s*<\s*int32_t\s*>\s*(parent|parents|depth)",
    r"std::vector\s*<\s*float\s*>\s*cum_?log",
    r"(?<!real_draft_head_topk_tree_)parent_indices\s*\.",
    r"(?<!real_draft_head_topk_tree_)token_ids\s*\.",
    r"cum_logprob\s*\.",
    r"ancestor_matrix\s*\.",
]


class P5QTreeBuildDescriptorError(ValueError):
    """Raised when P5Q source-slice constraints are violated."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5QTreeBuildDescriptorError(f"missing required file: {rel}")
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
        raise P5QTreeBuildDescriptorError("cannot isolate draft-jetspec implementation slice")
    return source[start:end]


def _dirty_p5q_hits() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(["git", "diff", "--name-only", "--diff-filter=ACMR"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError:
        return []
    hits: list[dict[str, Any]] = []
    if proc.returncode != 0:
        return [{"path": "<git diff failed>", "token": proc.stderr.strip()}]
    for line in proc.stdout.splitlines():
        rel = pathlib.Path(line.strip())
        if not rel or rel in P5Q_ALLOWED_FILES or str(rel).startswith("experiments/jetspec/"):
            continue
        diff = subprocess.run(["git", "diff", "--", str(rel)], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False).stdout
        for token in P5Q_TOKENS:
            if token in diff:
                hits.append({"path": str(rel), "token": token})
    return hits


def validate_p5q_tree_build_descriptor() -> dict[str, Any]:
    errors: list[str] = []

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5QTreeBuildDescriptorError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    token_hits: list[dict[str, Any]] = []
    for rel in _iter_source_files():
        text = _read(rel)
        for token in P5Q_TOKENS:
            if token in text:
                token_hits.append({"path": str(rel), "token": token})
                if rel not in P5Q_ALLOWED_FILES:
                    errors.append(f"P5Q token {token!r} appears outside allowlist: {rel}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5Q must not add explicit CMake wiring in {rel}: {matched}")

    dirty_p5q_hits = _dirty_p5q_hits()
    for hit in dirty_p5q_hits:
        errors.append(f"dirty diff contains P5Q token outside allowlist: {hit}")

    try:
        source = _read(pathlib.Path("common/speculative.cpp"))
        impl = _impl_slice(source)
    except P5QTreeBuildDescriptorError as exc:
        errors.append(str(exc))
        impl = ""

    for token in FORBIDDEN_IMPL_TOKENS + FORBIDDEN_TREE_ARRAY_TOKENS:
        if token in impl:
            errors.append(f"draft-jetspec P5Q implementation must not contain {token!r}")
    for pattern in FORBIDDEN_IMPL_REGEXES:
        if re.search(pattern, impl):
            errors.append(f"draft-jetspec P5Q implementation matched forbidden pattern {pattern!r}")
    if "build_tree_build_descriptor()" not in impl:
        errors.append("P5Q must build a tree-build descriptor")
    if "tree_build_actual_nodes_last = 0" not in impl:
        errors.append("P5Q must keep actual tree nodes at zero")
    if "disable_runtime_state(jetspec_runtime_failure::invalid_tree_build_descriptor" not in impl:
        errors.append("P5Q must fail closed with invalid_tree_build_descriptor")
    if "// fail closed: do not emit draft tokens" not in impl:
        errors.append("P5Q must keep the no-draft fail-closed comment")

    docs = _read(pathlib.Path("docs/speculative.md"))
    for token in ["actual_tree_nodes=0", "no real tree build", "no tree arrays", "no verify mask", "no accept", "no draft tokens"]:
        if token not in docs:
            errors.append(f"docs/speculative.md missing P5Q boundary token: {token}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5q_tree_build_descriptor_validated" if not errors else "p5q_tree_build_descriptor_invalid",
        "allowed_files": sorted(str(path) for path in P5Q_ALLOWED_FILES),
        "token_hits": token_hits,
        "cmake_hits": cmake_hits,
        "dirty_p5q_hits": dirty_p5q_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    parse_args(argv)
    out = validate_p5q_tree_build_descriptor()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main([]))
