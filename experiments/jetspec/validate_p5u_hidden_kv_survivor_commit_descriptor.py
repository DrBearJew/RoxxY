#!/usr/bin/env python3
"""Validate the P5U JetSpec hidden/KV survivor commit descriptor source slice."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

P5_ALLOWED_FILES = {
    pathlib.Path("common/speculative.cpp"),
    pathlib.Path("docs/speculative.md"),
}

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("common/speculative.cpp"): ['JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_PHASE', 'JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_ROLLBACK_POINT', 'JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_DESCRIPTOR', 'commit_hidden_kv_survivors', 'after_hidden_kv_commit', 'hidden_kv_survivor_commit_descriptor_only', 'hidden_kv_survivor_commit_descriptor_ready', 'invalid_hidden_kv_survivor_commit_descriptor', 'n_hidden_kv_survivor_commit_descriptors', 'hidden_kv_survivor_commit_descriptor_hash_last', 'build_hidden_kv_survivor_commit_descriptor', '!token_commit_descriptor_ready || token_commit_descriptor_hash_last == 0', 'actual_survivor_pages_committed_last = 0', 'common_speculative_fnv1a64(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_PHASE', 'common_speculative_fnv1a64(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_ROLLBACK_POINT', 'disable_runtime_state(jetspec_runtime_failure::invalid_hidden_kv_survivor_commit_descriptor', 'hidden_kv_survivor_commit_descriptor_ready=%d', 'hidden_kv_survivor_commit_descriptor_hash=%016', 'commit_hidden_kv_survivors', 'actual_survivor_pages_committed=0', 'hidden_kv_survivor_commit_descriptor_only=1', 'no_real_hidden_kv_commit=1', 'no_kv_mutation=1', 'no_publish=1', 'no_draft_tokens=1', 'hidden_kv_survivor_commit_phase=commit_hidden_kv_survivors', 'rollback_point=after_hidden_kv_commit'],
    pathlib.Path("docs/speculative.md"): ['P5U hidden/KV survivor commit descriptor', 'commit_hidden_kv_survivors', 'hidden/KV survivor commit descriptor', 'actual_survivor_pages_committed=0', 'no real hidden/KV commit', 'no KV mutation', 'no publish', 'no draft tokens', 'no draft tokens'],
    pathlib.Path("experiments/jetspec/jetspec_p5u_hidden_kv_survivor_commit_descriptor_candidate.md"): ['JetSpec P5U hidden/KV survivor commit descriptor candidate', 'approved bounded production-source slice', 'default-off and non-drafting', 'common/speculative.cpp', 'docs/speculative.md', 'hidden_kv_survivor_commit_descriptor_ready', 'invalid_hidden_kv_survivor_commit_descriptor', 'actual_survivor_pages_committed=0', 'no real hidden/KV commit', 'no KV mutation', 'no publish', 'no draft tokens', 'no draft-head graph execution', 'no draft tokens emitted', 'no real KV mutation', 'no CUDA dispatch', 'no server route', 'no public API', 'no CMake wiring'],
}

P5_TOKENS = ['P5U hidden/KV survivor commit descriptor', 'jetspec_p5u_hidden_kv_survivor_commit_descriptor', 'validate_p5u_hidden_kv_survivor_commit_descriptor', 'test_p5u_hidden_kv_survivor_commit_descriptor', 'JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_PHASE', 'JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_ROLLBACK_POINT', 'JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_DESCRIPTOR', 'hidden_kv_survivor_commit_descriptor_ready', 'invalid_hidden_kv_survivor_commit_descriptor', 'hidden_kv_survivor_commit_descriptor_hash_last']

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
    "jetspec_p5u_hidden_kv_survivor_commit_descriptor",
    "validate_p5u_hidden_kv_survivor_commit_descriptor",
    "test_p5u_hidden_kv_survivor_commit_descriptor",
    "JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_PHASE",
    "hidden_kv_survivor_commit_descriptor_ready",
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

FORBIDDEN_IMPL_REGEXES = [
    r"std::vector\s*<\s*llama_token\s*>",
    r"std::vector\s*<\s*uint8_t\s*>\s*(verify|ancestor|mask)",
    r"std::vector\s*<\s*float\s*>\s*cum_?log",
    r"(?<!real_draft_head_topk_tree_)parent_indices\s*\.",
    r"(?<!real_draft_head_topk_tree_)token_ids\s*\.",
    r"cum_logprob\s*\.",
    r"ancestor_matrix\s*\.",
    r"mask_tensor\s*\.",
]


class P5DescriptorError(ValueError):
    """Raised when the source-slice constraints are violated."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5DescriptorError(f"missing required file: {rel}")
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
        raise P5DescriptorError("cannot isolate draft-jetspec implementation slice")
    return source[start:end]


def _dirty_p5_hits() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(["git", "diff", "--name-only", "--diff-filter=ACMR"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError:
        return []
    hits: list[dict[str, Any]] = []
    if proc.returncode != 0:
        return [{"path": "<git diff failed>", "token": proc.stderr.strip()}]
    for line in proc.stdout.splitlines():
        rel = pathlib.Path(line.strip())
        if not rel or rel in P5_ALLOWED_FILES or str(rel).startswith("experiments/jetspec/"):
            continue
        diff = subprocess.run(["git", "diff", "--", str(rel)], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False).stdout
        for token in P5_TOKENS:
            if token in diff:
                hits.append({"path": str(rel), "token": token})
    return hits


def validate_p5u_hidden_kv_survivor_commit_descriptor() -> dict[str, Any]:
    errors: list[str] = []

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5DescriptorError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    token_hits: list[dict[str, Any]] = []
    for rel in _iter_source_files():
        text = _read(rel)
        for token in P5_TOKENS:
            if token in text:
                token_hits.append({"path": str(rel), "token": token})
                if rel not in P5_ALLOWED_FILES:
                    errors.append(f"P5 token {token!r} appears outside allowlist: {rel}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"descriptor must not add explicit CMake wiring in {rel}: {matched}")

    dirty_p5_hits = _dirty_p5_hits()
    for hit in dirty_p5_hits:
        errors.append(f"dirty diff contains descriptor token outside allowlist: {hit}")

    try:
        source = _read(pathlib.Path("common/speculative.cpp"))
        impl = _impl_slice(source)
    except P5DescriptorError as exc:
        errors.append(str(exc))
        impl = ""

    for token in FORBIDDEN_IMPL_TOKENS:
        if token in impl:
            errors.append(f"draft-jetspec descriptor implementation must not contain {token!r}")
    for pattern in FORBIDDEN_IMPL_REGEXES:
        if re.search(pattern, impl):
            errors.append(f"draft-jetspec descriptor implementation matched forbidden pattern {pattern!r}")
    if "build_hidden_kv_survivor_commit_descriptor()" not in impl:
        errors.append("descriptor builder is missing")
    for token in ['actual_survivor_pages_committed_last']:
        if f"{token} = 0" not in impl:
            errors.append(f"descriptor must keep {token} at zero")
    if "disable_runtime_state(jetspec_runtime_failure::invalid_hidden_kv_survivor_commit_descriptor" not in impl:
        errors.append("descriptor must fail closed with its invalid reason")
    if "// fail closed: do not emit draft tokens" not in impl:
        errors.append("descriptor must keep the no-draft fail-closed comment")

    docs = _read(pathlib.Path("docs/speculative.md"))
    for token in ['actual_survivor_pages_committed=0', 'no real hidden/KV commit', 'no KV mutation', 'no publish', 'no draft tokens', 'no draft tokens']:
        if token not in docs:
            errors.append(f"docs/speculative.md missing boundary token: {token}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5u_hidden_kv_survivor_commit_descriptor_validated" if not errors else "p5u_hidden_kv_survivor_commit_descriptor_invalid",
        "allowed_files": sorted(str(path) for path in P5_ALLOWED_FILES),
        "token_hits": token_hits,
        "cmake_hits": cmake_hits,
        "dirty_p5_hits": dirty_p5_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    parse_args(argv)
    out = validate_p5u_hidden_kv_survivor_commit_descriptor()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main([]))
