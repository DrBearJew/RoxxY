#!/usr/bin/env python3
"""Validate the P5X JetSpec root-only runtime tree source slice."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

P5X_ALLOWED_FILES = {
    pathlib.Path("common/speculative.cpp"),
    pathlib.Path("docs/speculative.md"),
}

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("common/speculative.cpp"): [
        "LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY",
        "JETSPEC_ROOT_TREE_RUNTIME_PHASE",
        "root_tree_runtime",
        "tree_build_runtime_ready",
        "invalid_root_tree_runtime",
        "n_root_tree_runtime_builds",
        "root_tree_runtime_hash_last",
        "root_tree_runtime_seq_id_last",
        "root_tree_runtime_ready",
        "pre_round_root_token_last",
        "tree_token_ids",
        "tree_parent_indices",
        "tree_depth",
        "tree_rank",
        "tree_cum_logprob",
        "build_root_only_runtime_tree",
        "prompt.empty()",
        "p5x_root_tree_enabled && prompt.empty()",
        "pre_round_root_token_last = prompt.empty() ? -1 : prompt.back()",
        "if (!p5x_root_tree_enabled) {",
        "if (!tree_build_descriptor_ready || tree_build_descriptor_hash_last == 0) {",
        "if (pre_round_root_token_last < 0) {",
        "if (tree_build_actual_nodes_last != 0) {",
        "tree_build_actual_nodes_last = 1",
        "tree_token_ids[0] = pre_round_root_token_last",
        "tree_parent_indices[0] = JETSPEC_TREE_ROOT_PARENT",
        "tree_depth[0] = JETSPEC_TREE_ROOT_DEPTH",
        "tree_rank[0] = -1",
        "tree_cum_logprob[0] = 0.0f",
        "tree_build_actual_nodes_last > tree_build_node_budget_last",
        "runtime_phase = jetspec_runtime_phase::tree_build_runtime_ready",
        "n_root_tree_runtime_builds++",
        "disable_runtime_state(jetspec_runtime_failure::invalid_root_tree_runtime",
        "p5x_root_tree_runtime",
        "actual_tree_nodes=1",
        "tree_token_ids=[%d]",
        "tree_parent_indices=[%d]",
        "tree_depth=[%d]",
        "tree_cum_logprob=[%.1f]",
        "root_parent=%d",
        "root_depth=%d",
        "parent_before_child=1",
        "num_nodes_lte_budget=1",
        "root_tree_runtime_ready=%d",
        "root_tree_runtime_hash=%016",
        "actual_tree_nodes=%d",
        "no_draft_head_graph=1",
        "no_verify_mask=1",
        "no_accept=1",
        "no_token_commit=1",
        "no_hidden_kv_commit=1",
        "no_rejected_branch_discard=1",
        "no_publish=1",
        "no_visible_state_change=1",
        "no_draft_tokens=1",
    ],
    pathlib.Path("docs/speculative.md"): [
        "P5X root-only runtime tree materialization",
        "LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1",
        "actual_tree_nodes=1",
        "tree_token_ids=[root_token]",
        "tree_parent_indices=[-1]",
        "tree_depth=[0]",
        "tree_rank=[-1]",
        "tree_cum_logprob=[0.0]",
        "tree_build_runtime_ready",
        "parent_before_child=1",
        "num_nodes_lte_budget=1",
        "no draft-head graph execution",
        "no top-k/non-root tree expansion",
        "no verify mask",
        "no accept",
        "no token commit",
        "no hidden/KV commit",
        "no rejected-branch discard",
        "no publish",
        "no visible state change",
        "no KV mutation",
        "no draft tokens",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5x_root_tree_runtime_candidate.md"): [
        "JetSpec P5X root-only runtime tree candidate",
        "approved bounded production-source slice",
        "LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1",
        "actual_tree_nodes=1",
        "tree_token_ids=[root_token]",
        "tree_parent_indices=[-1]",
        "tree_depth=[0]",
        "tree_rank=[-1]",
        "tree_cum_logprob=[0.0]",
        "returns before P5R",
        "no draft-head graph execution",
        "no top-k or non-root tree expansion",
        "no verify mask",
        "no accept path",
        "no token commit",
        "no hidden/KV survivor commit",
        "no rejected-branch discard",
        "no publish",
        "no visible state change",
        "no KV mutation",
        "no draft tokens",
    ],
}

P5X_TOKENS = [
    "P5X root-only runtime tree materialization",
    "jetspec_p5x_root_tree_runtime",
    "validate_p5x_root_tree_runtime",
    "test_p5x_root_tree_runtime",
    "JETSPEC_ROOT_TREE_RUNTIME_PHASE",
    "LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY",
    "tree_build_runtime_ready",
    "root_tree_runtime_ready",
    "build_root_only_runtime_tree",
    "actual_tree_nodes=1",
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
    "jetspec_p5x_root_tree_runtime",
    "validate_p5x_root_tree_runtime",
    "test_p5x_root_tree_runtime",
    "JETSPEC_ROOT_TREE_RUNTIME_PHASE",
    "LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY",
]

FORBIDDEN_IMPL_TOKENS = [
    "llama_decode",
    "llama_graph",
    "tree_accept",
    "llama_kv_cache",
    "result->push_back",
    "common_sampler_sample",
]

FORBIDDEN_P5X_BRANCH_TOKENS = [
    "build_verify_mask_descriptor()",
    "build_accept_path_descriptor()",
    "build_token_commit_descriptor()",
    "build_hidden_kv_survivor_commit_descriptor()",
    "build_rejected_branch_discard_descriptor()",
    "build_publish_gate_descriptor()",
]


class P5XRootTreeRuntimeError(ValueError):
    """Raised when P5X source-slice constraints are violated."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5XRootTreeRuntimeError(f"missing required file: {rel}")
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
        raise P5XRootTreeRuntimeError("cannot isolate draft-jetspec implementation slice")
    return source[start:end]


def _p5x_branch(impl: str) -> str:
    start = impl.find("if (p5x_root_tree_enabled) {\n            if (!build_root_only_runtime_tree())")
    end = impl.find("        if (!build_verify_mask_descriptor())", start)
    if start < 0 or end < 0 or end <= start:
        raise P5XRootTreeRuntimeError("cannot isolate P5X process branch before verify-mask descriptor")
    return impl[start:end]


def _dirty_p5x_hits() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(["git", "diff", "--name-only", "--diff-filter=ACMR"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError:
        return []
    hits: list[dict[str, Any]] = []
    if proc.returncode != 0:
        return [{"path": "<git diff failed>", "token": proc.stderr.strip()}]
    for line in proc.stdout.splitlines():
        rel = pathlib.Path(line.strip())
        if not rel or rel in P5X_ALLOWED_FILES or str(rel).startswith("experiments/jetspec/"):
            continue
        diff = subprocess.run(["git", "diff", "--", str(rel)], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False).stdout
        for token in P5X_TOKENS:
            if token in diff:
                hits.append({"path": str(rel), "token": token})
    return hits


def validate_p5x_root_tree_runtime() -> dict[str, Any]:
    errors: list[str] = []

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5XRootTreeRuntimeError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    token_hits: list[dict[str, Any]] = []
    for rel in _iter_source_files():
        text = _read(rel)
        for token in P5X_TOKENS:
            if token in text:
                token_hits.append({"path": str(rel), "token": token})
                if rel not in P5X_ALLOWED_FILES:
                    errors.append(f"P5X token {token!r} appears outside allowlist: {rel}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5X must not add explicit CMake wiring in {rel}: {matched}")

    dirty_p5x_hits = _dirty_p5x_hits()
    for hit in dirty_p5x_hits:
        errors.append(f"dirty diff contains P5X token outside allowlist: {hit}")

    try:
        source = _read(pathlib.Path("common/speculative.cpp"))
        impl = _impl_slice(source)
        branch = _p5x_branch(impl)
    except P5XRootTreeRuntimeError as exc:
        errors.append(str(exc))
        impl = ""
        branch = ""

    for token in FORBIDDEN_IMPL_TOKENS:
        if token in impl:
            errors.append(f"P5X implementation must not contain {token!r}")
    for token in FORBIDDEN_P5X_BRANCH_TOKENS:
        if token in branch:
            errors.append(f"P5X branch must return before {token}")
    if branch and "return true;" not in branch:
        errors.append("P5X branch must return before P5R verify-mask descriptor")
    if "runtime_supported=true" in impl:
        errors.append("P5X must not claim runtime_supported=true")
    if re.search(r"common_sampler_sample|llama_sampler", impl):
        errors.append("P5X must not sample draft-head logits")

    docs = _read(pathlib.Path("docs/speculative.md"))
    for token in [
        "actual_tree_nodes=1",
        "tree_token_ids=[root_token]",
        "no draft-head graph execution",
        "no verify mask",
        "no accept",
        "no token commit",
        "no hidden/KV commit",
        "no rejected-branch discard",
        "no publish",
        "no visible state change",
        "no KV mutation",
        "no draft tokens",
    ]:
        if token not in docs:
            errors.append(f"docs/speculative.md missing P5X boundary token: {token}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5x_root_tree_runtime_validated" if not errors else "p5x_root_tree_runtime_invalid",
        "allowed_files": sorted(str(path) for path in P5X_ALLOWED_FILES),
        "token_hits": token_hits,
        "cmake_hits": cmake_hits,
        "dirty_p5x_hits": dirty_p5x_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out = validate_p5x_root_tree_runtime()
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    elif out["ok"]:
        print("P5X root-only runtime tree validation passed")
    else:
        print("P5X root-only runtime tree validation failed", file=__import__("sys").stderr)
        for error in out["errors"]:
            print(f"- {error}", file=__import__("sys").stderr)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
