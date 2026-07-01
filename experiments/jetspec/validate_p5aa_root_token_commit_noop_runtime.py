#!/usr/bin/env python3
"""Validate the P5AA JetSpec root-token-commit no-op runtime source slice."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

P5AA_ALLOWED_FILES = {
    pathlib.Path("common/speculative.cpp"),
    pathlib.Path("docs/speculative.md"),
}

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("common/speculative.cpp"): [
        "LLAMA_JETSPEC_ROOT_TOKEN_COMMIT_NOOP_ONLY",
        "JETSPEC_ROOT_TOKEN_COMMIT_NOOP_RUNTIME_PHASE",
        "root_token_commit_noop_runtime",
        "token_commit_runtime_ready",
        "invalid_root_token_commit_noop_runtime",
        "n_root_token_commit_noop_runtime_builds",
        "root_token_commit_noop_runtime_hash_last",
        "root_token_commit_noop_runtime_seq_id_last",
        "root_token_commit_noop_runtime_ready",
        "build_root_token_commit_noop_runtime",
        "p5aa_root_token_commit_noop_enabled && (!p5x_root_tree_enabled || !p5y_root_verify_mask_enabled || !p5z_root_anchor_accept_path_enabled)",
        "!p5z_root_anchor_accept_path_enabled || !root_anchor_accept_path_runtime_ready || root_anchor_accept_path_runtime_hash_last == 0",
        "root_verified_anchor_last != 1 || accept_path_len_last != 0 || actual_accepted_nodes_last != 0 || correction_token_present_last != 0",
        "actual_committed_tokens_last = 0",
        "runtime_phase = jetspec_runtime_phase::token_commit_runtime_ready",
        "n_root_token_commit_noop_runtime_builds++",
        "disable_runtime_state(jetspec_runtime_failure::invalid_root_token_commit_noop_runtime",
        "p5aa_root_token_commit_noop_runtime",
        "root_token_commit_noop_runtime_ready=%d",
        "actual_committed_tokens=%d",
        "no_real_token_commit=1",
        "no_visible_token_publish=1",
        "no_hidden_kv_commit=1",
        "no_rejected_branch_discard=1",
        "no_publish=1",
        "no_visible_state_change=1",
        "no_kv_mutation=1",
        "no_draft_head_graph=1",
        "no_draft_tokens=1",
    ],
    pathlib.Path("docs/speculative.md"): [
        "P5AA root-token-commit no-op materialization",
        "LLAMA_JETSPEC_ROOT_TOKEN_COMMIT_NOOP_ONLY=1",
        "root_token_commit_noop_runtime_ready=1",
        "actual_committed_tokens=0",
        "root_verified_anchor=1",
        "accept_path_len=0",
        "actual_accepted_nodes=0",
        "correction_token_present=0",
        "returns before P5U",
        "no real token commit",
        "no visible token publish",
        "no hidden/KV commit",
        "no rejected-branch discard",
        "no publish",
        "no visible state change",
        "no KV mutation",
        "no draft tokens",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5aa_root_token_commit_noop_runtime_candidate.md"): [
        "JetSpec P5AA root-token-commit no-op runtime candidate",
        "approved bounded production-source slice",
        "LLAMA_JETSPEC_ROOT_TOKEN_COMMIT_NOOP_ONLY=1",
        "root_token_commit_noop_runtime_ready=1",
        "actual_committed_tokens=0",
        "root_verified_anchor=1",
        "accept_path_len=0",
        "actual_accepted_nodes=0",
        "correction_token_present=0",
        "returns before P5U",
        "does not set",
        "token_commit_descriptor_ready",
        "does not call the P5T descriptor builder",
        "no real token commit",
        "no visible token publish",
        "no hidden/KV survivor commit",
        "no rejected-branch discard",
        "no publish",
        "no visible state change",
        "no KV mutation",
        "no draft-head graph execution",
        "no draft tokens",
    ],
}

P5AA_TOKENS = [
    "P5AA root-token-commit no-op materialization",
    "jetspec_p5aa_root_token_commit_noop_runtime",
    "validate_p5aa_root_token_commit_noop_runtime",
    "test_p5aa_root_token_commit_noop_runtime",
    "probe_p5aa_root_token_commit_noop_trace",
    "test_p5aa_root_token_commit_noop_trace_probe",
    "JETSPEC_ROOT_TOKEN_COMMIT_NOOP_RUNTIME_PHASE",
    "LLAMA_JETSPEC_ROOT_TOKEN_COMMIT_NOOP_ONLY",
    "token_commit_runtime_ready",
    "root_token_commit_noop_runtime_ready",
    "build_root_token_commit_noop_runtime",
    "actual_committed_tokens=0",
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
    "jetspec_p5aa_root_token_commit_noop_runtime",
    "validate_p5aa_root_token_commit_noop_runtime",
    "test_p5aa_root_token_commit_noop_runtime",
    "probe_p5aa_root_token_commit_noop_trace",
    "test_p5aa_root_token_commit_noop_trace_probe",
    "JETSPEC_ROOT_TOKEN_COMMIT_NOOP_RUNTIME_PHASE",
    "LLAMA_JETSPEC_ROOT_TOKEN_COMMIT_NOOP_ONLY",
]

FORBIDDEN_IMPL_TOKENS = [
    "llama_decode",
    "llama_graph",
    "tree_accept",
    "llama_kv_cache",
    "result->push_back",
    "common_sampler_sample",
    "seq_cp",
    "seq_rm",
    "seq_import_physical",
]

FORBIDDEN_P5AA_BRANCH_TOKENS = [
    "build_token_commit_descriptor()",
    "build_hidden_kv_survivor_commit_descriptor()",
    "build_rejected_branch_discard_descriptor()",
    "build_publish_gate_descriptor()",
    "token_commit_descriptor_ready = true",
    "hidden_kv_survivor_commit_descriptor_ready = true",
    "rejected_branch_discard_descriptor_ready = true",
    "publish_gate_descriptor_ready = true",
]


class P5AARootTokenCommitNoopRuntimeError(ValueError):
    """Raised when P5AA source-slice constraints are violated."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5AARootTokenCommitNoopRuntimeError(f"missing required file: {rel}")
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
        raise P5AARootTokenCommitNoopRuntimeError("cannot isolate draft-jetspec implementation slice")
    return source[start:end]


def _builder_slice(impl: str) -> str:
    start = impl.find("bool build_root_token_commit_noop_runtime()")
    end = impl.find("    bool build_accept_path_descriptor()", start)
    if start < 0 or end < 0 or end <= start:
        raise P5AARootTokenCommitNoopRuntimeError("cannot isolate P5AA builder before P5T descriptor")
    return impl[start:end]


def _p5aa_branch(impl: str) -> str:
    start = impl.find("if (p5aa_root_token_commit_noop_enabled) {\n                        if (!build_root_token_commit_noop_runtime())")
    end = impl.find("                    if (trace_taps) {\n                        LOG_INF(\"%s: draft-jetspec p5z_root_anchor_accept_path_runtime", start)
    if start < 0 or end < 0 or end <= start:
        raise P5AARootTokenCommitNoopRuntimeError("cannot isolate P5AA branch before P5Z return")
    return impl[start:end]


def _dirty_p5aa_hits() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(["git", "diff", "--name-only", "--diff-filter=ACMR"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError:
        return []
    hits: list[dict[str, Any]] = []
    if proc.returncode != 0:
        return [{"path": "<git diff failed>", "token": proc.stderr.strip()}]
    for line in proc.stdout.splitlines():
        rel = pathlib.Path(line.strip())
        if not rel or rel in P5AA_ALLOWED_FILES or str(rel).startswith("experiments/jetspec/"):
            continue
        diff = subprocess.run(["git", "diff", "--", str(rel)], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False).stdout
        for token in P5AA_TOKENS:
            if token in diff:
                hits.append({"path": str(rel), "token": token})
    return hits


def validate_p5aa_root_token_commit_noop_runtime() -> dict[str, Any]:
    errors: list[str] = []

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5AARootTokenCommitNoopRuntimeError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    token_hits: list[dict[str, Any]] = []
    for rel in _iter_source_files():
        text = _read(rel)
        for token in P5AA_TOKENS:
            if token in text:
                token_hits.append({"path": str(rel), "token": token})
                if rel not in P5AA_ALLOWED_FILES:
                    errors.append(f"P5AA token {token!r} appears outside allowlist: {rel}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5AA must not add explicit CMake wiring in {rel}: {matched}")

    dirty_p5aa_hits = _dirty_p5aa_hits()
    for hit in dirty_p5aa_hits:
        errors.append(f"dirty diff contains P5AA token outside allowlist: {hit}")

    try:
        source = _read(pathlib.Path("common/speculative.cpp"))
        impl = _impl_slice(source)
        builder = _builder_slice(impl)
        branch = _p5aa_branch(impl)
    except P5AARootTokenCommitNoopRuntimeError as exc:
        errors.append(str(exc))
        impl = ""
        builder = ""
        branch = ""

    for token in FORBIDDEN_IMPL_TOKENS:
        if token in impl:
            errors.append(f"P5AA implementation must not contain {token!r}")
    for token in FORBIDDEN_P5AA_BRANCH_TOKENS:
        if token in builder or token in branch:
            errors.append(f"P5AA builder/branch must not contain {token}")
    if branch and "return true;" not in branch:
        errors.append("P5AA branch must return before hidden/KV commit, discard, and publish")
    if builder and "build_token_commit_descriptor()" in builder:
        errors.append("P5AA must build the root no-op runtime object, not call the P5T descriptor builder")
    if builder and "token_commit_descriptor_ready = true" in builder:
        errors.append("P5AA must not mark the P5T descriptor ready")
    if "runtime_supported=true" in impl:
        errors.append("P5AA must not claim runtime_supported=true")
    if re.search(r"common_sampler_sample|llama_sampler", impl):
        errors.append("P5AA must not sample draft-head logits")

    docs = _read(pathlib.Path("docs/speculative.md"))
    for token in [
        "root_token_commit_noop_runtime_ready=1",
        "actual_committed_tokens=0",
        "root_verified_anchor=1",
        "accept_path_len=0",
        "actual_accepted_nodes=0",
        "correction_token_present=0",
        "no real token commit",
        "no visible token publish",
        "no hidden/KV commit",
        "no rejected-branch discard",
        "no publish",
        "no visible state change",
        "no KV mutation",
        "no draft tokens",
    ]:
        if token not in docs:
            errors.append(f"docs/speculative.md missing P5AA boundary token: {token}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5aa_root_token_commit_noop_runtime_validated" if not errors else "p5aa_root_token_commit_noop_runtime_invalid",
        "allowed_files": sorted(str(path) for path in P5AA_ALLOWED_FILES),
        "token_hits": token_hits,
        "cmake_hits": cmake_hits,
        "dirty_p5aa_hits": dirty_p5aa_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out = validate_p5aa_root_token_commit_noop_runtime()
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    elif out["ok"]:
        print("P5AA root-token-commit no-op runtime validation passed")
    else:
        print("P5AA root-token-commit no-op runtime validation failed", file=sys.stderr)
        for error in out["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
