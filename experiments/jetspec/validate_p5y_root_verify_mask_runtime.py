#!/usr/bin/env python3
"""Validate the P5Y JetSpec root-only verify-mask runtime source slice."""

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

P5Y_ALLOWED_FILES = {
    pathlib.Path("common/speculative.cpp"),
    pathlib.Path("docs/speculative.md"),
}

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("common/speculative.cpp"): [
        "LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY",
        "JETSPEC_ROOT_VERIFY_MASK_RUNTIME_PHASE",
        "root_verify_mask_runtime",
        "verify_mask_runtime_ready",
        "invalid_root_verify_mask_runtime",
        "n_root_verify_mask_runtime_builds",
        "root_verify_mask_runtime_hash_last",
        "root_verify_mask_runtime_seq_id_last",
        "root_verify_mask_runtime_ready",
        "root_verify_mask_rows",
        "root_verify_mask_cols",
        "root_verify_mask_values",
        "build_root_only_verify_mask_runtime",
        "p5y_root_verify_mask_enabled && !p5x_root_tree_enabled",
        "!p5x_root_tree_enabled || !root_tree_runtime_ready || root_tree_runtime_hash_last == 0",
        "tree_build_actual_nodes_last != 1",
        "tree_token_ids[0] < 0",
        "actual_verify_mask_entries_last = 1",
        "root_verify_mask_rows[0] = 0",
        "root_verify_mask_cols[0] = 0",
        "root_verify_mask_values[0] = 1",
        "actual_verify_mask_entries_last > tree_build_actual_nodes_last * tree_build_actual_nodes_last",
        "runtime_phase = jetspec_runtime_phase::verify_mask_runtime_ready",
        "n_root_verify_mask_runtime_builds++",
        "disable_runtime_state(jetspec_runtime_failure::invalid_root_verify_mask_runtime",
        "p5y_root_verify_mask_runtime",
        "actual_tree_nodes=%d",
        "actual_verify_mask_entries=1",
        "verify_mask_rows=1",
        "verify_mask_cols=1",
        "root_attends_self=%d",
        "root_mask_row=%d",
        "root_mask_col=%d",
        "prefix_visible=1",
        "ancestor_only=1",
        "sibling_visible=0",
        "descendant_visible=0",
        "no_draft_head_graph=1",
        "no_mask_tensor=1",
        "no_accept=1",
        "no_token_commit=1",
        "no_hidden_kv_commit=1",
        "no_rejected_branch_discard=1",
        "no_publish=1",
        "no_visible_state_change=1",
        "no_kv_mutation=1",
        "no_draft_tokens=1",
    ],
    pathlib.Path("docs/speculative.md"): [
        "P5Y root-only verify-mask materialization",
        "LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY=1",
        "requires `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`",
        "actual_verify_mask_entries=1",
        "root_attends_self=1",
        "prefix_visible=1",
        "ancestor_only=1",
        "sibling_visible=0",
        "descendant_visible=0",
        "no mask tensor",
        "no accept",
        "no token commit",
        "no hidden/KV commit",
        "no rejected-branch discard",
        "no publish",
        "no visible state change",
        "no KV mutation",
        "no draft tokens",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5y_root_verify_mask_runtime_candidate.md"): [
        "JetSpec P5Y root-only verify-mask runtime candidate",
        "approved bounded production-source slice",
        "LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY=1",
        "requires `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`",
        "actual_verify_mask_entries=1",
        "root_attends_self=1",
        "prefix_visible=1",
        "ancestor_only=1",
        "sibling_visible=0",
        "descendant_visible=0",
        "returns before P5S",
        "no draft-head graph execution",
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

P5Y_TOKENS = [
    "P5Y root-only verify-mask materialization",
    "jetspec_p5y_root_verify_mask_runtime",
    "validate_p5y_root_verify_mask_runtime",
    "test_p5y_root_verify_mask_runtime",
    "probe_p5y_root_verify_mask_trace",
    "test_p5y_root_verify_mask_trace_probe",
    "JETSPEC_ROOT_VERIFY_MASK_RUNTIME_PHASE",
    "LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY",
    "verify_mask_runtime_ready",
    "root_verify_mask_runtime_ready",
    "build_root_only_verify_mask_runtime",
    "actual_verify_mask_entries=1",
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
    "jetspec_p5y_root_verify_mask_runtime",
    "validate_p5y_root_verify_mask_runtime",
    "test_p5y_root_verify_mask_runtime",
    "probe_p5y_root_verify_mask_trace",
    "test_p5y_root_verify_mask_trace_probe",
    "JETSPEC_ROOT_VERIFY_MASK_RUNTIME_PHASE",
    "LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY",
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

FORBIDDEN_P5Y_BRANCH_TOKENS = [
    "build_accept_path_descriptor()",
    "build_token_commit_descriptor()",
    "build_hidden_kv_survivor_commit_descriptor()",
    "build_rejected_branch_discard_descriptor()",
    "build_publish_gate_descriptor()",
]


class P5YRootVerifyMaskRuntimeError(ValueError):
    """Raised when P5Y source-slice constraints are violated."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5YRootVerifyMaskRuntimeError(f"missing required file: {rel}")
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
        raise P5YRootVerifyMaskRuntimeError("cannot isolate draft-jetspec implementation slice")
    return source[start:end]


def _p5y_branch(impl: str) -> str:
    start = impl.find("if (p5y_root_verify_mask_enabled) {\n                if (!build_root_only_verify_mask_runtime())")
    end = impl.find("            if (trace_taps) {\n                LOG_INF(\"%s: draft-jetspec p5x_root_tree_runtime", start)
    if start < 0 or end < 0 or end <= start:
        raise P5YRootVerifyMaskRuntimeError("cannot isolate P5Y branch before P5S accept path")
    return impl[start:end]


def _dirty_p5y_hits() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(["git", "diff", "--name-only", "--diff-filter=ACMR"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError:
        return []
    hits: list[dict[str, Any]] = []
    if proc.returncode != 0:
        return [{"path": "<git diff failed>", "token": proc.stderr.strip()}]
    for line in proc.stdout.splitlines():
        rel = pathlib.Path(line.strip())
        if not rel or rel in P5Y_ALLOWED_FILES or str(rel).startswith("experiments/jetspec/"):
            continue
        diff = subprocess.run(["git", "diff", "--", str(rel)], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False).stdout
        for token in P5Y_TOKENS:
            if token in diff:
                hits.append({"path": str(rel), "token": token})
    return hits


def validate_p5y_root_verify_mask_runtime() -> dict[str, Any]:
    errors: list[str] = []

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5YRootVerifyMaskRuntimeError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    token_hits: list[dict[str, Any]] = []
    for rel in _iter_source_files():
        text = _read(rel)
        for token in P5Y_TOKENS:
            if token in text:
                token_hits.append({"path": str(rel), "token": token})
                if rel not in P5Y_ALLOWED_FILES:
                    errors.append(f"P5Y token {token!r} appears outside allowlist: {rel}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5Y must not add explicit CMake wiring in {rel}: {matched}")

    dirty_p5y_hits = _dirty_p5y_hits()
    for hit in dirty_p5y_hits:
        errors.append(f"dirty diff contains P5Y token outside allowlist: {hit}")

    try:
        source = _read(pathlib.Path("common/speculative.cpp"))
        impl = _impl_slice(source)
        branch = _p5y_branch(impl)
    except P5YRootVerifyMaskRuntimeError as exc:
        errors.append(str(exc))
        impl = ""
        branch = ""

    for token in FORBIDDEN_IMPL_TOKENS:
        if token in impl:
            errors.append(f"P5Y implementation must not contain {token!r}")
    for token in FORBIDDEN_P5Y_BRANCH_TOKENS:
        if token in branch:
            errors.append(f"P5Y branch must return before {token}")
    if branch and "return true;" not in branch:
        errors.append("P5Y branch must return before P5S accept path")
    if "runtime_supported=true" in impl:
        errors.append("P5Y must not claim runtime_supported=true")
    if re.search(r"common_sampler_sample|llama_sampler", impl):
        errors.append("P5Y must not sample draft-head logits")
    if "build_verify_mask_descriptor()" in branch:
        errors.append("P5Y must build the root-only runtime mask, not call the P5R descriptor builder")

    docs = _read(pathlib.Path("docs/speculative.md"))
    for token in [
        "actual_verify_mask_entries=1",
        "root_attends_self=1",
        "prefix_visible=1",
        "no mask tensor",
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
            errors.append(f"docs/speculative.md missing P5Y boundary token: {token}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5y_root_verify_mask_runtime_validated" if not errors else "p5y_root_verify_mask_runtime_invalid",
        "allowed_files": sorted(str(path) for path in P5Y_ALLOWED_FILES),
        "token_hits": token_hits,
        "cmake_hits": cmake_hits,
        "dirty_p5y_hits": dirty_p5y_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out = validate_p5y_root_verify_mask_runtime()
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    elif out["ok"]:
        print("P5Y root-only verify-mask runtime validation passed")
    else:
        print("P5Y root-only verify-mask runtime validation failed", file=sys.stderr)
        for error in out["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
