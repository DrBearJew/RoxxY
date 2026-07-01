#!/usr/bin/env python3
"""Validate JetSpec P5AB-P5AD root no-op readiness source slices."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

REQUIRED: dict[pathlib.Path, list[str]] = {
    pathlib.Path("common/speculative.cpp"): [
        "LLAMA_JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_ONLY",
        "LLAMA_JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_ONLY",
        "LLAMA_JETSPEC_ROOT_PUBLISH_GATE_NOOP_ONLY",
        "JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_RUNTIME_PHASE",
        "JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_RUNTIME_PHASE",
        "JETSPEC_ROOT_PUBLISH_GATE_NOOP_RUNTIME_PHASE",
        "build_root_hidden_kv_commit_noop_runtime",
        "build_root_rejected_branch_discard_noop_runtime",
        "build_root_publish_gate_noop_runtime",
        "root_hidden_kv_commit_noop_runtime_ready",
        "root_rejected_branch_discard_noop_runtime_ready",
        "root_publish_gate_noop_runtime_ready",
        "root_runtime_ready_for_real_test_last = 1",
        "actual_survivor_pages_committed_last = 0",
        "actual_pages_discarded_last = 0",
        "rejected_branch_pages_reachable_after_discard_last = 0",
        "actual_publish_visible_state_last = 0",
        "invalid_root_hidden_kv_commit_noop_runtime",
        "invalid_root_rejected_branch_discard_noop_runtime",
        "invalid_root_publish_gate_noop_runtime",
        "p5ad_root_publish_gate_noop_enabled && (!p5x_root_tree_enabled || !p5y_root_verify_mask_enabled || !p5z_root_anchor_accept_path_enabled || !p5aa_root_token_commit_noop_enabled || !p5ab_root_hidden_kv_commit_noop_enabled || !p5ac_root_rejected_branch_discard_noop_enabled)",
        "draft-jetspec p5ad_root_publish_gate_noop_runtime",
        "root_runtime_ready_for_real_test=%d",
        "publish_after_commit_and_discard_only=1",
        "no_real_hidden_kv_commit=1",
        "no_real_rejected_branch_discard=1",
        "no_real_publish=1",
        "no_visible_state_change=1",
        "no_kv_mutation=1",
        "no_draft_tokens=1",
    ],
    pathlib.Path("docs/speculative.md"): [
        "P5AB/P5AC/P5AD complete the root-only no-op round tail",
        "LLAMA_JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_ONLY=1",
        "LLAMA_JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_ONLY=1",
        "LLAMA_JETSPEC_ROOT_PUBLISH_GATE_NOOP_ONLY=1",
        "root_runtime_ready_for_real_test=1",
        "ready to start a real root-only test",
        "no real hidden/KV commit",
        "no real rejected-branch discard",
        "no real publish",
        "no visible state change",
        "no KV mutation",
        "no draft tokens",
    ],
    pathlib.Path("experiments/jetspec/jetspec_p5ab_to_p5ad_root_noop_readiness_candidate.md"): [
        "JetSpec P5AB-P5AD root no-op readiness candidate",
        "ready to start a real/live root-only trace test",
        "LLAMA_JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_ONLY=1",
        "LLAMA_JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_ONLY=1",
        "LLAMA_JETSPEC_ROOT_PUBLISH_GATE_NOOP_ONLY=1",
        "root_runtime_ready_for_real_test=1",
        "do not mutate KV",
        "do not publish visible state",
        "do not emit draft tokens",
    ],
}

FORBIDDEN = [
    "llama_kv_cache",
    "seq_cp",
    "seq_rm",
    "seq_import_physical",
    "common_sampler_sample",
    "llama_sampler",
    "result->push_back",
]

BRANCH_FORBIDDEN = [
    "build_hidden_kv_survivor_commit_descriptor()",
    "build_rejected_branch_discard_descriptor()",
    "build_publish_gate_descriptor()",
    "hidden_kv_survivor_commit_descriptor_ready = true",
    "rejected_branch_discard_descriptor_ready = true",
    "publish_gate_descriptor_ready = true",
]


def _read(rel: pathlib.Path) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")


def _impl() -> str:
    text = _read(pathlib.Path("common/speculative.cpp"))
    start = text.find("struct common_speculative_impl_draft_jetspec")
    end = text.find("struct common_speculative_impl_draft_mtp")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("cannot isolate JetSpec implementation")
    return text[start:end]


def validate_p5ab_to_p5ad_root_noop_readiness() -> dict[str, Any]:
    errors: list[str] = []
    token_hits: list[dict[str, str]] = []
    for rel, tokens in REQUIRED.items():
        path = REPO_ROOT / rel
        if not path.exists():
            errors.append(f"missing required file: {rel}")
            continue
        text = _read(rel)
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")
            else:
                token_hits.append({"path": str(rel), "token": token})

    try:
        impl = _impl()
    except RuntimeError as exc:
        errors.append(str(exc))
        impl = ""
    for token in FORBIDDEN:
        if token in impl:
            errors.append(f"P5AB-P5AD implementation must not contain {token!r}")
    start = impl.find("if (p5ab_root_hidden_kv_commit_noop_enabled) {")
    end = impl.find("if (trace_taps) {\n                            LOG_INF(\"%s: draft-jetspec p5aa_root_token_commit_noop_runtime", start)
    branch = impl[start:end] if start >= 0 and end > start else ""
    if not branch:
        errors.append("cannot isolate P5AB-P5AD root no-op branch")
    for token in BRANCH_FORBIDDEN:
        if token in branch:
            errors.append(f"P5AB-P5AD branch must not contain {token}")
    if branch and "return true;" not in branch:
        errors.append("P5AD branch must return before legacy descriptor chain")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5ab_to_p5ad_root_noop_readiness_validated" if not errors else "p5ab_to_p5ad_root_noop_readiness_invalid",
        "token_hits": token_hits,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = validate_p5ab_to_p5ad_root_noop_readiness()
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
