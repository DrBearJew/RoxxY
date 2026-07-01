#!/usr/bin/env python3
"""Validate the inert P5AH JetSpec draft-head top-k readiness descriptor."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import jetspec_draft_head_topk_readiness as readiness


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
FIXTURE = HERE / "fixtures" / "jetspec_draft_head_topk_readiness_smoke.json"
EXPECTED_OUT = HERE / "fixtures" / "jetspec_draft_head_topk_readiness_smoke.out.json"

REQUIRED_FILES = [
    pathlib.Path("experiments/jetspec/jetspec_p5ah_draft_head_topk_readiness.md"),
    pathlib.Path("experiments/jetspec/jetspec_draft_head_topk_readiness.py"),
    pathlib.Path("experiments/jetspec/validate_p5ah_draft_head_topk_readiness.py"),
    pathlib.Path("experiments/jetspec/test_p5ah_draft_head_topk_readiness.py"),
    pathlib.Path("experiments/jetspec/fixtures/jetspec_draft_head_topk_readiness_smoke.json"),
    pathlib.Path("experiments/jetspec/fixtures/jetspec_draft_head_topk_readiness_smoke.out.json"),
]

REQUIRED_TOKENS: dict[pathlib.Path, list[str]] = {
    pathlib.Path("experiments/jetspec/jetspec_p5ah_draft_head_topk_readiness.md"): [
        "P5AH draft-head top-k construction readiness descriptor",
        "inert readiness descriptor only",
        "not production source approval",
        "experiments/jetspec/",
        "runtime_supported=false",
        "ctx_dft=nullptr",
        "no `llama_decode`",
        "no draft-head graph execution",
        "no logits buffer read",
        "no sampler",
        "no target logits walk",
        "no KV mutation",
        "no CUDA",
        "LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1",
        "LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1",
        "LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1",
        "LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1",
        "future_logits_source=draft_head_full_vocab_logits",
        "synthetic_full_vocab_softmax",
        "topk_only_renormalization",
        "planned_draft_head_logits_rows=1",
        "actual_verified_logits_rows=0",
        "parent_logits_rows=[{logits_row:0,parent_node:0,candidate_nodes:[1,2]}]",
        "rank_semantics=rank_stable_descending_logprob",
        "draft_head_topk_readiness_verified_not_executed",
        "separate explicit production-source approval",
    ],
    pathlib.Path("experiments/jetspec/jetspec_draft_head_topk_readiness.py"): [
        "draft_head_topk_readiness_verified_not_executed",
        "REQUIRED_CHAIN_GATES",
        "LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1",
        "draft_head_full_vocab_logits",
        "FORBIDDEN_SOURCES",
        "target_logits",
        "sampler",
        "synthetic_full_vocab_softmax",
        "topk_only_renormalization",
        "EXPECTED_PLANNED_DRAFT_LOGITS_ROWS = 1",
        "actual_verified_logits_rows",
        "ctx_dft_null",
        "draft_head_graph_executed",
        "llama_decode_called",
        "draft_logits_buffer_read",
        "production_source_approval must be false",
        "explicit production-source approval required before real draft-head logits/top-k construction",
    ],
    pathlib.Path("experiments/jetspec/test_p5ah_draft_head_topk_readiness.py"): [
        "test_smoke_fixture_matches_expected_output",
        "test_rejects_missing_p5ag_gate",
        "test_rejects_actual_logits_rows_without_approval",
        "test_rejects_target_logits_source",
        "test_rejects_draft_context_creation",
        "test_rejects_production_path_touch",
        "test_validator_passes",
    ],
}

P5AH_TOKENS = [
    "P5AH draft-head top-k construction readiness descriptor",
    "draft_head_topk_readiness_verified_not_executed",
    "jetspec_draft_head_topk_readiness",
    "validate_p5ah_draft_head_topk_readiness",
    "test_p5ah_draft_head_topk_readiness",
]

FORBIDDEN_ROOTS = [
    pathlib.Path("common"),
    pathlib.Path("src"),
    pathlib.Path("include"),
    pathlib.Path("tools/server"),
    pathlib.Path("tests"),
    pathlib.Path("examples"),
    pathlib.Path("pocs"),
    pathlib.Path("ggml/src"),
    pathlib.Path("docs"),
]

CMAKE_TOKENS = [
    "experiments/jetspec/jetspec_draft_head_topk_readiness",
    "jetspec_draft_head_topk_readiness",
    "validate_p5ah_draft_head_topk_readiness",
    "test_p5ah_draft_head_topk_readiness",
    "draft_head_topk_readiness_verified_not_executed",
]


class P5AHDraftHeadTopKReadinessError(ValueError):
    """Raised when the P5AH readiness descriptor is invalid."""


def _read(rel: pathlib.Path) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise P5AHDraftHeadTopKReadinessError(f"missing required file: {rel}")
    return path.read_text(encoding="utf-8", errors="replace")


def _cmake_files() -> list[pathlib.Path]:
    files = list(REPO_ROOT.rglob("CMakeLists.txt"))
    files.extend(REPO_ROOT.rglob("*.cmake"))
    return sorted(path.relative_to(REPO_ROOT) for path in files if path.is_file())


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
            matched = [token for token in P5AH_TOKENS if token in text]
            if matched:
                hits.append({"path": str(rel), "tokens": matched})
    return hits


def validate_p5ah_draft_head_topk_readiness() -> dict[str, Any]:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"missing required P5AH file: {rel}")

    for rel, tokens in REQUIRED_TOKENS.items():
        try:
            text = _read(rel)
        except P5AHDraftHeadTopKReadinessError as exc:
            errors.append(str(exc))
            continue
        for token in tokens:
            if token not in text:
                errors.append(f"{rel} missing required token: {token}")

    try:
        fixture_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        actual = readiness.evaluate_fixture(fixture_data)
        expected = json.loads(EXPECTED_OUT.read_text(encoding="utf-8"))
        if actual != expected:
            errors.append("P5AH smoke output differs from expected fixture output")
        if actual.get("status") != readiness.STATUS:
            errors.append(f"P5AH smoke status mismatch: {actual.get('status')}")
        boundary = actual.get("runtime_boundary", {})
        if boundary.get("no_runtime_execution") is not True:
            errors.append("P5AH runtime boundary must report no_runtime_execution=true")
        contract = actual.get("future_topk_contract", {})
        if contract.get("actual_verified_logits_rows") != 0:
            errors.append("P5AH must keep actual_verified_logits_rows=0")
        if contract.get("planned_draft_head_logits_rows") != 1:
            errors.append("P5AH must plan exactly one future draft-head logits row")
        if actual.get("production_source_approval") is not False:
            errors.append("P5AH must not approve production source")
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        errors.append(f"P5AH smoke fixture failed: {exc}")

    forbidden_hits = _scan_forbidden_roots()
    for hit in forbidden_hits:
        errors.append(f"P5AH token appears in forbidden production path: {hit}")

    cmake_hits: list[dict[str, Any]] = []
    for rel in _cmake_files():
        text = _read(rel)
        matched = [token for token in CMAKE_TOKENS if token in text]
        if matched:
            cmake_hits.append({"path": str(rel), "tokens": matched})
            errors.append(f"P5AH must not add explicit CMake wiring in {rel}: {matched}")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5ah_draft_head_topk_readiness_validated" if not errors else "p5ah_draft_head_topk_readiness_invalid",
        "files_checked": [str(path) for path in REQUIRED_FILES],
        "forbidden_root_hits": forbidden_hits,
        "cmake_hits": cmake_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON output (default; accepted for consistency with other validators).")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    parse_args(argv)
    out = validate_p5ah_draft_head_topk_readiness()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
