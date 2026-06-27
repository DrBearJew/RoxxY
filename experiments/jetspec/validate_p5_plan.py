#!/usr/bin/env python3
"""Validate the inert JetSpec P5 default-off candidate plan."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any


DEFAULT_PLAN = "jetspec_p5_default_off_plan.md"

REQUIRED_SECTIONS = [
    "# JetSpec P5 default-off production candidate plan",
    "## Context",
    "## Evidence checked",
    "## Non-goals",
    "## Default-off controls",
    "## Candidate sequence",
    "### P5A: loader-registration candidate, default-off and fail-closed",
    "### P5B: target hidden tap capture candidate, default-off",
    "### P5C: speculative type integration candidate, explicit opt-in only",
    "## Preview-GGUF rejection contract",
    "## No-default-behavior-change proof",
    "## Verification matrix",
    "## Rejected alternatives",
    "## Approval gate",
]

REQUIRED_TOKENS = [
    "Status: proposed inert plan only; not approved for production edits.",
    "Do not edit production paths in this plan.",
    "LLAMA_JETSPEC_EXPERIMENTAL=1",
    "LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1",
    "--spec-type draft-jetspec",
    "COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC",
    "LLM_ARCH_JETSPEC_QWEN3_DRAFT_HEAD",
    "jetspec_qwen3_draft_head",
    "runtime_supported=false",
    "preview_not_allowed",
    "unsupported_runtime",
    "metadata_only=true",
    "exactly 91 BF16 tensors",
    "raw BF16",
    "default disabled",
    "no default behavior change",
    "llama-server",
    "draft-mtp",
    "ngram",
    "src/CMakeLists.txt",
    "file(GLOB LLAMA_MODELS_SOURCES \"models/*.cpp\")",
    "src/llama-arch.h",
    "src/llama-arch.cpp",
    "src/models/models.h",
    "src/models/jetspec_qwen3_draft_head.cpp",
    "src/llama-model.cpp",
    "common/common.h",
    "common/speculative.cpp",
    "common/speculative.h",
    "common/arg.cpp",
    "tools/server/server-context.cpp",
    "docs/speculative.md",
    "include/llama.h",
    "rollback: revert",
    "P5 production edits are approved",
]

REQUIRED_ORDER = ["P5A", "P5B", "P5C"]

FORBIDDEN_PATTERNS = [
    r"Status:\s*accepted",
    r"Status:\s*approved",
    r"default-enable",
    r"production-ready now",
    r"runtime_supported=true",
    r"auto-enable",
]


class P5PlanError(ValueError):
    """Raised when the P5 plan is invalid."""


def validate_plan(path: pathlib.Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lower = text.lower()
    errors: list[str] = []

    for section in REQUIRED_SECTIONS:
        if section not in text:
            errors.append(f"missing required section: {section}")

    for token in REQUIRED_TOKENS:
        if token not in text:
            errors.append(f"missing required token: {token}")

    positions: list[tuple[str, int]] = []
    for phase in REQUIRED_ORDER:
        match = re.search(rf"###\s+{phase}\b", text)
        if not match:
            errors.append(f"missing phase heading: {phase}")
        else:
            positions.append((phase, match.start()))
    if positions != sorted(positions, key=lambda item: item[1]):
        errors.append(f"P5 subphases are out of order: {positions}")

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, lower):
            errors.append(f"forbidden plan claim matched: {pattern}")

    if "This plan is not an approval to edit production paths." not in text:
        errors.append("approval gate must explicitly block production edits")
    if "If approval is absent, keep all work under `experiments/jetspec/`." not in text:
        errors.append("approval gate must keep non-approved work under experiments/jetspec")

    return {
        "ok": not errors,
        "errors": errors,
        "path": str(path),
        "sections_checked": len(REQUIRED_SECTIONS),
        "tokens_checked": len(REQUIRED_TOKENS),
        "phase_order": [phase for phase, _ in positions],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, default=here / DEFAULT_PLAN)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        out = validate_plan(args.plan.resolve())
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
