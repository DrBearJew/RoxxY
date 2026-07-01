#!/usr/bin/env python3
"""No-model P5C/P5F JetSpec speculative route gate probe.

This source probe verifies that `draft-jetspec` remains explicit opt-in,
fail-closed before runtime execution, and does not silently fall back to
`draft-simple`. It does not instantiate llama contexts or run a model.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SOURCE = REPO_ROOT / "common/speculative.cpp"


class RouteGateProbeError(RuntimeError):
    """Raised when source probing cannot proceed."""


def _source() -> str:
    if not SOURCE.exists():
        raise RouteGateProbeError(f"missing source: {SOURCE}")
    return SOURCE.read_text(encoding="utf-8", errors="replace")


def _line_no(text: str, needle: str) -> int:
    pos = text.find(needle)
    if pos < 0:
        return -1
    return text.count("\n", 0, pos) + 1


def _slice(text: str, start_token: str, end_token: str) -> str:
    start = text.find(start_token)
    end = text.find(end_token, start + len(start_token))
    if start < 0 or end < 0 or end <= start:
        raise RouteGateProbeError(f"cannot isolate slice {start_token!r}..{end_token!r}")
    return text[start:end]


def probe_p5c_route_gate() -> dict[str, Any]:
    errors: list[str] = []
    text = _source()
    init = _slice(text, "common_speculative * common_speculative_init", "std::vector<std::unique_ptr<common_speculative_impl>> impls")
    impl = _slice(text, "struct common_speculative_impl_draft_jetspec", "struct common_speculative_impl_draft_mtp")

    required_tokens = [
        '{"draft-jetspec", COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC}',
        '{"jetspec",       COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC}',
        "bool has_draft_jetspec",
        "LLAMA_JETSPEC_EXPERIMENTAL",
        "draft-jetspec requires LLAMA_JETSPEC_EXPERIMENTAL=1; disabling JetSpec before runtime execution",
        "draft-jetspec requires a target context and a loaded draft-head model; disabling JetSpec before runtime execution",
        "common_speculative_jetspec_preflight",
        "draft-jetspec preflight failed",
        "configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC, params))",
        "!has_draft_jetspec",
    ]
    for token in required_tokens:
        if token not in text:
            errors.append(f"missing route-gate token: {token}")

    env_pos = init.find("LLAMA_JETSPEC_EXPERIMENTAL")
    ctx_pos = init.find("requires a target context and a loaded draft-head model")
    preflight_pos = init.find("common_speculative_jetspec_preflight")
    push_pos = init.find("configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC, params))")
    fallback_pos = init.find("has_draft_model_path && !has_mtp && !has_draft_eagle3 && !has_draft_jetspec")
    if min(env_pos, ctx_pos, preflight_pos, push_pos, fallback_pos) < 0:
        errors.append("could not locate all route-gate positions")
    else:
        if not (fallback_pos < env_pos < ctx_pos < preflight_pos < push_pos):
            errors.append("route-gate order must be fallback guard -> env gate -> context gate -> preflight -> JetSpec config push")

    simple_enable_slice = _slice(init, "if (has_draft_simple) {", "if (has_draft_jetspec) {")
    if "has_draft_jetspec" not in simple_enable_slice:
        errors.append("draft-simple auto-enable guard must explicitly exclude has_draft_jetspec")
    next_simple_push = init.find("if (has_draft_simple) {", env_pos)
    jetspec_gate_slice = init[env_pos:next_simple_push if next_simple_push > env_pos else push_pos]
    if "COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE" in jetspec_gate_slice:
        errors.append("draft-jetspec gate block must not push draft-simple as a fallback")
    if "result->push_back" in impl:
        errors.append("draft-jetspec implementation must not emit draft tokens")
    if "llama_decode" in impl or "tree_accept" in impl:
        errors.append("draft-jetspec no-model route must not run decode/tree_accept")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "p5c_route_gate_verified_no_model" if not errors else "p5c_route_gate_invalid",
        "runtime_executed": False,
        "draft_context_created": False,
        "line_map": {
            "env_gate": _line_no(text, "LLAMA_JETSPEC_EXPERIMENTAL"),
            "missing_context_gate": _line_no(text, "requires a target context and a loaded draft-head model"),
            "preflight_gate": _line_no(text, "common_speculative_jetspec_preflight(params.draft"),
            "jetspec_config_push": _line_no(text, "configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC"),
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        result = probe_p5c_route_gate()
    except RouteGateProbeError as exc:
        result = {"ok": False, "status": "p5c_route_gate_probe_error", "errors": [str(exc)]}
    if args.json or not result["ok"]:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"P5C route gate probe passed status={result['status']} runtime_executed=false")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
