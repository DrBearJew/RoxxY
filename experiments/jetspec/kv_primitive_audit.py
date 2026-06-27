#!/usr/bin/env python3
"""P5J inert JetSpec KV/runtime primitive audit.

This reads source text and classifies P5I missing runtime primitives. It does not
edit production files, instantiate llama_context, execute a draft-head graph,
emit draft tokens, mutate real KV state, add server behavior, or claim
performance/promotion readiness.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
STATUS = "kv_primitive_audit_verified_not_executed"

REQUIRED_ACTIONS = [
    "hidden_kv_survivor_commit",
    "rejected_branch_discard",
    "cross_sequence_isolation",
]

ALLOWED_CLASSIFICATIONS = {
    "exact_existing_primitive_candidate",
    "exact_missing_primitive",
    "blocked_pending_explicit_approval",
}

IMPLICIT_FORBIDDEN_MAPPINGS = {
    "seq_cp",
    "seq_rm",
    "seq_import_physical",
    "llama_kv_cache::seq_cp",
    "llama_kv_cache::seq_rm",
    "llama_kv_cache::seq_import_physical",
}

RUNTIME_FALSE_KEYS = [
    "llama_context_created",
    "draft_head_graph_executed",
    "draft_tokens_emitted",
    "real_kv_mutated",
    "server_route_added",
    "runtime_supported",
    "performance_claimed",
    "promotion_claimed",
]


class KVPrimitiveAuditError(ValueError):
    """Raised when a P5J audit fixture violates the contract."""


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_source(rel: str) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise KVPrimitiveAuditError(f"source file missing: {rel}")
    return path.read_text(encoding="utf-8", errors="replace")


def _find_symbol_locations(source_files: Sequence[str], symbol: str) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    needle = re.compile(rf"\b{re.escape(symbol)}\b")
    for rel in source_files:
        text = _read_source(rel)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if needle.search(line):
                locations.append({"path": rel, "line": lineno, "text": line.strip()})
    return locations


def _validate_runtime_boundary(boundary: Mapping[str, Any]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for key in RUNTIME_FALSE_KEYS:
        value = bool(boundary.get(key, True))
        if value:
            raise KVPrimitiveAuditError(f"runtime boundary requires {key}=false")
        out[key] = False
    out["source_text_read_only"] = True
    return out


def _validate_existing_symbol(symbol_obj: Mapping[str, Any], source_files: Sequence[str]) -> dict[str, Any]:
    symbol = str(symbol_obj["symbol"])
    purpose = str(symbol_obj.get("purpose", "")).strip()
    if not purpose:
        raise KVPrimitiveAuditError(f"existing symbol {symbol} requires purpose")
    locations = _find_symbol_locations(source_files, symbol)
    if not locations:
        raise KVPrimitiveAuditError(f"expected existing symbol not found in source text: {symbol}")
    return {
        "symbol": symbol,
        "purpose": purpose,
        "locations": locations[:8],
        "location_count": len(locations),
    }


def _validate_action(action: Mapping[str, Any], source_files: Sequence[str]) -> dict[str, Any]:
    action_id = str(action["id"])
    if action_id not in REQUIRED_ACTIONS:
        raise KVPrimitiveAuditError(f"unknown P5J action: {action_id}")
    classification = str(action["classification"])
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise KVPrimitiveAuditError(f"{action_id}: invalid classification {classification}")
    primitive = str(action["primitive"])
    if primitive in IMPLICIT_FORBIDDEN_MAPPINGS:
        raise KVPrimitiveAuditError(f"{action_id}: implicit mapping to {primitive} is forbidden")
    if classification == "exact_missing_primitive" and primitive != "missing primitive":
        raise KVPrimitiveAuditError(f"{action_id}: exact_missing_primitive must use primitive='missing primitive'")
    if classification == "exact_existing_primitive_candidate":
        if primitive == "missing primitive":
            raise KVPrimitiveAuditError(f"{action_id}: existing primitive candidate must name a primitive")
        locations = _find_symbol_locations(source_files, primitive)
        if not locations:
            raise KVPrimitiveAuditError(f"{action_id}: primitive candidate not found: {primitive}")
    else:
        locations = []
    blockers = [str(item) for item in action.get("blockers", [])]
    if classification != "exact_existing_primitive_candidate" and not blockers:
        raise KVPrimitiveAuditError(f"{action_id}: non-existing classification requires blockers")
    if bool(action.get("touches_production_now", True)):
        raise KVPrimitiveAuditError(f"{action_id}: touches_production_now must be false")
    if bool(action.get("runtime_executed_now", True)):
        raise KVPrimitiveAuditError(f"{action_id}: runtime_executed_now must be false")
    if bool(action.get("claims_promotion", True)) or bool(action.get("claims_performance", True)):
        raise KVPrimitiveAuditError(f"{action_id}: perf/promotion claims are forbidden")
    evidence = [str(item) for item in action.get("evidence", [])]
    if not evidence:
        raise KVPrimitiveAuditError(f"{action_id}: evidence is required")
    return {
        "id": action_id,
        "classification": classification,
        "primitive": primitive,
        "locations": locations[:8],
        "blockers": blockers,
        "touches_production_now": False,
        "runtime_executed_now": False,
        "claims_performance": False,
        "claims_promotion": False,
        "evidence": evidence,
    }


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("status") != STATUS:
        raise KVPrimitiveAuditError(f"fixture status must be {STATUS!r}")
    if str(data.get("scope")) != "experiments/jetspec only":
        raise KVPrimitiveAuditError("scope must be experiments/jetspec only")
    if data.get("production_paths_touched", []) != []:
        raise KVPrimitiveAuditError("P5J audit must not touch production paths")
    source_files = [str(path) for path in data["source_files"]]
    if not source_files:
        raise KVPrimitiveAuditError("source_files must be non-empty")
    runtime_boundary = _validate_runtime_boundary(data["runtime_boundary"])
    existing_symbols = [_validate_existing_symbol(symbol, source_files) for symbol in data["existing_symbols_to_audit"]]
    actions = [_validate_action(action, source_files) for action in data["actions"]]
    ids = [action["id"] for action in actions]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise KVPrimitiveAuditError(f"duplicate action ids: {duplicates}")
    missing = [action_id for action_id in REQUIRED_ACTIONS if action_id not in ids]
    if missing:
        raise KVPrimitiveAuditError(f"missing required P5J actions: {missing}")
    classifications = {name: 0 for name in sorted(ALLOWED_CLASSIFICATIONS)}
    for action in actions:
        classifications[action["classification"]] += 1
    return {
        "ok": True,
        "status": STATUS,
        "scope": "experiments/jetspec only",
        "source_files": source_files,
        "production_paths_touched": [],
        "runtime_boundary": {**runtime_boundary, "no_runtime_execution": True},
        "existing_symbols_audited": existing_symbols,
        "actions": sorted(actions, key=lambda item: REQUIRED_ACTIONS.index(item["id"])),
        "coverage": {
            "required_actions": REQUIRED_ACTIONS,
            "required_actions_present": True,
            "classification_counts": classifications,
            "implicit_mappings_forbidden": sorted(IMPLICIT_FORBIDDEN_MAPPINGS),
            "no_action_touches_production_now": True,
            "no_action_executes_runtime_now": True,
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        out = evaluate_fixture(_load_json(args.fixture.resolve()))
    except (OSError, KeyError, TypeError, ValueError, KVPrimitiveAuditError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
