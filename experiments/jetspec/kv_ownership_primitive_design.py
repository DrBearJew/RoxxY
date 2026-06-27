#!/usr/bin/env python3
"""P5K inert JetSpec KV ownership primitive design packet.

This turns the P5J exact-missing primitive gaps into explicit design-only
contracts. It reads source text for audited helper names, but does not edit
production files, instantiate llama_context, execute a draft-head graph, emit
draft tokens, mutate KV state, add server behavior, or claim performance/
promotion readiness.
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
STATUS = "kv_ownership_primitive_design_verified_not_executed"

REQUIRED_HELPERS = [
    "seq_rm",
    "seq_cp",
    "seq_import_physical",
    "seq_keep",
    "find_slot",
    "apply_ubatch",
]

REQUIRED_DESIGNS = [
    "hidden_kv_survivor_commit",
    "rejected_branch_discard",
    "cross_sequence_isolation",
]

REQUIRED_INVARIANTS = {
    "hidden_kv_survivor_commit": [
        "accepted_path physical gather/compact",
        "[root | accepted] only",
        "correction hidden deferred",
        "committed tail compact",
    ],
    "rejected_branch_discard": [
        "rejected transient tree slots unreachable",
        "accepted path preserved",
        "not range removal only",
        "rollback restores pre-round state",
    ],
    "cross_sequence_isolation": [
        "other-sequence slots unchanged",
        "seq_to_stream isolation",
        "no shared-slot corruption",
        "rollback preserves other sequences",
    ],
}

FORBIDDEN_IMPLICIT_HELPERS = [
    "seq_cp",
    "seq_rm",
    "seq_import_physical",
]

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


class KVOwnershipPrimitiveDesignError(ValueError):
    """Raised when a P5K primitive design fixture violates the contract."""


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_source(rel: str) -> str:
    path = REPO_ROOT / rel
    if not path.exists():
        raise KVOwnershipPrimitiveDesignError(f"source file missing: {rel}")
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
            raise KVOwnershipPrimitiveDesignError(f"runtime boundary requires {key}=false")
        out[key] = False
    out["source_text_read_only"] = True
    out["design_only"] = True
    return out


def _validate_helper(helper: Mapping[str, Any], source_files: Sequence[str]) -> dict[str, Any]:
    symbol = str(helper["symbol"])
    if symbol not in REQUIRED_HELPERS:
        raise KVOwnershipPrimitiveDesignError(f"unexpected audited helper: {symbol}")
    verdict = str(helper.get("verdict", ""))
    if verdict != "audited_non_exact_helper":
        raise KVOwnershipPrimitiveDesignError(f"{symbol}: verdict must be audited_non_exact_helper")
    reason = str(helper.get("reason", ""))
    if "not exact" not in reason:
        raise KVOwnershipPrimitiveDesignError(f"{symbol}: reason must explain it is not exact")
    exact_for_actions = helper.get("exact_for_actions", [])
    if exact_for_actions != []:
        raise KVOwnershipPrimitiveDesignError(f"{symbol}: exact_for_actions must be empty")
    locations = _find_symbol_locations(source_files, symbol)
    if not locations:
        raise KVOwnershipPrimitiveDesignError(f"{symbol}: source location not found")
    return {
        "symbol": symbol,
        "verdict": verdict,
        "reason": reason,
        "exact_for_actions": [],
        "locations": locations[:8],
        "location_count": len(locations),
    }


def _validate_design(design: Mapping[str, Any]) -> dict[str, Any]:
    design_id = str(design["id"])
    if design_id not in REQUIRED_DESIGNS:
        raise KVOwnershipPrimitiveDesignError(f"unexpected primitive design: {design_id}")
    proposed = str(design["proposed_primitive"])
    if not proposed.startswith("llama_kv_cache_jetspec_"):
        raise KVOwnershipPrimitiveDesignError(f"{design_id}: proposed primitive must use llama_kv_cache_jetspec_ prefix")
    if bool(design.get("implementation_approved", True)):
        raise KVOwnershipPrimitiveDesignError(f"{design_id}: implementation_approved must be false")
    if bool(design.get("exact_existing_primitive_proven", True)):
        raise KVOwnershipPrimitiveDesignError(f"{design_id}: exact_existing_primitive_proven must be false")
    if bool(design.get("touches_production_now", True)):
        raise KVOwnershipPrimitiveDesignError(f"{design_id}: touches_production_now must be false")
    if bool(design.get("runtime_executed_now", True)):
        raise KVOwnershipPrimitiveDesignError(f"{design_id}: runtime_executed_now must be false")
    if bool(design.get("claims_performance", True)) or bool(design.get("claims_promotion", True)):
        raise KVOwnershipPrimitiveDesignError(f"{design_id}: performance/promotion claims are forbidden")
    status = str(design.get("status", ""))
    if status != "design_only_missing_implementation":
        raise KVOwnershipPrimitiveDesignError(f"{design_id}: status must be design_only_missing_implementation")
    must_not_use = [str(item) for item in design.get("must_not_use_as_implicit_mapping", [])]
    missing_forbidden = [item for item in FORBIDDEN_IMPLICIT_HELPERS if item not in must_not_use]
    if missing_forbidden:
        raise KVOwnershipPrimitiveDesignError(f"{design_id}: must forbid implicit helpers {missing_forbidden}")
    invariants = [str(item) for item in design.get("invariants", [])]
    missing_invariants = [item for item in REQUIRED_INVARIANTS[design_id] if item not in invariants]
    if missing_invariants:
        raise KVOwnershipPrimitiveDesignError(f"{design_id}: missing invariants {missing_invariants}")
    evidence = [str(item) for item in design.get("source_backing", [])]
    if not any("P5J" in item for item in evidence):
        raise KVOwnershipPrimitiveDesignError(f"{design_id}: source_backing must cite P5J")
    if not any("P5H" in item or "P5G" in item for item in evidence):
        raise KVOwnershipPrimitiveDesignError(f"{design_id}: source_backing must cite prior readiness evidence")
    return {
        "id": design_id,
        "proposed_primitive": proposed,
        "status": status,
        "implementation_approved": False,
        "exact_existing_primitive_proven": False,
        "touches_production_now": False,
        "runtime_executed_now": False,
        "claims_performance": False,
        "claims_promotion": False,
        "must_not_use_as_implicit_mapping": must_not_use,
        "invariants": invariants,
        "source_backing": evidence,
    }


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("status") != STATUS:
        raise KVOwnershipPrimitiveDesignError(f"fixture status must be {STATUS!r}")
    if str(data.get("scope")) != "experiments/jetspec only":
        raise KVOwnershipPrimitiveDesignError("scope must be experiments/jetspec only")
    if data.get("production_paths_touched", []) != []:
        raise KVOwnershipPrimitiveDesignError("P5K design must not touch production paths")
    source_files = [str(path) for path in data["source_files"]]
    boundary = _validate_runtime_boundary(data["runtime_boundary"])
    helpers = [_validate_helper(helper, source_files) for helper in data["audited_non_exact_helpers"]]
    helper_symbols = [helper["symbol"] for helper in helpers]
    missing_helpers = [symbol for symbol in REQUIRED_HELPERS if symbol not in helper_symbols]
    if missing_helpers:
        raise KVOwnershipPrimitiveDesignError(f"missing audited helpers: {missing_helpers}")
    designs = [_validate_design(design) for design in data["primitive_designs"]]
    design_ids = [design["id"] for design in designs]
    missing_designs = [design_id for design_id in REQUIRED_DESIGNS if design_id not in design_ids]
    if missing_designs:
        raise KVOwnershipPrimitiveDesignError(f"missing required primitive designs: {missing_designs}")
    duplicates = sorted({item for item in design_ids if design_ids.count(item) > 1})
    if duplicates:
        raise KVOwnershipPrimitiveDesignError(f"duplicate primitive designs: {duplicates}")
    return {
        "ok": True,
        "status": STATUS,
        "scope": "experiments/jetspec only",
        "source_files": source_files,
        "production_paths_touched": [],
        "runtime_boundary": {**boundary, "no_runtime_execution": True},
        "audited_non_exact_helpers": sorted(helpers, key=lambda item: REQUIRED_HELPERS.index(item["symbol"])),
        "primitive_designs": sorted(designs, key=lambda item: REQUIRED_DESIGNS.index(item["id"])),
        "coverage": {
            "required_helpers": REQUIRED_HELPERS,
            "required_helpers_present": True,
            "required_designs": REQUIRED_DESIGNS,
            "required_designs_present": True,
            "design_count": len(designs),
            "all_designs_missing_implementation": all(design["status"] == "design_only_missing_implementation" for design in designs),
            "implicit_mappings_forbidden": FORBIDDEN_IMPLICIT_HELPERS,
            "no_runtime_or_production_touch": True,
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
    except (OSError, KeyError, TypeError, ValueError, KVOwnershipPrimitiveDesignError) as exc:
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
