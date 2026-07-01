#!/usr/bin/env python3
"""P5AH inert JetSpec draft-head top-k construction readiness descriptor.

This prepares the contract for replacing the P5AE synthetic top-k ABI source with
future real draft-head full-vocab logits. It does not create a draft context,
execute a draft-head graph, walk logits, emit draft tokens, mutate KV, add server
behavior, or approve production source wiring.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


STATUS = "draft_head_topk_readiness_verified_not_executed"

REQUIRED_CHAIN_GATES = [
    "LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1",
    "LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1",
    "LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1",
    "LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1",
]

EXPECTED_TARGET_LAYER_IDS = [1, 10, 19, 28, 37]
EXPECTED_TAP_WIDTH = 10240
EXPECTED_EMBEDDING_LENGTH = 2048
EXPECTED_VOCAB_SIZE = 248320
EXPECTED_TOPK_WIDTH = 2
EXPECTED_TREE_DEPTH = 1
EXPECTED_CANDIDATE_NODES = 2
EXPECTED_TREE_NODES = 3
EXPECTED_PLANNED_DRAFT_LOGITS_ROWS = 1

FORBIDDEN_SOURCES = {
    "target_logits",
    "sampler",
    "synthetic_full_vocab_softmax",
    "topk_only_renormalization",
}

FALSE_BOUNDARY_KEYS = [
    "runtime_supported",
    "draft_context_created",
    "draft_head_graph_executed",
    "llama_decode_called",
    "draft_logits_buffer_read",
    "sampler_used",
    "target_logits_walked",
    "target_accept_walked",
    "draft_tokens_emitted",
    "token_commit_performed",
    "hidden_kv_commit_performed",
    "rejected_branch_discard_performed",
    "kv_mutated",
    "visible_state_published",
    "cuda_dispatched",
    "server_route_added",
    "public_api_added",
    "cmake_wired",
    "performance_claimed",
    "promotion_claimed",
]

TRUE_BOUNDARY_KEYS = [
    "ctx_dft_null",
    "model_only_binding_preserved",
]

REQUIRED_FAILURE_MODES = [
    "missing_p5ag_accept_boundary",
    "missing_p5ag_hash",
    "runtime_supported_true",
    "ctx_dft_non_null",
    "draft_context_created",
    "draft_head_graph_execution_attempted",
    "llama_decode_attempted",
    "draft_logits_rows_nonzero_without_approval",
    "target_logits_source_attempted",
    "sampler_source_attempted",
    "synthetic_source_used_as_real_topk",
]

REQUIRED_APPROVAL_GATES = [
    "aggregate_contracts",
    "default_build_only",
    "disabled_no_spec_path",
    "existing_mtp_path",
    "jetspec_opt_in_fail_closed",
    "real_draft_head_logits_source_approval",
    "runtime_correctness_matrix",
    "baseline_benchmark_before_promotion",
]


class DraftHeadTopKReadinessError(ValueError):
    """Raised when the P5AH readiness descriptor is invalid."""


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_bool(mapping: Mapping[str, Any], key: str, expected: bool, *, what: str) -> bool:
    value = bool(mapping.get(key, not expected))
    if value is not expected:
        raise DraftHeadTopKReadinessError(f"{what}.{key} must be {str(expected).lower()}")
    return expected


def _validate_scope(data: Mapping[str, Any]) -> None:
    if data.get("status") != STATUS:
        raise DraftHeadTopKReadinessError(f"fixture status must be {STATUS!r}")
    if str(data.get("scope")) != "experiments/jetspec only":
        raise DraftHeadTopKReadinessError("scope must be experiments/jetspec only")
    if bool(data.get("production_source_approval", True)):
        raise DraftHeadTopKReadinessError("production_source_approval must be false for P5AH readiness")
    touched = [str(path) for path in data.get("production_paths_touched", [])]
    if touched:
        raise DraftHeadTopKReadinessError(f"P5AH must not touch production paths: {touched}")


def _validate_chain(chain: Mapping[str, Any]) -> dict[str, Any]:
    gates = [str(gate) for gate in chain.get("requires_gates", [])]
    missing = [gate for gate in REQUIRED_CHAIN_GATES if gate not in gates]
    if missing:
        raise DraftHeadTopKReadinessError(f"missing required P5AG chain gates: {missing}")
    _require_bool(chain, "p5ag_accept_boundary_ready_required", True, what="chain")
    _require_bool(chain, "p5ag_accept_boundary_hash_required", True, what="chain")
    _require_bool(chain, "topk_abi_root_tail_conflict_rejected", True, what="chain")
    return {
        "requires_gates": REQUIRED_CHAIN_GATES,
        "p5ag_accept_boundary_ready_required": True,
        "p5ag_accept_boundary_hash_required": True,
        "topk_abi_root_tail_conflict_rejected": True,
    }


def _validate_parent_row_map(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if len(rows) != EXPECTED_PLANNED_DRAFT_LOGITS_ROWS:
        raise DraftHeadTopKReadinessError("parent_logits_rows must contain exactly one planned root row")
    for row in rows:
        logits_row = int(row.get("logits_row", -1))
        parent_node = int(row.get("parent_node", -1))
        candidate_nodes = [int(node) for node in row.get("candidate_nodes", [])]
        if logits_row != 0 or parent_node != 0 or candidate_nodes != [1, 2]:
            raise DraftHeadTopKReadinessError("planned logits row 0 must map parent node 0 to candidate nodes [1,2]")
        normalized.append({"logits_row": 0, "parent_node": 0, "candidate_nodes": [1, 2]})
    return normalized


def _validate_topk_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    source = str(contract.get("future_logits_source", ""))
    if source != "draft_head_full_vocab_logits":
        raise DraftHeadTopKReadinessError("future_logits_source must be draft_head_full_vocab_logits")
    listed_forbidden = {str(item) for item in contract.get("forbidden_sources", [])}
    missing_forbidden = sorted(FORBIDDEN_SOURCES - listed_forbidden)
    if missing_forbidden:
        raise DraftHeadTopKReadinessError(f"missing forbidden logits sources: {missing_forbidden}")
    if source in listed_forbidden:
        raise DraftHeadTopKReadinessError("future real draft-head source must not be listed as forbidden")
    _require_bool(contract, "full_vocab_softmax_required", True, what="topk_contract")
    if bool(contract.get("topk_only_renormalization_allowed", True)):
        raise DraftHeadTopKReadinessError("topk_only_renormalization_allowed must be false")

    checks = {
        "target_tap_width": EXPECTED_TAP_WIDTH,
        "embedding_length": EXPECTED_EMBEDDING_LENGTH,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "topk_width": EXPECTED_TOPK_WIDTH,
        "tree_depth": EXPECTED_TREE_DEPTH,
        "candidate_nodes": EXPECTED_CANDIDATE_NODES,
        "output_tree_nodes": EXPECTED_TREE_NODES,
        "planned_draft_head_logits_rows": EXPECTED_PLANNED_DRAFT_LOGITS_ROWS,
        "actual_verified_logits_rows": 0,
        "actual_accepted_nodes": 0,
        "correction_token_present": 0,
    }
    for key, expected in checks.items():
        if int(contract.get(key, -999)) != expected:
            raise DraftHeadTopKReadinessError(f"{key} must be {expected}")
    target_layer_ids = [int(item) for item in contract.get("target_layer_ids", [])]
    if target_layer_ids != EXPECTED_TARGET_LAYER_IDS:
        raise DraftHeadTopKReadinessError("target_layer_ids must be [1,10,19,28,37]")
    if str(contract.get("rank_semantics")) != "rank_stable_descending_logprob":
        raise DraftHeadTopKReadinessError("rank_semantics must be rank_stable_descending_logprob")
    rows = _validate_parent_row_map(contract.get("parent_logits_rows", []))
    return {
        "future_logits_source": source,
        "forbidden_sources": sorted(FORBIDDEN_SOURCES),
        "full_vocab_softmax_required": True,
        "topk_only_renormalization_allowed": False,
        "target_layer_ids": EXPECTED_TARGET_LAYER_IDS,
        "target_tap_width": EXPECTED_TAP_WIDTH,
        "embedding_length": EXPECTED_EMBEDDING_LENGTH,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "topk_width": EXPECTED_TOPK_WIDTH,
        "tree_depth": EXPECTED_TREE_DEPTH,
        "candidate_nodes": EXPECTED_CANDIDATE_NODES,
        "output_tree_nodes": EXPECTED_TREE_NODES,
        "planned_draft_head_logits_rows": EXPECTED_PLANNED_DRAFT_LOGITS_ROWS,
        "actual_verified_logits_rows": 0,
        "actual_accepted_nodes": 0,
        "correction_token_present": 0,
        "rank_semantics": "rank_stable_descending_logprob",
        "parent_logits_rows": rows,
    }


def _validate_boundary(boundary: Mapping[str, Any]) -> dict[str, bool]:
    normalized: dict[str, bool] = {}
    for key in FALSE_BOUNDARY_KEYS:
        normalized[key] = _require_bool(boundary, key, False, what="boundary")
    for key in TRUE_BOUNDARY_KEYS:
        normalized[key] = _require_bool(boundary, key, True, what="boundary")
    normalized["no_runtime_execution"] = True
    return normalized


def _validate_failure_modes(modes: Sequence[Any]) -> list[str]:
    normalized = [str(mode) for mode in modes]
    missing = [mode for mode in REQUIRED_FAILURE_MODES if mode not in normalized]
    if missing:
        raise DraftHeadTopKReadinessError(f"missing required failure modes: {missing}")
    return REQUIRED_FAILURE_MODES


def _validate_approval_gates(gates: Sequence[Any]) -> list[str]:
    normalized = [str(gate) for gate in gates]
    missing = [gate for gate in REQUIRED_APPROVAL_GATES if gate not in normalized]
    if missing:
        raise DraftHeadTopKReadinessError(f"missing required approval gates: {missing}")
    return REQUIRED_APPROVAL_GATES


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    _validate_scope(data)
    chain = _validate_chain(data["chain"])
    topk_contract = _validate_topk_contract(data["future_topk_contract"])
    boundary = _validate_boundary(data["runtime_boundary"])
    failure_modes = _validate_failure_modes(data.get("failure_modes", []))
    approval_gates = _validate_approval_gates(data.get("approval_gates", []))

    return {
        "ok": True,
        "status": STATUS,
        "scope": "experiments/jetspec only",
        "production_source_approval": False,
        "production_paths_touched": [],
        "chain": chain,
        "future_topk_contract": topk_contract,
        "failure_modes": failure_modes,
        "runtime_boundary": boundary,
        "approval_gates": approval_gates,
        "next_allowed_work": "explicit production-source approval required before real draft-head logits/top-k construction",
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
    except (OSError, KeyError, TypeError, ValueError, DraftHeadTopKReadinessError) as exc:
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
