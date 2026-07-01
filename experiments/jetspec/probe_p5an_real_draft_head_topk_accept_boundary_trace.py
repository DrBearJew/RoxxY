#!/usr/bin/env python3
"""Fast no-model P5AN real draft-head top-k accept-boundary ABI trace contract probe."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DOC = REPO_ROOT / "docs/speculative.md"
CANDIDATE = HERE / "jetspec_p5an_real_draft_head_topk_accept_boundary_runtime_candidate.md"

REQUIRED_TRACE_TOKENS = [
    "draft-jetspec p5an_real_draft_head_topk_accept_boundary_runtime",
    "phase=real_draft_head_topk_accept_boundary_ready",
    "real_topk_accept_boundary_runtime_ready=1",
    "real_topk_verify_mask_runtime_ready=1",
    "real_topk_tree_runtime_ready=1",
    "real_topk_candidate_runtime_ready=1",
    "topk_accept_boundary_runtime_ready=1",
    "topk_verify_mask_runtime_ready=1",
    "topk_tree_runtime_ready=1",
    "logits_source=draft_head_full_vocab_logits",
    "ctx_dft_present=1",
    "decode_rc=0",
    "logits_rows=1",
    "logits_width=248320",
    "actual_verified_logits_rows=1",
    "topk_k=2",
    "actual_tree_nodes=3",
    "real_tree_token_ids=[",
    "real_tree_parent_indices=[-1,0,0]",
    "real_tree_depth=[0,1,1]",
    "real_tree_rank=[-1,0,1]",
    "real_tree_logits=[",
    "parent_node=0",
    "candidate_nodes=2",
    "candidate_ids=[",
    "candidate_logits=[",
    "rank_semantics=rank_stable_descending_logit",
    "actual_verify_mask_entries=5",
    "verify_mask_rows=3",
    "verify_mask_cols=3",
    "allowed_edges=[0:0,1:0,1:1,2:0,2:2]",
    "real_verify_mask_rows=[0,1,1,2,2]",
    "real_verify_mask_cols=[0,0,1,0,2]",
    "real_verify_mask_values=[1,1,1,1,1]",
    "accept_boundary_candidate_nodes=2",
    "accept_boundary_verified_edges=5",
    "accept_decision_source=none_no_target_logits",
    "accept_path_len=0",
    "actual_accepted_nodes=0",
    "correction_token_present=0",
    "actual_committed_tokens=0",
    "actual_survivor_pages_committed=0",
    "actual_pages_discarded=0",
    "actual_publish_visible_state=0",
    "no_target_logits_walk=1",
    "no_target_accept_walk=1",
    "no_accept=1",
    "no_mask_tensor=1",
    "no_token_commit=1",
    "no_hidden_kv_commit=1",
    "no_rejected_branch_discard=1",
    "no_publish=1",
    "no_visible_state_change=1",
    "no_kv_mutation=1",
    "no_draft_tokens=1",
]

FORBIDDEN_TRACE_TOKENS = [
    " target_logits ",
    " sampler ",
    "tree_accept",
    "topk_only_renormalization",
    "real_topk_accept_boundary_runtime_ready=0",
    "real_topk_verify_mask_runtime_ready=0",
    "real_topk_tree_runtime_ready=0",
    "real_topk_candidate_runtime_ready=0",
    "actual_verified_logits_rows=0",
    "actual_tree_nodes=0",
    "actual_verify_mask_entries=0",
    "accept_boundary_candidate_nodes=0",
    "accept_boundary_verified_edges=0",
    "accept_path_len=1",
    "actual_accepted_nodes=1",
    "correction_token_present=1",
    "actual_committed_tokens=1",
    "actual_survivor_pages_committed=1",
    "actual_pages_discarded=1",
    "actual_publish_visible_state=1",
    "no_target_logits_walk=0",
    "no_target_accept_walk=0",
    "no_accept=0",
    "no_mask_tensor=0",
    "no_token_commit=0",
    "no_kv_mutation=0",
    "no_draft_tokens=0",
    "actual_committed_tokens=1",
    "actual_publish_visible_state=1",
    "#gen drafts = 1",
    "#gen tokens = 1",
]

SELF_TEST_TRACE = (
    "common_speculative_impl_draft_jetspec::process: draft-jetspec p5an_real_draft_head_topk_accept_boundary_runtime "
    "phase=real_draft_head_topk_accept_boundary_ready real_topk_accept_boundary_runtime_ready=1 "
    "real_topk_accept_boundary_runtime_hash=9999999999999999 real_topk_verify_mask_runtime_ready=1 "
    "real_topk_verify_mask_runtime_hash=ffffffffffffffff real_topk_tree_runtime_ready=1 "
    "real_topk_tree_runtime_hash=eeeeeeeeeeeeeeee real_topk_candidate_runtime_ready=1 "
    "real_topk_candidate_runtime_hash=dddddddddddddddd topk_accept_boundary_runtime_ready=1 "
    "topk_accept_boundary_runtime_hash=cccccccccccccccc topk_verify_mask_runtime_ready=1 "
    "topk_verify_mask_runtime_hash=bbbbbbbbbbbbbbbb topk_tree_runtime_ready=1 "
    "topk_tree_runtime_hash=aaaaaaaaaaaaaaaa logits_source=draft_head_full_vocab_logits "
    "ctx_dft_present=1 decode_rc=0 logits_rows=1 logits_width=248320 actual_verified_logits_rows=1 "
    "topk_k=2 actual_tree_nodes=3 real_tree_token_ids=[13,76279,46746] "
    "real_tree_parent_indices=[-1,0,0] real_tree_depth=[0,1,1] real_tree_rank=[-1,0,1] "
    "real_tree_logits=[0,6.25,5.75] parent_node=0 candidate_nodes=2 "
    "candidate_ids=[76279,46746] candidate_logits=[6.25,5.75] "
    "rank_semantics=rank_stable_descending_logit actual_verify_mask_entries=5 verify_mask_rows=3 verify_mask_cols=3 "
    "allowed_edges=[0:0,1:0,1:1,2:0,2:2] real_verify_mask_rows=[0,1,1,2,2] "
    "real_verify_mask_cols=[0,0,1,0,2] real_verify_mask_values=[1,1,1,1,1] "
    "accept_boundary_candidate_nodes=2 accept_boundary_verified_edges=5 accept_decision_source=none_no_target_logits "
    "accept_path_len=0 actual_accepted_nodes=0 correction_token_present=0 "
    "actual_committed_tokens=0 actual_survivor_pages_committed=0 actual_pages_discarded=0 actual_publish_visible_state=0 "
    "no_target_logits_walk=1 no_target_accept_walk=1 no_accept=1 no_mask_tensor=1 no_token_commit=1 "
    "no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 "
    "no_kv_mutation=1 no_draft_tokens=1"
)


def _parse_scalar(line: str, key: str) -> int | None:
    m = re.search(rf"{re.escape(key)}=(-?\d+)", line)
    return None if m is None else int(m.group(1))


def _parse_int_list(line: str, key: str) -> list[int] | None:
    m = re.search(rf"{re.escape(key)}=\[([^\]]+)\]", line)
    if m is None:
        return None
    try:
        return [int(part) for part in m.group(1).split(",")]
    except ValueError:
        return None


def _parse_float_list(line: str, key: str) -> list[float] | None:
    m = re.search(rf"{re.escape(key)}=\[([^\]]+)\]", line)
    if m is None:
        return None
    try:
        return [float(part) for part in m.group(1).split(",")]
    except ValueError:
        return None


def validate_trace_line(line: str) -> dict[str, Any]:
    errors: list[str] = []
    missing = [token for token in REQUIRED_TRACE_TOKENS if token not in line]
    forbidden = [token for token in FORBIDDEN_TRACE_TOKENS if token in line]
    errors.extend(f"missing P5AN trace token: {token}" for token in missing)
    errors.extend(f"forbidden P5AN trace token present: {token}" for token in forbidden)

    parsed = {key: _parse_scalar(line, key) for key in [
        "ctx_dft_present", "decode_rc", "logits_rows", "logits_width",
        "actual_verified_logits_rows", "topk_k", "actual_tree_nodes", "parent_node",
        "candidate_nodes", "actual_verify_mask_entries", "accept_boundary_candidate_nodes",
        "accept_boundary_verified_edges", "accept_path_len", "actual_accepted_nodes",
        "correction_token_present", "actual_committed_tokens", "actual_survivor_pages_committed",
        "actual_pages_discarded", "actual_publish_visible_state",
    ]}
    expected_exact = {
        "ctx_dft_present": 1,
        "decode_rc": 0,
        "logits_rows": 1,
        "logits_width": 248320,
        "actual_verified_logits_rows": 1,
        "topk_k": 2,
        "actual_tree_nodes": 3,
        "parent_node": 0,
        "candidate_nodes": 2,
        "actual_verify_mask_entries": 5,
        "accept_boundary_candidate_nodes": 2,
        "accept_boundary_verified_edges": 5,
        "accept_path_len": 0,
        "actual_accepted_nodes": 0,
        "correction_token_present": 0,
        "actual_committed_tokens": 0,
        "actual_survivor_pages_committed": 0,
        "actual_pages_discarded": 0,
        "actual_publish_visible_state": 0,
    }
    for key, value in expected_exact.items():
        if parsed[key] != value:
            errors.append(f"P5AN trace {key} must be {value}")

    candidate_ids = _parse_int_list(line, "candidate_ids")
    tree_ids = _parse_int_list(line, "real_tree_token_ids")
    parent_indices = _parse_int_list(line, "real_tree_parent_indices")
    tree_depth = _parse_int_list(line, "real_tree_depth")
    tree_rank = _parse_int_list(line, "real_tree_rank")
    candidate_logits = _parse_float_list(line, "candidate_logits")
    real_tree_logits = _parse_float_list(line, "real_tree_logits")
    rows = _parse_int_list(line, "real_verify_mask_rows")
    cols = _parse_int_list(line, "real_verify_mask_cols")
    values = _parse_int_list(line, "real_verify_mask_values")
    if candidate_ids is None or len(candidate_ids) != 2 or candidate_ids[0] < 0 or candidate_ids[1] < 0 or candidate_ids[0] == candidate_ids[1]:
        errors.append("P5AN trace candidate_ids must contain two distinct non-negative ids")
    if tree_ids is None or len(tree_ids) != 3:
        errors.append("P5AN trace real_tree_token_ids must contain root plus two candidates")
    elif candidate_ids is not None and tree_ids[1:] != candidate_ids:
        errors.append("P5AN trace real_tree_token_ids[1:] must equal P5AK candidate_ids")
    if parent_indices != [-1, 0, 0]:
        errors.append("P5AN trace real_tree_parent_indices must be [-1,0,0]")
    if tree_depth != [0, 1, 1]:
        errors.append("P5AN trace real_tree_depth must be [0,1,1]")
    if tree_rank != [-1, 0, 1]:
        errors.append("P5AN trace real_tree_rank must be [-1,0,1]")
    if candidate_logits is None or len(candidate_logits) != 2 or candidate_logits[0] < candidate_logits[1]:
        errors.append("P5AN trace candidate_logits must be descending rank logits")
    if real_tree_logits is None or len(real_tree_logits) != 3:
        errors.append("P5AN trace real_tree_logits must include root plus candidate logits")
    elif candidate_logits is not None and real_tree_logits[1:] != candidate_logits:
        errors.append("P5AN trace real_tree_logits[1:] must equal candidate_logits")
    if rows != [0, 1, 1, 2, 2]:
        errors.append("P5AN real_verify_mask_rows must be [0,1,1,2,2]")
    if cols != [0, 0, 1, 0, 2]:
        errors.append("P5AN real_verify_mask_cols must be [0,0,1,0,2]")
    if values != [1, 1, 1, 1, 1]:
        errors.append("P5AN real_verify_mask_values must be all ones")

    return {
        "ok": not errors,
        "errors": errors,
        "missing_tokens": missing,
        "forbidden_hits": forbidden,
        "candidate_ids": candidate_ids,
        "candidate_logits": candidate_logits,
        "real_tree_token_ids": tree_ids,
        "real_tree_logits": real_tree_logits,
        "real_tree_parent_indices": parent_indices,
        "real_tree_depth": tree_depth,
        "real_tree_rank": tree_rank,
        "real_verify_mask_rows": rows,
        "real_verify_mask_cols": cols,
        "real_verify_mask_values": values,
        **parsed,
    }


def _contract_documents() -> dict[str, Any]:
    errors: list[str] = []
    for path in [DOC, CANDIDATE]:
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        for token in [
            "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_BOUNDARY_ABI_ONLY=1",
            "P5AN real draft-head top-k accept-boundary ABI",
            "p5an_real_draft_head_topk_accept_boundary_runtime",
            "phase=real_draft_head_topk_accept_boundary_ready",
            "actual_verified_logits_rows=1",
            "allowed_edges=[0:0,1:0,1:1,2:0,2:2]",
            "accept_path_len=0",
            "actual_accepted_nodes=0",
            "correction_token_present=0",
            "no target logits walk",
            "no target accept walk",
            "no token commit",
            "no KV mutation",
            "no publish",
            "no draft tokens",
        ]:
            if token not in text:
                errors.append(f"{path.name} missing P5AN contract token: {token}")
    return {"ok": not errors, "errors": errors, "runtime_executed": False, "model_loaded": False, "draft_tokens_emitted": False}


def _trace_from_log(path: pathlib.Path) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "draft-jetspec p5an_real_draft_head_topk_accept_boundary_runtime" in line:
            return line
    return None


def probe_p5an_real_draft_head_topk_accept_boundary_trace(trace_log: pathlib.Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    docs = _contract_documents()
    errors.extend(docs["errors"])
    self_test = validate_trace_line(SELF_TEST_TRACE)
    errors.extend(f"self-test trace invalid: {err}" for err in self_test["errors"])
    live = None
    live_runtime = False
    if trace_log is not None:
        if not trace_log.exists():
            errors.append(f"missing trace log: {trace_log}")
        else:
            line = _trace_from_log(trace_log)
            if line is None:
                errors.append("trace log does not contain draft-jetspec p5an_real_draft_head_topk_accept_boundary_runtime")
            else:
                live_runtime = True
                live = validate_trace_line(line)
                live["line"] = line
                errors.extend(live["errors"])
    return {
        "ok": not errors,
        "status": "p5an_real_draft_head_topk_accept_boundary_trace_contract_verified" if not errors else "p5an_real_draft_head_topk_accept_boundary_trace_contract_invalid",
        "errors": errors,
        "runtime_executed": live_runtime,
        "model_loaded": live_runtime,
        "context_created": live_runtime,
        "draft_tokens_emitted": False,
        "contract_documents": docs,
        "self_test_trace": self_test,
        "live_trace": live,
        "limitations": [
            "default path is no-model and validates docs plus trace parser contract only",
            "P5AN binds P5AM shadow real-tree verify-mask metadata to an accept boundary without source/runtime edits in this lane",
            "does not walk target logits, accept, commit tokens, mutate KV, publish visible state, or emit draft tokens",
        ],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-log", type=pathlib.Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = probe_p5an_real_draft_head_topk_accept_boundary_trace(args.trace_log)
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
