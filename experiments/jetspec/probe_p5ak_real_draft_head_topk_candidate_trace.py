#!/usr/bin/env python3
"""Fast no-model P5AK real draft-head top-k candidate ABI trace contract probe."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SOURCE = REPO_ROOT / "common/speculative.cpp"

REQUIRED_TRACE_TOKENS = [
    "draft-jetspec p5ak_real_draft_head_topk_candidate_runtime",
    "phase=real_draft_head_topk_candidate_ready",
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
    "parent_node=0",
    "candidate_nodes=2",
    "rank_semantics=rank_stable_descending_logit",
    "accept_path_len=0",
    "actual_accepted_nodes=0",
    "correction_token_present=0",
    "no_external_logits_walk=1",
    "no_target_accept_walk=1",
    "no_accept=1",
    "no_token_commit=1",
    "no_hidden_kv_commit=1",
    "no_rejected_branch_discard=1",
    "no_publish=1",
    "no_visible_state_change=1",
    "no_kv_mutation=1",
    "no_draft_tokens=1",
]

FORBIDDEN_TRACE_TOKENS = [
    "target_logits",
    "sampler",
    "synthetic_full_vocab_softmax",
    "topk_only_renormalization",
    "ctx_dft_present=0",
    "decode_rc=1",
    "logits_rows=0",
    "actual_verified_logits_rows=0",
    "topk_k=0",
    "candidate_nodes=0",
    "no_accept=0",
    "no_token_commit=0",
    "no_kv_mutation=0",
    "no_draft_tokens=0",
    "actual_committed_tokens=1",
    "actual_publish_visible_state=1",
    "#gen drafts = 1",
    "#gen tokens = 1",
]

SELF_TEST_TRACE = (
    "common_speculative_impl_draft_jetspec::process: draft-jetspec p5ak_real_draft_head_topk_candidate_runtime "
    "phase=real_draft_head_topk_candidate_ready real_topk_candidate_runtime_ready=1 "
    "real_topk_candidate_runtime_hash=dddddddddddddddd topk_accept_boundary_runtime_ready=1 "
    "topk_accept_boundary_runtime_hash=cccccccccccccccc topk_verify_mask_runtime_ready=1 "
    "topk_verify_mask_runtime_hash=bbbbbbbbbbbbbbbb topk_tree_runtime_ready=1 "
    "topk_tree_runtime_hash=aaaaaaaaaaaaaaaa logits_source=draft_head_full_vocab_logits "
    "ctx_dft_present=1 decode_rc=0 logits_rows=1 logits_width=248320 actual_verified_logits_rows=1 "
    "topk_k=2 parent_node=0 candidate_nodes=2 candidate_ids=[46746,128519] "
    "candidate_logits=[5.72088,5.48563] rank_semantics=rank_stable_descending_logit "
    "accept_path_len=0 actual_accepted_nodes=0 correction_token_present=0 no_external_logits_walk=1 "
    "no_target_accept_walk=1 no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 "
    "no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1"
)


def _parse_scalar(line: str, key: str) -> int | None:
    m = re.search(rf"{re.escape(key)}=(-?\d+)", line)
    return None if m is None else int(m.group(1))


def _parse_candidate_ids(line: str) -> list[int] | None:
    m = re.search(r"candidate_ids=\[(-?\d+),(-?\d+)\]", line)
    if m is None:
        return None
    return [int(m.group(1)), int(m.group(2))]


def _parse_candidate_logits(line: str) -> list[float] | None:
    m = re.search(r"candidate_logits=\[([-+0-9.eE]+),([-+0-9.eE]+)\]", line)
    if m is None:
        return None
    return [float(m.group(1)), float(m.group(2))]


def validate_trace_line(line: str) -> dict[str, Any]:
    errors: list[str] = []
    missing = [token for token in REQUIRED_TRACE_TOKENS if token not in line]
    forbidden = [token for token in FORBIDDEN_TRACE_TOKENS if token in line]
    errors.extend(f"missing P5AK trace token: {token}" for token in missing)
    errors.extend(f"forbidden P5AK trace token present: {token}" for token in forbidden)

    parsed = {key: _parse_scalar(line, key) for key in [
        "ctx_dft_present",
        "decode_rc",
        "logits_rows",
        "logits_width",
        "actual_verified_logits_rows",
        "topk_k",
        "parent_node",
        "candidate_nodes",
        "accept_path_len",
        "actual_accepted_nodes",
        "correction_token_present",
    ]}
    expected_exact = {
        "ctx_dft_present": 1,
        "decode_rc": 0,
        "logits_width": 248320,
        "topk_k": 2,
        "parent_node": 0,
        "candidate_nodes": 2,
        "accept_path_len": 0,
        "actual_accepted_nodes": 0,
        "correction_token_present": 0,
    }
    for key, value in expected_exact.items():
        if parsed[key] != value:
            errors.append(f"P5AK trace {key} must be {value}")
    if parsed["logits_rows"] is None or parsed["logits_rows"] <= 0:
        errors.append("P5AK trace logits_rows must be > 0")
    if parsed["actual_verified_logits_rows"] != parsed["logits_rows"]:
        errors.append("P5AK trace actual_verified_logits_rows must equal logits_rows")

    candidate_ids = _parse_candidate_ids(line)
    if candidate_ids is None:
        errors.append("P5AK trace candidate_ids must be present")
    elif len(candidate_ids) != 2 or candidate_ids[0] < 0 or candidate_ids[1] < 0 or candidate_ids[0] == candidate_ids[1]:
        errors.append("P5AK trace candidate_ids must contain two distinct non-negative ids")
    candidate_logits = _parse_candidate_logits(line)
    if candidate_logits is None:
        errors.append("P5AK trace candidate_logits must be present")
    elif candidate_logits[0] < candidate_logits[1]:
        errors.append("P5AK trace candidate_logits must be descending by rank")

    return {
        "ok": not errors,
        "errors": errors,
        "missing_tokens": missing,
        "forbidden_hits": forbidden,
        "candidate_ids": candidate_ids,
        "candidate_logits": candidate_logits,
        **parsed,
    }


def _source_contract() -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8", errors="replace")
    errors: list[str] = []
    for token in [
        'common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY")',
        "JETSPEC_REAL_DRAFT_HEAD_TOPK_CANDIDATE_RUNTIME_PHASE",
        "JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE",
        "JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS",
        "real_draft_head_topk_candidate_ready",
        "invalid_real_draft_head_topk_candidate_runtime",
        "p5ak_real_draft_head_topk_candidate_enabled",
        "real_draft_head_topk_candidate_runtime_ready",
        "build_real_draft_head_topk_candidate_runtime",
        "!p5aj_real_draft_head_logits_canary_enabled",
        "!p5ag_topk_accept_boundary_enabled",
        "real_draft_head_logits_canary_ready",
        "topk_accept_boundary_runtime_ready",
        "real_draft_head_topk_candidate_ids[0] = real_draft_head_canary_top1_id_last",
        "real_draft_head_topk_candidate_ids[1] = real_draft_head_canary_top2_id_last",
        "p5ak_real_draft_head_topk_candidate_runtime",
        "logits_source=%s",
        "candidate_ids=[%d,%d]",
        "rank_semantics=%s",
        "no_draft_tokens=1",
    ]:
        if token not in text:
            errors.append(f"missing P5AK source token: {token}")

    branch_start = text.find("bool build_real_draft_head_topk_candidate_runtime()")
    branch_end = text.find("bool build_pre_round_snapshot", branch_start)
    branch = text[branch_start:branch_end] if branch_start >= 0 and branch_end > branch_start else ""
    if not branch:
        errors.append("cannot isolate P5AK real top-k candidate builder")
    for token in ["llama_decode", "llama_kv_cache", "result->push_back", "tree_accept", "common_sampler_sample", "llama_sampler"]:
        if token in branch:
            errors.append(f"forbidden P5AK builder token present: {token}")
    return {"ok": not errors, "errors": errors, "runtime_executed": False, "model_loaded": False, "draft_tokens_emitted": False}


def _trace_from_log(path: pathlib.Path) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "draft-jetspec p5ak_real_draft_head_topk_candidate_runtime" in line:
            return line
    return None


def probe_p5ak_real_draft_head_topk_candidate_trace(trace_log: pathlib.Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    source = _source_contract()
    errors.extend(source["errors"])
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
                errors.append("trace log does not contain draft-jetspec p5ak_real_draft_head_topk_candidate_runtime")
            else:
                live_runtime = True
                live = validate_trace_line(line)
                live["line"] = line
                errors.extend(live["errors"])
    return {
        "ok": not errors,
        "status": "p5ak_real_draft_head_topk_candidate_trace_contract_verified" if not errors else "p5ak_real_draft_head_topk_candidate_trace_contract_invalid",
        "errors": errors,
        "runtime_executed": live_runtime,
        "model_loaded": live_runtime,
        "context_created": live_runtime,
        "draft_tokens_emitted": False,
        "source_contract": source,
        "self_test_trace": self_test,
        "live_trace": live,
        "limitations": [
            "default path is no-model and validates source plus trace parser contract only",
            "P5AK records top-k candidate metadata from P5AJ logits but does not replace the synthetic tree yet",
            "does not accept, commit tokens, mutate KV, publish visible state, or emit draft tokens",
        ],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-log", type=pathlib.Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = probe_p5ak_real_draft_head_topk_candidate_trace(args.trace_log)
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
