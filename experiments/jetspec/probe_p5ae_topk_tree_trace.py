#!/usr/bin/env python3
"""Fast no-model P5AE top-k tree ABI trace contract probe."""

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
    "draft-jetspec p5ae_topk_tree_runtime",
    "phase=tree_build_runtime_ready",
    "topk_tree_runtime_ready=1",
    "topk_logprob_source=synthetic_full_vocab_softmax",
    "topk_width=2",
    "topk_depth=1",
    "actual_tree_nodes=3",
    "tree_parent_indices=[-1,0,0]",
    "tree_depth=[0,1,1]",
    "tree_rank=[-1,0,1]",
    "tree_cum_logprob=[0.0,-0.1,-0.3]",
    "parent_before_child=1",
    "num_nodes_lte_budget=1",
    "non_root_nodes=2",
    "no_draft_head_graph=1",
    "no_draft_logits=1",
    "no_verify_mask=1",
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
    "draft_token=",
    "draft token",
    "no_draft_tokens=0",
    "no_kv_mutation=0",
    "no_publish=0",
]

SELF_TEST_TRACE = (
    "common_speculative_impl_draft_jetspec::process: draft-jetspec p5ae_topk_tree_runtime "
    "phase=tree_build_runtime_ready topk_tree_runtime_ready=1 topk_tree_runtime_hash=aaaaaaaaaaaaaaaa "
    "topk_logprob_source=synthetic_full_vocab_softmax topk_width=2 topk_depth=1 actual_tree_nodes=3 "
    "tree_token_ids=[42,43,44] tree_parent_indices=[-1,0,0] tree_depth=[0,1,1] "
    "tree_rank=[-1,0,1] tree_cum_logprob=[0.0,-0.1,-0.3] parent_before_child=1 "
    "num_nodes_lte_budget=1 non_root_nodes=2 no_draft_head_graph=1 no_draft_logits=1 "
    "no_verify_mask=1 no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 "
    "no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1"
)


def _parse_scalar(line: str, key: str) -> int | None:
    m = re.search(rf"{re.escape(key)}=(-?\d+)", line)
    return None if m is None else int(m.group(1))


def validate_trace_line(line: str) -> dict[str, Any]:
    errors: list[str] = []
    missing = [token for token in REQUIRED_TRACE_TOKENS if token not in line]
    forbidden = [token for token in FORBIDDEN_TRACE_TOKENS if token in line]
    errors.extend(f"missing P5AE trace token: {token}" for token in missing)
    errors.extend(f"forbidden P5AE trace token present: {token}" for token in forbidden)
    expected = {"topk_width": 2, "topk_depth": 1, "actual_tree_nodes": 3, "non_root_nodes": 2}
    parsed = {key: _parse_scalar(line, key) for key in expected}
    for key, value in expected.items():
        if parsed[key] != value:
            errors.append(f"P5AE trace {key} must be {value}")
    return {"ok": not errors, "errors": errors, "missing_tokens": missing, "forbidden_hits": forbidden, **parsed}


def _source_contract() -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8", errors="replace")
    errors: list[str] = []
    for token in [
        'common_speculative_env_enabled("LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY")',
        "build_topk_tree_runtime",
        "topk_tree_runtime_ready",
        "JETSPEC_SYNTHETIC_FULL_VOCAB_SOFTMAX",
        "tree_build_actual_nodes_last = JETSPEC_TOPK_ABI_NODES",
        "p5ae_topk_tree_runtime",
        "topk_abi_root_tail_conflict()",
    ]:
        if token not in text:
            errors.append(f"missing P5AE source token: {token}")
    branch_start = text.find("bool build_topk_tree_runtime()")
    branch_end = text.find("bool build_topk_verify_mask_runtime()", branch_start)
    branch = text[branch_start:branch_end]
    for token in ["llama_decode", "llama_kv_cache", "result->push_back", "tree_accept"]:
        if token in branch:
            errors.append(f"forbidden P5AE source token present: {token}")
    return {"ok": not errors, "errors": errors, "runtime_executed": False, "model_loaded": False, "draft_tokens_emitted": False}


def _trace_from_log(path: pathlib.Path) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "draft-jetspec p5ae_topk_tree_runtime" in line:
            return line
    return None


def probe_p5ae_topk_tree_trace(trace_log: pathlib.Path | None = None) -> dict[str, Any]:
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
                errors.append("trace log does not contain draft-jetspec p5ae_topk_tree_runtime")
            else:
                live_runtime = True
                live = validate_trace_line(line)
                live["line"] = line
                errors.extend(live["errors"])
    return {
        "ok": not errors,
        "status": "p5ae_topk_tree_trace_contract_verified" if not errors else "p5ae_topk_tree_trace_contract_invalid",
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
            "synthetic top-k children are ABI sentinels, not draft-head logits",
            "does not execute draft-head graph, verify, accept, commit, KV mutation, publish, or draft tokens",
        ],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-log", type=pathlib.Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = probe_p5ae_topk_tree_trace(args.trace_log)
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
