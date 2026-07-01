#!/usr/bin/env python3
"""Fast no-model P5AD root-ready trace contract probe."""

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
    "draft-jetspec p5ad_root_publish_gate_noop_runtime",
    "phase=publish_gate_runtime_ready",
    "root_publish_gate_noop_runtime_ready=1",
    "root_rejected_branch_discard_noop_runtime_ready=1",
    "root_hidden_kv_commit_noop_runtime_ready=1",
    "root_token_commit_noop_runtime_ready=1",
    "root_runtime_ready_for_real_test=1",
    "actual_committed_tokens=0",
    "actual_survivor_pages_committed=0",
    "actual_pages_discarded=0",
    "rejected_branch_pages_reachable_after_discard=0",
    "actual_publish_visible_state=0",
    "publish_after_commit_and_discard_only=1",
    "no_real_token_commit=1",
    "no_real_hidden_kv_commit=1",
    "no_real_rejected_branch_discard=1",
    "no_real_publish=1",
    "no_visible_state_change=1",
    "no_kv_mutation=1",
    "no_draft_head_graph=1",
    "no_draft_tokens=1",
]

FORBIDDEN_TRACE_TOKENS = [
    "actual_committed_tokens=1",
    "actual_survivor_pages_committed=1",
    "actual_pages_discarded=1",
    "actual_publish_visible_state=1",
    "draft token",
]

SELF_TEST_TRACE = (
    "common_speculative_impl_draft_jetspec::process: draft-jetspec p5ad_root_publish_gate_noop_runtime "
    "phase=publish_gate_runtime_ready root_publish_gate_noop_runtime_ready=1 "
    "root_publish_gate_noop_runtime_hash=9999999999999999 root_rejected_branch_discard_noop_runtime_ready=1 "
    "root_rejected_branch_discard_noop_runtime_hash=8888888888888888 root_hidden_kv_commit_noop_runtime_ready=1 "
    "root_hidden_kv_commit_noop_runtime_hash=7777777777777777 root_token_commit_noop_runtime_ready=1 "
    "root_token_commit_noop_runtime_hash=6666666666666666 root_runtime_ready_for_real_test=1 "
    "actual_committed_tokens=0 actual_survivor_pages_committed=0 actual_pages_discarded=0 "
    "rejected_branch_pages_reachable_after_discard=0 actual_publish_visible_state=0 "
    "publish_after_commit_and_discard_only=1 no_real_token_commit=1 no_real_hidden_kv_commit=1 "
    "no_real_rejected_branch_discard=1 no_real_publish=1 no_visible_state_change=1 "
    "no_kv_mutation=1 no_draft_head_graph=1 no_draft_tokens=1"
)


def _parse_scalar(line: str, key: str) -> int | None:
    m = re.search(rf"{re.escape(key)}=(-?\d+)", line)
    return None if m is None else int(m.group(1))


def validate_trace_line(line: str) -> dict[str, Any]:
    errors: list[str] = []
    missing = [token for token in REQUIRED_TRACE_TOKENS if token not in line]
    forbidden = [token for token in FORBIDDEN_TRACE_TOKENS if token in line]
    errors.extend(f"missing P5AD trace token: {token}" for token in missing)
    errors.extend(f"forbidden P5AD trace token present: {token}" for token in forbidden)
    expected = {
        "root_runtime_ready_for_real_test": 1,
        "actual_committed_tokens": 0,
        "actual_survivor_pages_committed": 0,
        "actual_pages_discarded": 0,
        "rejected_branch_pages_reachable_after_discard": 0,
        "actual_publish_visible_state": 0,
    }
    parsed = {key: _parse_scalar(line, key) for key in expected}
    for key, value in expected.items():
        if parsed[key] != value:
            errors.append(f"P5AD trace {key} must be {value}")
    return {"ok": not errors, "errors": errors, "missing_tokens": missing, "forbidden_hits": forbidden, **parsed}


def _source_contract() -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8", errors="replace")
    errors: list[str] = []
    for token in [
        'common_speculative_env_enabled("LLAMA_JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_ONLY")',
        'common_speculative_env_enabled("LLAMA_JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_ONLY")',
        'common_speculative_env_enabled("LLAMA_JETSPEC_ROOT_PUBLISH_GATE_NOOP_ONLY")',
        "build_root_hidden_kv_commit_noop_runtime",
        "build_root_rejected_branch_discard_noop_runtime",
        "build_root_publish_gate_noop_runtime",
        "root_runtime_ready_for_real_test_last = 1",
        "draft-jetspec p5ad_root_publish_gate_noop_runtime",
    ]:
        if token not in text:
            errors.append(f"missing P5AD source token: {token}")
    for token in ["llama_kv_cache", "seq_cp", "seq_rm", "seq_import_physical", "result->push_back"]:
        if token in text[text.find("struct common_speculative_impl_draft_jetspec"):text.find("struct common_speculative_impl_draft_mtp")]:
            errors.append(f"forbidden runtime primitive token present: {token}")
    return {"ok": not errors, "errors": errors, "runtime_executed": False, "model_loaded": False, "draft_tokens_emitted": False}


def _trace_from_log(path: pathlib.Path) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "draft-jetspec p5ad_root_publish_gate_noop_runtime" in line:
            return line
    return None


def probe_p5ad_root_ready_trace(trace_log: pathlib.Path | None = None) -> dict[str, Any]:
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
                errors.append("trace log does not contain draft-jetspec p5ad_root_publish_gate_noop_runtime")
            else:
                live_runtime = True
                live = validate_trace_line(line)
                live["line"] = line
                errors.extend(live["errors"])
    return {
        "ok": not errors,
        "status": "p5ad_root_ready_trace_contract_verified" if not errors else "p5ad_root_ready_trace_contract_invalid",
        "errors": errors,
        "runtime_executed": live_runtime,
        "model_loaded": live_runtime,
        "context_created": live_runtime,
        "draft_tokens_emitted": False,
        "source_contract": source,
        "self_test_trace": self_test,
        "live_trace": live,
        "ready_to_start_real_root_only_test": not errors,
        "limitations": [
            "default path is no-model and validates source plus trace parser contract only",
            "pass --trace-log to validate a separately captured live target+draft trace",
            "does not execute draft-head graph, real token commit, KV mutation, publish, or draft tokens",
        ],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-log", type=pathlib.Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = probe_p5ad_root_ready_trace(args.trace_log)
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
