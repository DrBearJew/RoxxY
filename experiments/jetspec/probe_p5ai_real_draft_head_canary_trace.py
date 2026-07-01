#!/usr/bin/env python3
"""Fast no-model P5AI real draft-head graph canary trace contract probe."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SPEC_SOURCE = REPO_ROOT / "common/speculative.cpp"
MODEL_SOURCE = REPO_ROOT / "src/models/jetspec_qwen3_draft_head.cpp"
SERVER_SOURCE = REPO_ROOT / "tools/server/server-context.cpp"

REQUIRED_TRACE_TOKENS = [
    "draft-jetspec p5ai_real_draft_head_canary",
    "phase=real_draft_head_canary_ready",
    "canary_enabled=1",
    "ctx_dft_present=1",
    "decode_rc=0",
    "input_width=10240",
    "output_width=2048",
    "actual_draft_head_logits_rows=0",
    "actual_topk_rows=0",
    "real_draft_head_tensors_bound=1",
    "hidden_taps_source=target_tap_capture",
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
    "decode_rc=1",
    "ctx_dft_present=0",
    "actual_draft_head_logits_rows=1",
    "actual_topk_rows=1",
    "no_accept=0",
    "no_token_commit=0",
    "no_kv_mutation=0",
    "no_draft_tokens=0",
    "#gen drafts = 1",
    "#gen tokens = 1",
]

SELF_TEST_TRACE = (
    "common_speculative_impl_draft_jetspec::process: draft-jetspec p5ai_real_draft_head_canary "
    "phase=real_draft_head_canary_ready canary_enabled=1 ctx_dft_present=1 decode_rc=0 "
    "input_rows=1 input_width=10240 output_rows=1 output_width=2048 graph_hash=4684ed9f8e47d33a "
    "actual_draft_head_graph_rows=1 actual_draft_head_logits_rows=0 actual_topk_rows=0 "
    "real_draft_head_tensors_bound=1 hidden_taps_source=target_tap_capture no_accept=1 "
    "no_token_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 "
    "no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1"
)


def _parse_scalar(line: str, key: str) -> int | None:
    m = re.search(rf"{re.escape(key)}=(-?\d+)", line)
    return None if m is None else int(m.group(1))


def validate_trace_line(line: str) -> dict[str, Any]:
    errors: list[str] = []
    missing = [token for token in REQUIRED_TRACE_TOKENS if token not in line]
    forbidden = [token for token in FORBIDDEN_TRACE_TOKENS if token in line]
    errors.extend(f"missing P5AI trace token: {token}" for token in missing)
    errors.extend(f"forbidden P5AI trace token present: {token}" for token in forbidden)

    expected_exact = {
        "ctx_dft_present": 1,
        "decode_rc": 0,
        "input_width": 10240,
        "output_width": 2048,
        "actual_draft_head_logits_rows": 0,
        "actual_topk_rows": 0,
    }
    parsed = {key: _parse_scalar(line, key) for key in [
        "ctx_dft_present",
        "decode_rc",
        "input_rows",
        "input_width",
        "output_rows",
        "output_width",
        "actual_draft_head_graph_rows",
        "actual_draft_head_logits_rows",
        "actual_topk_rows",
    ]}
    for key, value in expected_exact.items():
        if parsed[key] != value:
            errors.append(f"P5AI trace {key} must be {value}")
    if parsed["input_rows"] is None or parsed["input_rows"] <= 0:
        errors.append("P5AI trace input_rows must be > 0")
    if parsed["output_rows"] != parsed["input_rows"]:
        errors.append("P5AI trace output_rows must equal input_rows")
    if parsed["actual_draft_head_graph_rows"] != parsed["input_rows"]:
        errors.append("P5AI trace actual_draft_head_graph_rows must equal input_rows")
    return {"ok": not errors, "errors": errors, "missing_tokens": missing, "forbidden_hits": forbidden, **parsed}


def _source_contract() -> dict[str, Any]:
    errors: list[str] = []
    spec = SPEC_SOURCE.read_text(encoding="utf-8", errors="replace")
    model = MODEL_SOURCE.read_text(encoding="utf-8", errors="replace")
    server = SERVER_SOURCE.read_text(encoding="utf-8", errors="replace")

    for token in [
        'common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_CANARY")',
        "common_speculative_jetspec_real_draft_head_canary_eval",
        "run_real_draft_head_canary_decode",
        "p5ai_real_draft_head_canary",
        "actual_draft_head_logits_rows=%d",
        "no_draft_tokens=1",
    ]:
        if token not in spec:
            errors.append(f"missing P5AI speculative source token: {token}")

    for token in [
        "LLAMA_JETSPEC_REAL_DRAFT_HEAD_CANARY",
        "loaded JetSpec draft-head canary context",
        "llama_set_embeddings(ctx_dft.get(), true)",
        "draft token emission remains disabled",
    ]:
        if token not in server:
            errors.append(f"missing P5AI server source token: {token}")

    for token in [
        "LLM_TENSOR_JETSPEC_FC",
        "draft.fc",
        "JETSPEC_QWEN36_TARGET_TAP_WIDTH",
        "n_deepstack_layers",
        "jetspec_fc_canary",
        "llama_model_graph_build_forward_expand",
        "No LM head logits, sampling, accept, token commit, KV mutation, publish, or",
    ]:
        if token not in model:
            errors.append(f"missing P5AI model source token: {token}")

    impl_start = spec.find("struct common_speculative_impl_draft_jetspec")
    impl_end = spec.find("struct common_speculative_impl_draft_mtp", impl_start)
    impl = spec[impl_start:impl_end] if impl_start >= 0 and impl_end > impl_start else ""
    if not impl:
        errors.append("cannot isolate draft-jetspec implementation slice")
    for token in ["result->push_back", "dp.result->push_back", "common_sampler_sample", "llama_kv_cache"]:
        if token in impl:
            errors.append(f"forbidden P5AI implementation token present: {token}")

    return {"ok": not errors, "errors": errors, "runtime_executed": False, "model_loaded": False, "draft_tokens_emitted": False}


def _trace_from_log(path: pathlib.Path) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "draft-jetspec p5ai_real_draft_head_canary" in line:
            return line
    return None


def probe_p5ai_real_draft_head_canary_trace(trace_log: pathlib.Path | None = None) -> dict[str, Any]:
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
                errors.append("trace log does not contain draft-jetspec p5ai_real_draft_head_canary")
            else:
                live_runtime = True
                live = validate_trace_line(line)
                live["line"] = line
                errors.extend(live["errors"])
    return {
        "ok": not errors,
        "status": "p5ai_real_draft_head_canary_trace_contract_verified" if not errors else "p5ai_real_draft_head_canary_trace_contract_invalid",
        "errors": errors,
        "runtime_executed": live_runtime,
        "model_loaded": live_runtime,
        "context_created": live_runtime,
        "draft_tokens_emitted": False,
        "source_contract": source,
        "self_test_trace": self_test,
        "live_trace": live,
        "limitations": [
            "default path validates source plus trace parser contract only",
            "P5AI executes draft.fc projection/embedding canary, not full-vocab logits/top-k",
            "does not accept, commit tokens, mutate KV, publish visible state, or emit draft tokens",
        ],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-log", type=pathlib.Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = probe_p5ai_real_draft_head_canary_trace(args.trace_log)
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
