#!/usr/bin/env python3
"""Fast no-model P5AJ real draft-head logits/top-k canary trace contract probe."""

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
    "draft-jetspec p5aj_real_draft_head_logits_canary",
    "phase=real_draft_head_logits_canary_ready",
    "logits_canary_enabled=1",
    "ctx_dft_present=1",
    "decode_rc=0",
    "input_width=10240",
    "logits_rows=1",
    "logits_width=248320",
    "topk_rows=1",
    "topk_k=2",
    "logits_extract_source=private_embeddings_output",
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
    "logits_rows=0",
    "topk_rows=0",
    "topk_k=0",
    "top1_id=-1",
    "top2_id=-1",
    "no_accept=0",
    "no_token_commit=0",
    "no_kv_mutation=0",
    "no_draft_tokens=0",
    "#gen drafts = 1",
    "#gen tokens = 1",
]

SELF_TEST_TRACE = (
    "common_speculative_impl_draft_jetspec::process: draft-jetspec p5aj_real_draft_head_logits_canary "
    "phase=real_draft_head_logits_canary_ready logits_canary_enabled=1 ctx_dft_present=1 decode_rc=0 "
    "input_rows=1 input_width=10240 logits_rows=1 logits_width=248320 logits_hash=ec60e99edc5f3a7f "
    "topk_rows=1 topk_k=2 top1_id=46746 top2_id=128519 top1_logit=5.72088 top2_logit=5.48563 "
    "logits_extract_source=private_embeddings_output real_draft_head_tensors_bound=1 "
    "hidden_taps_source=target_tap_capture no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 "
    "no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1"
)


def _parse_scalar(line: str, key: str) -> int | None:
    m = re.search(rf"{re.escape(key)}=(-?\d+)", line)
    return None if m is None else int(m.group(1))


def validate_trace_line(line: str) -> dict[str, Any]:
    errors: list[str] = []
    missing = [token for token in REQUIRED_TRACE_TOKENS if token not in line]
    forbidden = [token for token in FORBIDDEN_TRACE_TOKENS if token in line]
    errors.extend(f"missing P5AJ trace token: {token}" for token in missing)
    errors.extend(f"forbidden P5AJ trace token present: {token}" for token in forbidden)

    parsed = {key: _parse_scalar(line, key) for key in [
        "ctx_dft_present",
        "decode_rc",
        "input_rows",
        "input_width",
        "logits_rows",
        "logits_width",
        "topk_rows",
        "topk_k",
        "top1_id",
        "top2_id",
    ]}
    expected_exact = {
        "ctx_dft_present": 1,
        "decode_rc": 0,
        "input_width": 10240,
        "logits_width": 248320,
        "topk_k": 2,
    }
    for key, value in expected_exact.items():
        if parsed[key] != value:
            errors.append(f"P5AJ trace {key} must be {value}")
    if parsed["input_rows"] is None or parsed["input_rows"] <= 0:
        errors.append("P5AJ trace input_rows must be > 0")
    if parsed["logits_rows"] != parsed["input_rows"]:
        errors.append("P5AJ trace logits_rows must equal input_rows")
    if parsed["topk_rows"] != parsed["input_rows"]:
        errors.append("P5AJ trace topk_rows must equal input_rows")
    if parsed["top1_id"] is None or parsed["top1_id"] < 0:
        errors.append("P5AJ trace top1_id must be non-negative")
    if parsed["top2_id"] is None or parsed["top2_id"] < 0:
        errors.append("P5AJ trace top2_id must be non-negative")
    return {"ok": not errors, "errors": errors, "missing_tokens": missing, "forbidden_hits": forbidden, **parsed}


def _source_contract() -> dict[str, Any]:
    errors: list[str] = []
    spec = SPEC_SOURCE.read_text(encoding="utf-8", errors="replace")
    model = MODEL_SOURCE.read_text(encoding="utf-8", errors="replace")
    server = SERVER_SOURCE.read_text(encoding="utf-8", errors="replace")

    for token in [
        'common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY")',
        "p5aj_real_draft_head_logits_canary",
        "real_draft_head_logits_canary_ready",
        "logits_extract_source=private_embeddings_output",
        "real_draft_head_canary_top1_id_last",
        "real_draft_head_canary_top2_id_last",
        "no_draft_tokens=1",
    ]:
        if token not in spec:
            errors.append(f"missing P5AJ speculative source token: {token}")

    for token in [
        "LLAMA_JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY",
        "loaded JetSpec draft-head logits canary context",
        "draft token emission remains disabled",
    ]:
        if token not in server:
            errors.append(f"missing P5AJ server source token: {token}")

    for token in [
        "LLAMA_JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY",
        "hparams.n_embd_out_impl = meta.vocab_size",
        "jetspec_logits_canary",
        "build_lora_mm(model.output, cur)",
        "res->t_logits = cur",
        "private embeddings buffer",
    ]:
        if token not in model:
            errors.append(f"missing P5AJ model source token: {token}")

    impl_start = spec.find("struct common_speculative_impl_draft_jetspec")
    impl_end = spec.find("struct common_speculative_impl_draft_mtp", impl_start)
    impl = spec[impl_start:impl_end] if impl_start >= 0 and impl_end > impl_start else ""
    if not impl:
        errors.append("cannot isolate draft-jetspec implementation slice")
    for token in ["result->push_back", "dp.result->push_back", "common_sampler_sample", "llama_kv_cache"]:
        if token in impl:
            errors.append(f"forbidden P5AJ implementation token present: {token}")

    return {"ok": not errors, "errors": errors, "runtime_executed": False, "model_loaded": False, "draft_tokens_emitted": False}


def _trace_from_log(path: pathlib.Path) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "draft-jetspec p5aj_real_draft_head_logits_canary" in line:
            return line
    return None


def probe_p5aj_real_draft_head_logits_canary_trace(trace_log: pathlib.Path | None = None) -> dict[str, Any]:
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
                errors.append("trace log does not contain draft-jetspec p5aj_real_draft_head_logits_canary")
            else:
                live_runtime = True
                live = validate_trace_line(line)
                live["line"] = line
                errors.extend(live["errors"])
    return {
        "ok": not errors,
        "status": "p5aj_real_draft_head_logits_canary_trace_contract_verified" if not errors else "p5aj_real_draft_head_logits_canary_trace_contract_invalid",
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
            "P5AJ computes draft.fc plus shared target output.weight; BF16 norm/full draft stack still out of scope",
            "does not accept, commit tokens, mutate KV, publish visible state, or emit draft tokens",
        ],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-log", type=pathlib.Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    out = probe_p5aj_real_draft_head_logits_canary_trace(args.trace_log)
    if args.json or not out["ok"]:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out["status"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
