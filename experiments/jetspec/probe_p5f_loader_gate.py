#!/usr/bin/env python3
"""Probe the P5F/P5A live loader gate with a metadata-only JetSpec GGUF.

This is a fail-closed runtime loader probe, not JetSpec draft execution. It writes
the approved zero-tensor metadata preview, optionally invokes a built llama-cli,
and requires both loader gates to fail before any graph/runtime path can exist:

1. default load fails as preview_not_allowed;
2. LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1 plus LLAMA_JETSPEC_EXPERIMENTAL=1 load still fails as unsupported_runtime.

This metadata-only probe remains blocked before model-only binding: no target/draft
llama_context pair exists and no draft head, tree, verify, or rollback runtime
executes. The separate 91-tensor model-only binding path remains non-drafting.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import Any

import convert_jetspec_head_to_gguf as gguf_writer

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_PLAN = HERE / "conversion_plans/JetSpec_jetspec-Qwen3.6-35B-A3B_gguf_plan.json"
DEFAULT_LLAMA_CLI = REPO_ROOT / "build-rocm-qwen35-dev/bin/llama-cli"

FORBIDDEN_RUNTIME_TOKENS = [
    "draft-jetspec accepted as P5F",
    "no draft tokens will be generated",
    "build_arch_graph",
    "tree_accept",
]


class ProbeError(RuntimeError):
    """Raised for malformed probe setup."""


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_preview(plan_path: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    plan = _load_json(plan_path)
    return gguf_writer.write_gguf(output, plan, force=True)


def _run_case(
    name: str,
    llama_cli: pathlib.Path,
    preview_path: pathlib.Path,
    expected_token: str,
    *,
    allow_preview: bool,
    experimental: bool,
    timeout_s: float,
) -> dict[str, Any]:
    env = os.environ.copy()
    if allow_preview:
        env["LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD"] = "1"
    else:
        env.pop("LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD", None)
    if experimental:
        env["LLAMA_JETSPEC_EXPERIMENTAL"] = "1"
    else:
        env.pop("LLAMA_JETSPEC_EXPERIMENTAL", None)

    cmd = [str(llama_cli), "-m", str(preview_path), "-p", "jetspec-loader-gate", "-n", "1"]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "name": name,
            "ok": False,
            "timed_out": True,
            "returncode": None,
            "expected_token": expected_token,
            "token_found": expected_token in stdout or expected_token in stderr,
            "forbidden_runtime_hits": [],
            "elapsed_s": round(time.perf_counter() - started, 4),
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }

    combined = proc.stdout + proc.stderr
    forbidden_hits = [token for token in FORBIDDEN_RUNTIME_TOKENS if token in combined]
    token_found = expected_token in combined
    ok = proc.returncode != 0 and token_found and not forbidden_hits
    return {
        "name": name,
        "ok": ok,
        "timed_out": timed_out,
        "returncode": proc.returncode,
        "expected_token": expected_token,
        "token_found": token_found,
        "forbidden_runtime_hits": forbidden_hits,
        "elapsed_s": round(time.perf_counter() - started, 4),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _source_runtime_supported_true_rejected() -> dict[str, Any]:
    source_path = REPO_ROOT / "src/models/jetspec_qwen3_draft_head.cpp"
    text = source_path.read_text(encoding="utf-8", errors="replace")
    reject_token = "jetspec_expect(!meta.runtime_supported"
    message_token = "jetspec.experimental.runtime_supported must remain false until JetSpec draft-head graph execution is implemented"
    optional_gate = "if (!meta.runtime_supported)"
    reject_pos = text.find(reject_token)
    optional_pos = text.find(optional_gate)
    ok = reject_pos >= 0 and message_token in text and optional_pos >= 0 and reject_pos < optional_pos
    return {
        "name": "runtime_supported_true_source_rejected",
        "ok": ok,
        "source": str(source_path.relative_to(REPO_ROOT)),
        "reject_token_found": reject_pos >= 0,
        "message_token_found": message_token in text,
        "reject_before_optional_load_gate": reject_pos >= 0 and optional_pos >= 0 and reject_pos < optional_pos,
    }


def probe_loader_gate(
    *,
    llama_cli: pathlib.Path = DEFAULT_LLAMA_CLI,
    plan_path: pathlib.Path = DEFAULT_PLAN,
    run_binary: bool = True,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    errors: list[str] = []
    cases: list[dict[str, Any]] = []

    if not plan_path.exists():
        errors.append(f"missing conversion plan: {plan_path}")
        return {"ok": False, "status": "loader_gate_setup_failed", "errors": errors, "cases": cases}

    if run_binary and not llama_cli.exists():
        errors.append(f"missing llama-cli binary: {llama_cli}")
        return {"ok": False, "status": "loader_gate_setup_failed", "errors": errors, "cases": cases}

    with tempfile.TemporaryDirectory(prefix="jetspec-p5f-loader-gate-") as tmp_s:
        tmp = pathlib.Path(tmp_s)
        preview_path = tmp / "JetSpec.metadata-only.gguf"
        try:
            preview = _write_preview(plan_path, preview_path)
        except Exception as exc:  # noqa: BLE001 - report contract failure as data
            errors.append(f"failed to write metadata-only preview: {exc}")
            return {"ok": False, "status": "loader_gate_setup_failed", "errors": errors, "cases": cases}

        if run_binary:
            cases.append(
                _run_case(
                    "default_preview_rejected",
                    llama_cli,
                    preview_path,
                    "preview_not_allowed",
                    allow_preview=False,
                    experimental=False,
                    timeout_s=timeout_s,
                )
            )
            cases.append(
                _run_case(
                    "preview_allowed_runtime_still_unsupported",
                    llama_cli,
                    preview_path,
                    "unsupported_runtime",
                    allow_preview=True,
                    experimental=True,
                    timeout_s=timeout_s,
                )
            )
        else:
            cases.append(
                {
                    "name": "binary_probe_skipped",
                    "ok": True,
                    "returncode": None,
                    "expected_token": None,
                    "token_found": False,
                    "forbidden_runtime_hits": [],
                }
            )

    runtime_supported_true_negative = _source_runtime_supported_true_rejected()
    if not runtime_supported_true_negative.get("ok"):
        errors.append("loader source must reject runtime_supported=true before optional load gates")

    for case in cases:
        if not case.get("ok"):
            errors.append(f"loader gate case failed: {case.get('name')}")

    ok = not errors
    return {
        "ok": ok,
        "status": "loader_gate_verified_preflight_still_blocked" if ok and run_binary else "loader_gate_contract_ready_binary_not_run",
        "errors": errors,
        "runtime_executed": False,
        "draft_context_created": False,
        "p5f_preflight_executed": False,
        "preview_metadata": preview,
        "runtime_supported_true_negative": runtime_supported_true_negative,
        "cases": cases,
        "limitations": [
            "probes live loader rejection only",
            "does not instantiate target/draft llama_context pair",
            "does not execute common_speculative_jetspec_preflight",
            "does not execute draft-head graph/tree/rollback runtime",
        ],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cli", type=pathlib.Path, default=DEFAULT_LLAMA_CLI)
    parser.add_argument("--plan", type=pathlib.Path, default=DEFAULT_PLAN)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--no-binary", action="store_true", help="only check preview generation contract; do not invoke llama-cli")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = probe_loader_gate(
        llama_cli=args.llama_cli.resolve(),
        plan_path=args.plan.resolve(),
        run_binary=not args.no_binary,
        timeout_s=args.timeout,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print(
            "P5F loader-gate probe passed "
            f"status={result['status']} "
            f"runtime_executed={result['runtime_executed']} "
            f"cases={len(result['cases'])}"
        )
    else:
        print("P5F loader-gate probe failed", file=sys.stderr)
        for error in result["errors"]:
            print(f"- {error}", file=sys.stderr)
        for case in result.get("cases", []):
            if not case.get("ok"):
                print(json.dumps(case, indent=2, sort_keys=True), file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
