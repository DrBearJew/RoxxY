#!/usr/bin/env python3
"""Run every inert JetSpec staging contract check.

This script is intentionally staged under experiments/jetspec/ and is not wired
into CMake or llama-server. It runs stdlib-only contract tests, fixture validators,
py_compile, and a CMake isolation scan so future promotion work has one local gate.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

UNIT_TESTS = [
    "test_tree_semantics.py",
    "test_tree_verify_mask.py",
    "test_jetspec_round_contract.py",
    "test_runtime_contract.py",
    "test_promotion_checklist.py",
    "test_p5_plan.py",
    "test_p5a_loader_candidate.py",
    "test_p5b_target_taps.py",
    "test_p5c_speculative_type.py",
    "test_p5d_target_tap_ingestion.py",
    "test_p5e_runtime_state.py",
    "test_p5f_binding_preflight.py",
    "test_p5f_artifact_binding.py",
    "test_p5f_loader_gate_probe.py",
    "test_p5g_tree_runtime_readiness.py",
    "test_p5h_kv_commit_readiness.py",
    "test_p5i_tree_runtime_approval_packet.py",
    "test_p5j_kv_primitive_audit.py",
    "test_p5k_kv_ownership_primitive_design.py",
    "test_p5l_page_map_ownership_oracle.py",
    "test_p5m_transaction_plan_oracle.py",
    "test_p5n_transaction_scaffold.py",
    "test_p5o_pre_round_snapshot.py",
    "test_p5p_transient_reservation_descriptor.py",
    "test_p5q_tree_build_descriptor.py",
    "test_draft_head_loader_prototype.py",
    "test_bf16_payload_parity.py",
    "test_gguf_preview.py",
    "test_target_hidden_taps.py",
    "test_committed_hidden_cache.py",
]

CONTRACT_COMMANDS = [
    ("gguf conversion plan", ["plan_gguf_conversion.py"]),
    ("gguf writer inspect summary", ["convert_jetspec_head_to_gguf.py", "--json"]),
    ("gguf writer self-test", ["convert_jetspec_head_to_gguf.py", "--self-test"]),
    ("gguf preview parser self-test", ["parse_gguf_preview.py", "--self-test"]),
    ("standalone loader prototype self-test", ["draft_head_loader_prototype.py", "--self-test"]),
    ("BF16 payload parity fixture", ["bf16_payload_parity.py", "--fixture", "fixtures/bf16_payload_parity_smoke.json"]),
    ("BF16 payload parity self-test", ["bf16_payload_parity.py", "--self-test"]),
    ("target-hidden tap summary", ["target_hidden_taps.py", "--json"]),
    ("target-hidden tap parity fixture", ["target_hidden_taps.py", "--fixture", "fixtures/target_hidden_tap_parity_smoke.json", "--json"]),
    ("committed hidden-cache fixture", ["committed_hidden_cache.py", "--fixture", "fixtures/committed_hidden_cache_smoke.json"]),
    ("tree verify-mask fixture", ["tree_verify_mask.py", "--fixture", "fixtures/tree_verify_mask_smoke.json"]),
    ("single-round fixture", ["jetspec_round_contract.py", "--fixture", "fixtures/jetspec_round_smoke.json"]),
    ("tree verify rollback parity fixture", ["jetspec_round_contract.py", "--fixture", "fixtures/jetspec_round_parity_smoke.json"]),
    ("runtime header validator", ["validate_runtime_contract.py"]),
    ("P5 default-off plan validator", ["validate_p5_plan.py"]),
    ("P5A loader candidate validator", ["validate_p5a_loader_candidate.py"]),
    ("P5B target hidden taps validator", ["validate_p5b_target_taps.py"]),
    ("P5C speculative type validator", ["validate_p5c_speculative_type.py"]),
    ("P5D target tap ingestion validator", ["validate_p5d_target_tap_ingestion.py"]),
    ("P5E runtime state validator", ["validate_p5e_runtime_state.py"]),
    ("P5F binding preflight validator", ["validate_p5f_binding_preflight.py"]),
    ("P5F artifact binding validator", ["validate_p5f_artifact_binding.py"]),
    ("P5F loader gate probe", ["probe_p5f_loader_gate.py"]),
    ("P5G tree-runtime readiness fixture", ["jetspec_tree_runtime_readiness.py", "--fixture", "fixtures/jetspec_tree_runtime_readiness_smoke.json"]),
    ("P5G tree-runtime readiness validator", ["validate_p5g_tree_runtime_readiness.py"]),
    ("P5H KV commit readiness fixture", ["jetspec_kv_commit_readiness.py", "--fixture", "fixtures/jetspec_kv_commit_readiness_smoke.json"]),
    ("P5H KV commit readiness validator", ["validate_p5h_kv_commit_readiness.py"]),
    ("P5I tree-runtime approval matrix fixture", ["tree_runtime_approval_matrix.py", "--fixture", "fixtures/tree_runtime_approval_matrix_smoke.json"]),
    ("P5I tree-runtime approval packet validator", ["validate_p5i_tree_runtime_approval_packet.py"]),
    ("P5J KV primitive audit fixture", ["kv_primitive_audit.py", "--fixture", "fixtures/kv_primitive_audit_smoke.json"]),
    ("P5J KV primitive audit validator", ["validate_p5j_kv_primitive_audit.py"]),
    ("P5K KV ownership primitive design fixture", ["kv_ownership_primitive_design.py", "--fixture", "fixtures/kv_ownership_primitive_design_smoke.json"]),
    ("P5K KV ownership primitive design validator", ["validate_p5k_kv_ownership_primitive_design.py"]),
    ("P5L page-map ownership oracle fixture", ["page_map_ownership_oracle.py", "--fixture", "fixtures/page_map_ownership_oracle_smoke.json"]),
    ("P5L page-map ownership oracle validator", ["validate_p5l_page_map_ownership_oracle.py"]),
    ("P5M transaction plan oracle fixture", ["transaction_plan_oracle.py", "--fixture", "fixtures/transaction_plan_oracle_smoke.json"]),
    ("P5M transaction plan oracle validator", ["validate_p5m_transaction_plan_oracle.py"]),
    ("P5N transaction scaffold validator", ["validate_p5n_transaction_scaffold.py"]),
    ("P5O pre-round snapshot validator", ["validate_p5o_pre_round_snapshot.py"]),
    ("P5P transient reservation descriptor validator", ["validate_p5p_transient_reservation_descriptor.py"]),
    ("P5Q tree build descriptor validator", ["validate_p5q_tree_build_descriptor.py"]),
    ("promotion checklist validator", ["validate_promotion_checklist.py"]),
]

CMAKE_FORBIDDEN_TOKENS = [
    "experiments/jetspec",
    "run_all_jetspec_contracts",
    "jetspec_runtime_contract",
    "jetspec_runtime_state_contract",
    "validate_runtime_contract",
    "test_runtime_contract",
    "jetspec_promotion_checklist",
    "validate_promotion_checklist",
    "test_promotion_checklist",
    "jetspec_p5_default_off_plan",
    "jetspec_p5a_loader_candidate",
    "validate_p5_plan",
    "test_p5_plan",
    "validate_p5a_loader_candidate",
    "test_p5a_loader_candidate",
    "validate_p5b_target_taps",
    "test_p5b_target_taps",
    "jetspec_p5c_speculative_type",
    "validate_p5c_speculative_type",
    "test_p5c_speculative_type",
    "jetspec_p5d_target_tap_ingestion",
    "validate_p5d_target_tap_ingestion",
    "test_p5d_target_tap_ingestion",
    "jetspec_p5e_runtime_state",
    "validate_p5e_runtime_state",
    "test_p5e_runtime_state",
    "jetspec_p5f_binding_preflight",
    "validate_p5f_binding_preflight",
    "test_p5f_binding_preflight",
    "validate_p5f_artifact_binding",
    "test_p5f_artifact_binding",
    "probe_p5f_loader_gate",
    "test_p5f_loader_gate_probe",
    "jetspec_tree_runtime_readiness",
    "validate_p5g_tree_runtime_readiness",
    "test_p5g_tree_runtime_readiness",
    "tree_runtime_readiness_verified_not_executed",
    "jetspec_kv_commit_readiness",
    "validate_p5h_kv_commit_readiness",
    "test_p5h_kv_commit_readiness",
    "kv_commit_readiness_verified_not_executed",
    "tree_runtime_approval_matrix",
    "validate_p5i_tree_runtime_approval_packet",
    "test_p5i_tree_runtime_approval_packet",
    "tree_runtime_approval_packet_verified_not_executed",
    "kv_primitive_audit",
    "validate_p5j_kv_primitive_audit",
    "test_p5j_kv_primitive_audit",
    "kv_primitive_audit_verified_not_executed",
    "kv_ownership_primitive_design",
    "validate_p5k_kv_ownership_primitive_design",
    "test_p5k_kv_ownership_primitive_design",
    "kv_ownership_primitive_design_verified_not_executed",
    "page_map_ownership_oracle",
    "validate_p5l_page_map_ownership_oracle",
    "test_p5l_page_map_ownership_oracle",
    "page_map_ownership_oracle_verified_not_executed",
    "transaction_plan_oracle",
    "validate_p5m_transaction_plan_oracle",
    "test_p5m_transaction_plan_oracle",
    "transaction_plan_oracle_verified_not_executed",
    "jetspec_p5n_transaction_scaffold",
    "validate_p5n_transaction_scaffold",
    "test_p5n_transaction_scaffold",
    "JETSPEC_TRANSACTION_PHASE_ORDER",
    "transaction_plan_scaffold_ready",
    "invalid_transaction_plan",
    "jetspec_p5o_pre_round_snapshot",
    "validate_p5o_pre_round_snapshot",
    "test_p5o_pre_round_snapshot",
    "JETSPEC_PRE_ROUND_SNAPSHOT_PHASE",
    "pre_round_snapshot_ready",
    "invalid_pre_round_snapshot",
    "jetspec_p5p_transient_reservation_descriptor",
    "validate_p5p_transient_reservation_descriptor",
    "test_p5p_transient_reservation_descriptor",
    "JETSPEC_TRANSIENT_RESERVATION_PHASE",
    "transient_reservation_descriptor_ready",
    "invalid_transient_reservation_descriptor",
    "jetspec_p5q_tree_build_descriptor",
    "validate_p5q_tree_build_descriptor",
    "test_p5q_tree_build_descriptor",
    "JETSPEC_TREE_BUILD_PHASE",
    "tree_build_descriptor_ready",
    "invalid_tree_build_descriptor",
    "draft_head_loader_prototype",
    "test_draft_head_loader_prototype",
    "bf16_payload_parity",
    "test_bf16_payload_parity",
    "jetspec_round_contract",
    "jetspec_round_parity",
    "tree_verify_mask",
    "jetspec_tree_verify_mask",
    "committed_hidden_cache",
    "jetspec_hidden_cache_contract",
    "target_hidden_taps",
    "jetspec_target_hidden_taps",
    "target_hidden_tap_parity",
    "convert_jetspec_head_to_gguf",
    "parse_gguf_preview",
    "plan_gguf_conversion",
    "tree_semantics",
    "inspect_hf_head",
    "jetspec_loader_contract",
]


@dataclasses.dataclass
class StepResult:
    name: str
    ok: bool
    elapsed_s: float
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "elapsed_s": round(self.elapsed_s, 4),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "detail": self.detail,
        }


def _run_subprocess(name: str, args: Sequence[str]) -> StepResult:
    start = time.perf_counter()
    cmd = [sys.executable, *args]
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, check=False)
    return StepResult(
        name=name,
        ok=proc.returncode == 0,
        elapsed_s=time.perf_counter() - start,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _cmake_files(repo_root: pathlib.Path) -> list[pathlib.Path]:
    files = list(repo_root.rglob("CMakeLists.txt"))
    files.extend(repo_root.rglob("*.cmake"))
    return sorted(path for path in files if path.is_file())


def _check_cmake_isolation() -> StepResult:
    start = time.perf_counter()
    hits: list[dict[str, Any]] = []
    files = _cmake_files(REPO_ROOT)
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            hits.append({"path": str(path), "error": str(exc)})
            continue
        rel = str(path.relative_to(REPO_ROOT))
        for lineno, line in enumerate(text.splitlines(), start=1):
            matched = [token for token in CMAKE_FORBIDDEN_TOKENS if token in line]
            if matched:
                hits.append({"path": rel, "line": lineno, "tokens": matched, "text": line.strip()})
    return StepResult(
        name="CMake isolation scan",
        ok=not hits,
        elapsed_s=time.perf_counter() - start,
        detail={"cmake_files_scanned": len(files), "hits": hits},
        stderr="" if not hits else json.dumps(hits, indent=2, sort_keys=True),
    )


def _py_compile_step() -> StepResult:
    py_files = sorted(path.name for path in HERE.glob("*.py"))
    return _run_subprocess("py_compile staged Python", ["-m", "py_compile", *py_files])


def _all_steps() -> list[tuple[str, Callable[[], StepResult]]]:
    steps: list[tuple[str, Callable[[], StepResult]]] = []
    for test_file in UNIT_TESTS:
        steps.append((f"unit {test_file}", lambda test_file=test_file: _run_subprocess(f"unit {test_file}", [test_file])))
    for name, args in CONTRACT_COMMANDS:
        steps.append((name, lambda name=name, args=args: _run_subprocess(name, args)))
    steps.append(("py_compile staged Python", _py_compile_step))
    steps.append(("CMake isolation scan", _check_cmake_isolation))
    return steps


def _print_step(result: StepResult, *, verbose: bool) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status} {result.name} ({result.elapsed_s:.3f}s)")
    if verbose or not result.ok:
        if result.stdout:
            print("stdout:")
            print(result.stdout.rstrip())
        if result.stderr:
            print("stderr:")
            print(result.stderr.rstrip())
        if result.detail and (verbose or not result.ok):
            print("detail:")
            print(json.dumps(result.detail, indent=2, sort_keys=True))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail-fast", action="store_true", help="stop at the first failing step")
    parser.add_argument("--json", action="store_true", help="print machine-readable summary")
    parser.add_argument("--verbose", action="store_true", help="print stdout/stderr for passing steps too")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    results: list[StepResult] = []
    total_start = time.perf_counter()

    for _name, run_step in _all_steps():
        result = run_step()
        results.append(result)
        if not args.json:
            _print_step(result, verbose=args.verbose)
        if args.fail_fast and not result.ok:
            break

    ok = all(result.ok for result in results)
    summary = {
        "ok": ok,
        "steps_run": len(results),
        "steps_total": len(_all_steps()),
        "failed": [result.name for result in results if not result.ok],
        "elapsed_s": round(time.perf_counter() - total_start, 4),
        "results": [result.as_dict() for result in results],
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"SUMMARY {'PASS' if ok else 'FAIL'} {summary['steps_run']}/{summary['steps_total']} steps ({summary['elapsed_s']:.3f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
