#!/usr/bin/env python3
"""Validate a live packed16/I32 MTP DP16 route-smoke artifact.

This is intentionally log/artifact based: it verifies the product route contract
without treating legacy `rocm_q8k_dot4_kq` labels as proof of a q8_0 K cache.
"""

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BAD_LOG_MARKERS = [
    "GGML_ASSERT",
    "ABORT",
    "ROCm error",
    "missing_packed16_k_sidecar",
    "bad_packed16_k_sidecar_shape",
    "bad FA V layout",
    "required rocm_fa2_packed16_dot4_decode route was not selected",
]

DP16_PLAN_RE = re.compile(
    r"dp16_fa_plan[^\n]*"
    r"status=selected[^\n]*"
    r"inst=mtp_draft_decode[^\n]*"
    r"backend=fa2_packed16_dot4_decode[^\n]*"
    r"route=rocm_fa2_packed16_dot4_decode[^\n]*"
    r"k_type=i32[^\n]*"
    r"v_type=q4_0[^\n]*"
    r"k_layout=packed16_i32_scaled[^\n]*"
    r"k_repr=packed16_i32_persistent[^\n]*"
    r"graph_key=0x[0-9a-fA-F]+"
)

FINAL_SELECT_RE = re.compile(
    r"fa_final_select:[^\n]*"
    r"inst=mtp_draft_decode_qk[^\n]*"
    r"selected=rocm_q8k_dot4_kq[^\n]*"
    r"K=i32[^\n]*V=q4_0"
)

DOT4_LAUNCH_RE = re.compile(
    r"fa_dot4_launch:[^\n]*"
    r"fa_inst=3[^\n]*"
    r"K=i32[^\n]*V=q4_0"
)

K_PHYS_RE = re.compile(
    r"(k_phys=packed16_q8_sidechannel_i32_f16scales|physical=packed16_q8_sidechannel_i32_f16scales|k_layout=packed16_i32_scaled)"
)

GRAPH_REPLAY_RE = re.compile(
    r"cuda_graph_trace mode=(capture|graph_launch_after_capture|graph_launch)|graph_capture_replay|graph_launch_after_capture|graph_launch"
)

REAL_FA_OUTPUT_RE = re.compile(
    r"q8k_dot4_kq_timing[^\n]*"
    r"variant=blockfa_recthist_v4_single_auto[^\n]*"
    r"zero_ms=0\.000000[^\n]*"
    r"full_fa=1"
)

PROBE_OUTPUT_RE = re.compile(
    r"q8k_dot4_kq_timing[^\n]*"
    r"variant=packed16[^\n]*"
    r"full_fa=0"
)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostics path
        return {"_json_error": str(exc)}


def infer_capture(case_dir: Path) -> Optional[str]:
    name = case_dir.name.lower()
    if "nocapture" in name:
        return "0"
    if "capture" in name:
        return "1"
    env = load_json(case_dir / "env.json")
    if env.get("GGML_CUDA_DISABLE_GRAPHS") == "1":
        return "0"
    return None


def command_has_k_cache_flag(cmd_text: str) -> bool:
    try:
        tokens = shlex.split(cmd_text)
    except ValueError:
        tokens = cmd_text.split()
    return any(tok == "--cache-type-k" or tok.startswith("--cache-type-k") for tok in tokens)


def validate_case(case_dir: Path, expect_capture: Optional[str], route_require_mode: str, require_real_fa_output: bool) -> Tuple[bool, Dict[str, Any]]:
    failures: List[str] = []
    warnings: List[str] = []

    cmd_path = case_dir / "cmd.txt"
    env_path = case_dir / "env.json"
    summary_path = case_dir / "summary.json"
    log_path = case_dir / "server.stderr.log"

    cmd_text = cmd_path.read_text(encoding="utf-8", errors="replace") if cmd_path.exists() else ""
    env = load_json(env_path)
    summary = load_json(summary_path)
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

    if not cmd_text:
        failures.append("missing cmd.txt")
    if not env:
        failures.append("missing or empty env.json")
    if not summary:
        failures.append("missing or empty summary.json")
    if not log:
        failures.append("missing or empty server.stderr.log")

    if command_has_k_cache_flag(cmd_text):
        failures.append("cmd contains --cache-type-k / --cache-type-k*; packed16/I32 route must not use a K cache-type flag")

    bad_env_keys = [k for k in env if k.startswith("LLAMA_ARG_CACHE_TYPE_K")]
    if bad_env_keys:
        failures.append(f"env contains K cache override keys: {bad_env_keys}")

    required_route = env.get("GGML_CUDA_FA_ROUTE_REQUIRE")
    route_force_keys = sorted(
        k for k, v in env.items()
        if v is not None and (
            k == "GGML_CUDA_FA_ROUTE_REQUIRE" or
            k == "GGML_CUDA_DP16_ROUTE_REQUIRE" or
            k == "GGML_CUDA_FA_ROUTE_REQUIRE_DOT4" or
            "ROUTE_REQUIRE" in k))
    if route_require_mode == "exact" and required_route != "rocm_fa2_packed16_dot4_decode":
        failures.append(
            f"GGML_CUDA_FA_ROUTE_REQUIRE is {required_route!r}, expected 'rocm_fa2_packed16_dot4_decode'")
    elif route_require_mode == "absent" and route_force_keys:
        failures.append(
            f"route-forcing env keys present for auto-route validation: {route_force_keys}")

    if summary:
        if summary.get("status") != "ok":
            failures.append(f"summary status is {summary.get('status')!r}, expected 'ok'")
        if summary.get("server_rc") not in (0, None):
            failures.append(f"server_rc is {summary.get('server_rc')!r}, expected 0")
        if summary.get("negative_markers"):
            failures.append(f"summary negative_markers not empty: {summary.get('negative_markers')!r}")
        if require_real_fa_output:
            request_rows = summary.get("requests") if isinstance(summary.get("requests"), list) else []
            if request_rows:
                bad_requests = [r for r in request_rows if (r.get("draft_n") or 0) <= 0 or (r.get("draft_n_accepted") or 0) <= 0]
                if bad_requests:
                    failures.append(f"summary has request rows without nonzero packed16 MTP acceptance: {bad_requests!r}")
            else:
                draft_n = summary.get("draft_n") or 0
                draft_acc = summary.get("draft_n_accepted") or 0
                if draft_n <= 0:
                    failures.append(f"summary draft_n is {draft_n!r}, expected > 0 for semantic packed16 MTP validation")
                if draft_acc <= 0:
                    failures.append(f"summary draft_n_accepted is {draft_acc!r}, expected > 0 for semantic packed16 MTP validation")

    for marker in BAD_LOG_MARKERS:
        if marker in log:
            failures.append(f"bad log marker present: {marker}")

    bad_h_values = [int(v) for v in re.findall(r"bad_h=(\d+)", log)]
    nonzero_bad_h = [v for v in bad_h_values if v != 0]
    if nonzero_bad_h:
        failures.append(f"nonzero MTP_INPUT_REAL bad_h values: {nonzero_bad_h[:8]}")
    elif not bad_h_values:
        warnings.append("no MTP_INPUT_REAL bad_h values found")

    plan_matches = DP16_PLAN_RE.findall(log)
    if not plan_matches:
        failures.append("missing canonical dp16_fa_plan for fa2_packed16_dot4_decode with graph_key and packed16_i32_scaled K")

    expected = expect_capture if expect_capture is not None else infer_capture(case_dir)
    if expected is not None and plan_matches:
        if not any(f"capture={expected}" in line for line in plan_matches):
            failures.append(f"no canonical dp16_fa_plan line has capture={expected}")

    graph_keys = sorted(set(re.findall(r"graph_key=(0x[0-9a-fA-F]+)", "\n".join(plan_matches))))
    if not graph_keys:
        failures.append("missing graph_key on packed16 dp16_fa_plan")

    if not FINAL_SELECT_RE.search(log):
        failures.append("missing draft decode final selection evidence: inst=mtp_draft_decode_qk selected=rocm_q8k_dot4_kq K=i32 V=q4_0")

    if not DOT4_LAUNCH_RE.search(log):
        failures.append("missing DOT4 launch evidence for draft decode: fa_inst=3 K=i32 V=q4_0")

    if not K_PHYS_RE.search(log):
        failures.append("missing packed16 physical-K evidence: k_phys/physical sidechannel marker or packed16_i32_scaled layout")

    if require_real_fa_output:
        if PROBE_OUTPUT_RE.search(log):
            failures.append("packed16 MTP draft decode used probe output path: q8k_dot4_kq_timing variant=packed16 full_fa=0")
        if not REAL_FA_OUTPUT_RE.search(log):
            failures.append("missing real FA output evidence: q8k_dot4_kq_timing variant=blockfa_recthist_v4_single_auto zero_ms=0 full_fa=1")

    if expected == "1" and not GRAPH_REPLAY_RE.search(log):
        failures.append("capture row missing graph capture/replay evidence")

    report = {
        "case_dir": str(case_dir),
        "ok": not failures,
        "expect_capture": expected,
        "graph_keys": graph_keys,
        "plan_count": len(plan_matches),
        "bad_h_count": len(bad_h_values),
        "bad_h_nonzero_count": len(nonzero_bad_h),
        "route_require_mode": route_require_mode,
        "route_require": required_route,
        "route_force_keys": route_force_keys,
        "require_real_fa_output": require_real_fa_output,
        "warnings": warnings,
        "failures": failures,
    }
    return not failures, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dirs", nargs="+", help="Artifact case directories to validate")
    parser.add_argument("--expect-capture", choices=["0", "1", "auto"], default="auto")
    parser.add_argument(
        "--route-require-mode",
        choices=["exact", "absent", "any"],
        default="exact",
        help="Validate GGML_CUDA_FA_ROUTE_REQUIRE provenance: exact packed16 route, absent for auto-route, or any.")
    parser.add_argument(
        "--require-real-fa-output",
        action="store_true",
        help="Also require semantic packed16 MTP draft output: nonzero acceptance and q8k real FULL_FA timing evidence.")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    reports = []
    all_ok = True
    for case in args.case_dirs:
        expect = None if args.expect_capture == "auto" else args.expect_capture
        ok, report = validate_case(Path(case), expect, args.route_require_mode, args.require_real_fa_output)
        reports.append(report)
        all_ok = all_ok and ok

    payload = {"ok": all_ok, "cases": reports}
    text = json.dumps(payload, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
