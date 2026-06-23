#!/usr/bin/env python3
"""AMD/RDNA3 INT8 route helper for llama.cpp MTP QKV/FFN work.

This is intentionally project-specific. It complements AMD's official Matrix
Instruction Calculator by parsing llama.cpp route logs and estimating the shapes
that AMD's static instruction calculator cannot know about.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

QK8 = 32
Q8_0_BYTES_PER_ELEM = (32 + 2) / 32.0   # block_q8_0: qs[32] + half scale
Q8_1_BYTES_PER_ELEM = (32 + 4) / 32.0   # block_q8_1: qs[32] + half2 scale/sum
F32_BYTES_PER_ELEM = 4
PACKED16_COMPILE_MAX_N = 16
PACKED16_PROD_MAX_N = 8
PACKED16_REUSE_N_COMPILE_MAX_N = 8
PACKED16_REUSE_N_ROUTE_MAX_N = 4
PACKED16_REUSE_N_ROWS_PER_BLOCK = 4
PACKED16_LDS_M2N_ROWS_TILE = 2
PACKED16_LDS_M2N_QB_TILE = 32
PACKED16_LDS_M2N_ROUTE_MAX_N = 4
PACKED16_LDS_M2N_FORCE_MAX_N = 5
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
DEFAULT_AMD_CALC = Path("/home/mrtrent/.harness/research/amd_matrix_instruction_calculator/matrix_calculator.py")
DEFAULT_WMMA_INSTRUCTION = "v_wmma_i32_16x16x16_iu8"

KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<val>[^\s]+)")
ROUTE_LINE_MARKERS = ("mtp_weight_route", "mtp_weight_timing", "DP16 accept", "DP16 reject", "dp16_fa_plan")
NEGATIVE_MARKERS = (
    "ROCm error: operation not permitted when stream is capturing",
    "cross_stream_unordered",
    "capture_packed_weight_unverified",
    "ngram",
    "n/gram",
)
SAFETY_FALLBACK_MARKERS = (
    "capture_cache_pointer_unstable",
)

ROUTE_GROUPS = {
    "rocm_mtp_i8_qkv_proj_dot4": "qkv",
    "rocm_mtp_i8_ffn_gate_up_dot4": "ffn_gate_up",
    "rocm_mtp_i8_ffn_down_dot4": "ffn_down",
    "rocm_mtp_i8_ffn_down_dot4_reuse_n": "ffn_down",
    "rocm_mtp_i8_ffn_down_dot4_lds_m2n": "ffn_down",
    "rocm_mtp_i8_attn_out_dot4": "attn_out",
    "rocm_mtp_i8_attn_out_dot4_reuse_n": "attn_out",
    "rocm_mtp_i8_attn_out_dot4_lds_m2n": "attn_out",
    "rocm_mtp_i8_eh_proj_dot4": "eh_proj",
    "rocm_mtp_i8_qkv_act_reuse": "qkv_act_reuse",
    "rocm_q8_dot4_mmvq": "generic_q8_dot4",
    "rocm_packed16_dot4_mmvq": "generic_packed16",
    "rocm_packed16_dot4_mmq": "fa2_pdmq",
    "rocm_fa2_packed16_dot4_decode": "fa2_packed16_dot4_decode",
    "rocm_mtp_f16k_to_packed16_dot4_decode": "fa2_f16k_adapt_dot4_decode",
    "rocm_mtp_f16k_q4v_decode": "fa1_vec_fallback_historical",
    "rocm_q8k_dot4_kq": "fa2_q8k_dot4",
    "rocm_q8k_dot4_packed16_vec": "fa_packed16_vec",
    "rocm_packed16_wmma_tile": "fa_packed16_wmma",
    "rocm_fattn_vec": "fa_vec",
}


def parse_value(v: str) -> Any:
    if v == "-":
        return v
    try:
        if v.startswith("0x"):
            return int(v, 16)
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def parse_kv(line: str) -> dict[str, Any]:
    return {m.group("key"): parse_value(m.group("val").rstrip(",")) for m in KV_RE.finditer(line)}


def route_group(route: str) -> str:
    if route in ROUTE_GROUPS:
        return ROUTE_GROUPS[route]
    if "qkv" in route:
        return "qkv"
    if "ffn_gate" in route or "gate_up" in route:
        return "ffn_gate_up"
    if "ffn_down" in route:
        return "ffn_down"
    return "other"


def summarize_routes(logs: list[Path]) -> dict[str, Any]:
    by_route: dict[str, Counter] = defaultdict(Counter)
    by_group: dict[str, Counter] = defaultdict(Counter)
    reject_reasons: Counter = Counter()
    reject_reason_by_route: dict[str, Counter] = defaultdict(Counter)
    tensors_by_route: dict[str, Counter] = defaultdict(Counter)
    ncols_by_route: dict[str, Counter] = defaultdict(Counter)
    fusion_by_route: dict[str, Counter] = defaultdict(Counter)
    selected_fusion_by_route: dict[str, Counter] = defaultdict(Counter)
    reject_fusion_by_route: dict[str, Counter] = defaultdict(Counter)
    capture_by_route: dict[str, Counter] = defaultdict(Counter)
    selected_capture_by_route: dict[str, Counter] = defaultdict(Counter)
    reject_capture_by_route: dict[str, Counter] = defaultdict(Counter)
    timing_by_route: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "last_ms_sum": 0.0, "max_ms": 0.0})
    dp16_fa_selected: Counter = Counter()
    dp16_fa_rejected: Counter = Counter()
    dp16_fa_shapes: Counter = Counter()
    dp16_fa_capture: Counter = Counter()
    dp16_fa_reasons: Counter = Counter()
    negative_hits: Counter = Counter()
    safety_fallback_hits: Counter = Counter()
    total_lines = 0
    matched_lines = 0

    for path in logs:
        for line in path.read_text(errors="replace").splitlines():
            total_lines += 1
            for marker in NEGATIVE_MARKERS:
                if marker in line:
                    negative_hits[marker] += 1
            for marker in SAFETY_FALLBACK_MARKERS:
                if marker in line:
                    safety_fallback_hits[marker] += 1
            if not any(marker in line for marker in ROUTE_LINE_MARKERS):
                continue
            kv = parse_kv(line)
            route = str(kv.get("route", "unknown"))
            status = str(kv.get("status", "accept" if "DP16 accept" in line else "reject" if "DP16 reject" in line else "unknown"))
            group = route_group(route)
            matched_lines += 1
            by_route[route][status] += 1
            by_route[route]["total"] += 1
            by_group[group][status] += 1
            by_group[group]["total"] += 1
            if status == "reject" and "reject" in kv:
                reject_reason = str(kv["reject"])
                reject_reasons[reject_reason] += 1
                reject_reason_by_route[route][reject_reason] += 1
            elif status == "reject" and "reject=" in line:
                reject_reasons["unknown"] += 1
                reject_reason_by_route[route]["unknown"] += 1
            if "tensor" in kv:
                tensors_by_route[route][str(kv["tensor"])] += 1
            if "ncols_dst" in kv:
                ncols_by_route[route][str(kv["ncols_dst"])] += 1
            if "fusion" in kv:
                fusion_key = str(kv["fusion"])
                fusion_by_route[route][fusion_key] += 1
                if status == "selected":
                    selected_fusion_by_route[route][fusion_key] += 1
                elif status == "reject":
                    reject_fusion_by_route[route][fusion_key] += 1
            if "capture" in kv:
                capture_key = str(kv["capture"])
                capture_by_route[route][capture_key] += 1
                if status == "selected":
                    selected_capture_by_route[route][capture_key] += 1
                elif status == "reject":
                    reject_capture_by_route[route][capture_key] += 1
            if "dp16_fa_plan" in line:
                backend = str(kv.get("backend", "unknown"))
                plane = str(kv.get("plane", "unknown"))
                k_repr = str(kv.get("k_repr", "unknown"))
                shape = str(kv.get("shape", "unknown"))
                vpath = str(kv.get("vpath", "unknown"))
                fa_key = (
                    f"plane={plane} backend={backend} route={route} nq={kv.get('nq', 'unknown')} "
                    f"d={kv.get('d', 'unknown')} gqa={kv.get('gqa', 'unknown')} "
                    f"k_repr={k_repr} shape={shape} vpath={vpath} qtok={kv.get('qtok_tile', 'unknown')} "
                    f"gqatile={kv.get('gqa_tile', 'unknown')} ktile={kv.get('k_tile', 'unknown')} "
                    f"experimental={kv.get('experimental', 'unknown')} fallback={kv.get('fallback', 'unknown')}"
                )
                if status == "selected":
                    dp16_fa_selected[fa_key] += 1
                    dp16_fa_shapes[fa_key] += 1
                else:
                    dp16_fa_rejected[f"backend={backend} route={route} reason={kv.get('reason', 'unknown')}"] += 1
                dp16_fa_capture[
                    f"backend={backend} capture={kv.get('capture', 'unknown')} capture_safe={kv.get('capture_safe', 'unknown')}"
                ] += 1
                dp16_fa_reasons[f"backend={backend} route={route} reason={kv.get('reason', 'unknown')}"] += 1
            if "mtp_weight_timing" in line:
                t = timing_by_route[route]
                t["count"] += 1
                if "last_ms" in kv:
                    t["last_ms_sum"] += float(kv["last_ms"])
                if "max_ms" in kv:
                    t["max_ms"] = max(t["max_ms"], float(kv["max_ms"]))

    timing = {}
    for route, t in timing_by_route.items():
        count = int(t["count"])
        timing[route] = {
            "samples": count,
            "last_ms_avg": (t["last_ms_sum"] / count) if count else None,
            "max_ms_seen": t["max_ms"],
        }

    return {
        "logs": [str(p) for p in logs],
        "total_lines": total_lines,
        "matched_route_lines": matched_lines,
        "by_route": {k: dict(v) for k, v in sorted(by_route.items())},
        "by_group": {k: dict(v) for k, v in sorted(by_group.items())},
        "reject_reasons": dict(reject_reasons.most_common()),
        "reject_reason_by_route": {k: dict(v.most_common()) for k, v in sorted(reject_reason_by_route.items())},
        "ncols_dst_by_route": {k: dict(v) for k, v in sorted(ncols_by_route.items())},
        "fusion_by_route": {k: dict(v) for k, v in sorted(fusion_by_route.items())},
        "selected_fusion_by_route": {k: dict(v) for k, v in sorted(selected_fusion_by_route.items())},
        "reject_fusion_by_route": {k: dict(v) for k, v in sorted(reject_fusion_by_route.items())},
        "capture_by_route": {k: dict(v) for k, v in sorted(capture_by_route.items())},
        "selected_capture_by_route": {k: dict(v) for k, v in sorted(selected_capture_by_route.items())},
        "reject_capture_by_route": {k: dict(v) for k, v in sorted(reject_capture_by_route.items())},
        "top_tensors_by_route": {k: dict(v.most_common(12)) for k, v in sorted(tensors_by_route.items())},
        "timing_by_route": timing,
        "dp16_fa_selected": dict(dp16_fa_selected.most_common()),
        "dp16_fa_rejected": dict(dp16_fa_rejected.most_common()),
        "dp16_fa_shapes": dict(dp16_fa_shapes.most_common()),
        "dp16_fa_capture": dict(dp16_fa_capture.most_common()),
        "dp16_fa_reasons": dict(dp16_fa_reasons.most_common()),
        "negative_markers": dict(negative_hits.most_common()),
        "safety_fallback_markers": dict(safety_fallback_hits.most_common()),
    }


def shape_cost(args: argparse.Namespace) -> dict[str, Any]:
    n = args.ncols_dst
    k = args.n_embd
    q_rows = args.q_rows
    k_rows = args.k_rows
    v_rows = args.v_rows
    n_ff = args.n_ff
    layers = args.mtp_layers

    qkv_rows = q_rows + k_rows + v_rows
    ffn_gate_up_rows = 2 * n_ff
    ffn_down_rows = k
    qkv_macs = qkv_rows * k * n * layers
    ffn_gate_up_macs = ffn_gate_up_rows * k * n * layers
    ffn_down_macs = ffn_down_rows * n_ff * n * layers
    total_macs = qkv_macs + ffn_gate_up_macs + ffn_down_macs

    act_q8_bytes_per_projection = k * n * Q8_1_BYTES_PER_ELEM
    qkv_act_quant_no_reuse = 3 * act_q8_bytes_per_projection * layers
    qkv_act_quant_reuse = act_q8_bytes_per_projection * layers
    qkv_act_quant_saved = qkv_act_quant_no_reuse - qkv_act_quant_reuse

    weight_bytes_per_n = {
        "qkv": qkv_rows * k * Q8_0_BYTES_PER_ELEM * n * layers,
        "ffn_gate_up": ffn_gate_up_rows * k * Q8_0_BYTES_PER_ELEM * n * layers,
        "ffn_down": ffn_down_rows * n_ff * Q8_0_BYTES_PER_ELEM * n * layers,
    }
    ideal_weight_bytes_if_reused_across_n = {kk: vv / max(n, 1) for kk, vv in weight_bytes_per_n.items()}

    launches = {
        "qkv_separate": 3 * layers,
        "qkv_with_graph_level_fusion_goal": 1 * layers,
        "ffn_gate_up_fused": 1 * layers,
        "ffn_down": 1 * layers,
        "qkv_ffn_current_packed_total": 5 * layers,
        "qkv_ffn_ideal_graph_level_total": 3 * layers,
    }

    return {
        "inputs": vars(args),
        "macs": {
            "qkv": qkv_macs,
            "ffn_gate_up": ffn_gate_up_macs,
            "ffn_down": ffn_down_macs,
            "total": total_macs,
        },
        "activation_quantization_bytes": {
            "qkv_no_reuse": qkv_act_quant_no_reuse,
            "qkv_with_reuse": qkv_act_quant_reuse,
            "qkv_reuse_saved": qkv_act_quant_saved,
        },
        "weight_stream_bytes_current_kernel_model": weight_bytes_per_n,
        "weight_stream_bytes_ideal_reuse_across_n": ideal_weight_bytes_if_reused_across_n,
        "launch_counts": launches,
        "notes": [
            "Weight bytes current-kernel model scales by N because current packed GEMV maps one wave row per output column.",
            "Ideal reuse-across-N is a lower bound for future tiled/WMMA-style kernels, not current behavior.",
        ],
    }


def packed16_reuse_n_cost(args: argparse.Namespace) -> dict[str, Any]:
    rows = args.rows
    k = args.K
    n = args.N
    qblocks = math.ceil(k / QK8)

    # Approximate q8 block bytes. packed16 sidecar keeps q8_0-equivalent payload
    # and one half scale per 32-value block; q8_1 activation carries payload plus
    # a half2 scale/sum. This models the repeated-weight traffic removed by one
    # wave computing all N columns for a row.
    w_payload = 32
    w_scale = 2
    a_payload = 32
    a_scale = 4

    current = rows * qblocks * n * (w_payload + w_scale + a_payload + a_scale)
    reuse_n = rows * qblocks * (w_payload + w_scale + n * (a_payload + a_scale))

    return {
        "inputs": {"rows": rows, "K": k, "N": n},
        "qblocks": qblocks,
        "bytes_per_qblock": {
            "weight_payload": w_payload,
            "weight_scale": w_scale,
            "activation_payload": a_payload,
            "activation_scale": a_scale,
        },
        "current_bytes_est": current,
        "reuse_n_bytes_est": reuse_n,
        "saved_bytes_est": current - reuse_n,
        "saved_frac": 0.0 if current == 0 else (current - reuse_n) / current,
        "current_geometry": "block=(32,N,1), one wave per (row,N-column)",
        "reuse_n_geometry": f"block=(32,{PACKED16_REUSE_N_ROWS_PER_BLOCK},1), one wave per row with acc[N] registers",
        "reuse_n_compile_max_n": PACKED16_REUSE_N_COMPILE_MAX_N,
        "reuse_n_route_max_n": PACKED16_REUSE_N_ROUTE_MAX_N,
        "valid_for_reuse_n_route": 2 <= n <= PACKED16_REUSE_N_ROUTE_MAX_N,
        "generic_reuse_n_env": os.environ.get("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ_GENERIC_REUSE_N") not in (None, "", "0"),
        "mtp_reuse_n_env": os.environ.get("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ_REUSE_N") not in (None, "", "0"),
        "lds_m2n_env": os.environ.get("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ_LDS_M2N") not in (None, "", "0"),
        "lds_m2n_n5_env": os.environ.get("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ_LDS_M2N_N5") not in (None, "", "0"),
        "lds_m2n_geometry": f"block=(32,{PACKED16_LDS_M2N_ROWS_TILE}*N,1), one wave per (row,col), QB_TILE={PACKED16_LDS_M2N_QB_TILE}",
        "lds_m2n_route_max_n": PACKED16_LDS_M2N_ROUTE_MAX_N,
        "lds_m2n_force_max_n": PACKED16_LDS_M2N_FORCE_MAX_N,
        "valid_for_lds_m2n_route": False,
        "valid_for_lds_m2n_force_n5": False,
        "notes": [
            "This estimates load traffic before cache effects; realized speedup may be lower if repeated weights hit cache.",
            "The no-LDS acc[N] reuse-N prototype keeps templates through N=8, but tight real-MTP profiling made it a production no-go; MTP route selection is capped at N<=4 and generic route exposure requires GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ_GENERIC_REUSE_N=1.",
            "The LDS M2xN replacement prototype passed route/correctness/safety but failed real-MTP performance for N=2..5; MTP ffn_down/attn_out across-N routing is killed, and these fields are retained for historical artifact parsing only.",
        ],
    }


def wide_n(args: argparse.Namespace) -> dict[str, Any]:
    k = args.K
    m = args.M
    n = args.N
    q8_blocks = math.ceil(k / QK8)
    wide_n_enabled = os.environ.get("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ_WIDE_N") not in (None, "", "0")
    runtime_max_n = PACKED16_COMPILE_MAX_N if wide_n_enabled else PACKED16_PROD_MAX_N
    return {
        "inputs": {"K": k, "M": m, "N": n},
        "compile_max_n": PACKED16_COMPILE_MAX_N,
        "prod_max_n": PACKED16_PROD_MAX_N,
        "runtime_max_n": runtime_max_n,
        "wide_n_env": wide_n_enabled,
        "valid_for_current_packed16": 1 <= n <= runtime_max_n and k % 256 == 0,
        "compiled_template_available": 1 <= n <= PACKED16_COMPILE_MAX_N and k % 256 == 0,
        "k_aligned_256": k % 256 == 0,
        "threads_per_block": 32 * n,
        "waves_per_block": n,
        "row_blocks": m,
        "q8_blocks_per_row": q8_blocks,
        "loop_iterations_per_wave": math.ceil(q8_blocks / 32),
        "occupancy_risk": "high" if n >= 16 else "medium" if n >= 8 else "low",
        "notes": [
            "Current packed16 kernel block shape is dim3(32, N, 1).",
            "N=9..16 templates are compiled for prototype validation but production selection requires GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ_WIDE_N=1.",
            "N=16 reaches 512 threads/block; validate occupancy and register pressure with profiler.",
        ],
    }


def run_amd_matrix_calculator(calculator: Path, arch: str, instruction: str) -> dict[str, Any]:
    if not calculator.exists():
        return {"available": False, "error": f"calculator not found: {calculator}"}
    cmd = [sys.executable, str(calculator), "-a", arch, "-i", instruction, "-d"]
    env = os.environ.copy()
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.returncode == 0:
        return {"available": True, "command": cmd, "returncode": 0, "stdout": proc.stdout, "stderr": proc.stderr}

    # Local research clone may not have Python dependencies installed. Inject a
    # tiny tabulate shim so the official calculator can still print facts.
    shim = r'''
import runpy, sys, types
m = types.ModuleType("tabulate")
def tabulate(rows, headers=None, **kwargs):
    out = []
    if headers:
        out.append("\t".join(map(str, headers)))
    for row in rows:
        out.append("\t".join(map(str, row if isinstance(row, (list, tuple)) else [row])))
    return "\n".join(out)
m.tabulate = tabulate
sys.modules["tabulate"] = m
sys.argv = [CALC, "-a", ARCH, "-i", INST, "-d"]
runpy.run_path(CALC, run_name="__main__")
'''.replace("CALC", repr(str(calculator))).replace("ARCH", repr(arch)).replace("INST", repr(instruction))
    proc2 = subprocess.run([sys.executable, "-c", shim], text=True, capture_output=True, env=env)
    return {"available": proc2.returncode == 0, "command": cmd, "returncode": proc2.returncode, "stdout": proc2.stdout, "stderr": proc2.stderr or proc.stderr}


def wmma(args: argparse.Namespace) -> dict[str, Any]:
    k = args.K
    m = args.M
    n = args.N
    tiles_m = math.ceil(m / WMMA_M)
    tiles_n = math.ceil(n / WMMA_N)
    tiles_k = math.ceil(k / WMMA_K)
    padded_ops = tiles_m * WMMA_M * tiles_n * WMMA_N * tiles_k * WMMA_K
    useful_ops = m * n * k
    utilization = useful_ops / padded_ops if padded_ops else 0.0
    verdict = "poor" if utilization < 0.25 else "mixed" if utilization < 0.60 else "good"
    out: dict[str, Any] = {
        "inputs": {"K": k, "M": m, "N": n, "arch": args.arch, "instruction": args.instruction},
        "wmma_tile": {"M": WMMA_M, "N": WMMA_N, "K": WMMA_K},
        "tiles": {"M": tiles_m, "N": tiles_n, "K": tiles_k, "total": tiles_m * tiles_n * tiles_k},
        "utilization": utilization,
        "verdict": verdict,
        "notes": [
            "RDNA3 IU8 WMMA is attractive only when M and N can fill a meaningful fraction of 16x16.",
            "Decode GEMV-like M=1/N<=8 shapes are usually poor WMMA fits unless QKV/FFN are batched/fused into larger tiles.",
        ],
    }
    if args.run_amd_calculator:
        out["amd_matrix_instruction_calculator"] = run_amd_matrix_calculator(args.calculator, args.arch, args.instruction)
    return out


def print_result(data: dict[str, Any], text: bool = False) -> None:
    if not text:
        print(json.dumps(data, indent=2, sort_keys=True))
        return
    # Compact text mode for the routes subcommand.
    print(json.dumps(data, indent=2, sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser(description="Project-specific AMD/RDNA3 INT8 QKV/FFN route and shape calculator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_routes = sub.add_parser("routes", help="Parse llama.cpp MTP/DP16 route logs")
    p_routes.add_argument("logs", nargs="+", type=Path)
    p_routes.add_argument("--text", action="store_true", help="reserved; JSON is still emitted for scriptability")

    p_cost = sub.add_parser("cost", help="Estimate QKV/FFN per-token costs")
    p_cost.add_argument("--n-embd", type=int, required=True)
    p_cost.add_argument("--q-rows", type=int, required=True)
    p_cost.add_argument("--k-rows", type=int, required=True)
    p_cost.add_argument("--v-rows", type=int, required=True)
    p_cost.add_argument("--n-ff", type=int, required=True)
    p_cost.add_argument("--ncols-dst", type=int, required=True)
    p_cost.add_argument("--mtp-layers", type=int, default=1)

    p_reuse = sub.add_parser("reuse-n-cost", help="Estimate packed16 DOT4 reuse-N memory traffic")
    p_reuse.add_argument("--rows", type=int, required=True)
    p_reuse.add_argument("--K", type=int, required=True)
    p_reuse.add_argument("--N", type=int, required=True)

    p_wide = sub.add_parser("wide-n", help="Check packed16 wide-N launch geometry")
    p_wide.add_argument("--K", type=int, required=True)
    p_wide.add_argument("--M", type=int, required=True)
    p_wide.add_argument("--N", type=int, required=True)

    p_wmma = sub.add_parser("wmma", help="Check RDNA3 IU8 WMMA tile fit; optionally call AMD calculator")
    p_wmma.add_argument("--K", type=int, required=True)
    p_wmma.add_argument("--M", type=int, required=True)
    p_wmma.add_argument("--N", type=int, required=True)
    p_wmma.add_argument("--arch", default="gfx1100")
    p_wmma.add_argument("--instruction", default=DEFAULT_WMMA_INSTRUCTION)
    p_wmma.add_argument("--calculator", type=Path, default=DEFAULT_AMD_CALC)
    p_wmma.add_argument("--run-amd-calculator", action="store_true")

    args = ap.parse_args()
    if args.cmd == "routes":
        print_result(summarize_routes(args.logs), args.text)
    elif args.cmd == "cost":
        print_result(shape_cost(args))
    elif args.cmd == "reuse-n-cost":
        print_result(packed16_reuse_n_cost(args))
    elif args.cmd == "wide-n":
        print_result(wide_n(args))
    elif args.cmd == "wmma":
        print_result(wmma(args))
    else:
        ap.error(f"unknown command {args.cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
