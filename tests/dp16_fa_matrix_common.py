#!/usr/bin/env python3
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

DP16_ROUTE_FA_PACKED16_MMQ = "rocm_packed16_dot4_mmq"
DP16_ROUTE_FA2_PACKED16_DOT4_DECODE = "rocm_fa2_packed16_dot4_decode"
DP16_ROUTE_FA2_F16K_ADAPT_DOT4_DECODE = "rocm_mtp_f16k_to_packed16_dot4_decode"
DP16_ROUTE_FA_Q8K_DOT4_KQ = "rocm_q8k_dot4_kq"
DP16_ROUTE_FA_PACKED16_WMMA_TILE = "rocm_packed16_wmma_tile"
DP16_ROUTE_FA1_VEC_FALLBACK = "rocm_fattn_vec"

DP16_BACKEND_NONE = "none"
DP16_BACKEND_FA2_Q8K_DOT4_DECODE = "fa2_q8k_dot4_decode"
DP16_BACKEND_FA2_PACKED16_DOT4_DECODE = "fa2_packed16_dot4_decode"
DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE = "fa2_f16k_adapt_dot4_decode"
DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY = "fa2_packed16_dot4_mmq_verify"
DP16_BACKEND_FA2_PACKED16_WMMA_PREFILL = "fa2_packed16_wmma_prefill"
DP16_BACKEND_FA1_VEC_FALLBACK = "fa1_vec_fallback"

DP16_FA_PLANE_NONE = "none"
DP16_FA_PLANE_FA2_DOT4 = "fa2_dot4"
DP16_FA_PLANE_FA2_PDMQ = "fa2_pdmq"
DP16_FA_PLANE_FA2_PWMMA = "fa2_pwmma"
DP16_FA_PLANE_FA1_VEC_FALLBACK = "fa1_vec_fallback"

DP16_K_REPR_UNKNOWN = "unknown"
DP16_K_REPR_Q8_BLOCK32 = "q8_block32"
DP16_K_REPR_PACKED16_I32_PERSISTENT = "packed16_i32_persistent"
DP16_K_REPR_F16_TO_PACKED16_EXPERIMENTAL = "f16_to_packed16_experimental"
DP16_K_REPR_EXACT_F16_FALLBACK = "exact_f16_fallback"

DP16_LAYOUT_F32 = "f32"
DP16_LAYOUT_F16 = "f16"
DP16_LAYOUT_Q8_BLOCK32 = "q8_block32"
DP16_LAYOUT_PACKED16_I32_SCALED = "packed16_i32_scaled"
DP16_LAYOUT_Q4_0_BLOCK32 = "q4_0_block32"

DP16_FA_INST_MTP_DRAFT_DECODE = "mtp_draft_decode"
DP16_FA_INST_MTP_VERIFY = "mtp_verify"
DP16_FA_INST_PREFILL = "prefill"

DP16_FA_SHAPE_NONE = "none"
DP16_FA_SHAPE_1X64 = "1x64"
DP16_FA_SHAPE_16X16 = "16x16"
DP16_FA_SHAPE_Q2_GQA2_K32 = "q2_gqa2_k32"

DP16_FA_VPATH_NONE = "none"
DP16_FA_VPATH_DIRECT_PV = "direct_pv"
DP16_FA_VPATH_RAW_LDS_Q4 = "raw_lds_q4"

FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211

QK4_0 = 32
QK8_0 = 32
Q4_0_BLOCK_BYTES = 2 + (QK4_0 // 2)
Q8_0_BLOCK_BYTES = 2 + QK8_0
WMMA_IU8_M = 16
WMMA_IU8_N = 16
WMMA_IU8_K = 16
WMMA_IU8_CYCLES = 32
WMMA_IU8_OPS = 8192
WMMA_IU8_WAVE32_A_GPRS = 4
WMMA_IU8_WAVE32_B_GPRS = 4
WMMA_IU8_WAVE32_C_GPRS = 8
WMMA_IU8_WAVE32_D_GPRS = 8


@dataclass
class Problem:
    inst: str
    nq: int
    nk_bucket: int
    d_head: int
    n_heads_q: int
    n_heads_kv: int
    gqa_ratio: int
    batch: int
    q_type: str
    k_type: str
    v_type: str
    q_layout: str
    k_layout: str
    v_layout: str
    causal: bool
    has_mask: bool
    has_sliding_window: bool
    has_sink: bool
    is_mtp: bool
    is_draft_decode: bool
    is_verify: bool
    is_prefill: bool
    packed16_k_ready: bool
    capture: bool
    route_required: bool


@dataclass
class Plan:
    valid: bool = False
    selected: bool = False
    default_allowed: bool = False
    capture_safe: bool = False
    experimental: bool = False
    fallback: bool = False
    plane: str = DP16_FA_PLANE_NONE
    backend: str = DP16_BACKEND_NONE
    k_repr: str = DP16_K_REPR_UNKNOWN
    shape: str = DP16_FA_SHAPE_NONE
    vpath: str = DP16_FA_VPATH_NONE
    qtok_tile: int = 0
    gqa_tile: int = 0
    logical_q: int = 0
    k_tile: int = 0
    route: str = "none"
    reason: str = "none"


@dataclass
class WmmaInstructionSummary:
    architecture: str = "rdna3/gfx1100"
    instruction: str = "v_wmma_i32_16x16x16_iu8"
    m: int = WMMA_IU8_M
    n: int = WMMA_IU8_N
    k: int = WMMA_IU8_K
    cycles: int = WMMA_IU8_CYCLES
    ops: int = WMMA_IU8_OPS
    wave32_a_gprs: int = WMMA_IU8_WAVE32_A_GPRS
    wave32_b_gprs: int = WMMA_IU8_WAVE32_B_GPRS
    wave32_c_gprs: int = WMMA_IU8_WAVE32_C_GPRS
    wave32_d_gprs: int = WMMA_IU8_WAVE32_D_GPRS
    key_mapping: Dict[str, str] | None = None


LANE_ORDER = ["vec", "q8", "packed16", "f16adapt", "pdmq", "pwmma"]


def lane_canonical_name(lane: str) -> str:
    return {
        "vec": "fa1_vec_fallback",
        "q8": "fa2_q8k_dot4_decode",
        "packed16": "fa2_packed16_dot4_decode",
        "f16adapt": "fa2_f16k_adapt_dot4_decode",
        "pdmq": "fa2_packed16_dot4_mmq_verify",
        "pwmma": "fa2_packed16_wmma_prefill",
    }[lane]


def scenario_name(lane: str, capture: bool) -> str:
    return f"{lane}_{'capture' if capture else 'nocapture'}"


def env_enabled(name: str) -> bool:
    v = os.getenv(name)
    return v is not None and v != "" and v != "0"


def mtp_force_vec_fallback() -> bool:
    return env_enabled("LLAMA_MTP_FORCE_FA1_VEC") or env_enabled("LLAMA_MTP_FORCE_VEC_FALLBACK")


def mtp_enable_dot4_fa2() -> bool:
    return env_enabled("LLAMA_MTP_ENABLE_DOT4_FA2") or env_enabled("LLAMA_MTP_ENABLE_PACKED16_FA") or env_enabled("GGML_CUDA_FA_ROUTE_REQUIRE_DOT4")


def mtp_enable_f16_adapt_dot4() -> bool:
    return env_enabled("LLAMA_MTP_ENABLE_F16K_ADAPT_DOT4_FA2")


@contextmanager
def scoped_env(name: str, value: str | None):
    old = os.getenv(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


@contextmanager
def scoped_env_map(mapping: Dict[str, str | None]):
    olds = {k: os.getenv(k) for k in mapping}
    try:
        for key, value in mapping.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, old in olds.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def make_fa1_vec_fallback(reason: str) -> Plan:
    return Plan(
        valid=True,
        selected=True,
        default_allowed=True,
        capture_safe=True,
        experimental=False,
        fallback=True,
        plane=DP16_FA_PLANE_FA1_VEC_FALLBACK,
        backend=DP16_BACKEND_FA1_VEC_FALLBACK,
        k_repr=DP16_K_REPR_EXACT_F16_FALLBACK,
        route=DP16_ROUTE_FA1_VEC_FALLBACK,
        reason=reason,
    )


def plan_mtp_fa(p: Problem) -> Plan:
    if mtp_force_vec_fallback():
        return make_fa1_vec_fallback("forced_vec_safety_fallback")

    is_decode = p.is_draft_decode or p.inst == DP16_FA_INST_MTP_DRAFT_DECODE
    is_verify = p.is_verify or p.inst == DP16_FA_INST_MTP_VERIFY
    is_prefill = p.is_prefill
    dot4_enabled = p.route_required or mtp_enable_dot4_fa2()

    if p.is_mtp and is_decode and p.nq == 1 and p.d_head == 256:
        if dot4_enabled and p.k_layout == DP16_LAYOUT_PACKED16_I32_SCALED and p.packed16_k_ready:
            return Plan(
                valid=True, selected=True, capture_safe=True, experimental=True,
                plane=DP16_FA_PLANE_FA2_DOT4, backend=DP16_BACKEND_FA2_PACKED16_DOT4_DECODE,
                k_repr=DP16_K_REPR_PACKED16_I32_PERSISTENT, shape=DP16_FA_SHAPE_1X64,
                vpath=DP16_FA_VPATH_DIRECT_PV, route=DP16_ROUTE_FA2_PACKED16_DOT4_DECODE,
                reason="mtp_draft_decode_packed16_dot4", k_tile=64,
            )
        if dot4_enabled and p.k_layout == DP16_LAYOUT_Q8_BLOCK32:
            return Plan(
                valid=True, selected=True, capture_safe=True, experimental=True,
                plane=DP16_FA_PLANE_FA2_DOT4, backend=DP16_BACKEND_FA2_Q8K_DOT4_DECODE,
                k_repr=DP16_K_REPR_Q8_BLOCK32, shape=DP16_FA_SHAPE_1X64,
                vpath=DP16_FA_VPATH_DIRECT_PV, route=DP16_ROUTE_FA_Q8K_DOT4_KQ,
                reason="mtp_draft_decode_q8k_dot4", k_tile=64,
            )
        if dot4_enabled and mtp_enable_f16_adapt_dot4() and p.k_type == "f16" and p.v_type == "q4_0" and not p.capture:
            return Plan(
                valid=True, selected=True, capture_safe=False, experimental=True,
                plane=DP16_FA_PLANE_FA2_DOT4, backend=DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE,
                k_repr=DP16_K_REPR_F16_TO_PACKED16_EXPERIMENTAL, shape=DP16_FA_SHAPE_1X64,
                vpath=DP16_FA_VPATH_DIRECT_PV, route=DP16_ROUTE_FA2_F16K_ADAPT_DOT4_DECODE,
                reason="mtp_draft_decode_f16k_adapt_dot4_experimental", k_tile=64,
            )
        return make_fa1_vec_fallback("dot4_decode_not_available_fallback_vec")

    if p.is_mtp and is_verify and 2 <= p.nq <= 4 and p.d_head == 256 and p.gqa_ratio in (6, 8) and p.v_type == "q4_0" and p.v_layout == DP16_LAYOUT_Q4_0_BLOCK32 and p.k_layout == DP16_LAYOUT_PACKED16_I32_SCALED and p.packed16_k_ready and dot4_enabled:
        return Plan(
            valid=True, selected=True, capture_safe=True, experimental=True,
            plane=DP16_FA_PLANE_FA2_PDMQ, backend=DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY,
            k_repr=DP16_K_REPR_PACKED16_I32_PERSISTENT, shape=DP16_FA_SHAPE_Q2_GQA2_K32,
            vpath=DP16_FA_VPATH_RAW_LDS_Q4, route=DP16_ROUTE_FA_PACKED16_MMQ,
            reason="mtp_verify_smallq_pdmq_q4", qtok_tile=2, gqa_tile=2, logical_q=4, k_tile=32,
        )

    if is_prefill and p.d_head == 256 and p.k_layout == DP16_LAYOUT_PACKED16_I32_SCALED and p.packed16_k_ready and dot4_enabled:
        return Plan(
            valid=True, selected=True, capture_safe=True, experimental=True,
            plane=DP16_FA_PLANE_FA2_PWMMA, backend=DP16_BACKEND_FA2_PACKED16_WMMA_PREFILL,
            k_repr=DP16_K_REPR_PACKED16_I32_PERSISTENT, shape=DP16_FA_SHAPE_16X16,
            vpath=DP16_FA_VPATH_DIRECT_PV, route=DP16_ROUTE_FA_PACKED16_WMMA_TILE,
            reason="prefill_packed16_wmma", k_tile=64,
        )

    return make_fa1_vec_fallback("unsupported_fallback_vec")


def graph_key(problem: Problem, plan: Plan) -> List[object]:
    return [
        problem.inst,
        plan.plane,
        plan.backend,
        plan.k_repr,
        plan.shape,
        plan.vpath,
        problem.nq,
        problem.nk_bucket,
        problem.d_head,
        problem.n_heads_q,
        problem.n_heads_kv,
        problem.gqa_ratio,
        problem.batch,
        problem.q_type,
        problem.k_type,
        problem.v_type,
        problem.q_layout,
        problem.k_layout,
        problem.v_layout,
        plan.qtok_tile,
        plan.gqa_tile,
        plan.logical_q,
        plan.k_tile,
        problem.causal,
        problem.has_mask,
        problem.has_sliding_window,
        problem.has_sink,
        plan.capture_safe,
        plan.experimental,
        plan.fallback,
    ]


def hash_graph_key(key: List[object]) -> int:
    h = FNV_OFFSET
    for item in key:
        payload = "1" if isinstance(item, bool) and item else "0" if isinstance(item, bool) else str(item)
        for b in payload.encode("utf-8"):
            h ^= b
            h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        h ^= 0xFF
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def graph_key_field_names() -> List[str]:
    return [
        "inst", "plane", "backend", "k_repr", "shape", "vpath",
        "nq", "nk_bucket", "d_head", "n_heads_q", "n_heads_kv", "gqa_ratio", "batch",
        "q_type", "k_type", "v_type", "q_layout", "k_layout", "v_layout",
        "qtok_tile", "gqa_tile", "logical_q", "k_tile",
        "causal", "has_mask", "has_sliding_window", "has_sink",
        "capture_safe", "experimental", "fallback",
    ]


def graph_key_dict(problem: Problem, plan: Plan) -> Dict[str, object]:
    return dict(zip(graph_key_field_names(), graph_key(problem, plan)))


def measure_plan(problem: Problem, repeats: int) -> Dict[str, float]:
    samples: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        plan = plan_mtp_fa(problem)
        _ = hash_graph_key(graph_key(problem, plan))
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1000.0)
    samples.sort()
    if not samples:
        return {"median_us": 0.0, "p05_us": 0.0, "p95_us": 0.0}

    def pct(p: float) -> float:
        idx = round((len(samples) - 1) * p)
        return samples[idx]

    return {"median_us": pct(0.50), "p05_us": pct(0.05), "p95_us": pct(0.95)}


def make_base_problem() -> Problem:
    return Problem(
        inst=DP16_FA_INST_MTP_DRAFT_DECODE,
        nq=1,
        nk_bucket=2048,
        d_head=256,
        n_heads_q=24,
        n_heads_kv=4,
        gqa_ratio=6,
        batch=1,
        q_type="f32",
        k_type="q8_0",
        v_type="q4_0",
        q_layout=DP16_LAYOUT_F32,
        k_layout=DP16_LAYOUT_Q8_BLOCK32,
        v_layout=DP16_LAYOUT_Q4_0_BLOCK32,
        causal=True,
        has_mask=True,
        has_sliding_window=False,
        has_sink=False,
        is_mtp=True,
        is_draft_decode=True,
        is_verify=False,
        is_prefill=False,
        packed16_k_ready=False,
        capture=False,
        route_required=True,
    )


def make_problem(lane: str, capture: bool) -> Problem:
    p = make_base_problem()
    p.capture = capture
    if lane == "vec":
        p.k_type = "f16"
        p.k_layout = DP16_LAYOUT_F16
        p.v_type = "q4_0"
        p.v_layout = DP16_LAYOUT_Q4_0_BLOCK32
        p.route_required = False
    elif lane == "q8":
        pass
    elif lane == "packed16":
        p.k_type = "i32"
        p.k_layout = DP16_LAYOUT_PACKED16_I32_SCALED
        p.packed16_k_ready = True
    elif lane == "f16adapt":
        p.k_type = "f16"
        p.k_layout = DP16_LAYOUT_F16
    elif lane == "pdmq":
        p.inst = DP16_FA_INST_MTP_VERIFY
        p.nq = 4
        p.nk_bucket = 4096
        p.k_type = "i32"
        p.k_layout = DP16_LAYOUT_PACKED16_I32_SCALED
        p.packed16_k_ready = True
        p.is_draft_decode = False
        p.is_verify = True
    elif lane == "pwmma":
        p.inst = DP16_FA_INST_PREFILL
        p.nq = 16
        p.k_type = "i32"
        p.k_layout = DP16_LAYOUT_PACKED16_I32_SCALED
        p.packed16_k_ready = True
        p.is_draft_decode = False
        p.is_prefill = True
        p.route_required = True
    else:
        raise ValueError(f"unknown lane {lane}")
    return p


def expected_backend(lane: str, capture: bool) -> str:
    if lane == "vec":
        return DP16_BACKEND_FA1_VEC_FALLBACK
    if lane == "q8":
        return DP16_BACKEND_FA2_Q8K_DOT4_DECODE
    if lane == "packed16":
        return DP16_BACKEND_FA2_PACKED16_DOT4_DECODE
    if lane == "f16adapt":
        return DP16_BACKEND_FA1_VEC_FALLBACK if capture else DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE
    if lane == "pdmq":
        return DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY
    if lane == "pwmma":
        return DP16_BACKEND_FA2_PACKED16_WMMA_PREFILL
    raise ValueError(lane)


def route_aliases_for_lane(lane: str) -> List[str]:
    return {
        "vec": [DP16_ROUTE_FA1_VEC_FALLBACK],
        "q8": [DP16_ROUTE_FA_Q8K_DOT4_KQ],
        "packed16": [DP16_ROUTE_FA2_PACKED16_DOT4_DECODE],
        "f16adapt": [DP16_ROUTE_FA2_F16K_ADAPT_DOT4_DECODE, "rocm_mtp_f16k_q4v_decode"],
        "pdmq": [DP16_ROUTE_FA_PACKED16_MMQ, "packed16_dot4_mmq"],
        "pwmma": [DP16_ROUTE_FA_PACKED16_WMMA_TILE, "packed16_wmma_tile"],
    }[lane]


def enable_env_for_lane(lane: str) -> Dict[str, str | None]:
    base: Dict[str, str | None] = {
        "LLAMA_MTP_FORCE_FA1_VEC": None,
        "LLAMA_MTP_FORCE_VEC_FALLBACK": None,
        "LLAMA_MTP_ENABLE_DOT4_FA2": None,
        "LLAMA_MTP_ENABLE_PACKED16_FA": None,
        "LLAMA_MTP_ENABLE_F16K_ADAPT_DOT4_FA2": None,
    }
    if lane == "vec":
        base["LLAMA_MTP_FORCE_FA1_VEC"] = "1"
    elif lane == "packed16":
        base["LLAMA_MTP_ENABLE_PACKED16_FA"] = "1"
    else:
        base["LLAMA_MTP_ENABLE_DOT4_FA2"] = "1"
    if lane == "f16adapt":
        base["LLAMA_MTP_ENABLE_F16K_ADAPT_DOT4_FA2"] = "1"
    return base


def rollback_env_matrix() -> List[Dict[str, str | None]]:
    return [
        {"LLAMA_MTP_FORCE_FA1_VEC": "1", "LLAMA_MTP_FORCE_VEC_FALLBACK": None},
        {"LLAMA_MTP_FORCE_FA1_VEC": None, "LLAMA_MTP_FORCE_VEC_FALLBACK": "1"},
    ]


def row_bytes_for_layout(layout: str, d_head: int) -> int:
    if layout == DP16_LAYOUT_F32:
        return d_head * 4
    if layout == DP16_LAYOUT_F16:
        return d_head * 2
    if layout == DP16_LAYOUT_Q8_BLOCK32:
        return (d_head // QK8_0) * Q8_0_BLOCK_BYTES
    if layout == DP16_LAYOUT_Q4_0_BLOCK32:
        return (d_head // QK4_0) * Q4_0_BLOCK_BYTES
    if layout == DP16_LAYOUT_PACKED16_I32_SCALED:
        payload_bytes = (d_head // 4) * 4
        scale_bytes = (d_head // QK8_0) * 2
        return payload_bytes + scale_bytes
    raise ValueError(f"unsupported layout {layout}")


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


def is_dot4_backend(backend: str) -> bool:
    return backend in {
        DP16_BACKEND_FA2_Q8K_DOT4_DECODE,
        DP16_BACKEND_FA2_PACKED16_DOT4_DECODE,
        DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE,
        DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY,
    }


def estimate_calculation(problem: Problem, plan: Plan) -> Dict[str, Any]:
    q_row_bytes = row_bytes_for_layout(problem.q_layout, problem.d_head)
    k_row_bytes = row_bytes_for_layout(problem.k_layout, problem.d_head)
    v_row_bytes = row_bytes_for_layout(problem.v_layout, problem.d_head)
    packed16_row_bytes = row_bytes_for_layout(DP16_LAYOUT_PACKED16_I32_SCALED, problem.d_head)
    out_row_bytes = problem.d_head * 4

    q_rows = problem.batch * problem.nq * problem.n_heads_q
    kv_rows = problem.batch * problem.nk_bucket * problem.n_heads_kv

    qk_macs = problem.batch * problem.nq * problem.n_heads_q * problem.nk_bucket * problem.d_head
    pv_macs = qk_macs

    q_bytes = q_rows * q_row_bytes
    k_bytes = kv_rows * k_row_bytes
    v_bytes = kv_rows * v_row_bytes
    output_bytes = q_rows * out_row_bytes
    sidecar_bytes = kv_rows * packed16_row_bytes if plan.backend == DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE else 0
    temp_conversion_bytes = sidecar_bytes
    total_bytes = q_bytes + k_bytes + v_bytes + output_bytes + sidecar_bytes

    dot4_per_key_row = problem.d_head // 4
    qtok_tile_count = div_up(problem.nq, plan.qtok_tile or problem.nq)
    head_tile_count = div_up(problem.n_heads_q, plan.gqa_tile or 1)
    k_tile_count = div_up(problem.nk_bucket, plan.k_tile or problem.nk_bucket)

    dot4_inst_per_tile = 0
    dot4_inst_qk_est = 0
    if is_dot4_backend(plan.backend):
        logical_q = plan.logical_q or problem.nq
        k_tile = plan.k_tile or problem.nk_bucket
        dot4_inst_per_tile = logical_q * k_tile * dot4_per_key_row
        dot4_inst_qk_est = dot4_inst_per_tile * qtok_tile_count * head_tile_count * k_tile_count * problem.batch

    wmma_k_steps = div_up(problem.d_head, WMMA_IU8_K)
    wmma_inst_qk_est = 0
    if plan.backend == DP16_BACKEND_FA2_PACKED16_WMMA_PREFILL:
        wmma_inst_qk_est = (
            problem.batch *
            div_up(problem.nq, WMMA_IU8_M) *
            div_up(problem.nk_bucket, WMMA_IU8_N) *
            wmma_k_steps *
            problem.n_heads_q
        )

    math_issue_lower_bound_cycles = 0
    if is_dot4_backend(plan.backend):
        math_issue_lower_bound_cycles = dot4_inst_qk_est
    elif plan.backend == DP16_BACKEND_FA2_PACKED16_WMMA_PREFILL:
        math_issue_lower_bound_cycles = wmma_inst_qk_est * WMMA_IU8_CYCLES

    memory_tx_128b_lower_bound = div_up(total_bytes, 128)

    if total_bytes >= 5_000_000:
        memory_pressure = "very_high"
    elif total_bytes >= 2_500_000:
        memory_pressure = "high"
    elif total_bytes >= 1_000_000:
        memory_pressure = "medium"
    else:
        memory_pressure = "low"

    expected_bottleneck = {
        DP16_BACKEND_FA1_VEC_FALLBACK: "vec_fallback_kv_bandwidth",
        DP16_BACKEND_FA2_Q8K_DOT4_DECODE: "q8_k_cache_bandwidth_and_dot4_issue",
        DP16_BACKEND_FA2_PACKED16_DOT4_DECODE: "packed16_k_cache_bandwidth",
        DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE: "op_local_packed16_sidecar_conversion",
        DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY: "lds_and_register_pressure",
        DP16_BACKEND_FA2_PACKED16_WMMA_PREFILL: "wmma_issue_and_v_bandwidth",
    }[plan.backend]

    if plan.backend == DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE:
        capture_policy = "nocapture_only_fallback_under_capture"
    elif plan.capture_safe:
        capture_policy = "capture_ok"
    else:
        capture_policy = "fallback_only"

    return {
        "qk_macs": qk_macs,
        "pv_macs": pv_macs,
        "dot4_per_key_row": dot4_per_key_row if is_dot4_backend(plan.backend) else 0,
        "dot4_inst_per_tile": dot4_inst_per_tile,
        "dot4_inst_qk_est": dot4_inst_qk_est,
        "wmma_inst_qk_est": wmma_inst_qk_est,
        "wmma_tile": {
            "m": WMMA_IU8_M,
            "n": WMMA_IU8_N,
            "k": WMMA_IU8_K,
            "k_steps_per_qk_tile": wmma_k_steps,
            "cycles_per_inst": WMMA_IU8_CYCLES,
            "ops_per_inst": WMMA_IU8_OPS,
        },
        "tile_shape": {
            "qtok_tile_count": qtok_tile_count,
            "head_tile_count": head_tile_count,
            "k_tile_count": k_tile_count,
            "logical_q": plan.logical_q or problem.nq,
            "qtok_tile": plan.qtok_tile or problem.nq,
            "gqa_tile": plan.gqa_tile or 1,
            "k_tile": plan.k_tile or problem.nk_bucket,
        },
        "bytes_q_est": q_bytes,
        "bytes_k_est": k_bytes,
        "bytes_v_est": v_bytes,
        "bytes_output_est": output_bytes,
        "bytes_sidecar_est": sidecar_bytes,
        "bytes_temp_conversion_est": temp_conversion_bytes,
        "bytes_total_est": total_bytes,
        "row_bytes": {
            "q": q_row_bytes,
            "k": k_row_bytes,
            "v": v_row_bytes,
            "output": out_row_bytes,
            "packed16": packed16_row_bytes,
        },
        "math_issue_lower_bound_cycles": math_issue_lower_bound_cycles,
        "memory_128b_tx_lower_bound": memory_tx_128b_lower_bound,
        "memory_pressure_class": memory_pressure,
        "expected_bottleneck": expected_bottleneck,
        "capture_policy": capture_policy,
        "notes": [
            "QK/PV MACs are exact problem-size counts.",
            "DOT4/WMMA estimates model the dominant QK stage only; PV is tracked separately via pv_macs.",
            "math_issue_lower_bound_cycles is an optimistic issue-floor, not a measured runtime.",
            "memory_128b_tx_lower_bound is a lower-bound transaction proxy, not a bandwidth-timed estimate.",
        ],
    }


def expected_negative_markers_contract(lane: str, capture: bool) -> List[str]:
    markers = [
        "backend_expected_*",
        "plan_not_selected",
        "graph_key_unstable",
    ]
    if lane == "f16adapt" and not capture:
        markers.extend([
            "f16_adapt_not_marked_experimental",
            "f16_adapt_unexpected_capture_safe",
        ])
    if lane == "f16adapt" and capture:
        markers.append("capture_expected_fallback")
    if lane not in ("vec", "f16adapt"):
        markers.append("unexpected_fastlane_fallback")
    if capture and lane != "f16adapt":
        markers.append("capture_safe_violation")
    return markers


def run_graph_key_proofs() -> List[Dict[str, Any]]:
    proofs: List[Dict[str, Any]] = []
    with scoped_env("LLAMA_MTP_ENABLE_DOT4_FA2", "1"):
        p = make_problem("q8", False)
        h0 = hash_graph_key(graph_key(p, plan_mtp_fa(p)))
        h1 = hash_graph_key(graph_key(p, plan_mtp_fa(p)))
        assert h0 == h1, "identical q8 plan hash must be stable"
        proofs.append({
            "case": "stable_q8_repeat",
            "lhs": f"0x{h0:016x}",
            "rhs": f"0x{h1:016x}",
            "passed": True,
        })

        p_q8 = make_problem("q8", False)
        p_p16 = make_problem("packed16", False)
        h_q8 = hash_graph_key(graph_key(p_q8, plan_mtp_fa(p_q8)))
        h_p16 = hash_graph_key(graph_key(p_p16, plan_mtp_fa(p_p16)))
        assert h_q8 != h_p16, "backend/k_repr shift must change graph hash"
        proofs.append({
            "case": "backend_krepr_shift_changes_hash",
            "lhs": f"0x{h_q8:016x}",
            "rhs": f"0x{h_p16:016x}",
            "passed": True,
        })

    with scoped_env("LLAMA_MTP_FORCE_FA1_VEC", "1"):
        p_vec = make_problem("vec", False)
        h_vec = hash_graph_key(graph_key(p_vec, plan_mtp_fa(p_vec)))
    with scoped_env_map({
        "LLAMA_MTP_ENABLE_DOT4_FA2": "1",
        "LLAMA_MTP_ENABLE_F16K_ADAPT_DOT4_FA2": "1",
    }):
        p_f16 = make_problem("f16adapt", False)
        h_f16 = hash_graph_key(graph_key(p_f16, plan_mtp_fa(p_f16)))
        assert h_vec != h_f16, "plane/experimental shift must change graph hash"
        proofs.append({
            "case": "plane_experimental_shift_changes_hash",
            "lhs": f"0x{h_vec:016x}",
            "rhs": f"0x{h_f16:016x}",
            "passed": True,
        })

        p_p16 = make_problem("packed16", False)
        p_pdmq = make_problem("pdmq", False)
        h_p16 = hash_graph_key(graph_key(p_p16, plan_mtp_fa(p_p16)))
        h_pdmq = hash_graph_key(graph_key(p_pdmq, plan_mtp_fa(p_pdmq)))
        assert h_p16 != h_pdmq, "shape/plane shift must change graph hash"
        proofs.append({
            "case": "shape_plane_shift_changes_hash",
            "lhs": f"0x{h_p16:016x}",
            "rhs": f"0x{h_pdmq:016x}",
            "passed": True,
        })

    return proofs


def run_rollback_proofs() -> List[Dict[str, Any]]:
    proofs: List[Dict[str, Any]] = []
    baseline_problem = make_problem("q8", False)
    for env_case in rollback_env_matrix():
        with scoped_env_map(env_case):
            plan = plan_mtp_fa(baseline_problem)
            assert plan.backend == DP16_BACKEND_FA1_VEC_FALLBACK, "rollback env must force vec fallback"
            proofs.append({
                "case": "force_vec_fallback",
                "env": {k: v for k, v in env_case.items() if v is not None},
                "backend": plan.backend,
                "route": plan.route,
                "passed": True,
            })
    return proofs


def run_scenario(lane: str, capture: bool, repeats: int) -> Dict[str, Any]:
    problem = make_problem(lane, capture)
    markers: Dict[str, bool] = {}
    env_map = enable_env_for_lane(lane)
    with scoped_env_map(env_map):
        plan = plan_mtp_fa(problem)
        key = graph_key(problem, plan)
        key_hash = hash_graph_key(key)
        timing = measure_plan(problem, repeats)
        if plan.backend != expected_backend(lane, capture):
            markers[f"backend_expected_{expected_backend(lane, capture)}_got_{plan.backend}"] = True
        if not plan.selected:
            markers["plan_not_selected"] = True
        if lane == "f16adapt" and not capture and not plan.experimental:
            markers["f16_adapt_not_marked_experimental"] = True
        if lane == "f16adapt" and not capture and plan.capture_safe:
            markers["f16_adapt_unexpected_capture_safe"] = True
        if lane == "f16adapt" and capture and not plan.fallback:
            markers["capture_expected_fallback"] = True
        if lane not in ("f16adapt", "vec") and plan.fallback:
            markers["unexpected_fastlane_fallback"] = True
        if capture and lane != "f16adapt" and not plan.capture_safe:
            markers["capture_safe_violation"] = True
        replay_hash = hash_graph_key(graph_key(problem, plan))
        if replay_hash != key_hash:
            markers["graph_key_unstable"] = True
        calculation = estimate_calculation(problem, plan)
        return {
            "name": scenario_name(lane, capture),
            "mode": "plan_only",
            "lane": lane,
            "lane_canonical": lane_canonical_name(lane),
            "problem": {
                "inst": problem.inst,
                "nq": problem.nq,
                "nk_bucket": problem.nk_bucket,
                "d": problem.d_head,
                "gqa": problem.gqa_ratio,
                "n_heads_q": problem.n_heads_q,
                "n_heads_kv": problem.n_heads_kv,
                "k_layout": problem.k_layout,
                "v_layout": problem.v_layout,
                "capture": problem.capture,
            },
            "plan": {
                "plane": plan.plane,
                "backend": plan.backend,
                "route": plan.route,
                "reason": plan.reason,
                "k_repr": plan.k_repr,
                "shape": plan.shape,
                "vpath": plan.vpath,
                "experimental": plan.experimental,
                "fallback": plan.fallback,
                "capture_safe": plan.capture_safe,
                "default_allowed": plan.default_allowed,
                "graph_key": f"0x{key_hash:016x}",
                "graph_key_fields": graph_key_dict(problem, plan),
            },
            "timing": timing,
            "calculation": calculation,
            "correct": not markers,
            "accepted": not markers,
            "negative_markers": markers,
            "collection_contract": {
                "expected_route_aliases": route_aliases_for_lane(lane),
                "expected_negative_markers": expected_negative_markers_contract(lane, capture),
                "enable_env": {k: v for k, v in env_map.items() if v is not None},
            },
        }


def make_lane_calculation_table(results: List[Dict[str, Any]], proofs: List[Dict[str, Any]], rollback: List[Dict[str, Any]]) -> Dict[str, Any]:
    lane_rows = []
    for lane in LANE_ORDER:
        lane_results = [r for r in results if r["lane"] == lane]
        if not lane_results:
            continue
        nocapture = next((r for r in lane_results if not r["problem"]["capture"]), lane_results[0])
        capture = next((r for r in lane_results if r["problem"]["capture"]), None)
        lane_rows.append({
            "lane": lane,
            "lane_canonical": lane_canonical_name(lane),
            "nocapture": {
                "backend": nocapture["plan"]["backend"],
                "route": nocapture["plan"]["route"],
                "graph_key": nocapture["plan"]["graph_key"],
                "calculation": nocapture["calculation"],
            },
            "capture": None if capture is None else {
                "backend": capture["plan"]["backend"],
                "route": capture["plan"]["route"],
                "graph_key": capture["plan"]["graph_key"],
                "capture_safe": capture["plan"]["capture_safe"],
                "fallback": capture["plan"]["fallback"],
            },
        })
    return {
        "metadata": {
            "phase": "dp16_fa_matrix_calculation",
            "wmma_instruction": asdict(WmmaInstructionSummary(
                key_mapping={
                    "a": "A[i][k] -> Src0, wave32 lanes i and i+16",
                    "b": "B[k][j] -> Src1, wave32 lanes j and j+16",
                    "d": "D[i][j] -> Vdst, wave32 GPR floor(i/2), lane ((16*i)%32)+j",
                },
            )),
            "row_bytes": {
                DP16_LAYOUT_F32: row_bytes_for_layout(DP16_LAYOUT_F32, 256),
                DP16_LAYOUT_F16: row_bytes_for_layout(DP16_LAYOUT_F16, 256),
                DP16_LAYOUT_Q8_BLOCK32: row_bytes_for_layout(DP16_LAYOUT_Q8_BLOCK32, 256),
                DP16_LAYOUT_Q4_0_BLOCK32: row_bytes_for_layout(DP16_LAYOUT_Q4_0_BLOCK32, 256),
                DP16_LAYOUT_PACKED16_I32_SCALED: row_bytes_for_layout(DP16_LAYOUT_PACKED16_I32_SCALED, 256),
            },
        },
        "graph_key_proofs": proofs,
        "rollback_proofs": rollback,
        "lanes": lane_rows,
    }


def make_route_capture_matrix(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for lane in LANE_ORDER:
        nocapture = next(r for r in results if r["lane"] == lane and not r["problem"]["capture"])
        rows.append({
            "lane": lane,
            "lane_canonical": lane_canonical_name(lane),
            "backend": nocapture["plan"]["backend"],
            "plane": nocapture["plan"]["plane"],
            "route_aliases": route_aliases_for_lane(lane),
            "default_allowed": nocapture["plan"]["default_allowed"],
            "experimental": nocapture["plan"]["experimental"],
            "force_fallback_env_any": ["LLAMA_MTP_FORCE_FA1_VEC", "LLAMA_MTP_FORCE_VEC_FALLBACK"],
            "enable_env": {k: v for k, v in enable_env_for_lane(lane).items() if v is not None},
            "capture_allowed": lane != "f16adapt",
            "capture_behavior": "fallback_to_vec" if lane == "f16adapt" else "allowed",
            "expected_negative_markers": expected_negative_markers_contract(lane, False),
        })
    return rows


def make_runtime_acceptance_matrix() -> List[Dict[str, Any]]:
    rows = []
    for lane in LANE_ORDER:
        capture_modes = [False, True]
        for capture in capture_modes:
            if lane == "pwmma":
                capture_modes = [False, True]
            rows.append({
                "id": scenario_name(lane, capture),
                "lane": lane,
                "lane_canonical": lane_canonical_name(lane),
                "capture": capture,
                "expected_backend": expected_backend(lane, capture),
                "expected_route_aliases": route_aliases_for_lane(lane),
                "enable_env": {k: v for k, v in enable_env_for_lane(lane).items() if v is not None},
                "planner_probe_command": f"python3 tests/test-fattn-mtp-graph-race.py --lane {lane} --capture {'capture' if capture else 'nocapture'} --repeats 128 --json-out <artifact_dir>/results-{scenario_name(lane, capture)}.json",
                "hip_replay_harness_command": f"python3 tests/test-fattn-mtp-hip-replay.py --lane {lane} --capture {'capture' if capture else 'nocapture'} --mode live --artifact-dir <artifact_dir> --command '<live decode/prefill command>'",
                "collect_fields": [
                    "route_log_tail",
                    "dp16_fa_plan_trace",
                    "legacy_route_label",
                    "graph_key_hash",
                    "correctness_summary",
                    "negative_markers",
                    "fallback_markers",
                    "timing_summary",
                    "capture_status",
                ],
                "live_status": "contract_defined",
            })
    return rows


def make_summary_markdown(
        artifact_dir: str,
        results: List[Dict[str, Any]],
        proofs: List[Dict[str, Any]],
        rollback: List[Dict[str, Any]],
        route_capture_matrix: List[Dict[str, Any]],
        runtime_matrix: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# DP16 FA Matrix Calculation Summary")
    lines.append("")
    lines.append(f"- artifact_dir: `{artifact_dir}`")
    lines.append("- scope: calculation/matrix-calculation expansion for DP16 FA lanes")
    lines.append("- planner harness remains plan-only; HIP replay is separated into a dedicated harness contract")
    lines.append("- WMMA reference instruction: `v_wmma_i32_16x16x16_iu8` on `gfx1100` (RDNA3)")
    lines.append("")
    lines.append("## WMMA Matrix-Calculator Facts")
    lines.append("")
    lines.append("- shape: 16x16x16")
    lines.append("- execution cycles per instruction: 32")
    lines.append("- ops per instruction: 8192")
    lines.append("- wave32 register usage: A=4, B=4, C=8, D=8 GPRs")
    lines.append("- wave32 A/B mapping: `A[i][k]` and `B[k][j]` lanes are duplicated across `lane` and `lane+16`")
    lines.append("- wave32 D mapping: `D[i][j] -> Vdst floor(i/2), lane ((16*i)%32)+j`")
    lines.append("")
    lines.append("## Lane Matrix")
    lines.append("")
    for lane in LANE_ORDER:
        row = next(r for r in results if r["lane"] == lane and not r["problem"]["capture"])
        calc = row["calculation"]
        lines.append(
            f"- `{lane}` → backend `{row['plan']['backend']}`, route `{row['plan']['route']}`, "
            f"qk_macs={calc['qk_macs']}, pv_macs={calc['pv_macs']}, total_bytes={calc['bytes_total_est']}, "
            f"bottleneck={calc['expected_bottleneck']}")
    lines.append("")
    lines.append("## Acceptance Gates")
    lines.append("")
    lines.append("- `negative_markers == {}` for all planner rows")
    lines.append("- `graph_key_proofs` must all pass")
    lines.append("- `LLAMA_MTP_FORCE_FA1_VEC=1` and `LLAMA_MTP_FORCE_VEC_FALLBACK=1` must force FA1/VEC")
    lines.append("- f16-adapt must remain non-capture-safe and must fallback under capture")
    lines.append("- no across-N reuse or LDS M2xN reintroduction in this phase")
    lines.append("")
    lines.append("## Profiling Strategy")
    lines.append("")
    lines.append("- Prefer planner/hash proof + lightweight HIP timing first")
    lines.append("- `rocprof-compute` remains blocked by missing Python deps in `/opt/rocm-7.2.3/libexec/rocprofiler-compute/requirements.txt`")
    lines.append("- Use route logs, HIP event timing, kernel-name traces, and existing acceptance scripts before deep profiler work")
    lines.append("")
    lines.append("## Route/Capture Matrix")
    lines.append("")
    for row in route_capture_matrix:
        lines.append(
            f"- `{row['lane']}`: capture_behavior={row['capture_behavior']}, experimental={row['experimental']}, "
            f"default_allowed={row['default_allowed']}, route_aliases={', '.join(row['route_aliases'])}")
    lines.append("")
    lines.append("## Runtime Acceptance Matrix")
    lines.append("")
    lines.append(f"- rows defined: {len(runtime_matrix)}")
    lines.append("- live runtime collection is contract-defined here; actual route/stderr capture should be populated by `tests/test-fattn-mtp-hip-replay.py --mode live`")
    lines.append("")
    lines.append("## Graph-Key Proofs")
    lines.append("")
    for proof in proofs:
        lines.append(f"- `{proof['case']}`: passed={proof['passed']} lhs={proof['lhs']} rhs={proof['rhs']}")
    lines.append("")
    lines.append("## Rollback Proofs")
    lines.append("")
    for proof in rollback:
        env_desc = ", ".join(f"{k}={v}" for k, v in proof["env"].items())
        lines.append(f"- `{env_desc}` -> backend `{proof['backend']}` route `{proof['route']}`")
    lines.append("")
    lines.append("## Go/No-Go")
    lines.append("")
    lines.append("- keep `fa1_vec_fallback` as the rollback/control lane")
    lines.append("- keep `fa2_f16k_adapt_dot4_decode` experimental and non-capture-only")
    lines.append("- keep `fa2_packed16_dot4_mmq_verify` shape-guarded")
    lines.append("- use the WMMA calculator evidence only for prefill/PWMMA reasoning, not for decode DOT4 promotion")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_json(path: str, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
