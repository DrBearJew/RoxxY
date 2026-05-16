#!/usr/bin/env python3
"""TBQ4 FlashAttention planning calculator.

Offline model for RDNA3 TBQ4_0 KV-cache optimization ideas.  It does not run a
GPU kernel; it estimates row storage, current VEC dequant load pressure,
materialization expansion, and FWHT rotation cost from constants used in the
HIP kernels.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


CURRENT_FORMAT_LABEL = "TBQ4_0"
FUTURE_FORMAT_LABEL = "future_format_v2_harness_only"
PRODUCTION_ROUTE_LABEL = "tbq4_vec"

QK_TBQ4 = 128
TBQ4_BLOCK_BYTES = 66      # ggml_half d + 64 packed 4-bit bytes
FATTN_KQ_STRIDE = 256
WARP_SIZE = 32


@dataclass
class RowEstimate:
    d: int
    blocks_per_row: int
    tbq4_raw_row_bytes: int
    tbq4_padded16_row_bytes: int
    f16_row_bytes: int
    q8_0_approx_row_bytes: int
    tbq4_bpw: float
    tbq4_vs_f16: float


@dataclass
class VecLoadEstimate:
    d: int
    k_current_bytes_per_row: int
    k_ideal_norm_hoist_bytes_per_row: int
    k_norm_redundant_bytes_per_row: int
    v_current_bytes_per_row: int
    v_ideal_norm_hoist_bytes_per_row: int
    v_norm_redundant_bytes_per_row: int
    kv_current_bytes_per_qk_pair: int
    kv_ideal_norm_hoist_bytes_per_qk_pair: int
    kv_norm_hoist_savings_per_pair: int
    savings_percent: float


@dataclass
class KqNormHoistEstimate:
    d: int
    nthreads_kq: int
    cpy_nb: int
    cpy_ne: int
    subwarp_groups_per_warp: int
    blocks_per_row: int
    half2_pairs_per_row: int
    current_norm_loads_per_k_row: int
    per_thread_cache_norm_loads_per_k_row: int
    subwarp_broadcast_norm_loads_per_k_row: int
    ideal_norm_loads_per_k_row: int
    current_k_bytes_per_row: int
    per_thread_cache_k_bytes_per_row: int
    subwarp_broadcast_k_bytes_per_row: int
    ideal_k_bytes_per_row: int
    per_thread_cache_savings_percent: float
    subwarp_broadcast_savings_percent: float
    ideal_savings_percent: float
    notes: str


@dataclass
class RotationEstimate:
    d: int
    blocks_per_row: int
    fwht128_addsub_per_row: int
    fwht128_barriers_per_row: int
    future32_addsub_per_row: int
    future32_barriers_per_row: int
    addsub_savings_percent: float


@dataclass
class AttentionEstimate:
    nq: int
    kv: int
    q_heads: int
    kv_heads: int
    gqa_ratio: float
    qk_pairs: int
    cache_tbq4_bytes: int
    cache_f16_bytes: int
    vec_current_logical_kv_bytes: int
    vec_norm_hoist_logical_kv_bytes: int
    vec_norm_hoist_savings_bytes: int
    vec_norm_hoist_savings_percent: float
    q_rotate_rows: int
    out_rotate_rows: int
    fwht128_barriers_total_if_kv_tbq4: int
    future32_barriers_total_if_kv_tbq4: int


@dataclass
class MaterializationEstimate:
    d: int
    rows: int
    raw_staging_bytes: int
    f16_tile_bytes: int
    total_lhs_bytes: int
    expansion_vs_raw: float
    fits_64k: bool


@dataclass
class GateDecision:
    route: str
    d: int
    decision: str
    priority: int
    reason: str
    required_next_evidence: str


@dataclass
class LdsRouteEstimate:
    route: str
    d: int
    bytes_per_row: int
    max_rows_48k: int
    max_rows_56k: int
    max_rows_64k: int
    cache_savings_preserved: bool
    notes: str


@dataclass
class LdsExperimentContract:
    route: str
    base_route: str
    stage: int
    role: str
    env_var: str
    env_value: str
    d: int
    tile_rows: int
    packed_stages: int
    packed_stride_bytes: int
    f16_stride_half2: int
    f16_stride_bytes: int
    lds_bytes_per_row: int
    total_lds_bytes: int
    fits_48k: bool
    fits_56k: bool
    fits_64k: bool
    sparse_v_tau_level: int
    sparse_v_threshold: str
    fallback_route: str
    supported_shape: str
    route_log_contract: str
    status: str
    next_evidence: str


def row_estimate(d: int) -> RowEstimate:
    if d % QK_TBQ4 != 0:
        raise ValueError("D must be divisible by 128 for TBQ4_0")
    blocks = d // QK_TBQ4
    raw = blocks * TBQ4_BLOCK_BYTES
    padded = (raw + 15) & ~15
    f16 = d * 2
    # q8_0 is 34 bytes / 32 values in ggml; keep approximate row math local.
    q8 = (d // 32) * 34
    return RowEstimate(
        d=d,
        blocks_per_row=blocks,
        tbq4_raw_row_bytes=raw,
        tbq4_padded16_row_bytes=padded,
        f16_row_bytes=f16,
        q8_0_approx_row_bytes=q8,
        tbq4_bpw=8.0 * raw / d,
        tbq4_vs_f16=raw / f16,
    )


def vec_load_estimate(d: int) -> VecLoadEstimate:
    blocks = d // QK_TBQ4
    qbytes = d // 2
    ideal_norm_bytes = blocks * 2

    # Current vec_dot_fattn_vec_KQ_tbq4_0 loads the same fp16 norm once per
    # half2 pair.  For D=128 that is 64 norm loads for one TBQ4 block.
    k_current_norm_bytes = (d // 2) * 2
    k_current = qbytes + k_current_norm_bytes
    k_ideal = qbytes + ideal_norm_bytes

    # Current dequantize_V_tbq4_0<ne=4> loads one norm per 4 values.
    v_current_norm_bytes = (d // 4) * 2
    v_current = qbytes + v_current_norm_bytes
    v_ideal = qbytes + ideal_norm_bytes

    current = k_current + v_current
    ideal = k_ideal + v_ideal
    savings = current - ideal
    return VecLoadEstimate(
        d=d,
        k_current_bytes_per_row=k_current,
        k_ideal_norm_hoist_bytes_per_row=k_ideal,
        k_norm_redundant_bytes_per_row=k_current - k_ideal,
        v_current_bytes_per_row=v_current,
        v_ideal_norm_hoist_bytes_per_row=v_ideal,
        v_norm_redundant_bytes_per_row=v_current - v_ideal,
        kv_current_bytes_per_qk_pair=current,
        kv_ideal_norm_hoist_bytes_per_qk_pair=ideal,
        kv_norm_hoist_savings_per_pair=savings,
        savings_percent=100.0 * savings / current if current else 0.0,
    )


def kq_norm_hoist_estimate(d: int, nthreads_kq: int = 8, cpy_nb: int = 16) -> KqNormHoistEstimate:
    if d % QK_TBQ4 != 0:
        raise ValueError("D must be divisible by 128 for TBQ4_0")
    if WARP_SIZE % nthreads_kq != 0:
        raise ValueError("nthreads_kq must divide WARP_SIZE")
    cpy_ne = cpy_nb // 4
    blocks = d // QK_TBQ4
    qbytes = d // 2
    half2_pairs = d // 2

    # HIP/RDNA TBQ4 K uses Q in registers, so fattn-vec.cuh sets
    # nthreads_KQ = 128 / ggml_cuda_get_max_cpy_bytes() = 8.  Each subwarp
    # lane touches every 128-value TBQ4 block for D=128/256, so a per-thread
    # cache still leaves one norm load per lane per block.  A subwarp broadcast
    # can reduce this to one norm load per block for the active K row.
    current_norm_loads = half2_pairs
    per_thread_norm_loads = blocks * nthreads_kq
    subwarp_norm_loads = blocks
    ideal_norm_loads = blocks

    current = qbytes + current_norm_loads * 2
    per_thread = qbytes + per_thread_norm_loads * 2
    subwarp = qbytes + subwarp_norm_loads * 2
    ideal = qbytes + ideal_norm_loads * 2
    return KqNormHoistEstimate(
        d=d,
        nthreads_kq=nthreads_kq,
        cpy_nb=cpy_nb,
        cpy_ne=cpy_ne,
        subwarp_groups_per_warp=WARP_SIZE // nthreads_kq,
        blocks_per_row=blocks,
        half2_pairs_per_row=half2_pairs,
        current_norm_loads_per_k_row=current_norm_loads,
        per_thread_cache_norm_loads_per_k_row=per_thread_norm_loads,
        subwarp_broadcast_norm_loads_per_k_row=subwarp_norm_loads,
        ideal_norm_loads_per_k_row=ideal_norm_loads,
        current_k_bytes_per_row=current,
        per_thread_cache_k_bytes_per_row=per_thread,
        subwarp_broadcast_k_bytes_per_row=subwarp,
        ideal_k_bytes_per_row=ideal,
        per_thread_cache_savings_percent=100.0 * (current - per_thread) / current if current else 0.0,
        subwarp_broadcast_savings_percent=100.0 * (current - subwarp) / current if current else 0.0,
        ideal_savings_percent=100.0 * (current - ideal) / current if current else 0.0,
        notes="HIP/RDNA TBQ4 VEC KQ ownership model; K-only estimate before V-path changes",
    )


def rotation_estimate(d: int) -> RotationEstimate:
    blocks = d // QK_TBQ4
    fwht128_addsub = blocks * 128 * 7
    fwht128_barriers = blocks * 3
    future32_addsub = blocks * 4 * 32 * 5
    future32_barriers = 0
    return RotationEstimate(
        d=d,
        blocks_per_row=blocks,
        fwht128_addsub_per_row=fwht128_addsub,
        fwht128_barriers_per_row=fwht128_barriers,
        future32_addsub_per_row=future32_addsub,
        future32_barriers_per_row=future32_barriers,
        addsub_savings_percent=100.0 * (fwht128_addsub - future32_addsub) / fwht128_addsub,
    )


def attention_estimate(d: int, nq: int, kv: int, q_heads: int, kv_heads: int, v_tbq4: bool) -> AttentionEstimate:
    row = row_estimate(d)
    vec = vec_load_estimate(d)
    rot = rotation_estimate(d)
    qk_pairs = nq * q_heads * kv
    cache_rows = kv * kv_heads * (2 if v_tbq4 else 1)
    cache_tbq4 = cache_rows * row.tbq4_raw_row_bytes
    cache_f16 = cache_rows * row.f16_row_bytes
    current = qk_pairs * (vec.k_current_bytes_per_row + (vec.v_current_bytes_per_row if v_tbq4 else 0))
    ideal = qk_pairs * (vec.k_ideal_norm_hoist_bytes_per_row + (vec.v_ideal_norm_hoist_bytes_per_row if v_tbq4 else 0))
    q_rotate_rows = nq * q_heads
    out_rotate_rows = q_rotate_rows if v_tbq4 else 0
    return AttentionEstimate(
        nq=nq,
        kv=kv,
        q_heads=q_heads,
        kv_heads=kv_heads,
        gqa_ratio=q_heads / kv_heads if kv_heads else 0.0,
        qk_pairs=qk_pairs,
        cache_tbq4_bytes=cache_tbq4,
        cache_f16_bytes=cache_f16,
        vec_current_logical_kv_bytes=current,
        vec_norm_hoist_logical_kv_bytes=ideal,
        vec_norm_hoist_savings_bytes=current - ideal,
        vec_norm_hoist_savings_percent=100.0 * (current - ideal) / current if current else 0.0,
        q_rotate_rows=q_rotate_rows,
        out_rotate_rows=out_rotate_rows,
        fwht128_barriers_total_if_kv_tbq4=(q_rotate_rows + out_rotate_rows) * rot.fwht128_barriers_per_row,
        future32_barriers_total_if_kv_tbq4=(q_rotate_rows + out_rotate_rows) * rot.future32_barriers_per_row,
    )


def materialization_estimate(d: int, rows: int) -> MaterializationEstimate:
    row = row_estimate(d)
    raw = rows * row.tbq4_padded16_row_bytes
    f16 = rows * row.f16_row_bytes
    total = raw + f16
    return MaterializationEstimate(
        d=d,
        rows=rows,
        raw_staging_bytes=raw,
        f16_tile_bytes=f16,
        total_lhs_bytes=total,
        expansion_vs_raw=total / (rows * row.tbq4_raw_row_bytes),
        fits_64k=total <= 65_536,
    )


def lds_route_estimates(d_values: list[int]) -> list[LdsRouteEstimate]:
    estimates: list[LdsRouteEstimate] = []
    for d in d_values:
        row = row_estimate(d)
        padded = row.tbq4_padded16_row_bytes
        f16 = row.f16_row_bytes
        route_rows = [
            (
                "tbq4_lds_route_a",
                2 * padded,
                True,
                "packed K+V TBQ4 staging in LDS; dequant to registers, no f16 tile stored",
            ),
            (
                "tbq4_lds_route_b",
                padded + f16,
                True,
                "single f16 materialization tile reused K -> V; needs online-softmax state",
            ),
            (
                "tbq4_lds_route_c",
                2 * (padded + f16),
                True,
                "dual K+V f16 materialization; LDS-heavy hold/negative-control route",
            ),
            (
                "tbq4_lds_route_d",
                2 * padded + f16,
                True,
                "ping-pong raw TBQ4 staging plus one f16 tile; barrier/scheduling risk",
            ),
            (
                "f16_mma_fa",
                2 * f16,
                False,
                "f16 K+V comparison route; no compressed KV savings",
            ),
        ]
        for route, bytes_per_row, preserves_cache, notes in route_rows:
            estimates.append(LdsRouteEstimate(
                route=route,
                d=d,
                bytes_per_row=bytes_per_row,
                max_rows_48k=49_152 // bytes_per_row,
                max_rows_56k=57_344 // bytes_per_row,
                max_rows_64k=65_536 // bytes_per_row,
                cache_savings_preserved=preserves_cache,
                notes=notes,
            ))
    return estimates


def lds_experiment_contracts() -> list[LdsExperimentContract]:
    """Concrete Stage-1 contracts from the LDS research report.

    These rows do not enable runtime behavior. They make the env gate, tile
    shape, LDS footprint, fallback, and sparse-V policy machine-checkable before
    Stage 2 adds a kernel branch.
    """
    row = row_estimate(128)
    packed_stride = row.tbq4_padded16_row_bytes
    f16_stride_half2 = 66
    f16_stride_bytes = f16_stride_half2 * 4

    def make(
            route: str,
            base_route: str,
            role: str,
            env_value: str,
            tile_rows: int,
            packed_stages: int,
            status: str,
            next_evidence: str) -> LdsExperimentContract:
        lds_bytes_per_row = packed_stages * packed_stride + f16_stride_bytes
        total = tile_rows * lds_bytes_per_row
        packed_part = f" packed_stride={packed_stride}" if packed_stages else " packed_stride=direct_global"
        return LdsExperimentContract(
            route=route,
            base_route=base_route,
            stage=1,
            role=role,
            env_var="GGML_CUDA_TBQ4_LDS_ROUTE",
            env_value=env_value,
            d=128,
            tile_rows=tile_rows,
            packed_stages=packed_stages,
            packed_stride_bytes=packed_stride,
            f16_stride_half2=f16_stride_half2,
            f16_stride_bytes=f16_stride_bytes,
            lds_bytes_per_row=lds_bytes_per_row,
            total_lds_bytes=total,
            fits_48k=total <= 49_152,
            fits_56k=total <= 57_344,
            fits_64k=total <= 65_536,
            sparse_v_tau_level=0,
            sparse_v_threshold="1e-6",
            fallback_route=PRODUCTION_ROUTE_LABEL,
            supported_shape="D=128 K=TBQ4_0 K-only; env absent or unsupported shape must fall back",
            route_log_contract=f"route={route} env=GGML_CUDA_TBQ4_LDS_ROUTE={env_value} d=128 tile_rows={tile_rows} f16_stride={f16_stride_half2}{packed_part} sparse_v_tau_level=0 fallback={PRODUCTION_ROUTE_LABEL}",
            status=status,
            next_evidence=next_evidence,
        )

    return [
        make(
            route="tbq4_lds_route_d_k_d128",
            base_route="tbq4_lds_route_d",
            role="first_experiment",
            env_value="D_K",
            tile_rows=96,
            packed_stages=2,
            status="stage1_contract_locked",
            next_evidence="Stage 2: implement default-off D128 K-only route and prove fallback/route logs before canaries",
        ),
        make(
            route="tbq4_lds_route_b_k_d128",
            base_route="tbq4_lds_route_b",
            role="backup_diagnostic",
            env_value="B_K",
            tile_rows=96,
            packed_stages=0,
            status="backup_contract_locked",
            next_evidence="Use only if D-lite fails from complexity or needs a simpler f16-materialization diagnostic",
        ),
        make(
            route="tbq4_lds_route_b_k_d128_rows128",
            base_route="tbq4_lds_route_b",
            role="backup_diagnostic_larger_tile",
            env_value="B_K",
            tile_rows=128,
            packed_stages=0,
            status="backup_contract_locked",
            next_evidence="Try after rows96 if correctness is clean and profiler says LDS budget/occupancy remain safe",
        ),
    ]


def gate_decisions(d_values: list[int], tile_rows: list[int]) -> list[GateDecision]:
    decisions: list[GateDecision] = []
    for d in d_values:
        vec = vec_load_estimate(d)
        kq = kq_norm_hoist_estimate(d)
        mats = [materialization_estimate(d, rows) for rows in tile_rows]
        fit_rows = [m.rows for m in mats if m.fits_64k]
        max_fit = max(fit_rows) if fit_rows else 0
        decisions.extend([
            GateDecision(
                route=PRODUCTION_ROUTE_LABEL,
                d=d,
                decision="baseline",
                priority=0,
                reason="current production-compatible TBQ4_0 VEC route",
                required_next_evidence="keep as fallback for every experiment",
            ),
            GateDecision(
                route="tbq4_vec_norm_hoist",
                d=d,
                decision="next_candidate",
                priority=1,
                reason=f"same TBQ4_0 format; KQ-only model saves {kq.per_thread_cache_savings_percent:.1f}% with per-thread cache or {kq.subwarp_broadcast_savings_percent:.1f}% with subwarp broadcast before later V work (full K+V ideal {vec.savings_percent:.1f}%)",
                required_next_evidence="design/code review of norm load ownership, then correctness canaries before benchmarks",
            ),
            *([
                GateDecision(
                    route="tbq4_lds_route_d_k_d128",
                    d=d,
                    decision="stage1_first_experiment",
                    priority=2,
                    reason="research report recommends Route D-lite D128 K-only: ping-pong packed TBQ4 K staging plus one padded f16/half2 K tile, V unchanged",
                    required_next_evidence="Stage 2 default-off implementation for GGML_CUDA_TBQ4_LDS_ROUTE=D_K with route log and fallback proof",
                ),
                GateDecision(
                    route="tbq4_lds_route_b_k_d128",
                    d=d,
                    decision="stage1_backup_diagnostic",
                    priority=3,
                    reason="backup B-lite D128 K-only isolates whether direct f16 K materialization helps without ping-pong complexity",
                    required_next_evidence="implement only if D-lite fails from complexity or profiler needs simpler diagnostic comparison",
                ),
            ] if d == 128 else []),
            GateDecision(
                route="tbq4_lds_route_a",
                d=d,
                decision="model_before_code",
                priority=4,
                reason="packed TBQ4 LDS staging may preserve cache savings but register pressure is not modeled yet",
                required_next_evidence="add LDS/register gate rows for packed staging plus VGPR estimate",
            ),
            GateDecision(
                route="tbq4_lds_route_b",
                d=d,
                decision="model_before_code",
                priority=5,
                reason=f"single f16 materialization can fit up to {max_fit} rows under 64KiB in this simple model, but scratch/padding/occupancy are not modeled yet",
                required_next_evidence="generic route only; Stage-1 concrete backup is tbq4_lds_route_b_k_d128",
            ),
            GateDecision(
                route="tbq4_lds_route_c",
                d=d,
                decision="hold_negative_control",
                priority=9,
                reason="dual K+V f16 materialization is LDS-heavy and previous TBQ4 WMMA/materialized sweep was slower",
                required_next_evidence="only revisit after Route A/B/D evidence or as a bounded negative-control probe",
            ),
            GateDecision(
                route="tbq4_lds_route_d",
                d=d,
                decision="model_before_code",
                priority=6,
                reason="raw ping-pong plus one f16 tile may overlap loads without doubling f16 LDS cost, but scheduling/barrier cost is unknown",
                required_next_evidence="generic route only; Stage-1 concrete first experiment is tbq4_lds_route_d_k_d128",
            ),
            GateDecision(
                route="f16_mma_fa",
                d=d,
                decision="comparison_only",
                priority=8,
                reason="useful tensor-core FA ceiling comparison, but loses compressed KV savings",
                required_next_evidence="benchmark only as reference; do not promote as compressed-KV optimization",
            ),
            GateDecision(
                route=FUTURE_FORMAT_LABEL,
                d=d,
                decision="future_format_only",
                priority=99,
                reason="future-format-only and incompatible with current GGUF/cache rows",
                required_next_evidence="separate format contract, quantizer/converter, and quality canaries",
            ),
        ])
    return decisions


def fmt_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n/1024**3:.2f} GiB"
    if n >= 1024**2:
        return f"{n/1024**2:.2f} MiB"
    if n >= 1024:
        return f"{n/1024:.1f} KiB"
    return f"{n} B"


def print_markdown(args: argparse.Namespace) -> None:
    rows = [row_estimate(d) for d in args.d]
    vecs = [vec_load_estimate(d) for d in args.d]
    kqs = [kq_norm_hoist_estimate(d, args.nthreads_kq, args.cpy_nb) for d in args.d]
    rots = [rotation_estimate(d) for d in args.d]
    mats = [materialization_estimate(d, r) for d in args.d for r in args.tile_rows]
    attn = attention_estimate(args.d[0], args.nq, args.kv, args.q_heads, args.kv_heads, args.v_tbq4)

    print("# TBQ4 FlashAttention calculator")
    print()
    print("Offline planning model; values are estimates from kernel constants, not benchmark results.")
    print()
    print("## Scope labels")
    print("| label | status | rule |")
    print("|---|---|---|")
    print(f"| `{CURRENT_FORMAT_LABEL}` | current format | Production-compatible compressed KV cache format. Near-term optimizations must preserve this contract. |")
    print(f"| `{PRODUCTION_ROUTE_LABEL}` | current default route | Baseline VEC FlashAttention route; do not replace without env-gated canaries + sweep evidence. |")
    print("| `tbq4_vec_norm_hoist` | proposed current-format optimization | Same TBQ4_0 data contract; hoist repeated norm loads before trying heavier LDS/WMMA paths. |")
    print("| `tbq4_lds_route_a/b/c/d` | proposed env-gated experiments | LDS materialization/staging experiments only; default-off until benchmark and coherence gates pass. |")
    print(f"| `{FUTURE_FORMAT_LABEL}` | future-format research placeholder | Not TBQ4_0, not a runtime flag, and not compatible with current GGUF/cache rows. Exact future-format label stays in harness context. |")
    print()
    print("## Row storage")
    print("| D | blocks | TBQ4 raw row | TBQ4 padded16 | f16 row | q8_0 approx row | TBQ4 bpw | TBQ4/f16 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r.d} | {r.blocks_per_row} | {r.tbq4_raw_row_bytes} | {r.tbq4_padded16_row_bytes} | {r.f16_row_bytes} | {r.q8_0_approx_row_bytes} | {r.tbq4_bpw:.3f} | {r.tbq4_vs_f16:.3f} |")

    print("\n## Current VEC dequant load model")
    print("Current KQ TBQ4 VEC loads the block norm once per half2 pair; norm-hoist means one norm per 128-value TBQ4 block.")
    print("| D | K current B/row | K hoisted B/row | V current B/row | V hoisted B/row | K+V current | K+V hoisted | savings |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in vecs:
        print(f"| {v.d} | {v.k_current_bytes_per_row} | {v.k_ideal_norm_hoist_bytes_per_row} | {v.v_current_bytes_per_row} | {v.v_ideal_norm_hoist_bytes_per_row} | {v.kv_current_bytes_per_qk_pair} | {v.kv_ideal_norm_hoist_bytes_per_qk_pair} | {v.savings_percent:.1f}% |")

    print("\n## KQ norm-hoist ownership model")
    print("HIP/RDNA TBQ4 K uses Q in registers and currently maps `nthreads_KQ = 128 / cpy_nb`; defaults here are cpy_nb=16, nthreads_KQ=8.")
    print("Per-thread cache is safer but leaves one norm load per lane per TBQ4 block; subwarp broadcast targets one norm load per block for the active K row.")
    print("| D | nthreads_KQ | cpy_ne | blocks | current norm loads | per-thread norm loads | subwarp norm loads | current K B | per-thread K B | subwarp K B | ideal K B | per-thread savings | subwarp savings |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for kq in kqs:
        print(f"| {kq.d} | {kq.nthreads_kq} | {kq.cpy_ne} | {kq.blocks_per_row} | {kq.current_norm_loads_per_k_row} | {kq.per_thread_cache_norm_loads_per_k_row} | {kq.subwarp_broadcast_norm_loads_per_k_row} | {kq.current_k_bytes_per_row} | {kq.per_thread_cache_k_bytes_per_row} | {kq.subwarp_broadcast_k_bytes_per_row} | {kq.ideal_k_bytes_per_row} | {kq.per_thread_cache_savings_percent:.1f}% | {kq.subwarp_broadcast_savings_percent:.1f}% |")

    print("\n## Rotation model")
    print("Future 4x32 transform estimates are format-incompatible with current TBQ4 GGUFs; exact future-format labels stay in harness context.")
    print("| D | FWHT128 add/sub | FWHT128 barriers | future32 add/sub | future32 barriers | add/sub savings |")
    print("|---:|---:|---:|---:|---:|---:|")
    for r in rots:
        print(f"| {r.d} | {r.fwht128_addsub_per_row} | {r.fwht128_barriers_per_row} | {r.future32_addsub_per_row} | {r.future32_barriers_per_row} | {r.addsub_savings_percent:.1f}% |")

    print("\n## Materialized tile / WMMA expansion")
    print("Shows why direct TBQ4 WMMA must be carefully gated: compressed rows expand to f16 tiles in LDS.")
    print("| D | rows | raw staging | f16 tile | total LDS-ish | expansion vs raw | fits 64KiB |")
    print("|---:|---:|---:|---:|---:|---:|:---:|")
    for m in mats:
        print(f"| {m.d} | {m.rows} | {m.raw_staging_bytes} | {m.f16_tile_bytes} | {m.total_lhs_bytes} | {m.expansion_vs_raw:.2f}x | {'yes' if m.fits_64k else 'NO'} |")

    print("\n## LDS route gates")
    print("Rows are max resident tile rows under practical 48KiB/56KiB budgets and the hard 64KiB limit. Scratch, padding, and VGPR pressure still need route-specific validation.")
    print("| route | D | bytes/row | rows @48KiB | rows @56KiB | rows @64KiB | preserves compressed KV | notes |")
    print("|---|---:|---:|---:|---:|---:|:---:|---|")
    for lds in lds_route_estimates(args.d):
        print(f"| `{lds.route}` | {lds.d} | {lds.bytes_per_row} | {lds.max_rows_48k} | {lds.max_rows_56k} | {lds.max_rows_64k} | {'yes' if lds.cache_savings_preserved else 'no'} | {lds.notes} |")

    print("\n## Stage 1 LDS experiment contracts")
    print("Concrete route contracts from `tbq4-lds-research-report.md`. These are planning rows only: no runtime behavior changes until Stage 2.")
    print("| route | role | env | D | tile rows | packed stages | packed stride | f16 stride half2 | LDS bytes/row | total LDS | fits 48/56/64KiB | sparse tau | fallback | status |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|---|---|")
    for c in lds_experiment_contracts():
        fits = f"{'yes' if c.fits_48k else 'no'}/{'yes' if c.fits_56k else 'no'}/{'yes' if c.fits_64k else 'no'}"
        print(f"| `{c.route}` | {c.role} | `{c.env_var}={c.env_value}` | {c.d} | {c.tile_rows} | {c.packed_stages} | {c.packed_stride_bytes} | {c.f16_stride_half2} | {c.lds_bytes_per_row} | {fmt_bytes(c.total_lds_bytes)} | {fits} | tau{c.sparse_v_tau_level} `{c.sparse_v_threshold}` | `{c.fallback_route}` | {c.status} |")
    print("\nRoute log contracts:")
    for c in lds_experiment_contracts():
        print(f"- `{c.route_log_contract}`")

    print("\n## FlashAttention route model")
    print("This includes the newer FA route families as calculator lanes, even when they stay env-gated.")
    print("| route | current TBQ4_0 compatible | cache savings | estimated target | calculator gate | status |")
    print("|---|:---:|:---:|---|---|---|")
    for d in args.d:
        row = row_estimate(d)
        vec = vec_load_estimate(d)
        fit_rows = [m.rows for m in mats if m.d == d and m.fits_64k]
        max_fit = max(fit_rows) if fit_rows else 0
        print(f"| VEC inline D{d} | yes | yes | baseline; K+V logical load {vec.kv_current_bytes_per_qk_pair} B/pair | none | production default |")
        print(f"| VEC norm-hoist D{d} | yes | yes | reduce repeated norm loads; model saves {vec.savings_percent:.1f}% of K+V TBQ4 dequant load bytes | no format change | best first optimization candidate |")
        if d == 128:
            print("| LDS D-lite K-only D128 | yes | yes | first Stage-1 contract: ping-pong packed TBQ4 K staging + padded f16 K tile; V unchanged | `GGML_CUDA_TBQ4_LDS_ROUTE=D_K`, tile_rows=96, f16_stride=66 | Stage 2 implementation target; default-off |")
            print("| LDS B-lite K-only D128 | yes | yes | backup diagnostic: direct TBQ4 K -> padded f16 K tile without ping-pong | `GGML_CUDA_TBQ4_LDS_ROUTE=B_K`, tile_rows=96/128, f16_stride=66 | backup only if D-lite needs diagnostic fallback |")
        print(f"| TBQ4 WMMA/materialized FA D{d} | yes | yes | new FA path with f16 tile materialization | max materialized rows within 64KiB: {max_fit} | env-gated; previous sweep slower |")
        print(f"| f16 MMA FA D{d} | no | no | compare tensor-core FA ceiling with f16 KV | f16 row {row.f16_row_bytes} B vs TBQ4 row {row.tbq4_raw_row_bytes} B | benchmark comparison only |")
        print(f"| {FUTURE_FORMAT_LABEL} VEC D{d} | no | yes | estimate future 4x32 transform upside, {rotation_estimate(d).addsub_savings_percent:.1f}% fewer add/sub ops | new format + quality canaries | harness-only future format experiment |")

    print("\n## Calculator decision gate")
    print("This table is the current routing gate. Lower priority means earlier work; `baseline` remains fallback.")
    print("| priority | route | D | decision | reason | next evidence |")
    print("|---:|---|---:|---|---|---|")
    for g in sorted(gate_decisions(args.d, args.tile_rows), key=lambda item: (item.priority, item.d, item.route)):
        print(f"| {g.priority} | `{g.route}` | {g.d} | {g.decision} | {g.reason} | {g.required_next_evidence} |")

    print("\n## Request-scale estimate")
    print(f"Using D={args.d[0]}, nq={args.nq}, kv={args.kv}, q_heads={args.q_heads}, kv_heads={args.kv_heads}, V_TBQ4={args.v_tbq4}.")
    print("| metric | value |")
    print("|---|---:|")
    for key, value in asdict(attn).items():
        if key.endswith("bytes"):
            shown = fmt_bytes(value)
        else:
            shown = f"{value:.3f}" if isinstance(value, float) else str(value)
        print(f"| {key} | {shown} |")

    print("\n## Stage execution hypotheses")
    print("1. **Stage 1 contract gate**: calculator rows lock `tbq4_lds_route_d_k_d128` as first experiment and `tbq4_lds_route_b_k_d128` as backup; no runtime behavior changes.")
    print("2. **Stage 2 D-lite implementation**: add `GGML_CUDA_TBQ4_LDS_ROUTE=D_K` for D128 K-only with fallback to `tbq4_vec` for env-absent/unsupported shapes.")
    print("3. **Stage 3 canaries before benchmarks**: route log, tiny decode, OOB/tile-tail, GQA/mask/KV_max, sparse-V tau0 unchanged, NIAH/coherence.")
    print("4. **Stage 4 profiler-driven iteration**: compare against `tbq4_vec`, `q8k_tbq4v_sparsev` tau0, and opt-in norm-hoist; sweep tile rows and f16 stride.")
    print("5. **Stage 5 expansion only after D128 success**: D256 then optional V integration with sparse-V tau0 pre-dequant skip; Route C remains negative-control only.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", default="128,256", help="comma-separated head dimensions; TBQ4 requires multiples of 128")
    p.add_argument("--nq", type=int, default=512, help="query rows/tokens in the attention call")
    p.add_argument("--kv", type=int, default=4096, help="KV rows/context length")
    p.add_argument("--q-heads", type=int, default=1)
    p.add_argument("--kv-heads", type=int, default=1)
    p.add_argument("--v-tbq4", action="store_true", default=True, help="model V as TBQ4 too; default true")
    p.add_argument("--tile-rows", default="16,32,64,128,256", help="comma-separated materialized KV tile rows")
    p.add_argument("--nthreads-kq", type=int, default=8, help="KQ subwarp lanes for TBQ4 VEC ownership estimates; HIP/RDNA default is 8")
    p.add_argument("--cpy-nb", type=int, default=16, help="bytes per vector copy for KQ ownership estimates; HIP default is 16")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    args.d = [int(x.strip()) for x in args.d.split(",") if x.strip()]
    args.tile_rows = [int(x.strip()) for x in args.tile_rows.split(",") if x.strip()]

    data = {
        "scope": {
            "current_format_label": CURRENT_FORMAT_LABEL,
            "future_format_label": FUTURE_FORMAT_LABEL,
            "production_route_label": PRODUCTION_ROUTE_LABEL,
            "rules": [
                "TBQ4_0 is the only current-format target for near-term runtime work.",
                "Future-format labels are harness-only research labels, not TBQ4_0 routes or runtime flags.",
                "Experimental LDS/materialized routes must stay default-off and env-gated until canaries and sweeps pass.",
                "Stage 1 LDS contract rows do not change runtime behavior.",
                "Sparse-V remains fixed at tau0 (GGML_CUDA_SPARSE_V_TAU_LEVEL=0, threshold 1e-6) during LDS Stage 1/2.",
            ],
        },
        "row_estimates": [asdict(row_estimate(d)) for d in args.d],
        "vec_load_estimates": [asdict(vec_load_estimate(d)) for d in args.d],
        "kq_norm_hoist_estimates": [asdict(kq_norm_hoist_estimate(d, args.nthreads_kq, args.cpy_nb)) for d in args.d],
        "rotation_estimates": [asdict(rotation_estimate(d)) for d in args.d],
        "materialization_estimates": [asdict(materialization_estimate(d, r)) for d in args.d for r in args.tile_rows],
        "lds_route_estimates": [asdict(lds) for lds in lds_route_estimates(args.d)],
        "lds_experiment_contracts": [asdict(c) for c in lds_experiment_contracts()],
        "gate_decisions": [asdict(g) for g in gate_decisions(args.d, args.tile_rows)],
        "attention_estimate": asdict(attention_estimate(args.d[0], args.nq, args.kv, args.q_heads, args.kv_heads, args.v_tbq4)),
    }
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print_markdown(args)


if __name__ == "__main__":
    main()
