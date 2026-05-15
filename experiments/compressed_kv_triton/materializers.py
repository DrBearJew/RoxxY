#!/usr/bin/env python3
"""Materializer-only Triton kernels for compressed KV formats.

The kernels here deliberately do one thing: materialize a logical compressed KV
row tile as f16 values. They do not integrate with llama.cpp and are not wired
into production CMake.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl

REPO_ROOT = Path(__file__).resolve().parents[2]
TBQ4_HEADER = REPO_ROOT / "ggml/src/ggml-cuda/tbq4-cuda.cuh"
PLANAR_ISO_HEADER = REPO_ROOT / "ggml/src/ggml-cuda/planar-iso-constants.cuh"

QK = 128


def _parse_float_array(path: Path, name: str) -> torch.Tensor:
    text = path.read_text()
    match = re.search(rf"{re.escape(name)}\s*\[[^\]]+\]\s*=\s*\{{(.*?)\}};", text, re.S)
    if not match:
        raise ValueError(f"array {name!r} not found in {path}")
    vals = [float(x.rstrip("f")) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?f?", match.group(1))]
    return torch.tensor(vals, dtype=torch.float32)


def load_constants(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    arrays = {
        "tbq4_centroids": _parse_float_array(TBQ4_HEADER, "d_tbq4_centroids"),
        "pi_centroids_3bit": _parse_float_array(PLANAR_ISO_HEADER, "PI_CENTROIDS_3BIT"),
        "pi_cos": _parse_float_array(PLANAR_ISO_HEADER, "PI_COS"),
        "pi_sin": _parse_float_array(PLANAR_ISO_HEADER, "PI_SIN"),
        "pi_qw": _parse_float_array(PLANAR_ISO_HEADER, "PI_QW"),
        "pi_qx": _parse_float_array(PLANAR_ISO_HEADER, "PI_QX"),
        "pi_qy": _parse_float_array(PLANAR_ISO_HEADER, "PI_QY"),
        "pi_qz": _parse_float_array(PLANAR_ISO_HEADER, "PI_QZ"),
    }
    return {k: v.to(device) for k, v in arrays.items()}


def synthetic_inputs(fmt: str, rows: int, d_head: int, *, seed: int = 1234) -> dict[str, torch.Tensor]:
    if d_head % QK != 0:
        raise ValueError(f"d_head must be a multiple of {QK}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    blocks = d_head // QK
    d = (torch.rand((rows, blocks), generator=gen, dtype=torch.float32) * 2.0 + 0.125).half()
    if fmt == "tbq4_0":
        return {
            "d": d,
            "qs": torch.randint(0, 256, (rows, blocks, 64), generator=gen, dtype=torch.uint8),
        }
    if fmt in {"planar3_0", "iso3_0"}:
        return {
            "d": d,
            "qs": torch.randint(0, 256, (rows, blocks, 32), generator=gen, dtype=torch.uint8),
            "signs": torch.randint(0, 256, (rows, blocks, 16), generator=gen, dtype=torch.uint8),
        }
    raise ValueError(f"unsupported synthetic format {fmt}")


def materialize_tbq4_ref(d: torch.Tensor, qs: torch.Tensor, constants: dict[str, torch.Tensor]) -> torch.Tensor:
    rows, blocks, _ = qs.shape
    centroids = constants["tbq4_centroids"].cpu()
    out = torch.empty((rows, blocks * QK), dtype=torch.float32)
    for row in range(rows):
        for block in range(blocks):
            norm = float(d[row, block])
            for j in range(QK):
                byte = int(qs[row, block, j // 2])
                idx = (byte >> 4) & 0xF if j & 1 else byte & 0xF
                out[row, block * QK + j] = float(centroids[idx]) * norm
    return out


def _unpack3(qs: torch.Tensor, signs: torch.Tensor, row: int, block: int, j: int) -> int:
    low = (int(qs[row, block, j // 4]) >> ((j % 4) * 2)) & 0x3
    hi = (int(signs[row, block, j // 8]) >> (j % 8)) & 0x1
    return low | (hi << 2)


def materialize_planar3_ref(d: torch.Tensor, qs: torch.Tensor, signs: torch.Tensor, constants: dict[str, torch.Tensor]) -> torch.Tensor:
    rows, blocks, _ = qs.shape
    centroids = constants["pi_centroids_3bit"].cpu()
    cos = constants["pi_cos"].cpu()
    sin = constants["pi_sin"].cpu()
    out = torch.empty((rows, blocks * QK), dtype=torch.float32)
    for row in range(rows):
        for block in range(blocks):
            norm = float(d[row, block])
            for j in range(QK):
                j_pair = j & ~1
                q0 = float(centroids[_unpack3(qs, signs, row, block, j_pair + 0)])
                q1 = float(centroids[_unpack3(qs, signs, row, block, j_pair + 1)])
                p = ((block * QK + j_pair) // 2) & 63
                c = float(cos[p])
                s = float(sin[p])
                val = (-s * q0 + c * q1) if (j & 1) else (c * q0 + s * q1)
                out[row, block * QK + j] = val * norm
    return out


def materialize_iso3_ref(d: torch.Tensor, qs: torch.Tensor, signs: torch.Tensor, constants: dict[str, torch.Tensor]) -> torch.Tensor:
    rows, blocks, _ = qs.shape
    centroids = constants["pi_centroids_3bit"].cpu()
    qw_arr = constants["pi_qw"].cpu()
    qx_arr = constants["pi_qx"].cpu()
    qy_arr = constants["pi_qy"].cpu()
    qz_arr = constants["pi_qz"].cpu()
    out = torch.empty((rows, blocks * QK), dtype=torch.float32)
    for row in range(rows):
        for block in range(blocks):
            norm = float(d[row, block])
            for j in range(QK):
                j_group = j & ~3
                qvals = [float(centroids[_unpack3(qs, signs, row, block, j_group + c)]) for c in range(4)]
                g = ((block * QK + j_group) // 4) & 31
                qw = float(qw_arr[g])
                qx = -float(qx_arr[g])
                qy = -float(qy_arr[g])
                qz = -float(qz_arr[g])
                rw = qw * qvals[0] - qx * qvals[1] - qy * qvals[2] - qz * qvals[3]
                rx = qw * qvals[1] + qx * qvals[0] + qy * qvals[3] - qz * qvals[2]
                ry = qw * qvals[2] - qx * qvals[3] + qy * qvals[0] + qz * qvals[1]
                rz = qw * qvals[3] + qx * qvals[2] - qy * qvals[1] + qz * qvals[0]
                out[row, block * QK + j] = [rw, rx, ry, rz][j & 3] * norm
    return out


@triton.jit
def materialize_tbq4_kernel(d, qs, centroids, out, D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    blocks_per_row: tl.constexpr = D // 128
    block = offs // 128
    j = offs - block * 128
    byte = tl.load(qs + row * blocks_per_row * 64 + block * 64 + j // 2, mask=mask, other=0).to(tl.int32)
    idx = tl.where((j & 1) == 1, (byte >> 4) & 0xF, byte & 0xF)
    norm = tl.load(d + row * blocks_per_row + block, mask=mask, other=0.0).to(tl.float32)
    val = tl.load(centroids + idx, mask=mask, other=0.0) * norm
    tl.store(out + row * D + offs, val, mask=mask)


@triton.jit
def _unpack3_tl(qs, signs, row, blocks_per_row: tl.constexpr, block, j):
    low_byte = tl.load(qs + row * blocks_per_row * 32 + block * 32 + j // 4).to(tl.int32)
    sign_byte = tl.load(signs + row * blocks_per_row * 16 + block * 16 + j // 8).to(tl.int32)
    low = (low_byte >> ((j & 3) * 2)) & 0x3
    hi = (sign_byte >> (j & 7)) & 0x1
    return low | (hi << 2)


@triton.jit
def materialize_planar3_kernel(d, qs, signs, centroids, cos, sin, out, D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    blocks_per_row: tl.constexpr = D // 128
    block = offs // 128
    j = offs - block * 128
    j_pair = (j // 2) * 2
    idx0 = _unpack3_tl(qs, signs, row, blocks_per_row, block, j_pair + 0)
    idx1 = _unpack3_tl(qs, signs, row, blocks_per_row, block, j_pair + 1)
    q0 = tl.load(centroids + idx0, mask=mask, other=0.0)
    q1 = tl.load(centroids + idx1, mask=mask, other=0.0)
    p = ((block * 128 + j_pair) // 2) & 63
    c = tl.load(cos + p, mask=mask, other=0.0)
    s = tl.load(sin + p, mask=mask, other=0.0)
    norm = tl.load(d + row * blocks_per_row + block, mask=mask, other=0.0).to(tl.float32)
    even = c * q0 + s * q1
    odd = -s * q0 + c * q1
    val = tl.where((j & 1) == 1, odd, even) * norm
    tl.store(out + row * D + offs, val, mask=mask)


@triton.jit
def materialize_iso3_kernel(d, qs, signs, centroids, qw_ptr, qx_ptr, qy_ptr, qz_ptr, out, D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    blocks_per_row: tl.constexpr = D // 128
    block = offs // 128
    j = offs - block * 128
    j_group = (j // 4) * 4
    idx0 = _unpack3_tl(qs, signs, row, blocks_per_row, block, j_group + 0)
    idx1 = _unpack3_tl(qs, signs, row, blocks_per_row, block, j_group + 1)
    idx2 = _unpack3_tl(qs, signs, row, blocks_per_row, block, j_group + 2)
    idx3 = _unpack3_tl(qs, signs, row, blocks_per_row, block, j_group + 3)
    v0 = tl.load(centroids + idx0, mask=mask, other=0.0)
    v1 = tl.load(centroids + idx1, mask=mask, other=0.0)
    v2 = tl.load(centroids + idx2, mask=mask, other=0.0)
    v3 = tl.load(centroids + idx3, mask=mask, other=0.0)
    g = ((block * 128 + j_group) // 4) & 31
    qw = tl.load(qw_ptr + g, mask=mask, other=0.0)
    qx = -tl.load(qx_ptr + g, mask=mask, other=0.0)
    qy = -tl.load(qy_ptr + g, mask=mask, other=0.0)
    qz = -tl.load(qz_ptr + g, mask=mask, other=0.0)
    rw = qw * v0 - qx * v1 - qy * v2 - qz * v3
    rx = qw * v1 + qx * v0 + qy * v3 - qz * v2
    ry = qw * v2 - qx * v3 + qy * v0 + qz * v1
    rz = qw * v3 + qx * v2 - qy * v1 + qz * v0
    off = j & 3
    val = tl.where(off == 0, rw, tl.where(off == 1, rx, tl.where(off == 2, ry, rz)))
    norm = tl.load(d + row * blocks_per_row + block, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + row * D + offs, val * norm, mask=mask)


def _launch_materializer(fmt: str, inputs_cpu: dict[str, torch.Tensor], constants_cpu: dict[str, torch.Tensor], d_head: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    constants = {k: v.to(device) for k, v in constants_cpu.items()}
    rows = inputs_cpu["d"].shape[0]
    d_gpu = inputs_cpu["d"].contiguous().to(device)
    out = torch.empty((rows, d_head), dtype=torch.float16, device=device)
    block_d = triton.next_power_of_2(d_head)
    grid = (rows,)

    if fmt == "tbq4_0":
        qs_gpu = inputs_cpu["qs"].contiguous().to(device)
        materialize_tbq4_kernel[grid](d_gpu, qs_gpu, constants["tbq4_centroids"], out, D=d_head, BLOCK_D=block_d)
        ref = materialize_tbq4_ref(inputs_cpu["d"], inputs_cpu["qs"], constants_cpu)
    elif fmt == "planar3_0":
        qs_gpu = inputs_cpu["qs"].contiguous().to(device)
        signs_gpu = inputs_cpu["signs"].contiguous().to(device)
        materialize_planar3_kernel[grid](d_gpu, qs_gpu, signs_gpu, constants["pi_centroids_3bit"], constants["pi_cos"], constants["pi_sin"], out, D=d_head, BLOCK_D=block_d)
        ref = materialize_planar3_ref(inputs_cpu["d"], inputs_cpu["qs"], inputs_cpu["signs"], constants_cpu)
    elif fmt == "iso3_0":
        qs_gpu = inputs_cpu["qs"].contiguous().to(device)
        signs_gpu = inputs_cpu["signs"].contiguous().to(device)
        materialize_iso3_kernel[grid](d_gpu, qs_gpu, signs_gpu, constants["pi_centroids_3bit"], constants["pi_qw"], constants["pi_qx"], constants["pi_qy"], constants["pi_qz"], out, D=d_head, BLOCK_D=block_d)
        ref = materialize_iso3_ref(inputs_cpu["d"], inputs_cpu["qs"], inputs_cpu["signs"], constants_cpu)
    else:
        raise ValueError(fmt)

    torch.cuda.synchronize()
    return out.float().cpu(), ref.half().float()


def run_materializer_checks(rows: int = 17, dims: tuple[int, ...] = (128, 256)) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Triton materializer checks require a HIP/CUDA-visible torch device")
    constants_cpu = load_constants("cpu")
    results: list[dict[str, object]] = []
    for d_head in dims:
        for fmt in ("planar3_0", "iso3_0", "tbq4_0"):
            inputs = synthetic_inputs(fmt, rows, d_head, seed=1000 + d_head + len(fmt))
            got, ref = _launch_materializer(fmt, inputs, constants_cpu, d_head)
            diff = (got - ref).abs()
            max_abs = float(diff.max().item())
            # Half output plus non-associative float expression ordering leaves a
            # small tolerance for Planar/Iso rotations while still catching index
            # and unpack errors immediately.
            tol = 1.5e-3
            passed = max_abs <= tol
            results.append({"format": fmt, "D": d_head, "rows": rows, "max_abs_err": max_abs, "tol": tol, "passed": passed})
    return {"result": "PASS" if all(r["passed"] for r in results) else "FAIL", "checks": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=17)
    parser.add_argument("--dims", type=int, nargs="+", default=[128, 256])
    args = parser.parse_args()
    report = run_materializer_checks(rows=args.rows, dims=tuple(args.dims))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
