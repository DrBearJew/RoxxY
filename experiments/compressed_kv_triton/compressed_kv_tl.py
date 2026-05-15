#!/usr/bin/env python3
"""Shared Triton helpers for compressed-KV experiment prototypes.

The helpers are intentionally small and format-specific. They model the contract
we want before touching production C++: callers provide a physical compressed row
index and dimension offsets; the format helper returns f16-rounded f32 values.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

FORMATS = ("planar3_0", "iso3_0", "tbq4_0")
QK = 128


@triton.jit
def paged_physical_rows(block_tables, seq_idx, logical_rows, row_mask, block_table_stride, BLOCK_SIZE: tl.constexpr):
    block_ids = logical_rows // BLOCK_SIZE
    in_block = logical_rows - block_ids * BLOCK_SIZE
    physical_block = tl.load(
        block_tables + seq_idx * block_table_stride + block_ids,
        mask=row_mask,
        other=0,
    ).to(tl.int64)
    return physical_block * BLOCK_SIZE + in_block


@triton.jit
def unpack3_values(qs, signs, physical_row, blocks_per_row: tl.constexpr, block, j, mask):
    low_byte = tl.load(
        qs + physical_row * blocks_per_row * 32 + block * 32 + j // 4,
        mask=mask,
        other=0,
    ).to(tl.int32)
    sign_byte = tl.load(
        signs + physical_row * blocks_per_row * 16 + block * 16 + j // 8,
        mask=mask,
        other=0,
    ).to(tl.int32)
    low = (low_byte >> ((j & 3) * 2)) & 0x3
    hi = (sign_byte >> (j & 7)) & 0x1
    return low | (hi << 2)


@triton.jit
def tbq4_values(d, qs, centroids, physical_row, offs, mask, D: tl.constexpr):
    blocks_per_row: tl.constexpr = D // 128
    block = offs // 128
    j = offs - block * 128
    byte = tl.load(
        qs + physical_row * blocks_per_row * 64 + block * 64 + j // 2,
        mask=mask,
        other=0,
    ).to(tl.int32)
    idx = tl.where((j & 1) == 1, (byte >> 4) & 0xF, byte & 0xF)
    norm = tl.load(
        d + physical_row * blocks_per_row + block,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    return (tl.load(centroids + idx, mask=mask, other=0.0) * norm).to(tl.float16).to(tl.float32)


@triton.jit
def planar3_values(d, qs, signs, centroids, cos, sin, physical_row, offs, mask, D: tl.constexpr):
    blocks_per_row: tl.constexpr = D // 128
    block = offs // 128
    j = offs - block * 128
    j_pair = (j // 2) * 2
    idx0 = unpack3_values(qs, signs, physical_row, blocks_per_row, block, j_pair + 0, mask)
    idx1 = unpack3_values(qs, signs, physical_row, blocks_per_row, block, j_pair + 1, mask)
    q0 = tl.load(centroids + idx0, mask=mask, other=0.0)
    q1 = tl.load(centroids + idx1, mask=mask, other=0.0)
    p = ((block * 128 + j_pair) // 2) & 63
    c = tl.load(cos + p, mask=mask, other=0.0)
    s = tl.load(sin + p, mask=mask, other=0.0)
    norm = tl.load(
        d + physical_row * blocks_per_row + block,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    even = c * q0 + s * q1
    odd = -s * q0 + c * q1
    return (tl.where((j & 1) == 1, odd, even) * norm).to(tl.float16).to(tl.float32)


@triton.jit
def iso3_values(d, qs, signs, centroids, qw_ptr, qx_ptr, qy_ptr, qz_ptr, physical_row, offs, mask, D: tl.constexpr):
    blocks_per_row: tl.constexpr = D // 128
    block = offs // 128
    j = offs - block * 128
    j_group = (j // 4) * 4
    idx0 = unpack3_values(qs, signs, physical_row, blocks_per_row, block, j_group + 0, mask)
    idx1 = unpack3_values(qs, signs, physical_row, blocks_per_row, block, j_group + 1, mask)
    idx2 = unpack3_values(qs, signs, physical_row, blocks_per_row, block, j_group + 2, mask)
    idx3 = unpack3_values(qs, signs, physical_row, blocks_per_row, block, j_group + 3, mask)
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
    norm = tl.load(
        d + physical_row * blocks_per_row + block,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    return (val * norm).to(tl.float16).to(tl.float32)


def deterministic_block_table(rows: int, block_size: int, *, extra_blocks: int = 2) -> torch.Tensor:
    logical_blocks = math.ceil(rows / block_size)
    physical_blocks = logical_blocks + extra_blocks
    ids = torch.arange(physical_blocks, dtype=torch.int64)
    return torch.roll(ids, shifts=1)[:logical_blocks].contiguous()


def physical_rows_for_table(rows: int, block_size: int, block_table: torch.Tensor) -> int:
    return (int(block_table.max().item()) + 1) * block_size


def physicalize_inputs(logical_inputs: dict[str, torch.Tensor], block_table: torch.Tensor, block_size: int) -> dict[str, torch.Tensor]:
    rows = logical_inputs["d"].shape[0]
    physical_rows = physical_rows_for_table(rows, block_size, block_table)
    physical: dict[str, torch.Tensor] = {}
    for name, tensor in logical_inputs.items():
        shape = (physical_rows, *tensor.shape[1:])
        out = torch.zeros(shape, dtype=tensor.dtype)
        for logical_row in range(rows):
            physical_row = int(block_table[logical_row // block_size]) * block_size + logical_row % block_size
            out[physical_row].copy_(tensor[logical_row])
        physical[name] = out.contiguous()
    return physical
