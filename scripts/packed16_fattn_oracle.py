#!/usr/bin/env python3
"""Local Python oracle for llama.cpp ROCm packed16 FlashAttention paths.

This is an executable correctness model for the current experimental ROCm
packed16 attention work.  It intentionally models semantics rather than GPU
scheduling:

* GQA head mapping: hq -> hk = hq // gqa
* packed16/q8 block-32 Q and K quantization used by the DOT4 kernels
* q4_0 V quantize/dequant layout used by ggml
* llama-style additive F16/F32 masks, or causal masking when no mask is passed
* full-softmax and online-softmax reference paths

The default CLI builds a harness-like random case and checks that the online
FlashAttention recurrence matches the full-softmax oracle.  It can also write
an .npz fixture for local debugging.

The implementation is deliberately dependency-light: NumPy only.  It does not
call CUDA/HIP, PyTorch, or Triton.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

D_DEFAULT = 256
QK8_0 = 32
QK4_0 = 32
RAND_MAX_GLIBC = 2147483647


@dataclass(frozen=True)
class Q8Block32:
    """Block-32 signed int8 payload plus per-block scale."""

    qs: np.ndarray      # int8, shape (..., d)
    scales: np.ndarray  # float16/float32, shape (..., d // 32)


@dataclass(frozen=True)
class Q4_0:
    """ggml q4_0 in unpacked form.

    codes holds nibble values 0..15 per element.  It is easier to inspect than
    ggml's byte-packed block_q4_0, and dequantizes identically.
    """

    d: np.ndarray      # float16, shape (..., d // 32)
    codes: np.ndarray  # uint8, shape (..., d), values 0..15


def _require_block_dim(d: int) -> None:
    if d % QK8_0 != 0 or d % QK4_0 != 0:
        raise ValueError(f"head dimension must be divisible by 32, got {d}")


def _round_nearest_even_i32(x: np.ndarray) -> np.ndarray:
    """Match lrintf under the default nearest-even rounding mode."""

    return np.rint(x).astype(np.int32)


def _trunc_i32(x: np.ndarray) -> np.ndarray:
    """Match C float-to-int truncation toward zero."""

    return np.trunc(x).astype(np.int32)


def quantize_q8_block32(
    x: np.ndarray,
    *,
    scale_dtype: np.dtype | type = np.float32,
    clamp_low: int = -128,
    clamp_high: int = 127,
) -> Q8Block32:
    """Quantize (..., d) rows to the packed16/DOT4 q8 block-32 contract.

    Q in `ggml_cuda_q8k_dot4_quant_q_packed16_kernel` uses float32 scales and
    clamps [-128, 127].  The test harness's pre-packed K sidecar stores float16
    scales and clamps [-127, 127].  This helper supports both through args.
    """

    x = np.asarray(x, dtype=np.float32)
    d = x.shape[-1]
    _require_block_dim(d)
    blocks = x.reshape(*x.shape[:-1], d // QK8_0, QK8_0)
    amax = np.max(np.abs(blocks), axis=-1)
    scale = np.where(amax > 0.0, amax / 127.0, 1.0).astype(scale_dtype)
    scale_f32 = scale.astype(np.float32)
    q = _round_nearest_even_i32(blocks / scale_f32[..., None])
    q = np.clip(q, clamp_low, clamp_high).astype(np.int8)
    return Q8Block32(q.reshape(x.shape), scale)


def dequantize_q8_block32(q: Q8Block32) -> np.ndarray:
    qs = np.asarray(q.qs, dtype=np.int8)
    scales = np.asarray(q.scales).astype(np.float32)
    d = qs.shape[-1]
    _require_block_dim(d)
    return (qs.reshape(*qs.shape[:-1], d // QK8_0, QK8_0).astype(np.float32) * scales[..., None]).reshape(qs.shape)


def pack_i8x4_words(qs: np.ndarray) -> np.ndarray:
    """Pack int8 lanes into int32 words like the DOT4 payload buffers."""

    qs = np.asarray(qs, dtype=np.int8)
    if qs.shape[-1] % 4 != 0:
        raise ValueError("last dimension must be divisible by 4")
    u = qs.reshape(*qs.shape[:-1], qs.shape[-1] // 4, 4).astype(np.uint8).astype(np.uint32)
    words = u[..., 0] | (u[..., 1] << 8) | (u[..., 2] << 16) | (u[..., 3] << 24)
    return words.astype(np.int32)


def unpack_i8x4_words(words: np.ndarray) -> np.ndarray:
    words_u = np.asarray(words, dtype=np.uint32)
    lanes = np.empty((*words_u.shape, 4), dtype=np.uint8)
    lanes[..., 0] = (words_u >> 0) & 0xFF
    lanes[..., 1] = (words_u >> 8) & 0xFF
    lanes[..., 2] = (words_u >> 16) & 0xFF
    lanes[..., 3] = (words_u >> 24) & 0xFF
    return lanes.view(np.int8).reshape(*words_u.shape[:-1], words_u.shape[-1] * 4)


def quantize_q4_0_rows(x: np.ndarray) -> Q4_0:
    """Quantize (..., d) rows with ggml's reference q4_0 formula.

    Reference: ggml/src/ggml-quants.c:quantize_row_q4_0_ref.
    """

    x = np.asarray(x, dtype=np.float32)
    d = x.shape[-1]
    _require_block_dim(d)
    blocks = x.reshape(*x.shape[:-1], d // QK4_0, QK4_0)

    abs_blocks = np.abs(blocks)
    imax = np.argmax(abs_blocks, axis=-1)
    max_signed = np.take_along_axis(blocks, imax[..., None], axis=-1)[..., 0]
    delta = (max_signed / -8.0).astype(np.float16)
    delta_f32 = delta.astype(np.float32)
    inv_delta = np.where(delta_f32 != 0.0, 1.0 / delta_f32, 0.0)

    scaled = blocks * inv_delta[..., None]
    # C processes first half and second half into one byte; the code value per
    # logical element is still trunc(x/d + 8.5), capped at 15.
    codes = _trunc_i32(scaled + np.float32(8.5))
    codes = np.clip(codes, 0, 15).astype(np.uint8)
    return Q4_0(delta, codes.reshape(x.shape))


def dequantize_q4_0(q: Q4_0) -> np.ndarray:
    codes = np.asarray(q.codes, dtype=np.uint8)
    d = codes.shape[-1]
    _require_block_dim(d)
    delta = np.asarray(q.d).astype(np.float32)
    centered = codes.reshape(*codes.shape[:-1], d // QK4_0, QK4_0).astype(np.int16) - 8
    return (centered.astype(np.float32) * delta[..., None]).reshape(codes.shape)


def pack_q4_0_bytes(q: Q4_0) -> tuple[np.ndarray, np.ndarray]:
    """Return (d, qs_bytes) matching block_q4_0's byte packing.

    qs_bytes shape is (..., d//32, 16).  Byte j contains element j in the low
    nibble and element j+16 in the high nibble.
    """

    codes = np.asarray(q.codes, dtype=np.uint8)
    d = codes.shape[-1]
    blocks = codes.reshape(*codes.shape[:-1], d // QK4_0, QK4_0)
    lo = blocks[..., :16]
    hi = blocks[..., 16:]
    return q.d, (lo | (hi << 4)).astype(np.uint8)


def unpack_q4_0_bytes(deltas: np.ndarray, qs_bytes: np.ndarray) -> Q4_0:
    qs_bytes = np.asarray(qs_bytes, dtype=np.uint8)
    lo = qs_bytes & 0x0F
    hi = qs_bytes >> 4
    codes = np.concatenate([lo, hi], axis=-1)
    leading = codes.shape[:-2]
    n_blocks = codes.shape[-2]
    return Q4_0(np.asarray(deltas, dtype=np.float16), codes.reshape(*leading, n_blocks * QK4_0))


def _normalize_q_shape(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    if q.ndim == 3:  # [hq, nq, d]
        q = q[None, ...]
    if q.ndim != 4:
        raise ValueError("Q must have shape [batch,hq,nq,d] or [hq,nq,d]")
    return q


def _normalize_kv_shape(x: np.ndarray, *, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3:  # [hk, nk, d]
        x = x[None, ...]
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape [batch,hk,nk,d] or [hk,nk,d]")
    return x


def _mask_for(mask: np.ndarray | None, *, b: int, q: int, nk: int) -> np.ndarray | None:
    if mask is None:
        return None
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 2:
        # Accept [nq,nk] (Python-natural) or [nk,nq] (ggml tensor-natural).
        if m.shape[1] == nk:
            return m[q, :]
        if m.shape[0] == nk:
            return m[:, q]
    if m.ndim == 3:
        # Accept [batch,nq,nk] or [batch,nk,nq].
        if m.shape[2] == nk:
            return m[b, q, :]
        if m.shape[1] == nk:
            return m[b, :, q]
    raise ValueError(f"unsupported mask shape {m.shape} for q={q}, nk={nk}")


def softmax_stable(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    finite = np.isfinite(scores)
    if not np.any(finite):
        return np.zeros_like(scores, dtype=np.float32)
    m = np.max(scores[finite])
    e = np.where(finite, np.exp(scores - m, dtype=np.float32), 0.0).astype(np.float32)
    denom = np.sum(e, dtype=np.float32)
    return e / denom if denom > 0.0 else np.zeros_like(scores, dtype=np.float32)


def dense_attention_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    gqa: int,
    mask: np.ndarray | None = None,
    q_offset: int = 0,
    causal_if_no_mask: bool = True,
    scale: float | None = None,
) -> np.ndarray:
    """Dense full-softmax GQA attention reference.

    Shapes:
      q: [batch,hq,nq,d]
      k: [batch,hk,nk,d]
      v: [batch,hk,nk,d]
      out: [batch,nq,hq,d] matching ggml_flash_attn_ext result order.
    """

    q = _normalize_q_shape(q).astype(np.float32)
    k = _normalize_kv_shape(k, name="K").astype(np.float32)
    v = _normalize_kv_shape(v, name="V").astype(np.float32)
    batch, hq, nq, d = q.shape
    batch_k, hk, nk, d_k = k.shape
    if v.shape != (batch_k, hk, nk, d):
        raise ValueError(f"V shape {v.shape} incompatible with K {(batch_k, hk, nk, d)}")
    if batch_k != batch or d_k != d:
        raise ValueError("Q/K batch or head-dim mismatch")
    if hq != hk * gqa:
        raise ValueError(f"expected hq == hk*gqa, got hq={hq} hk={hk} gqa={gqa}")
    sm_scale = (1.0 / math.sqrt(d)) if scale is None else float(scale)

    out = np.zeros((batch, nq, hq, d), dtype=np.float32)
    for b in range(batch):
        for h in range(hq):
            kh = h // gqa
            kt = k[b, kh]  # [nk,d]
            vt = v[b, kh]  # [nk,d]
            for qi in range(nq):
                scores = (kt @ q[b, h, qi].astype(np.float32)) * sm_scale
                mrow = _mask_for(mask, b=b, q=qi, nk=nk)
                if mrow is not None:
                    scores = scores + mrow
                elif causal_if_no_mask:
                    pos = q_offset + qi
                    scores = scores.copy()
                    scores[np.arange(nk) > pos] = -np.inf
                p = softmax_stable(scores)
                out[b, qi, h] = p @ vt
    return out


def online_attention_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    gqa: int,
    mask: np.ndarray | None = None,
    q_offset: int = 0,
    causal_if_no_mask: bool = True,
    scale: float | None = None,
    block_size: int = 128,
) -> np.ndarray:
    """Online-softmax reference matching FlashAttention's m/l/O recurrence."""

    q = _normalize_q_shape(q).astype(np.float32)
    k = _normalize_kv_shape(k, name="K").astype(np.float32)
    v = _normalize_kv_shape(v, name="V").astype(np.float32)
    batch, hq, nq, d = q.shape
    _, hk, nk, _ = k.shape
    if hq != hk * gqa:
        raise ValueError(f"expected hq == hk*gqa, got hq={hq} hk={hk} gqa={gqa}")
    sm_scale = (1.0 / math.sqrt(d)) if scale is None else float(scale)
    block_size = max(1, int(block_size))

    out = np.zeros((batch, nq, hq, d), dtype=np.float32)
    for b in range(batch):
        for h in range(hq):
            kh = h // gqa
            for qi in range(nq):
                row_m = -np.inf
                row_l = np.float32(0.0)
                row_o = np.zeros((d,), dtype=np.float32)
                for k0 in range(0, nk, block_size):
                    k1 = min(nk, k0 + block_size)
                    scores = (k[b, kh, k0:k1] @ q[b, h, qi]) * sm_scale
                    mrow = _mask_for(mask, b=b, q=qi, nk=nk)
                    if mrow is not None:
                        scores = scores + mrow[k0:k1]
                    elif causal_if_no_mask:
                        pos = q_offset + qi
                        scores = scores.copy()
                        scores[np.arange(k0, k1) > pos] = -np.inf
                    finite = np.isfinite(scores)
                    if not np.any(finite):
                        continue
                    tile_m = np.max(scores[finite]).astype(np.float32)
                    new_m = max(float(row_m), float(tile_m))
                    old_scale = np.exp(np.float32(row_m - new_m)) if row_l > 0.0 else np.float32(0.0)
                    e = np.where(finite, np.exp(scores - np.float32(new_m), dtype=np.float32), 0.0).astype(np.float32)
                    tile_l = np.sum(e, dtype=np.float32)
                    row_o = row_o * old_scale + e @ v[b, kh, k0:k1]
                    row_l = row_l * old_scale + tile_l
                    row_m = np.float32(new_m)
                out[b, qi, h] = row_o / row_l if row_l > 0.0 else 0.0
    return out


def _score_from_q8_blocks(qrow: Q8Block32, khead: Q8Block32, *, scale: float) -> np.ndarray:
    """Compute one query row against all K rows via int32 block dots."""

    q_i8 = qrow.qs.reshape(-1).astype(np.int32)
    q_sc = qrow.scales.reshape(-1).astype(np.float32)
    k_i8 = khead.qs.astype(np.int32)          # [nk,d]
    k_sc = khead.scales.astype(np.float32)    # [nk,nb]
    nk, d = k_i8.shape
    nb = d // QK8_0
    scores = np.zeros((nk,), dtype=np.float32)
    for ib in range(nb):
        q_blk = q_i8[ib * QK8_0:(ib + 1) * QK8_0]
        k_blk = k_i8[:, ib * QK8_0:(ib + 1) * QK8_0]
        acc = k_blk @ q_blk
        scores += acc.astype(np.float32) * q_sc[ib] * k_sc[:, ib]
    return scores * np.float32(scale)


def packed16_attention_oracle(
    q_f32: np.ndarray,
    k_f16_or_f32: np.ndarray,
    v_f32_or_q4: np.ndarray | Q4_0,
    *,
    gqa: int,
    mask: np.ndarray | None = None,
    q_offset: int = 0,
    causal_if_no_mask: bool = True,
    scale: float | None = None,
    online_block_size: int | None = None,
) -> np.ndarray:
    """Oracle for packed16 I32 K + q4_0 V verifier/decode semantics.

    The function quantizes Q like `ggml_cuda_q8k_dot4_quant_q_packed16_kernel`,
    quantizes K like the test harness sidecar (`make_packed16_from_f16`),
    q4_0-quantizes V unless a Q4_0 object is already supplied, then computes
    exact softmax attention over those quantized/dequantized operands.
    """

    q_f32 = _normalize_q_shape(q_f32).astype(np.float32)
    k = _normalize_kv_shape(k_f16_or_f32, name="K").astype(np.float32)
    if isinstance(v_f32_or_q4, Q4_0):
        v_q4 = v_f32_or_q4
        v_deq = dequantize_q4_0(v_q4)
        v_deq = _normalize_kv_shape(v_deq, name="V")
    else:
        v = _normalize_kv_shape(v_f32_or_q4, name="V").astype(np.float32)
        v_q4 = quantize_q4_0_rows(v.reshape(-1, v.shape[-1]))
        v_deq = dequantize_q4_0(v_q4).reshape(v.shape)

    batch, hq, nq, d = q_f32.shape
    _, hk, nk, _ = k.shape
    if hq != hk * gqa:
        raise ValueError(f"expected hq == hk*gqa, got hq={hq} hk={hk} gqa={gqa}")
    sm_scale = (1.0 / math.sqrt(d)) if scale is None else float(scale)

    q_q8 = quantize_q8_block32(q_f32, scale_dtype=np.float32, clamp_low=-128, clamp_high=127)
    # Harness sidecar K starts from F16 source and stores F16 scales.
    k_source = k.astype(np.float16).astype(np.float32)
    k_q8 = quantize_q8_block32(k_source, scale_dtype=np.float16, clamp_low=-127, clamp_high=127)

    out = np.zeros((batch, nq, hq, d), dtype=np.float32)
    if online_block_size is None:
        # Full-softmax implementation using int32 block-dot scores.
        for b in range(batch):
            for h in range(hq):
                kh = h // gqa
                khead = Q8Block32(k_q8.qs[b, kh], k_q8.scales[b, kh])
                for qi in range(nq):
                    qrow = Q8Block32(q_q8.qs[b, h, qi], q_q8.scales[b, h, qi])
                    scores = _score_from_q8_blocks(qrow, khead, scale=sm_scale)
                    mrow = _mask_for(mask, b=b, q=qi, nk=nk)
                    if mrow is not None:
                        scores = scores + mrow
                    elif causal_if_no_mask:
                        scores = scores.copy()
                        scores[np.arange(nk) > (q_offset + qi)] = -np.inf
                    p = softmax_stable(scores)
                    out[b, qi, h] = p @ v_deq[b, kh]
        return out

    # Online path over quantized scores, useful for debugging split/page reducers.
    online_block_size = max(1, int(online_block_size))
    for b in range(batch):
        for h in range(hq):
            kh = h // gqa
            khead_all = Q8Block32(k_q8.qs[b, kh], k_q8.scales[b, kh])
            for qi in range(nq):
                qrow = Q8Block32(q_q8.qs[b, h, qi], q_q8.scales[b, h, qi])
                row_m = -np.inf
                row_l = np.float32(0.0)
                row_o = np.zeros((d,), dtype=np.float32)
                for k0 in range(0, nk, online_block_size):
                    k1 = min(nk, k0 + online_block_size)
                    khead = Q8Block32(khead_all.qs[k0:k1], khead_all.scales[k0:k1])
                    scores = _score_from_q8_blocks(qrow, khead, scale=sm_scale)
                    mrow = _mask_for(mask, b=b, q=qi, nk=nk)
                    if mrow is not None:
                        scores = scores + mrow[k0:k1]
                    elif causal_if_no_mask:
                        scores = scores.copy()
                        scores[np.arange(k0, k1) > (q_offset + qi)] = -np.inf
                    finite = np.isfinite(scores)
                    if not np.any(finite):
                        continue
                    tile_m = np.max(scores[finite]).astype(np.float32)
                    new_m = max(float(row_m), float(tile_m))
                    old_scale = np.exp(np.float32(row_m - new_m)) if row_l > 0.0 else np.float32(0.0)
                    e = np.where(finite, np.exp(scores - np.float32(new_m), dtype=np.float32), 0.0).astype(np.float32)
                    row_o = row_o * old_scale + e @ v_deq[b, kh, k0:k1]
                    row_l = row_l * old_scale + np.sum(e, dtype=np.float32)
                    row_m = np.float32(new_m)
                out[b, qi, h] = row_o / row_l if row_l > 0.0 else 0.0
    return out


def packed16_attention_from_sidecars(
    q_f32: np.ndarray,
    k_payload_words: np.ndarray,
    k_scales: np.ndarray,
    v_q4: Q4_0,
    *,
    gqa: int,
    mask: np.ndarray | None = None,
    q_offset: int = 0,
    causal_if_no_mask: bool = True,
    scale: float | None = None,
    online_block_size: int | None = None,
) -> np.ndarray:
    """Oracle using dumped C++ harness sidecars instead of re-quantizing K.

    Inputs:
      q_f32: [batch,hq,nq,d]
      k_payload_words: [batch,hk,nk,d//4] int32 packed int8 lanes
      k_scales: [batch,hk,nk,d//32] float16/float32 scales
      v_q4: Q4_0 with codes [batch,hk,nk,d]

    Output layout is [batch,nq,hq,d], matching packed16 decode harness output.
    """

    q_f32 = _normalize_q_shape(q_f32).astype(np.float32)
    k_payload_words = np.asarray(k_payload_words, dtype=np.int32)
    k_i8 = unpack_i8x4_words(k_payload_words)
    k_scales = np.asarray(k_scales).astype(np.float32)
    v_deq = _normalize_kv_shape(dequantize_q4_0(v_q4), name="V").astype(np.float32)

    batch, hq, nq, d = q_f32.shape
    if k_i8.shape != (batch, hq // gqa, v_deq.shape[2], d):
        # Build a clearer error than a later indexing failure.
        raise ValueError(
            f"K payload shape {k_i8.shape} incompatible with Q {(batch, hq, nq, d)} and gqa={gqa}"
        )
    _, hk, nk, _ = k_i8.shape
    if hq != hk * gqa:
        raise ValueError(f"expected hq == hk*gqa, got hq={hq} hk={hk} gqa={gqa}")
    if k_scales.shape != (batch, hk, nk, d // QK8_0):
        raise ValueError(f"K scales shape {k_scales.shape} incompatible with K {(batch, hk, nk, d)}")
    if v_deq.shape != (batch, hk, nk, d):
        raise ValueError(f"V shape {v_deq.shape} incompatible with K {(batch, hk, nk, d)}")

    q_q8 = quantize_q8_block32(q_f32, scale_dtype=np.float32, clamp_low=-128, clamp_high=127)
    k_q8 = Q8Block32(k_i8, k_scales)
    sm_scale = (1.0 / math.sqrt(d)) if scale is None else float(scale)

    out = np.zeros((batch, nq, hq, d), dtype=np.float32)
    if online_block_size is None:
        for b in range(batch):
            for h in range(hq):
                kh = h // gqa
                khead = Q8Block32(k_q8.qs[b, kh], k_q8.scales[b, kh])
                for qi in range(nq):
                    qrow = Q8Block32(q_q8.qs[b, h, qi], q_q8.scales[b, h, qi])
                    scores = _score_from_q8_blocks(qrow, khead, scale=sm_scale)
                    mrow = _mask_for(mask, b=b, q=qi, nk=nk)
                    if mrow is not None:
                        scores = scores + mrow
                    elif causal_if_no_mask:
                        scores = scores.copy()
                        scores[np.arange(nk) > (q_offset + qi)] = -np.inf
                    p = softmax_stable(scores)
                    out[b, qi, h] = p @ v_deq[b, kh]
        return out

    online_block_size = max(1, int(online_block_size))
    for b in range(batch):
        for h in range(hq):
            kh = h // gqa
            khead_all = Q8Block32(k_q8.qs[b, kh], k_q8.scales[b, kh])
            for qi in range(nq):
                qrow = Q8Block32(q_q8.qs[b, h, qi], q_q8.scales[b, h, qi])
                row_m = -np.inf
                row_l = np.float32(0.0)
                row_o = np.zeros((d,), dtype=np.float32)
                for k0 in range(0, nk, online_block_size):
                    k1 = min(nk, k0 + online_block_size)
                    khead = Q8Block32(khead_all.qs[k0:k1], khead_all.scales[k0:k1])
                    scores = _score_from_q8_blocks(qrow, khead, scale=sm_scale)
                    mrow = _mask_for(mask, b=b, q=qi, nk=nk)
                    if mrow is not None:
                        scores = scores + mrow[k0:k1]
                    elif causal_if_no_mask:
                        scores = scores.copy()
                        scores[np.arange(k0, k1) > (q_offset + qi)] = -np.inf
                    finite = np.isfinite(scores)
                    if not np.any(finite):
                        continue
                    tile_m = np.max(scores[finite]).astype(np.float32)
                    new_m = max(float(row_m), float(tile_m))
                    old_scale = np.exp(np.float32(row_m - new_m)) if row_l > 0.0 else np.float32(0.0)
                    e = np.where(finite, np.exp(scores - np.float32(new_m), dtype=np.float32), 0.0).astype(np.float32)
                    row_o = row_o * old_scale + e @ v_deq[b, kh, k0:k1]
                    row_l = row_l * old_scale + np.sum(e, dtype=np.float32)
                    row_m = np.float32(new_m)
                out[b, qi, h] = row_o / row_l if row_l > 0.0 else 0.0
    return out


def _read_array(path: Path, dtype: np.dtype | type, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"{path} has {arr.size} elements, expected {expected} for shape {shape}")
    return arr.reshape(shape)


def _read_f16_bits(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    raw = _read_array(path, np.uint16, shape)
    return raw.view(np.float16)


def _read_q4_0_raw(path: Path, *, batch: int, hk: int, nk: int, d: int) -> Q4_0:
    nb = d // QK4_0
    raw = _read_array(path, np.uint8, (batch, hk, nk, nb, 18))
    deltas = np.ascontiguousarray(raw[..., 0:2]).view(np.float16).reshape(batch, hk, nk, nb)
    qs_bytes = raw[..., 2:18]
    return unpack_q4_0_bytes(deltas, qs_bytes)


def load_harness_fixture(fixture_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a fixture dumped by PACKED16_DECODE_TEST_DUMP_DIR."""

    root = Path(fixture_dir)
    meta = json.loads((root / "meta.json").read_text())
    batch = int(meta.get("batch", 1))
    nq = int(meta["nq"])
    nk = int(meta["nk"])
    hq = int(meta["hq"])
    hk = int(meta["hk"])
    gqa = int(meta["gqa"])
    d = int(meta.get("d", D_DEFAULT))
    files = meta["files"]

    q = _read_array(root / files["q_f32"], np.float32, (batch, hq, nq, d))
    k_payload = _read_array(root / files["k_payload_i32"], np.int32, (batch, hk, nk, d // 4))
    k_scales = _read_f16_bits(root / files["k_scales_f16"], (batch, hk, nk, d // QK8_0))
    v_q4 = _read_q4_0_raw(root / files["v_q4_0"], batch=batch, hk=hk, nk=nk, d=d)

    mask = None
    if "mask_f16" in files:
        mask = _read_f16_bits(root / files["mask_f16"], (nk, nq)).astype(np.float32)

    outputs: dict[str, np.ndarray] = {}
    for name, rel in meta.get("outputs", {}).items():
        outputs[name] = _read_array(root / rel, np.float32, (batch, nq, hq, d))

    return {
        "root": root,
        "meta": meta,
        "q": q,
        "k_payload": k_payload,
        "k_scales": k_scales,
        "v_q4": v_q4,
        "mask": mask,
        "outputs": outputs,
        "shape": {"batch": batch, "nq": nq, "nk": nk, "hq": hq, "hk": hk, "gqa": gqa, "d": d},
    }


def compare_harness_fixture(args: argparse.Namespace) -> dict[str, Any]:
    fixture = load_harness_fixture(args.compare_fixture)
    shape = fixture["shape"]
    meta = fixture["meta"]
    scale = float(meta.get("scale", 1.0 / math.sqrt(shape["d"])))
    oracle = packed16_attention_from_sidecars(
        fixture["q"],
        fixture["k_payload"],
        fixture["k_scales"],
        fixture["v_q4"],
        gqa=shape["gqa"],
        mask=fixture["mask"],
        scale=scale,
        online_block_size=None,
    )
    online = packed16_attention_from_sidecars(
        fixture["q"],
        fixture["k_payload"],
        fixture["k_scales"],
        fixture["v_q4"],
        gqa=shape["gqa"],
        mask=fixture["mask"],
        scale=scale,
        online_block_size=args.block_size,
    )
    online_max, online_rms = max_abs_rms(oracle, online)

    per_output: dict[str, Any] = {}
    all_pass = online_max <= 2e-5 and online_rms <= 2e-6
    names = args.variant if args.variant else sorted(fixture["outputs"].keys())
    for name in names:
        if name not in fixture["outputs"]:
            per_output[name] = {"present": False, "pass": False}
            all_pass = False
            continue
        out = fixture["outputs"][name]
        max_abs, rms = max_abs_rms(out, oracle)
        finite = bool(np.all(np.isfinite(out)))
        passed = finite and max_abs <= args.atol and rms <= args.rms_tol
        per_output[name] = {
            "present": True,
            "finite": finite,
            "max_abs": max_abs,
            "rms": rms,
            "pass": passed,
        }
        all_pass = all_pass and passed

    return {
        "fixture": str(fixture["root"]),
        **shape,
        "oracle_full_vs_online": {"max_abs": online_max, "rms": online_rms},
        "thresholds": {"atol": args.atol, "rms_tol": args.rms_tol},
        "outputs": per_output,
        "pass": all_pass,
    }


def _libc_rand_f32(count: int, *, seed: int, scale: float) -> np.ndarray:
    """Reproduce the C++ harness fill_f32() sequence through libc rand()."""

    libc = ctypes.CDLL(None)
    libc.srand(ctypes.c_uint(seed))
    libc.rand.restype = ctypes.c_int
    vals = np.empty((count,), dtype=np.float32)
    rand_max = np.float32(RAND_MAX_GLIBC)
    sc = np.float32(scale)
    for i in range(count):
        r = np.float32(libc.rand())
        vals[i] = ((r / rand_max) * np.float32(2.0) - np.float32(1.0)) * sc
    return vals


def make_harness_like_case(
    *,
    nq: int,
    nk: int,
    n_heads_k: int,
    gqa: int,
    d: int = D_DEFAULT,
    q_seed: int = 123,
    k_seed: int = 124,
    v_seed: int = 125,
    value_scale: float = 0.75,
    mask_value: float = 0.0,
) -> dict[str, Any]:
    """Build a Python version of tests/test-packed16-decode-variants.cpp data."""

    _require_block_dim(d)
    batch = 1
    n_heads_q = n_heads_k * gqa
    q = _libc_rand_f32(d * nq * n_heads_q * batch, seed=q_seed, scale=value_scale).reshape(batch, n_heads_q, nq, d)
    k_f32 = _libc_rand_f32(d * nk * n_heads_k * batch, seed=k_seed, scale=value_scale).reshape(batch, n_heads_k, nk, d)
    k_f16 = k_f32.astype(np.float16)
    v = _libc_rand_f32(d * nk * n_heads_k * batch, seed=v_seed, scale=value_scale).reshape(batch, n_heads_k, nk, d)
    mask = np.full((batch, nq, nk), np.float32(mask_value), dtype=np.float32)
    return {"q": q, "k_f16": k_f16, "v": v, "mask": mask, "gqa": gqa}


def max_abs_rms(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.max(np.abs(diff))), float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))


def run_self_test(args: argparse.Namespace) -> dict[str, Any]:
    case = make_harness_like_case(nq=args.nq, nk=args.nk, n_heads_k=args.hk, gqa=args.gqa, d=args.d)
    q = case["q"]
    k_f16 = case["k_f16"]
    v = case["v"]
    mask = case["mask"]
    gqa = int(case["gqa"])

    # Packed16/q4 oracle: full-softmax vs online-softmax over the same quantized operands.
    packed_full = packed16_attention_oracle(q, k_f16, v, gqa=gqa, mask=mask, online_block_size=None)
    packed_online = packed16_attention_oracle(q, k_f16, v, gqa=gqa, mask=mask, online_block_size=args.block_size)
    packed_max, packed_rms = max_abs_rms(packed_full, packed_online)

    # Dense dequantized sanity: direct dense and online should also match.
    v_q4 = quantize_q4_0_rows(v.reshape(-1, args.d))
    v_deq = dequantize_q4_0(v_q4).reshape(v.shape)
    k_deq = dequantize_q8_block32(quantize_q8_block32(k_f16.astype(np.float32), scale_dtype=np.float16, clamp_low=-127, clamp_high=127))
    q_deq = dequantize_q8_block32(quantize_q8_block32(q, scale_dtype=np.float32, clamp_low=-128, clamp_high=127))
    dense_full = dense_attention_reference(q_deq, k_deq, v_deq, gqa=gqa, mask=mask)
    dense_max, dense_rms = max_abs_rms(packed_full, dense_full)

    result = {
        "nq": args.nq,
        "nk": args.nk,
        "hq": args.hk * args.gqa,
        "hk": args.hk,
        "gqa": args.gqa,
        "d": args.d,
        "block_size": args.block_size,
        "packed_full_vs_online": {"max_abs": packed_max, "rms": packed_rms},
        "packed_blockdot_vs_dequant_dense": {"max_abs": dense_max, "rms": dense_rms},
        "pass": packed_max < 2e-5 and packed_rms < 2e-6 and dense_max < 2e-5 and dense_rms < 2e-6,
    }

    if args.dump_npz:
        out_path = Path(args.dump_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        q8_q = quantize_q8_block32(q, scale_dtype=np.float32, clamp_low=-128, clamp_high=127)
        q8_k = quantize_q8_block32(k_f16.astype(np.float32), scale_dtype=np.float16, clamp_low=-127, clamp_high=127)
        q4_v = quantize_q4_0_rows(v.reshape(-1, args.d))
        v_d, v_qs_bytes = pack_q4_0_bytes(q4_v)
        np.savez_compressed(
            out_path,
            q=q,
            k_f16=k_f16,
            v_f32=v,
            mask=mask,
            q_i8=q8_q.qs,
            q_scales=q8_q.scales,
            k_i8=q8_k.qs,
            k_scales=q8_k.scales,
            v_q4_d=v_d,
            v_q4_qs=v_qs_bytes,
            out_packed16=packed_full,
            out_packed16_online=packed_online,
        )
        result["dump_npz"] = str(out_path)

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Local oracle for ROCm packed16/q4_0 FlashAttention semantics")
    ap.add_argument("--nq", type=int, default=int(os.getenv("PACKED16_DECODE_TEST_NQ", "4")))
    ap.add_argument("--nk", type=int, default=int(os.getenv("PACKED16_DECODE_TEST_NK", "256")))
    ap.add_argument("--hk", type=int, default=int(os.getenv("PACKED16_DECODE_TEST_HK", "2")), help="number of KV heads")
    ap.add_argument("--gqa", type=int, default=int(os.getenv("PACKED16_DECODE_TEST_GQA", "4")))
    ap.add_argument("--d", type=int, default=D_DEFAULT)
    ap.add_argument("--block-size", type=int, default=128, help="online oracle K block size")
    ap.add_argument("--dump-npz", type=str, default="", help="optional compressed fixture output")
    ap.add_argument("--compare-fixture", type=str, default="", help="directory dumped by PACKED16_DECODE_TEST_DUMP_DIR")
    ap.add_argument("--variant", action="append", default=[], help="fixture output variant to compare; can be repeated")
    ap.add_argument("--atol", type=float, default=2.5e-2, help="GPU-output max_abs pass threshold")
    ap.add_argument("--rms-tol", type=float, default=5.0e-3, help="GPU-output RMS pass threshold")
    ap.add_argument("--json", action="store_true", help="print compact JSON only")
    args = ap.parse_args()

    result = compare_harness_fixture(args) if args.compare_fixture else run_self_test(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
