#!/usr/bin/env python3
"""
No-build formula oracle for the proposed transactional QBlock tail pager.

Mirrors source formulas from:
  - ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu
      GGML_CUDA_V4_K16D16_144_* constants
      ggml_cuda_pack_v4_k16d16_144_indexed_kernel()
      ggml_cuda_q8k_dot4_dequant_v4_k16d16_144()
  - ggml/src/ggml-cuda/fattn-packed16-dot4-mmq-impl.cuh
      pdmq_decode_v_v4_k16d16_144*()
  - ggml/src/ggml-cuda/dot4-packed16/dp16-packed-i8-desc.cuh
      dp16_packed_i8_payload_byte_offset()
      dp16_packed_i8_scale_byte_offset()
  - ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cuh
      ggml_cuda_packed16_k_*_index_from_desc()

CPU-only. No ROCm, model, or llama-server build required.
"""

from __future__ import annotations

import argparse
import random
import struct
from dataclasses import dataclass
from typing import Iterable

# V4_K16D16_144 constants.
D = 256
PAGE_TOKENS = 16
D32 = 32
D32_BLOCKS = D // D32
WORDS_PER_D = 2
PAYLOAD_WORDS = D * WORDS_PER_D
PAYLOAD_BYTES = PAYLOAD_WORDS * 4
SCALE_BYTES = D32_BLOCKS * PAGE_TOKENS * 2
PAGE_BYTES = PAYLOAD_BYTES + SCALE_BYTES
ROW_BYTES = PAGE_BYTES // PAGE_TOKENS

# packed16 K constants.
K_WORDS = D // 4
K_QBLOCKS = D // 32
K_D16 = 16
K_WORDS_PER_D16 = K_D16 // 4
I8X16_BYTES = 16
WORD_BYTES = 4
SCALE_BYTES_U16 = 2
K_ROW_BYTES = K_WORDS * WORD_BYTES + K_QBLOCKS * SCALE_BYTES_U16
K_PAGE_BYTES = K_ROW_BYTES * PAGE_TOKENS
KV_PAGE_BYTES = K_PAGE_BYTES + PAGE_BYTES

TAIL_ONLY = 1 << 0
BOUNDARY_COPY = 1 << 1
ALIGNED_ONLY = 1 << 2
DEBUG_ABORTS = 1 << 3


def f32_to_f16_bits(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", float(x)))[0]


def f16_bits_to_f32(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def c_int8(v: int) -> int:
    v &= 0xFF
    return v - 256 if v >= 128 else v


@dataclass
class Q4Row:
    codes: list[int]        # one unsigned q4 code per D element
    scale_bits: list[int]   # one f16 scale per D32 block


@dataclass
class V4Page:
    payload_words: list[int] # D * 2 u32 words, each word packs 8 token slots for one D
    scale_bits: list[int]    # D32_BLOCKS * PAGE_TOKENS f16 scales


def q4_pack_row_like_v4_kernel(vals: list[float]) -> Q4Row:
    """Pack one D=256 row with the same D32 q4_0-equivalent math used by the V4_144 packer."""
    assert len(vals) == D
    codes = [0] * D
    scale_bits = [0] * D32_BLOCKS
    for d32 in range(D32_BLOCKS):
        block = vals[d32 * D32:(d32 + 1) * D32]
        maxv = 0.0
        amax = 0.0
        for v in block:
            av = abs(v)
            if av > amax:
                amax = av
                maxv = v
        scale = maxv / -8.0
        inv_scale = 1.0 / scale if scale != 0.0 else 0.0
        scale_bits[d32] = f32_to_f16_bits(scale)
        for i, x in enumerate(block):
            # Mirrors: int q = (int8_t) (x * inv_scale + 8.5f); q = q > 15 ? 15 : q;
            q = c_int8(int(x * inv_scale + 8.5))
            if q > 15:
                q = 15
            codes[d32 * D32 + i] = q & 0x0F
    return Q4Row(codes, scale_bits)


def q4_decode(row: Q4Row, d: int) -> float:
    return float(row.codes[d] - 8) * f16_bits_to_f32(row.scale_bits[d // D32])


def pack_v4_page(rows: list[list[float]]) -> tuple[V4Page, list[Q4Row]]:
    assert len(rows) == PAGE_TOKENS
    payload = [0] * PAYLOAD_WORDS
    scales = [0] * (D32_BLOCKS * PAGE_TOKENS)
    q4_rows: list[Q4Row] = []
    for slot, vals in enumerate(rows):
        q4 = q4_pack_row_like_v4_kernel(vals)
        q4_rows.append(q4)
        word_lane = slot >> 3
        nib_shift = 4 * (slot & 7)
        for d, code in enumerate(q4.codes):
            wi = d * WORDS_PER_D + word_lane
            payload[wi] = (payload[wi] & ~(0x0F << nib_shift)) | ((code & 0x0F) << nib_shift)
        for d32, bits in enumerate(q4.scale_bits):
            scales[d32 * PAGE_TOKENS + slot] = bits
    return V4Page(payload, scales), q4_rows


def v4_payload_word_index(slot: int, d: int) -> int:
    return d * WORDS_PER_D + (slot >> 3)


def v4_scale_index(slot: int, d: int) -> int:
    return (d // D32) * PAGE_TOKENS + slot


def v4_decode(page: V4Page, slot: int, d: int) -> float:
    word = page.payload_words[v4_payload_word_index(slot, d)]
    q = ((word >> (4 * (slot & 7))) & 0x0F) - 8
    return float(q) * f16_bits_to_f32(page.scale_bits[v4_scale_index(slot, d)])


@dataclass
class TailDesc:
    logical_base_token: int
    logical_tokens: int
    valid_tail_tokens: int
    block_table: list[int]
    physical_pages: int
    page_tokens: int = PAGE_TOKENS
    d: int = D
    v4_page_stride_bytes: int = PAGE_BYTES
    batch: int = 1
    flags: int = TAIL_ONLY | ALIGNED_ONLY | DEBUG_ABORTS


def validate_tail_desc(desc: TailDesc) -> None:
    if not (desc.flags & TAIL_ONLY):
        raise ValueError("TAIL_ONLY flag is required")
    if desc.page_tokens != PAGE_TOKENS:
        raise ValueError("page_tokens must be 16")
    if desc.d != D:
        raise ValueError("D must be 256")
    if desc.v4_page_stride_bytes != PAGE_BYTES:
        raise ValueError("V4_144 page stride must be 2304")
    if desc.batch != 1:
        raise ValueError("v1 supports batch=1 only")
    if not desc.block_table:
        raise ValueError("block_table is required")
    if desc.logical_tokens > len(desc.block_table) * PAGE_TOKENS:
        raise ValueError("logical_tokens exceed block_table capacity")
    if desc.valid_tail_tokens > desc.logical_tokens:
        raise ValueError("valid_tail_tokens exceed logical_tokens")
    if (desc.flags & ALIGNED_ONLY) and (desc.logical_base_token % PAGE_TOKENS) != 0:
        raise ValueError("aligned-only tail starts mid K16 page")
    for pp in desc.block_table:
        if pp < 0 or pp >= desc.physical_pages:
            raise ValueError("physical page id out of range")


def logical_to_physical(desc: TailDesc, logical_token: int) -> tuple[int, int]:
    validate_tail_desc(desc)
    rel = logical_token - desc.logical_base_token
    if rel < 0 or rel >= desc.valid_tail_tokens:
        raise IndexError("logical token is not visible in this tail")
    lp = rel // PAGE_TOKENS
    slot = rel % PAGE_TOKENS
    return desc.block_table[lp], slot


def v4_offsets(physical_page: int, slot: int, d: int, *, head: int = 0, batch: int = 0,
               head_stride: int = PAGE_BYTES * 11, batch_stride: int = 0) -> tuple[int, int]:
    base = batch * batch_stride + head * head_stride + physical_page * PAGE_BYTES
    payload = base + v4_payload_word_index(slot, d) * 4
    scale = base + PAYLOAD_BYTES + v4_scale_index(slot, d) * 2
    return payload, scale


@dataclass
class PackedI8Desc:
    layout: str
    kv_capacity: int
    heads: int
    x_stride_bytes: int
    y_stride_bytes: int
    z_stride_bytes: int
    plane_stride_bytes: int
    scale_x_stride_bytes: int
    scale_y_stride_bytes: int
    scale_z_stride_bytes: int
    scale_plane_stride_bytes: int


def make_k_desc(layout: str, kv_capacity: int, heads: int) -> PackedI8Desc:
    z_stride = kv_capacity * K_WORDS * WORD_BYTES
    scale_z_stride = kv_capacity * K_QBLOCKS * SCALE_BYTES_U16
    if layout == "row":
        return PackedI8Desc(
            layout, kv_capacity, heads,
            x_stride_bytes=I8X16_BYTES,
            y_stride_bytes=K_WORDS * WORD_BYTES,
            z_stride_bytes=z_stride,
            plane_stride_bytes=0,
            scale_x_stride_bytes=SCALE_BYTES_U16,
            scale_y_stride_bytes=K_QBLOCKS * SCALE_BYTES_U16,
            scale_z_stride_bytes=scale_z_stride,
            scale_plane_stride_bytes=SCALE_BYTES_U16,
        )
    if layout == "d16_planar":
        return PackedI8Desc(
            layout, kv_capacity, heads,
            x_stride_bytes=kv_capacity * K_WORDS_PER_D16 * WORD_BYTES,
            y_stride_bytes=K_WORDS_PER_D16 * WORD_BYTES,
            z_stride_bytes=z_stride,
            plane_stride_bytes=kv_capacity * K_WORDS_PER_D16 * WORD_BYTES,
            scale_x_stride_bytes=kv_capacity * SCALE_BYTES_U16,
            scale_y_stride_bytes=SCALE_BYTES_U16,
            scale_z_stride_bytes=scale_z_stride,
            scale_plane_stride_bytes=kv_capacity * SCALE_BYTES_U16,
        )
    raise ValueError(layout)


def desc_payload_byte_offset(desc: PackedI8Desc, head: int, token: int, d_word: int) -> int:
    d16 = d_word // K_WORDS_PER_D16
    word = d_word - d16 * K_WORDS_PER_D16
    return (head * desc.z_stride_bytes + token * desc.y_stride_bytes +
            d16 * desc.x_stride_bytes + word * WORD_BYTES)


def desc_scale_byte_offset(desc: PackedI8Desc, head: int, token: int, qblock: int) -> int:
    return head * desc.scale_z_stride_bytes + token * desc.scale_y_stride_bytes + qblock * desc.scale_x_stride_bytes


def explicit_payload_byte_offset(layout: str, kv_capacity: int, head: int, token: int, d_word: int) -> int:
    if layout == "row":
        return ((head * kv_capacity + token) * K_WORDS + d_word) * WORD_BYTES
    d16 = d_word // K_WORDS_PER_D16
    word = d_word - d16 * K_WORDS_PER_D16
    return (head * kv_capacity * K_WORDS * WORD_BYTES +
            d16 * kv_capacity * K_WORDS_PER_D16 * WORD_BYTES +
            token * K_WORDS_PER_D16 * WORD_BYTES +
            word * WORD_BYTES)


def explicit_scale_byte_offset(layout: str, kv_capacity: int, head: int, token: int, qblock: int) -> int:
    if layout == "row":
        return ((head * kv_capacity + token) * K_QBLOCKS + qblock) * SCALE_BYTES_U16
    return (head * kv_capacity * K_QBLOCKS * SCALE_BYTES_U16 +
            qblock * kv_capacity * SCALE_BYTES_U16 +
            token * SCALE_BYTES_U16)


def rand_rows(rng: random.Random, n: int) -> list[list[float]]:
    rows = []
    for r in range(n):
        if r == 0:
            rows.append([0.0] * D)
        elif r == 1:
            rows.append([(-1.0) ** d * (d % 17) / 9.0 for d in range(D)])
        else:
            rows.append([rng.uniform(-8.0, 8.0) for _ in range(D)])
    return rows


def expect_raises(fn, needle: str) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001, test helper
        if needle not in str(exc):
            raise AssertionError(f"expected {needle!r}, got {exc!r}") from exc
        return
    raise AssertionError(f"expected exception containing {needle!r}")


def test_constants() -> None:
    assert D32_BLOCKS == 8
    assert PAYLOAD_BYTES == 2048
    assert SCALE_BYTES == 256
    assert PAGE_BYTES == 2304
    assert ROW_BYTES == 144
    assert K_ROW_BYTES == 272
    assert K_PAGE_BYTES == 4352
    assert KV_PAGE_BYTES == 6656


def test_v4_q4_equivalence(rng: random.Random, iters: int) -> None:
    for _ in range(iters):
        rows = rand_rows(rng, PAGE_TOKENS)
        page, q4_rows = pack_v4_page(rows)
        assert len(page.payload_words) == PAYLOAD_WORDS
        assert len(page.scale_bits) == D32_BLOCKS * PAGE_TOKENS
        for slot in range(PAGE_TOKENS):
            for d in [0, 1, 15, 16, 31, 32, 63, 64, 127, 128, 191, 192, 255]:
                assert v4_decode(page, slot, d) == q4_decode(q4_rows[slot], d)
        for _probe in range(128):
            slot = rng.randrange(PAGE_TOKENS)
            d = rng.randrange(D)
            assert v4_decode(page, slot, d) == q4_decode(q4_rows[slot], d)


def test_block_table_and_tail_visibility(rng: random.Random) -> None:
    block_table = [2, 0, 3, 1]
    desc = TailDesc(logical_base_token=64, logical_tokens=48, valid_tail_tokens=34,
                    block_table=block_table, physical_pages=4)
    validate_tail_desc(desc)
    assert logical_to_physical(desc, 64) == (2, 0)
    assert logical_to_physical(desc, 79) == (2, 15)
    assert logical_to_physical(desc, 80) == (0, 0)
    assert logical_to_physical(desc, 97) == (3, 1)
    expect_raises(lambda: logical_to_physical(desc, 98), "not visible")

    # Distinct physical pages prove lookup is through the table, not logical order.
    pages = []
    for pp in range(4):
        base = float(10 * (pp + 1))
        rows = [[base + slot + d / 1024.0 for d in range(D)] for slot in range(PAGE_TOKENS)]
        pages.append(pack_v4_page(rows)[0])
    for logical in [64, 80, 96, 97]:
        pp, slot = logical_to_physical(desc, logical)
        payload_off, scale_off = v4_offsets(pp, slot, 37, head=1)
        assert payload_off == PAGE_BYTES * 11 + pp * PAGE_BYTES + v4_payload_word_index(slot, 37) * 4
        assert scale_off == PAGE_BYTES * 11 + pp * PAGE_BYTES + PAYLOAD_BYTES + v4_scale_index(slot, 37) * 2
        _ = v4_decode(pages[pp], slot, 37)

    expect_raises(lambda: validate_tail_desc(TailDesc(65, 16, 16, [0], 1)), "aligned-only")
    validate_tail_desc(TailDesc(65, 16, 16, [0], 1, flags=TAIL_ONLY | BOUNDARY_COPY | DEBUG_ABORTS))
    expect_raises(lambda: validate_tail_desc(TailDesc(64, 33, 33, [0, 1], 2)), "capacity")
    expect_raises(lambda: validate_tail_desc(TailDesc(64, 16, 17, [0], 1)), "valid_tail_tokens")
    expect_raises(lambda: validate_tail_desc(TailDesc(64, 16, 16, [2], 2)), "out of range")


def test_packed16_k_descriptor_formulas(rng: random.Random, iters: int) -> None:
    for layout in ["row", "d16_planar"]:
        for kv_capacity in [16, 64, 257]:
            desc = make_k_desc(layout, kv_capacity, heads=3)
            for _ in range(iters):
                head = rng.randrange(3)
                token = rng.randrange(kv_capacity)
                d_word = rng.randrange(K_WORDS)
                qblock = rng.randrange(K_QBLOCKS)
                got_payload = desc_payload_byte_offset(desc, head, token, d_word)
                exp_payload = explicit_payload_byte_offset(layout, kv_capacity, head, token, d_word)
                got_scale = desc_scale_byte_offset(desc, head, token, qblock)
                exp_scale = explicit_scale_byte_offset(layout, kv_capacity, head, token, qblock)
                assert got_payload == exp_payload, (layout, kv_capacity, head, token, d_word, got_payload, exp_payload)
                assert got_scale == exp_scale, (layout, kv_capacity, head, token, qblock, got_scale, exp_scale)


def run(seed: int, iters: int) -> None:
    rng = random.Random(seed)
    test_constants()
    test_v4_q4_equivalence(rng, iters)
    test_block_table_and_tail_visibility(rng)
    test_packed16_k_descriptor_formulas(rng, iters)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CPU-only formula oracle for transactional QBlock V4_144 tail paging")
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0x51424C4F434B, help="random seed")
    ap.add_argument("--iters", type=int, default=128, help="randomized iterations per formula family")
    args = ap.parse_args(argv)
    run(args.seed, args.iters)
    print(f"transactional-v4-144-formula-oracle: PASS seed=0x{args.seed:x} iters={args.iters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
