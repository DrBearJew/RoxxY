#!/usr/bin/env python3
"""llama.cpp KV-cache layout model for compressed-KV Triton parity tests.

This module mirrors the layout rules used by src/llama-kv-cache.cpp and
src/llama-graph.cpp for the compressed KV formats exercised in this directory.
It is deliberately CPU-only: the goal is metadata parity, not speed.
"""

from __future__ import annotations

from dataclasses import dataclass


DOMAIN_ORIGINAL = "original"
DOMAIN_FWHT = "fwht"


@dataclass(frozen=True)
class FormatSpec:
    name: str
    qk: int
    type_size: int
    domain: str
    get_view_kind: str


# Sizes come from ggml/src/ggml-common.h static_asserts:
#   block_tbq4_0    = sizeof(ggml_half) + QK_TBQ4 / 2      = 66 bytes
#   block_planar3_0 = sizeof(ggml_half) + QK_PLANAR3/4 + /8 = 50 bytes
#   block_iso3_0    = same as planar3_0                     = 50 bytes
#   block_q8_0      = sizeof(ggml_half) + QK8_0             = 34 bytes
FORMAT_SPECS: dict[str, FormatSpec] = {
    "tbq4_0": FormatSpec(
        name="tbq4_0",
        qk=128,
        type_size=66,
        domain=DOMAIN_FWHT,
        # get_k/get_v first expose TBQ4 as [n_embd_gqa, n_kv, n_stream];
        # build_attn_mha then reshapes to [D, n_head_kv, n_kv, n_stream]
        # before permuting to the attention kernel view.
        get_view_kind="merged_3d_then_graph_reshape",
    ),
    "planar3_0": FormatSpec(
        name="planar3_0",
        qk=128,
        type_size=50,
        domain=DOMAIN_ORIGINAL,
        # get_k/get_v expose Planar/Iso as [D, n_head_kv, n_kv, n_stream]
        # directly because only TBQ3/TBQ4 take the merged 3D branch.
        get_view_kind="head_explicit_4d",
    ),
    "iso3_0": FormatSpec(
        name="iso3_0",
        qk=128,
        type_size=50,
        domain=DOMAIN_ORIGINAL,
        get_view_kind="head_explicit_4d",
    ),
    "q8_0": FormatSpec(
        name="q8_0",
        qk=32,
        type_size=34,
        domain=DOMAIN_ORIGINAL,
        get_view_kind="head_explicit_4d",
    ),
}

COMPRESSED_KV_FORMATS = ("planar3_0", "iso3_0", "tbq4_0")


def format_spec(fmt: str) -> FormatSpec:
    try:
        return FORMAT_SPECS[fmt]
    except KeyError as exc:
        raise ValueError(f"unsupported format {fmt!r}") from exc


def row_size_bytes(fmt: str, ncols: int) -> int:
    """Mirror ggml_row_size(type, ncols) for the formats modeled here."""
    spec = format_spec(fmt)
    if ncols % spec.qk != 0:
        raise ValueError(f"{fmt}: ncols={ncols} is not divisible by block size {spec.qk}")
    return (ncols // spec.qk) * spec.type_size


def n_embd_gqa(d_head: int, n_head_kv: int) -> int:
    return d_head * n_head_kv


@dataclass(frozen=True)
class AttentionView:
    fmt: str
    d_head: int
    n_head_kv: int
    kv_size: int
    n_stream: int
    # n_kv is the active attention length passed to get_k/get_v. kv_size is the
    # cache capacity returned by get_size(); production uses it for stream stride
    # and set_rows global indices even when n_kv < kv_size.
    n_kv: int | None = None
    # stream_base models sinfo.s0/view_offs. Kernel sequence IDs are local to the
    # view, while set_rows indices use absolute stream IDs from sinfo.strm[].
    stream_base: int = 0

    @property
    def active_n_kv(self) -> int:
        return self.kv_size if self.n_kv is None else self.n_kv

    @property
    def n_embd_gqa(self) -> int:
        return n_embd_gqa(self.d_head, self.n_head_kv)

    @property
    def head_stride_bytes(self) -> int:
        # After build_attn_mha permute(0, 2, 1, 3), nb12 selects the KV head.
        return row_size_bytes(self.fmt, self.d_head)

    @property
    def token_stride_bytes(self) -> int:
        # After build_attn_mha permute(0, 2, 1, 3), nb11 selects the token row.
        return row_size_bytes(self.fmt, self.n_embd_gqa)

    @property
    def stream_stride_bytes(self) -> int:
        # get_k/get_v offset stream s by row_size(type, n_embd_gqa * get_size()) * s.
        return row_size_bytes(self.fmt, self.n_embd_gqa * self.kv_size)

    @property
    def attention_ne(self) -> tuple[int, int, int, int]:
        # K/V shape seen by CUDA FA after graph permute: [D, n_kv, n_head_kv, ns].
        return (self.d_head, self.active_n_kv, self.n_head_kv, self.n_stream)

    @property
    def attention_nb(self) -> tuple[int | None, int, int, int]:
        # nb0 is type-size/block metadata for quantized tensors and is not used by row mapping.
        return (None, self.token_stride_bytes, self.head_stride_bytes, self.stream_stride_bytes)

    def absolute_stream(self, stream: int) -> int:
        if not 0 <= stream < self.n_stream:
            raise ValueError(f"local stream {stream} outside [0, {self.n_stream})")
        return self.stream_base + stream

    def row_offset_bytes(self, *, stream: int, head: int, slot: int) -> int:
        if not 0 <= head < self.n_head_kv:
            raise ValueError(f"head {head} outside [0, {self.n_head_kv})")
        if not 0 <= slot < self.active_n_kv:
            raise ValueError(f"slot {slot} outside active n_kv [0, {self.active_n_kv})")
        return (
            self.absolute_stream(stream) * self.stream_stride_bytes
            + head * self.head_stride_bytes
            + slot * self.token_stride_bytes
        )

    def set_rows_global_index(self, *, stream: int, slot: int) -> int:
        # set_input_k_idxs/set_input_v_idxs use absolute_stream*get_size() + sinfo.idxs[s][i]
        # when v_trans is false, and cpy_k/cpy_v reshape the cache across streams.
        if not 0 <= slot < self.kv_size:
            raise ValueError(f"slot {slot} outside cache size [0, {self.kv_size})")
        return self.absolute_stream(stream) * self.kv_size + slot

    def set_rows_offset_bytes(self, *, stream: int, head: int, slot: int) -> int:
        global_index = self.set_rows_global_index(stream=stream, slot=slot)
        return global_index * self.token_stride_bytes + head * self.head_stride_bytes


def block_table_slot(block_table: tuple[int, ...], logical_row: int, block_size: int) -> int:
    if logical_row < 0:
        raise ValueError("logical_row must be non-negative")
    block_id = logical_row // block_size
    if block_id >= len(block_table):
        raise ValueError(f"logical_row {logical_row} needs block {block_id}, table has {len(block_table)} blocks")
    in_block = logical_row % block_size
    return block_table[block_id] * block_size + in_block


def reconstruct_block_table(slot_idxs: tuple[int, ...], block_size: int) -> tuple[int, ...] | None:
    """Return a block table if slot_idxs are block-contiguous, else None.

    llama.cpp slot_info can describe arbitrary per-token cell indices. A paged
    block table can represent only groups whose in-block offsets are contiguous
    within each logical block. This helper documents that adapter boundary.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not slot_idxs:
        return ()

    table: list[int] = []
    for logical_block_start in range(0, len(slot_idxs), block_size):
        chunk = slot_idxs[logical_block_start:logical_block_start + block_size]
        first = chunk[0]
        if first % block_size != 0:
            return None
        physical_block = first // block_size
        for in_block, slot in enumerate(chunk):
            if slot != physical_block * block_size + in_block:
                return None
        table.append(physical_block)
    if len(set(table)) != len(table):
        return None
    return tuple(table)


def domain_policy(fmt: str) -> dict[str, object]:
    spec = format_spec(fmt)
    return {
        "format": fmt,
        "domain": spec.domain,
        "rotate_q_before_attention": spec.domain == DOMAIN_FWHT,
        "rotate_o_after_attention": spec.domain == DOMAIN_FWHT,
    }
