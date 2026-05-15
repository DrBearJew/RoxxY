#!/usr/bin/env python3
"""Block-table metadata parity against llama.cpp KV row offsets.

llama.cpp currently stores KV rows by slot_info cell indices. Future paged/block
metadata must map logical rows to the same physical slot offsets before a C++
row mapper is allowed to replace the contiguous mapper.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from llama_cpp_kv_layout import (
    COMPRESSED_KV_FORMATS,
    AttentionView,
    block_table_slot,
    reconstruct_block_table,
)


@dataclass(frozen=True)
class BlockTableCase:
    name: str
    rows: int
    block_size: int
    block_table: tuple[int, ...]
    stream: int


@dataclass(frozen=True)
class OffsetCase:
    fmt: str
    d_head: int
    n_head_kv: int
    kv_size: int
    n_stream: int


def default_block_cases() -> list[BlockTableCase]:
    return [
        BlockTableCase("seq0_nonmonotonic_tail", rows=17, block_size=4, block_table=(2, 0, 4, 1, 3), stream=0),
        BlockTableCase("seq1_exact_blocks_same_slots_other_stream", rows=24, block_size=8, block_table=(5, 3, 7), stream=1),
        BlockTableCase("seq2_single_partial", rows=3, block_size=8, block_table=(1,), stream=2),
    ]


def default_offset_cases() -> list[OffsetCase]:
    return [
        OffsetCase(fmt=fmt, d_head=d_head, n_head_kv=n_head_kv, kv_size=80, n_stream=3)
        for fmt in COMPRESSED_KV_FORMATS
        for d_head in (128, 256)
        for n_head_kv in (1, 4)
    ]


def _validate_block_case(case: BlockTableCase, offset_case: OffsetCase) -> dict[str, object]:
    view = AttentionView(
        fmt=offset_case.fmt,
        d_head=offset_case.d_head,
        n_head_kv=offset_case.n_head_kv,
        kv_size=offset_case.kv_size,
        n_stream=offset_case.n_stream,
    )

    needed_blocks = (case.rows + case.block_size - 1) // case.block_size
    failures: list[str] = []
    if len(case.block_table) != needed_blocks:
        failures.append(f"block_table length {len(case.block_table)} != needed blocks {needed_blocks}")
    if len(set(case.block_table)) != len(case.block_table):
        failures.append("block table aliases physical blocks")

    slots: list[int] = []
    offsets: dict[int, str] = {}
    samples: list[dict[str, object]] = []
    for logical_row in range(case.rows):
        slot = block_table_slot(case.block_table, logical_row, case.block_size)
        slots.append(slot)
        if slot >= offset_case.kv_size:
            failures.append(f"slot {slot} exceeds kv_size {offset_case.kv_size}")
            continue
        for head in range(offset_case.n_head_kv):
            block_table_off = view.row_offset_bytes(stream=case.stream, head=head, slot=slot)
            llama_slot_off = view.set_rows_offset_bytes(stream=case.stream, head=head, slot=slot)
            if block_table_off != llama_slot_off:
                failures.append(
                    f"offset mismatch logical={logical_row} head={head}: {block_table_off} != {llama_slot_off}"
                )
            tag = f"logical={logical_row},head={head},slot={slot}"
            previous = offsets.get(block_table_off)
            if previous is not None:
                failures.append(f"offset alias {block_table_off}: {previous} and {tag}")
            offsets[block_table_off] = tag
            if len(samples) < 10:
                samples.append({
                    "logical_row": logical_row,
                    "physical_slot": slot,
                    "head": head,
                    "offset_bytes": block_table_off,
                })

    reconstructed = reconstruct_block_table(tuple(slots), case.block_size)
    if reconstructed != case.block_table:
        failures.append(f"reconstructed table {reconstructed} != {case.block_table}")

    return {
        "name": case.name,
        "format": offset_case.fmt,
        "D": offset_case.d_head,
        "n_head_kv": offset_case.n_head_kv,
        "stream": case.stream,
        "rows": case.rows,
        "block_size": case.block_size,
        "block_table": list(case.block_table),
        "physical_slots": slots,
        "offset_samples": samples,
        "checked_offsets": len(offsets),
        "passed": not failures,
        "failures": failures,
    }


def _validate_cross_stream_isolation(block_cases: list[BlockTableCase], offset_case: OffsetCase) -> dict[str, object]:
    view = AttentionView(
        fmt=offset_case.fmt,
        d_head=offset_case.d_head,
        n_head_kv=offset_case.n_head_kv,
        kv_size=offset_case.kv_size,
        n_stream=offset_case.n_stream,
    )
    occupied: dict[int, str] = {}
    failures: list[str] = []
    for case in block_cases:
        for logical_row in range(case.rows):
            slot = block_table_slot(case.block_table, logical_row, case.block_size)
            for head in range(offset_case.n_head_kv):
                off = view.row_offset_bytes(stream=case.stream, head=head, slot=slot)
                tag = f"{case.name}:stream={case.stream}:logical={logical_row}:head={head}:slot={slot}"
                previous = occupied.get(off)
                if previous is not None:
                    failures.append(f"cross-sequence alias {off}: {previous} and {tag}")
                occupied[off] = tag
    return {
        "format": offset_case.fmt,
        "D": offset_case.d_head,
        "n_head_kv": offset_case.n_head_kv,
        "mapped_offsets": len(occupied),
        "passed": not failures,
        "failures": failures,
    }


def _validate_nonrepresentable_slot_info() -> dict[str, object]:
    # llama.cpp slot_info can be arbitrary. A page table with fixed BLOCK_SIZE can
    # represent only block-contiguous slot groups. This negative case documents
    # why C++ paged mapping remains gated until exact metadata parity exists.
    bad_slot_idxs = (0, 2, 1, 3)
    good_slot_idxs = (8, 9, 10, 11, 0, 1, 2, 3)
    bad = reconstruct_block_table(bad_slot_idxs, block_size=2)
    good = reconstruct_block_table(good_slot_idxs, block_size=4)
    return {
        "bad_slot_idxs": list(bad_slot_idxs),
        "bad_reconstructed_block_table": None if bad is None else list(bad),
        "good_slot_idxs": list(good_slot_idxs),
        "good_reconstructed_block_table": None if good is None else list(good),
        "passed": bad is None and good == (2, 0),
    }


def run() -> dict[str, object]:
    block_cases = default_block_cases()
    offset_cases = default_offset_cases()
    checks = [
        _validate_block_case(block_case, offset_case)
        for offset_case in offset_cases
        for block_case in block_cases
    ]
    isolation = [_validate_cross_stream_isolation(block_cases, offset_case) for offset_case in offset_cases]
    representability = _validate_nonrepresentable_slot_info()
    passed = all(c["passed"] for c in checks) and all(c["passed"] for c in isolation) and representability["passed"]
    return {
        "result": "PASS" if passed else "FAIL",
        "contract": "block table logical rows must resolve to the same stream/head/slot byte offsets as llama.cpp slot_info rows",
        "checks": checks,
        "cross_stream_isolation": isolation,
        "slot_info_representability": representability,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = run()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
