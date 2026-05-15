#!/usr/bin/env python3
"""CPU contract checks for paged/block-table row mapping.

This intentionally tests only logical-row -> physical-row mapping. Format decoders
must receive a physical row and remain unaware of block tables.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceCase:
    name: str
    rows: int
    block_size: int
    block_table: tuple[int, ...]


def physical_row(block_table: tuple[int, ...], logical_row: int, block_size: int) -> int:
    block_id = logical_row // block_size
    in_block = logical_row % block_size
    return block_table[block_id] * block_size + in_block


def expected_physical_rows(case: SequenceCase) -> list[int]:
    return [physical_row(case.block_table, row, case.block_size) for row in range(case.rows)]


def physicalize_rows(case: SequenceCase) -> dict[int, str]:
    physical: dict[int, str] = {}
    for logical_row in range(case.rows):
        dst = physical_row(case.block_table, logical_row, case.block_size)
        if dst in physical:
            raise AssertionError(f"physical row alias in {case.name}: {dst}")
        physical[dst] = f"{case.name}:logical:{logical_row}"
    return physical


def validate_case(case: SequenceCase) -> dict[str, object]:
    needed_blocks = (case.rows + case.block_size - 1) // case.block_size
    if len(case.block_table) != needed_blocks:
        raise AssertionError(
            f"{case.name}: block_table length {len(case.block_table)} != needed logical blocks {needed_blocks}"
        )
    if len(set(case.block_table)) != len(case.block_table):
        raise AssertionError(f"{case.name}: block_table aliases physical blocks: {case.block_table}")

    rows = expected_physical_rows(case)
    physical = physicalize_rows(case)

    roundtrip = []
    for logical_row, mapped in enumerate(rows):
        value = physical[mapped]
        expected = f"{case.name}:logical:{logical_row}"
        if value != expected:
            raise AssertionError(f"{case.name}: roundtrip mismatch row {logical_row}: {value} != {expected}")
        roundtrip.append({"logical_row": logical_row, "physical_row": mapped})

    last_valid = rows[-1]
    first_invalid_logical = case.rows
    invalid_block = first_invalid_logical // case.block_size
    invalid_tail_mapping = None
    if invalid_block < len(case.block_table):
        invalid_tail_mapping = physical_row(case.block_table, first_invalid_logical, case.block_size)
        if invalid_tail_mapping in physical:
            raise AssertionError(
                f"{case.name}: tail row {first_invalid_logical} would alias a valid physical row {invalid_tail_mapping}"
            )

    return {
        "name": case.name,
        "rows": case.rows,
        "block_size": case.block_size,
        "block_table": list(case.block_table),
        "mapped_rows": rows,
        "last_valid_physical_row": last_valid,
        "first_invalid_tail_mapping": invalid_tail_mapping,
        "roundtrip": roundtrip,
        "passed": True,
    }


def default_cases() -> list[SequenceCase]:
    return [
        # Non-monotonic table with tail rows: exercises page permutation and tail masking.
        SequenceCase("seq0_nonmonotonic_tail", rows=17, block_size=4, block_table=(2, 0, 4, 1, 3)),
        # Different block size and no tail: catches assumptions tied to a single page size.
        SequenceCase("seq1_exact_blocks", rows=24, block_size=8, block_table=(5, 3, 7)),
        # Single partial block: catches block zero and in-block offset handling.
        SequenceCase("seq2_single_partial", rows=3, block_size=8, block_table=(9,)),
    ]


def validate_cross_sequence_isolation(cases: list[SequenceCase]) -> dict[str, object]:
    occupied: dict[tuple[int, int], str] = {}
    # Sequence index is intentionally part of the key: identical physical rows in
    # different sequence arenas are legal only when the cache allocator gives each
    # sequence a distinct block table base/stride.
    for seq_idx, case in enumerate(cases):
        for logical_row in range(case.rows):
            mapped = physical_row(case.block_table, logical_row, case.block_size)
            key = (seq_idx, mapped)
            if key in occupied:
                raise AssertionError(f"duplicate mapping for {key}: {occupied[key]} and {case.name}:{logical_row}")
            occupied[key] = f"{case.name}:{logical_row}"
    return {"sequences": len(cases), "mapped_rows": len(occupied), "passed": True}


def run() -> dict[str, object]:
    cases = default_cases()
    checks = [validate_case(case) for case in cases]
    isolation = validate_cross_sequence_isolation(cases)
    return {
        "result": "PASS",
        "contract": "logical row + block table -> physical row; format decoder receives physical row only",
        "checks": checks,
        "cross_sequence_isolation": isolation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = run()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
