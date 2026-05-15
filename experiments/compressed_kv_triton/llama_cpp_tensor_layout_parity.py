#!/usr/bin/env python3
"""llama.cpp tensor-layout parity checks for compressed-KV Triton work.

The test models the production metadata path:
  llama_kv_cache allocation -> set_rows global indices -> get_k/get_v views ->
  build_attn_mha permute -> CUDA FA row pointer arithmetic.

It does not run Triton kernels. It freezes the byte-offset contract that future
paged/block-table row mappers must match before any C++ mapper is promoted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from llama_cpp_kv_layout import (
    COMPRESSED_KV_FORMATS,
    AttentionView,
    format_spec,
    row_size_bytes,
)


@dataclass(frozen=True)
class LayoutCase:
    name: str
    fmt: str
    d_head: int
    n_head_kv: int
    kv_size: int
    n_stream: int
    slots: tuple[int, ...]


def _validate_case(case: LayoutCase) -> dict[str, object]:
    view = AttentionView(
        fmt=case.fmt,
        d_head=case.d_head,
        n_head_kv=case.n_head_kv,
        kv_size=case.kv_size,
        n_stream=case.n_stream,
    )
    spec = format_spec(case.fmt)

    failures: list[str] = []
    offsets: dict[int, str] = {}
    samples: list[dict[str, object]] = []

    expected_head_stride = row_size_bytes(case.fmt, case.d_head)
    expected_token_stride = row_size_bytes(case.fmt, case.d_head * case.n_head_kv)
    expected_stream_stride = row_size_bytes(case.fmt, case.d_head * case.n_head_kv * case.kv_size)

    if view.head_stride_bytes != expected_head_stride:
        failures.append("head stride mismatch")
    if view.token_stride_bytes != expected_token_stride:
        failures.append("token stride mismatch")
    if view.stream_stride_bytes != expected_stream_stride:
        failures.append("stream stride mismatch")
    if view.head_stride_bytes * case.n_head_kv != view.token_stride_bytes:
        failures.append("head strides do not tile one token row")
    if view.token_stride_bytes * case.kv_size != view.stream_stride_bytes:
        failures.append("token strides do not tile one stream")

    for stream in range(case.n_stream):
        for head in range(case.n_head_kv):
            for slot in case.slots:
                attn_off = view.row_offset_bytes(stream=stream, head=head, slot=slot)
                set_rows_off = view.set_rows_offset_bytes(stream=stream, head=head, slot=slot)
                if attn_off != set_rows_off:
                    failures.append(
                        f"set_rows/attention offset mismatch stream={stream} head={head} slot={slot}: "
                        f"{set_rows_off} != {attn_off}"
                    )

                tag = f"stream={stream},head={head},slot={slot}"
                previous = offsets.get(attn_off)
                if previous is not None:
                    failures.append(f"offset alias {attn_off}: {previous} and {tag}")
                offsets[attn_off] = tag

                if len(samples) < 12:
                    samples.append({
                        "stream": stream,
                        "head": head,
                        "slot": slot,
                        "set_rows_global_index": view.set_rows_global_index(stream=stream, slot=slot),
                        "offset_bytes": attn_off,
                    })

    expected_view_kind = "merged_3d_then_graph_reshape" if case.fmt == "tbq4_0" else "head_explicit_4d"
    if spec.get_view_kind != expected_view_kind:
        failures.append(f"view kind mismatch: {spec.get_view_kind} != {expected_view_kind}")

    return {
        "name": case.name,
        "format": case.fmt,
        "domain": spec.domain,
        "get_view_kind": spec.get_view_kind,
        "d_head": case.d_head,
        "n_head_kv": case.n_head_kv,
        "kv_size": case.kv_size,
        "n_stream": case.n_stream,
        "attention_ne": list(view.attention_ne),
        "attention_nb": list(view.attention_nb),
        "head_stride_bytes": view.head_stride_bytes,
        "token_stride_bytes": view.token_stride_bytes,
        "stream_stride_bytes": view.stream_stride_bytes,
        "offset_samples": samples,
        "checked_offsets": len(offsets),
        "passed": not failures,
        "failures": failures,
    }


def default_cases() -> list[LayoutCase]:
    cases: list[LayoutCase] = []
    for fmt in COMPRESSED_KV_FORMATS:
        for d_head in (128, 256):
            for n_head_kv in (1, 2, 8):
                cases.append(LayoutCase(
                    name=f"{fmt}_D{d_head}_H{n_head_kv}_single_stream",
                    fmt=fmt,
                    d_head=d_head,
                    n_head_kv=n_head_kv,
                    kv_size=32,
                    n_stream=1,
                    slots=(0, 1, 3, 7, 16, 31),
                ))
                cases.append(LayoutCase(
                    name=f"{fmt}_D{d_head}_H{n_head_kv}_multi_stream",
                    fmt=fmt,
                    d_head=d_head,
                    n_head_kv=n_head_kv,
                    kv_size=64,
                    n_stream=3,
                    slots=(0, 2, 5, 17, 33, 63),
                ))
    return cases


def run() -> dict[str, object]:
    checks = [_validate_case(case) for case in default_cases()]
    return {
        "result": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "objective": "prove Triton row/layout contracts against llama.cpp KV-cache metadata before C++ paged mapping",
        "source_contract": [
            "src/llama-kv-cache.cpp: cache tensors are 3D [n_embd_gqa, kv_size, n_stream]",
            "src/llama-kv-cache.cpp: set_rows indices are stream*kv_size + slot when v_trans=false",
            "src/llama-kv-cache.cpp: TBQ4 get_k/get_v use merged 3D view; Planar/Iso use head-explicit 4D view",
            "src/llama-graph.cpp: build_attn_mha reshapes TBQ4 to [D,Hkv,n_kv,ns] then permutes K/V to [D,n_kv,Hkv,ns]",
        ],
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = run()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
