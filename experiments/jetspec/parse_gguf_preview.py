#!/usr/bin/env python3
"""Parse and validate experimental JetSpec GGUF previews.

This is an inert, stdlib-only fixture parser for files produced by
`convert_jetspec_head_to_gguf.py`. It is not a llama.cpp loader. Its purpose is
to freeze the future loader's required metadata contract before compiled code is
modified.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import tempfile
from typing import Any, BinaryIO


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
GGML_TYPE_BF16 = 30

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

_TYPE_NAMES = {
    GGUF_TYPE_UINT8: "uint8",
    GGUF_TYPE_INT8: "int8",
    GGUF_TYPE_UINT16: "uint16",
    GGUF_TYPE_INT16: "int16",
    GGUF_TYPE_UINT32: "uint32",
    GGUF_TYPE_INT32: "int32",
    GGUF_TYPE_FLOAT32: "float32",
    GGUF_TYPE_BOOL: "bool",
    GGUF_TYPE_STRING: "string",
    GGUF_TYPE_ARRAY: "array",
    GGUF_TYPE_UINT64: "uint64",
    GGUF_TYPE_INT64: "int64",
    GGUF_TYPE_FLOAT64: "float64",
}

_SIMPLE_PACK = {
    GGUF_TYPE_UINT8: "B",
    GGUF_TYPE_INT8: "b",
    GGUF_TYPE_UINT16: "H",
    GGUF_TYPE_INT16: "h",
    GGUF_TYPE_UINT32: "I",
    GGUF_TYPE_INT32: "i",
    GGUF_TYPE_FLOAT32: "f",
    GGUF_TYPE_BOOL: "?",
    GGUF_TYPE_UINT64: "Q",
    GGUF_TYPE_INT64: "q",
    GGUF_TYPE_FLOAT64: "d",
}


class GGUFParseError(RuntimeError):
    """Raised when a preview file violates the GGUF/JetSpec fixture contract."""


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise GGUFParseError(f"unexpected EOF: wanted {n} bytes, got {len(data)}")
    return data


def _unpack(f: BinaryIO, fmt: str) -> tuple[Any, ...]:
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, _read_exact(f, size))


def _read_string(f: BinaryIO) -> str:
    (nbytes,) = _unpack(f, "Q")
    if nbytes > 1024 * 1024 * 1024:
        raise GGUFParseError(f"unreasonable string length: {nbytes}")
    return _read_exact(f, int(nbytes)).decode("utf-8")


def _read_simple(f: BinaryIO, gguf_type: int) -> Any:
    fmt = _SIMPLE_PACK.get(gguf_type)
    if fmt is None:
        raise GGUFParseError(f"type {gguf_type} is not scalar")
    (value,) = _unpack(f, fmt)
    return bool(value) if gguf_type == GGUF_TYPE_BOOL else value


def _read_value(f: BinaryIO) -> tuple[str, Any]:
    (gguf_type,) = _unpack(f, "I")
    if gguf_type == GGUF_TYPE_STRING:
        return "string", _read_string(f)
    if gguf_type == GGUF_TYPE_ARRAY:
        (subtype,) = _unpack(f, "I")
        (count,) = _unpack(f, "Q")
        if subtype == GGUF_TYPE_ARRAY:
            raise GGUFParseError("nested arrays are not supported by this fixture parser")
        if count > 10_000_000:
            raise GGUFParseError(f"unreasonable array length: {count}")
        if subtype == GGUF_TYPE_STRING:
            value = [_read_string(f) for _ in range(int(count))]
        else:
            value = [_read_simple(f, subtype) for _ in range(int(count))]
        return f"array:{_TYPE_NAMES.get(subtype, f'unknown:{subtype}')}", value
    if gguf_type in _SIMPLE_PACK:
        return _TYPE_NAMES[gguf_type], _read_simple(f, gguf_type)
    raise GGUFParseError(f"unknown GGUF value type: {gguf_type}")


def _pad(n: int, alignment: int = GGUF_ALIGNMENT) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _expected_data_start_after_tensor_info(f: BinaryIO) -> int:
    return _pad(f.tell())


def parse_gguf(path: pathlib.Path) -> dict[str, Any]:
    with path.open("rb") as f:
        magic = _read_exact(f, 4)
        if magic != GGUF_MAGIC:
            raise GGUFParseError(f"bad GGUF magic: {magic!r}")
        version, tensor_count, kv_count = _unpack(f, "IQQ")
        if version != GGUF_VERSION:
            raise GGUFParseError(f"unsupported GGUF version: {version}")

        metadata: dict[str, dict[str, Any]] = {}
        for _ in range(int(kv_count)):
            key = _read_string(f)
            type_name, value = _read_value(f)
            if key in metadata:
                raise GGUFParseError(f"duplicate metadata key: {key}")
            metadata[key] = {"type": type_name, "value": value}

        tensors: list[dict[str, Any]] = []
        for _ in range(int(tensor_count)):
            name = _read_string(f)
            (n_dims,) = _unpack(f, "I")
            if n_dims > 8:
                raise GGUFParseError(f"unreasonable tensor rank for {name}: {n_dims}")
            stored_dims = [_unpack(f, "Q")[0] for _ in range(int(n_dims))]
            shape = list(reversed([int(x) for x in stored_dims]))
            tensor_type, offset = _unpack(f, "IQ")
            tensors.append(
                {
                    "name": name,
                    "shape": shape,
                    "ggml_type": int(tensor_type),
                    "offset": int(offset),
                    "type_name": "BF16" if tensor_type == GGML_TYPE_BF16 else f"ggml:{tensor_type}",
                }
            )

        data_start = _expected_data_start_after_tensor_info(f)
        file_size = path.stat().st_size

    return {
        "path": str(path),
        "magic": "GGUF",
        "version": int(version),
        "tensor_count": int(tensor_count),
        "kv_count": int(kv_count),
        "metadata": metadata,
        "tensors": tensors,
        "data_start": data_start,
        "file_size": file_size,
    }


def _get(parsed: dict[str, Any], key: str) -> Any:
    try:
        return parsed["metadata"][key]["value"]
    except KeyError as exc:
        raise GGUFParseError(f"missing required JetSpec metadata key: {key}") from exc


def validate_tensor_payload_against_plan(parsed: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Validate a parsed 91-tensor JetSpec GGUF payload table against the plan.

    This is still no-model validation: it checks GGUF tensor-info names, shapes,
    BF16 types, monotonic offsets, and file-size bounds without executing a draft
    graph or inspecting tensor values.
    """

    errors: list[str] = []
    planned = list(plan.get("tensors") or [])
    parsed_tensors = list(parsed.get("tensors") or [])
    proposed = plan.get("proposed_output") or {}

    if parsed.get("tensor_count") != len(planned):
        errors.append(f"tensor_count mismatch: parsed={parsed.get('tensor_count')} plan={len(planned)}")
    if proposed.get("tensor_count") != len(planned):
        errors.append(f"plan tensor_count mismatch: proposed={proposed.get('tensor_count')} tensors={len(planned)}")

    expected_offset = 0
    seen_names: set[str] = set()
    checked: list[dict[str, Any]] = []
    for index, tensor in enumerate(planned):
        if index >= len(parsed_tensors):
            errors.append(f"missing parsed tensor at index {index}: {tensor.get('gguf_name')}")
            break
        actual = parsed_tensors[index]
        name = str(tensor.get("gguf_name"))
        if actual.get("name") != name:
            errors.append(f"tensor {index} name mismatch: parsed={actual.get('name')} plan={name}")
        if name in seen_names:
            errors.append(f"duplicate parsed tensor name: {name}")
        seen_names.add(name)
        shape = [int(x) for x in tensor.get("shape", [])]
        if actual.get("shape") != shape:
            errors.append(f"tensor {name} shape mismatch: parsed={actual.get('shape')} plan={shape}")
        if actual.get("ggml_type") != GGML_TYPE_BF16:
            errors.append(f"tensor {name} ggml_type must be BF16/{GGML_TYPE_BF16}, got {actual.get('ggml_type')}")
        if actual.get("offset") != expected_offset:
            errors.append(f"tensor {name} offset mismatch: parsed={actual.get('offset')} expected={expected_offset}")
        nbytes = int(tensor.get("nbytes", -1))
        if nbytes <= 0:
            errors.append(f"tensor {name} has invalid nbytes={nbytes}")
        checked.append({"name": name, "shape": shape, "offset": expected_offset, "nbytes": nbytes})
        expected_offset += _pad(nbytes)

    if len(parsed_tensors) > len(planned):
        errors.append(f"parsed tensor table has extra entries: {len(parsed_tensors)} > {len(planned)}")

    expected_payload_bytes = int(proposed.get("tensor_payload_bytes", expected_offset))
    if expected_offset != expected_payload_bytes:
        errors.append(f"payload byte total mismatch: offsets={expected_offset} proposed={expected_payload_bytes}")
    expected_file_floor = int(parsed.get("data_start", 0)) + expected_offset
    if int(parsed.get("file_size", 0)) < expected_file_floor:
        errors.append(f"GGUF file is truncated: size={parsed.get('file_size')} required>={expected_file_floor}")

    metadata_only = parsed.get("metadata", {}).get("jetspec.experimental.metadata_only", {}).get("value")
    if metadata_only is not False:
        errors.append("tensor payload GGUF must set jetspec.experimental.metadata_only=false")

    return {
        "ok": not errors,
        "errors": errors,
        "status": "tensor_payload_plan_validated_no_runtime" if not errors else "tensor_payload_plan_invalid",
        "tensor_count": parsed.get("tensor_count"),
        "planned_tensor_count": len(planned),
        "payload_bytes": expected_offset,
        "expected_file_floor": expected_file_floor,
        "checked_first": checked[:3],
        "checked_last": checked[-3:],
        "runtime_executed": False,
    }


def validate_jetspec_loader_contract(parsed: dict[str, Any], *, allow_tensor_payload: bool = False) -> dict[str, Any]:
    """Validate the future loader-visible JetSpec metadata contract."""

    errors: list[str] = []

    expected_values = {
        "general.architecture": "jetspec_qwen3_draft_head",
        "jetspec.architecture": "qwen3_draft_head",
        "jetspec.source_architecture": "DFlashDraftModel",
        "jetspec.block_size": 16,
        "jetspec.draft_depth": 15,
        "jetspec.causal_head": True,
        "jetspec.mask_token_id": 248070,
        "jetspec.target_layer_ids": [1, 10, 19, 28, 37],
        "jetspec.num_target_layers": 40,
        "jetspec.requires_target_embeddings": True,
        "jetspec.requires_target_lm_head": True,
        "jetspec.embedding_length": 2048,
        "jetspec.feed_forward_length": 6144,
        "jetspec.block_count": 8,
        "jetspec.attention.head_count": 32,
        "jetspec.attention.head_count_kv": 4,
        "jetspec.attention.key_length": 128,
        "jetspec.attention.value_length": 128,
        "jetspec.rope.freq_base": 10000000.0,
        "jetspec.attention.layer_norm_rms_epsilon": 0.000001,
        "jetspec.vocab_size": 248320,
        "jetspec.tensor_data_dtype": "bfloat16",
        "jetspec.experimental.preview": True,
        "jetspec.experimental.runtime_supported": False,
    }

    for key, expected in expected_values.items():
        try:
            actual = _get(parsed, key)
        except GGUFParseError as exc:
            errors.append(str(exc))
            continue
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-8:
                errors.append(f"{key}: expected {expected!r}, got {actual!r}")
        elif actual != expected:
            errors.append(f"{key}: expected {expected!r}, got {actual!r}")

    tensor_count = parsed.get("tensor_count")
    if tensor_count == 0:
        try:
            metadata_only = _get(parsed, "jetspec.experimental.metadata_only")
            if metadata_only is not True:
                errors.append("zero-tensor preview must set jetspec.experimental.metadata_only=true")
        except GGUFParseError as exc:
            errors.append(str(exc))
    elif tensor_count == 91 and allow_tensor_payload:
        bad_types = [t for t in parsed.get("tensors", []) if t.get("ggml_type") != GGML_TYPE_BF16]
        if bad_types:
            errors.append(f"tensor payload contains non-BF16 tensor info: {bad_types[:3]}")
        offsets = [t.get("offset") for t in parsed.get("tensors", [])]
        if offsets != sorted(offsets):
            errors.append("tensor payload offsets are not monotonic")
    else:
        errors.append(f"unexpected tensor_count={tensor_count}; expected 0 preview or 91 payload")

    block_size = parsed.get("metadata", {}).get("jetspec.block_size", {}).get("value")
    draft_depth = parsed.get("metadata", {}).get("jetspec.draft_depth", {}).get("value")
    if isinstance(block_size, int) and isinstance(draft_depth, int) and block_size != draft_depth + 1:
        errors.append(f"block_size/draft_depth mismatch: {block_size} != {draft_depth}+1")

    return {
        "ok": not errors,
        "errors": errors,
        "loader_contract": {
            "arch": parsed.get("metadata", {}).get("general.architecture", {}).get("value"),
            "head_arch": parsed.get("metadata", {}).get("jetspec.architecture", {}).get("value"),
            "target_layer_ids": parsed.get("metadata", {}).get("jetspec.target_layer_ids", {}).get("value"),
            "requires_target_embeddings": parsed.get("metadata", {}).get("jetspec.requires_target_embeddings", {}).get("value"),
            "requires_target_lm_head": parsed.get("metadata", {}).get("jetspec.requires_target_lm_head", {}).get("value"),
            "runtime_supported": parsed.get("metadata", {}).get("jetspec.experimental.runtime_supported", {}).get("value"),
            "tensor_count": parsed.get("tensor_count"),
        },
    }


def _build_preview_with_converter(converter: pathlib.Path, output: pathlib.Path) -> None:
    import subprocess

    cmd = [sys.executable, str(converter), "--write-metadata-only", "--output", str(output), "--force"]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def self_test() -> None:
    here = pathlib.Path(__file__).resolve().parent
    converter = here / "convert_jetspec_head_to_gguf.py"
    with tempfile.TemporaryDirectory(prefix="jetspec-gguf-parse-") as tmp_s:
        out = pathlib.Path(tmp_s) / "preview.gguf"
        _build_preview_with_converter(converter, out)
        parsed = parse_gguf(out)
        validation = validate_jetspec_loader_contract(parsed)
        if not validation["ok"]:
            raise AssertionError(validation["errors"])
        if parsed["kv_count"] != 30 or parsed["tensor_count"] != 0:
            raise AssertionError(f"unexpected preview counts: kv={parsed['kv_count']} tensors={parsed['tensor_count']}")
        print("self-test ok", json.dumps(validation["loader_contract"], sort_keys=True))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", nargs="?", type=pathlib.Path, help="GGUF preview to parse")
    parser.add_argument("--validate-jetspec-loader", action="store_true")
    parser.add_argument("--allow-tensor-payload", action="store_true", help="accept 91 tensor-info entries in addition to metadata-only previews")
    parser.add_argument("--validate-tensor-payload-plan", type=pathlib.Path, help="validate parsed 91-tensor GGUF table against a conversion plan")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.gguf is None:
        print("error: GGUF path is required unless --self-test is used", file=sys.stderr)
        return 2

    try:
        parsed = parse_gguf(args.gguf.resolve())
        output: dict[str, Any]
        if args.summary_only:
            output = {
                "path": parsed["path"],
                "version": parsed["version"],
                "kv_count": parsed["kv_count"],
                "tensor_count": parsed["tensor_count"],
                "file_size": parsed["file_size"],
                "data_start": parsed["data_start"],
                "metadata_keys": sorted(parsed["metadata"].keys()),
            }
        else:
            output = parsed
        if args.validate_tensor_payload_plan is not None:
            plan = json.loads(args.validate_tensor_payload_plan.read_text(encoding="utf-8"))
            validation = validate_tensor_payload_against_plan(parsed, plan)
            output = {"parsed": output, "validation": validation}
            if not validation["ok"]:
                print(json.dumps(output, indent=2, sort_keys=True))
                return 1
        if args.validate_jetspec_loader:
            validation = validate_jetspec_loader_contract(parsed, allow_tensor_payload=args.allow_tensor_payload)
            output = {"parsed": output, "validation": validation}
            if not validation["ok"]:
                print(json.dumps(output, indent=2, sort_keys=True))
                return 1
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except (OSError, GGUFParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
