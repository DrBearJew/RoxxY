#!/usr/bin/env python3
"""Experimental JetSpec draft-head GGUF writer.

This stays under experiments/jetspec and is not wired into llama.cpp builds.
Default behavior is inspect-only. Metadata-only preview writing is allowed because
it writes zero tensors and is useful for checking provisional GGUF keys. Raw BF16
tensor payload writing is gated behind both a local safetensors path and an
explicit experimental flag.

The writer is stdlib-only on purpose: the local gguf-py import path currently
requires numpy, while the staging environment may not have numpy installed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import struct
import sys
import tempfile
from typing import Any, BinaryIO


DEFAULT_PLAN = "conversion_plans/JetSpec_jetspec-Qwen3.6-35B-A3B_gguf_plan.json"
DEFAULT_METADATA_OUTPUT = "gguf_previews/JetSpec_jetspec-Qwen3.6-35B-A3B.metadata-only.gguf"

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
GGML_QUANT_VERSION = 2
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

_VALUE_TYPES = {
    "uint8": GGUF_TYPE_UINT8,
    "int8": GGUF_TYPE_INT8,
    "uint16": GGUF_TYPE_UINT16,
    "int16": GGUF_TYPE_INT16,
    "uint32": GGUF_TYPE_UINT32,
    "int32": GGUF_TYPE_INT32,
    "float32": GGUF_TYPE_FLOAT32,
    "bool": GGUF_TYPE_BOOL,
    "string": GGUF_TYPE_STRING,
    "uint64": GGUF_TYPE_UINT64,
    "int64": GGUF_TYPE_INT64,
    "float64": GGUF_TYPE_FLOAT64,
}

_SIMPLE_PACK = {
    "uint8": "B",
    "int8": "b",
    "uint16": "H",
    "int16": "h",
    "uint32": "I",
    "int32": "i",
    "float32": "f",
    "bool": "?",
    "uint64": "Q",
    "int64": "q",
    "float64": "d",
}


class ConversionError(RuntimeError):
    """Raised for invalid plans, safetensors files, or unsafe write requests."""


def _here() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pack(fmt: str, *values: Any) -> bytes:
    return struct.pack("<" + fmt, *values)


def _pad(n: int, alignment: int = GGUF_ALIGNMENT) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _write_padding(f: BinaryIO, n: int, alignment: int = GGUF_ALIGNMENT) -> None:
    f.write(b"\x00" * (_pad(n, alignment) - n))


def _write_string(f: BinaryIO, value: str) -> None:
    raw = value.encode("utf-8")
    f.write(_pack("Q", len(raw)))
    f.write(raw)


def _write_typed_scalar(f: BinaryIO, type_name: str, value: Any) -> None:
    fmt = _SIMPLE_PACK[type_name]
    if type_name.startswith("uint") or type_name.startswith("int"):
        value = int(value)
    elif type_name.startswith("float"):
        value = float(value)
    elif type_name == "bool":
        value = bool(value)
    f.write(_pack(fmt, value))


def _write_value(f: BinaryIO, type_name: str, value: Any) -> None:
    if type_name == "string":
        f.write(_pack("I", GGUF_TYPE_STRING))
        _write_string(f, str(value))
        return

    if type_name.startswith("array:"):
        subtype = type_name.split(":", 1)[1]
        if subtype not in _VALUE_TYPES or subtype == "string" or subtype.startswith("array"):
            raise ConversionError(f"unsupported GGUF array subtype: {subtype}")
        if not isinstance(value, list):
            raise ConversionError(f"metadata value for array field must be list, got {type(value).__name__}")
        f.write(_pack("I", GGUF_TYPE_ARRAY))
        f.write(_pack("I", _VALUE_TYPES[subtype]))
        f.write(_pack("Q", len(value)))
        for item in value:
            _write_typed_scalar(f, subtype, item)
        return

    if type_name not in _VALUE_TYPES:
        raise ConversionError(f"unsupported GGUF metadata type: {type_name}")
    f.write(_pack("I", _VALUE_TYPES[type_name]))
    _write_typed_scalar(f, type_name, value)


def _write_kv(f: BinaryIO, entry: dict[str, Any]) -> None:
    key = entry.get("key")
    type_name = entry.get("type")
    if not isinstance(key, str) or not key:
        raise ConversionError(f"invalid metadata key: {key!r}")
    if not isinstance(type_name, str) or not type_name:
        raise ConversionError(f"invalid metadata type for {key}: {type_name!r}")
    if entry.get("value") is None:
        raise ConversionError(f"metadata key {key} has null value")
    _write_string(f, key)
    _write_value(f, type_name, entry["value"])


def _extra_metadata(plan: dict[str, Any], *, metadata_only: bool) -> list[dict[str, Any]]:
    proposed = plan.get("proposed_output") or {}
    return [
        {"key": "general.quantization_version", "type": "uint32", "value": GGML_QUANT_VERSION},
        {"key": "jetspec.experimental.preview", "type": "bool", "value": True},
        {"key": "jetspec.experimental.metadata_only", "type": "bool", "value": metadata_only},
        {"key": "jetspec.experimental.runtime_supported", "type": "bool", "value": bool(proposed.get("runtime_supported"))},
        {"key": "jetspec.experimental.source_plan_schema", "type": "string", "value": str(plan.get("schema", ""))},
    ]


def _merged_metadata(plan: dict[str, Any], *, metadata_only: bool) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in list(plan.get("metadata") or []) + _extra_metadata(plan, metadata_only=metadata_only):
        key = entry.get("key")
        if key in seen:
            raise ConversionError(f"duplicate metadata key in GGUF output plan: {key}")
        seen.add(key)
        merged.append(entry)
    return merged


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("status") != "pass":
        raise ConversionError(f"conversion plan is not pass-status: {plan.get('status')!r}")
    validation = plan.get("validation") or {}
    if validation.get("ok") is not True:
        raise ConversionError(f"conversion plan validation failed: {validation.get('errors')}")
    proposed = plan.get("proposed_output") or {}
    if proposed.get("runtime_supported") is not False:
        raise ConversionError("expected runtime_supported=false for inert JetSpec staging")

    tensors = list(plan.get("tensors") or [])
    if proposed.get("tensor_count") != len(tensors):
        raise ConversionError("planned tensor_count does not match tensor list length")

    seen_hf: set[str] = set()
    seen_gguf: set[str] = set()
    total_payload = 0
    last_end = 0
    for index, tensor in enumerate(tensors):
        hf_name = tensor.get("hf_name")
        gguf_name = tensor.get("gguf_name")
        if not isinstance(hf_name, str) or not hf_name:
            raise ConversionError(f"tensor {index} has invalid hf_name")
        if not isinstance(gguf_name, str) or not gguf_name:
            raise ConversionError(f"tensor {index} has invalid gguf_name")
        if hf_name in seen_hf:
            raise ConversionError(f"duplicate hf tensor name: {hf_name}")
        if gguf_name in seen_gguf:
            raise ConversionError(f"duplicate GGUF tensor name: {gguf_name}")
        seen_hf.add(hf_name)
        seen_gguf.add(gguf_name)
        if tensor.get("dtype") != "BF16":
            raise ConversionError(f"tensor {hf_name} must remain raw BF16, got {tensor.get('dtype')}")
        shape = [int(x) for x in tensor.get("shape", [])]
        if not shape or any(dim <= 0 for dim in shape):
            raise ConversionError(f"tensor {hf_name} has invalid shape: {shape}")
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = int(tensor.get("nbytes", -1))
        if nbytes != numel * 2:
            raise ConversionError(f"tensor {hf_name} BF16 byte count mismatch: nbytes={nbytes} numel={numel}")
        offsets = [int(x) for x in tensor.get("source_data_offsets", [])]
        if len(offsets) != 2 or offsets[0] < 0 or offsets[1] <= offsets[0]:
            raise ConversionError(f"tensor {hf_name} has invalid source_data_offsets: {offsets}")
        if offsets[0] < last_end:
            raise ConversionError(f"tensor {hf_name} source offsets overlap or regress: {offsets} after {last_end}")
        if offsets[1] - offsets[0] != nbytes:
            raise ConversionError(f"tensor {hf_name} source offset length does not match nbytes")
        last_end = offsets[1]
        total_payload += nbytes
    if proposed.get("tensor_payload_bytes") is not None and int(proposed.get("tensor_payload_bytes")) != total_payload:
        raise ConversionError("planned tensor_payload_bytes does not match summed tensor nbytes")


def _read_safetensors_header(path: pathlib.Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ConversionError(f"{path} is too small to be a safetensors file")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 0 or header_len > 512 * 1024 * 1024:
            raise ConversionError(f"unreasonable safetensors header length: {header_len}")
        raw = f.read(header_len)
        if len(raw) != header_len:
            raise ConversionError(f"truncated safetensors header in {path}")
    return json.loads(raw.decode("utf-8")), 8 + header_len


def _validate_safetensors_against_plan(path: pathlib.Path, plan: dict[str, Any]) -> int:
    header, data_start = _read_safetensors_header(path)
    file_size = path.stat().st_size
    max_end = 0

    for tensor in plan.get("tensors") or []:
        hf_name = tensor["hf_name"]
        info = header.get(hf_name)
        if info is None:
            raise ConversionError(f"local safetensors file is missing tensor {hf_name}")
        if info.get("dtype") != tensor.get("dtype"):
            raise ConversionError(f"dtype mismatch for {hf_name}: file={info.get('dtype')} plan={tensor.get('dtype')}")
        if [int(x) for x in info.get("shape", [])] != [int(x) for x in tensor.get("shape", [])]:
            raise ConversionError(f"shape mismatch for {hf_name}: file={info.get('shape')} plan={tensor.get('shape')}")
        offsets = [int(x) for x in info.get("data_offsets", [])]
        if offsets != [int(x) for x in tensor.get("source_data_offsets", [])]:
            raise ConversionError(f"offset mismatch for {hf_name}: file={offsets} plan={tensor.get('source_data_offsets')}")
        if offsets[1] - offsets[0] != int(tensor.get("nbytes", -1)):
            raise ConversionError(f"byte-size mismatch for {hf_name}: offsets={offsets} nbytes={tensor.get('nbytes')}")
        max_end = max(max_end, offsets[1])

    required_size = data_start + max_end
    if file_size < required_size:
        raise ConversionError(f"{path} is truncated: size={file_size}, required>={required_size}")
    return data_start


def _copy_exact(src: BinaryIO, dst: BinaryIO, nbytes: int) -> None:
    remaining = nbytes
    while remaining:
        chunk = src.read(min(1024 * 1024, remaining))
        if not chunk:
            raise ConversionError(f"unexpected EOF while copying tensor payload, {remaining} bytes remain")
        dst.write(chunk)
        remaining -= len(chunk)


def write_gguf(path: pathlib.Path, plan: dict[str, Any], *, safetensors_path: pathlib.Path | None = None, force: bool = False) -> dict[str, Any]:
    """Write a metadata-only or raw-BF16 JetSpec draft-head GGUF preview."""

    _validate_plan(plan)
    metadata_only = safetensors_path is None
    metadata = _merged_metadata(plan, metadata_only=metadata_only)
    tensors = [] if metadata_only else list(plan.get("tensors") or [])
    data_start = None

    if path.exists() and not force:
        raise ConversionError(f"refusing to overwrite existing output without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if safetensors_path is not None:
        data_start = _validate_safetensors_against_plan(safetensors_path, plan)
        for tensor in tensors:
            if tensor.get("dtype") != "BF16":
                raise ConversionError(f"only raw BF16 payload copy is supported, got {tensor.get('dtype')}")

    with path.open("wb") as f:
        f.write(GGUF_MAGIC)
        f.write(_pack("I", GGUF_VERSION))
        f.write(_pack("Q", len(tensors)))
        f.write(_pack("Q", len(metadata)))

        for entry in metadata:
            _write_kv(f, entry)

        offset = 0
        for tensor in tensors:
            _write_string(f, tensor["gguf_name"])
            shape = [int(x) for x in tensor["shape"]]
            f.write(_pack("I", len(shape)))
            for dim in reversed(shape):
                f.write(_pack("Q", dim))
            f.write(_pack("I", GGML_TYPE_BF16))
            f.write(_pack("Q", offset))
            offset += _pad(int(tensor["nbytes"]))

        _write_padding(f, f.tell())

        if safetensors_path is not None:
            assert data_start is not None
            with safetensors_path.open("rb") as src:
                for tensor in tensors:
                    begin, end = [int(x) for x in tensor["source_data_offsets"]]
                    nbytes = end - begin
                    src.seek(data_start + begin)
                    _copy_exact(src, f, nbytes)
                    _write_padding(f, nbytes)

    return {
        "output": str(path),
        "metadata_count": len(metadata),
        "tensor_count": len(tensors),
        "metadata_only": metadata_only,
        "bytes_written": path.stat().st_size,
    }


def summarize(plan: dict[str, Any]) -> dict[str, Any]:
    _validate_plan(plan)
    proposed = plan["proposed_output"]
    metadata = _merged_metadata(plan, metadata_only=True)
    return {
        "status": plan["status"],
        "runtime_supported": proposed.get("runtime_supported"),
        "planned_tensors": proposed.get("tensor_count"),
        "planned_tensor_payload_bytes": proposed.get("tensor_payload_bytes"),
        "metadata_preview_entries": len(metadata),
        "default_mode": "inspect-only",
        "metadata_only_write_supported": True,
        "tensor_payload_write_requires": ["--write-tensor-payload", "--safetensors PATH", "--experimental-write-tensor-payload"],
    }


def _tiny_self_test_plan() -> dict[str, Any]:
    return {
        "schema": "llama.cpp.experiments.jetspec.gguf_conversion_plan.self_test",
        "status": "pass",
        "proposed_output": {"runtime_supported": False, "tensor_count": 1, "tensor_payload_bytes": 4},
        "metadata": [
            {"key": "general.architecture", "type": "string", "value": "jetspec_qwen3_draft_head"},
            {"key": "jetspec.target_layer_ids", "type": "array:uint32", "value": [1, 10, 19, 28, 37]},
            {"key": "jetspec.causal_head", "type": "bool", "value": True},
            {"key": "jetspec.rope.freq_base", "type": "float32", "value": 10000000.0},
        ],
        "validation": {"ok": True, "errors": []},
        "tensors": [
            {
                "hf_name": "tiny.weight",
                "gguf_name": "draft.tiny.weight",
                "dtype": "BF16",
                "shape": [2],
                "numel": 2,
                "nbytes": 4,
                "source_data_offsets": [0, 4],
            }
        ],
    }


def _write_tiny_safetensors(path: pathlib.Path) -> None:
    header = {"tiny.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(_pack("Q", len(raw)) + raw + b"abcd")


def self_test() -> None:
    plan = _tiny_self_test_plan()
    with tempfile.TemporaryDirectory(prefix="jetspec-gguf-selftest-") as tmp_s:
        tmp = pathlib.Path(tmp_s)
        metadata_out = tmp / "metadata-only.gguf"
        payload_out = tmp / "payload.gguf"
        st = tmp / "tiny.safetensors"
        _write_tiny_safetensors(st)
        meta = write_gguf(metadata_out, plan)
        payload = write_gguf(payload_out, plan, safetensors_path=st)
        for output, expected_tensors in [(metadata_out, 0), (payload_out, 1)]:
            data = output.read_bytes()
            if data[:4] != GGUF_MAGIC:
                raise AssertionError(f"bad magic in {output}")
            version, tensor_count, kv_count = struct.unpack_from("<IQQ", data, 4)
            if version != GGUF_VERSION:
                raise AssertionError(f"bad version in {output}: {version}")
            if tensor_count != expected_tensors:
                raise AssertionError(f"bad tensor_count in {output}: {tensor_count}")
            if kv_count < 4:
                raise AssertionError(f"unexpected kv_count in {output}: {kv_count}")
        if b"abcd" not in payload_out.read_bytes():
            raise AssertionError("payload bytes were not copied into tensor GGUF")
        print("self-test ok", json.dumps({"metadata": meta, "payload": payload}, sort_keys=True))


def parse_args(argv: list[str]) -> argparse.Namespace:
    here = _here()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, default=here / DEFAULT_PLAN)
    parser.add_argument("--output", type=pathlib.Path, default=here / DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--write-metadata-only", action="store_true", help="write a valid zero-tensor GGUF metadata preview")
    parser.add_argument("--write-tensor-payload", action="store_true", help="write raw BF16 tensor bytes from a local safetensors file")
    parser.add_argument("--safetensors", type=pathlib.Path, help="local model.safetensors path; required for tensor payload writing")
    parser.add_argument("--experimental-write-tensor-payload", action="store_true", help="required safety acknowledgement for tensor payload writing")
    parser.add_argument("--force", action="store_true", help="overwrite output if it already exists")
    parser.add_argument("--json", action="store_true", help="print machine-readable summary")
    parser.add_argument("--self-test", action="store_true", help="run stdlib-only writer smoke tests")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    plan = _load_json(args.plan.resolve())

    try:
        if args.write_tensor_payload:
            if not args.experimental_write_tensor_payload:
                raise ConversionError("--write-tensor-payload requires --experimental-write-tensor-payload")
            if args.safetensors is None:
                raise ConversionError("--write-tensor-payload requires --safetensors PATH")
            safetensors_path = args.safetensors.resolve()
            if not safetensors_path.is_file():
                raise ConversionError(f"safetensors path is not a file: {safetensors_path}")
            result = write_gguf(args.output.resolve(), plan, safetensors_path=safetensors_path, force=args.force)
        elif args.write_metadata_only:
            result = write_gguf(args.output.resolve(), plan, force=args.force)
        else:
            result = summarize(plan)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
