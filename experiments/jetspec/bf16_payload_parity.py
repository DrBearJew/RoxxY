#!/usr/bin/env python3
"""P2 BF16 payload converter/loader parity fixture for JetSpec staging.

This remains inert under experiments/jetspec/. It validates the payload contract
that a future draft-head GGUF loader must enforce before any production CMake or
runtime integration exists:

- all 91 tensor names, dtypes, shapes, byte sizes, and safetensors offsets match
  the conversion plan;
- GGUF tensor-info offsets are the raw BF16 payload offsets the writer would use;
- no quantization, packing, or dtype conversion is introduced;
- malformed/missing tensor metadata fails closed.

The default fixture uses a synthetic safetensors header derived from the staged
conversion plan. It does not download model weights and does not copy the 948 MB
payload.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import tempfile
from collections.abc import Mapping
from typing import Any

import convert_jetspec_head_to_gguf as converter


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_PLAN = HERE / "conversion_plans/JetSpec_jetspec-Qwen3.6-35B-A3B_gguf_plan.json"
GGML_TYPE_BF16 = 30
GGUF_ALIGNMENT = 32
COPY_POLICY = "raw_bf16_no_transform"


class BF16PayloadParityError(ValueError):
    """Raised when a BF16 payload parity fixture violates the P2 contract."""


def _pad(n: int, alignment: int = GGUF_ALIGNMENT) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def load_plan(path: pathlib.Path = DEFAULT_PLAN) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plan_tensors(plan: dict[str, Any]) -> list[dict[str, Any]]:
    tensors = list(plan.get("tensors") or [])
    if not tensors:
        raise BF16PayloadParityError("conversion plan has no tensors")
    return tensors


def synthetic_safetensors_header(plan: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic safetensors header with all plan tensor entries."""

    return {
        str(tensor["hf_name"]): {
            "dtype": str(tensor["dtype"]),
            "shape": [int(x) for x in tensor["shape"]],
            "data_offsets": [int(x) for x in tensor["source_data_offsets"]],
        }
        for tensor in _plan_tensors(plan)
    }


def expected_gguf_tensor_offsets(plan: dict[str, Any]) -> dict[str, int]:
    """Return GGUF tensor payload offsets produced by the experimental writer."""

    offsets: dict[str, int] = {}
    offset = 0
    for tensor in _plan_tensors(plan):
        offsets[str(tensor["gguf_name"])] = offset
        offset += _pad(int(tensor["nbytes"]))
    return offsets


def _expected_by_hf(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_hf: dict[str, dict[str, Any]] = {}
    for tensor in _plan_tensors(plan):
        hf_name = str(tensor["hf_name"])
        if hf_name in by_hf:
            raise BF16PayloadParityError(f"duplicate HF tensor in plan: {hf_name}")
        by_hf[hf_name] = tensor
    return by_hf


def validate_header_against_plan(header: Mapping[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Validate safetensors header metadata against the 91-tensor conversion plan."""

    expected = _expected_by_hf(plan)
    expected_names = set(expected)
    observed_names = {str(key) for key in header if key != "__metadata__"}
    errors: list[str] = []

    missing = sorted(expected_names - observed_names)
    extra = sorted(observed_names - expected_names)
    if missing:
        errors.append(f"missing tensors: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if extra:
        errors.append(f"unexpected tensors: {extra[:5]}{'...' if len(extra) > 5 else ''}")

    dtype_set: set[str] = set()
    max_end = 0
    last_begin = -1
    monotonic_offsets = True

    for tensor in _plan_tensors(plan):
        hf_name = str(tensor["hf_name"])
        info = header.get(hf_name)
        if not isinstance(info, Mapping):
            continue
        dtype = str(info.get("dtype"))
        dtype_set.add(dtype)
        if dtype != "BF16" or str(tensor.get("dtype")) != "BF16":
            errors.append(f"{hf_name}: expected BF16 dtype, header={dtype!r}, plan={tensor.get('dtype')!r}")
        shape = [int(x) for x in info.get("shape", [])]
        expected_shape = [int(x) for x in tensor.get("shape", [])]
        if shape != expected_shape:
            errors.append(f"{hf_name}: shape mismatch header={shape} plan={expected_shape}")
        offsets = [int(x) for x in info.get("data_offsets", [])]
        expected_offsets = [int(x) for x in tensor.get("source_data_offsets", [])]
        if offsets != expected_offsets:
            errors.append(f"{hf_name}: data_offsets mismatch header={offsets} plan={expected_offsets}")
        if len(offsets) == 2:
            begin, end = offsets
            if begin < last_begin:
                monotonic_offsets = False
            last_begin = begin
            if end < begin:
                errors.append(f"{hf_name}: data_offsets end before begin: {offsets}")
            elif end - begin != int(tensor.get("nbytes", -1)):
                errors.append(f"{hf_name}: nbytes mismatch offsets={end - begin} plan={tensor.get('nbytes')}")
            max_end = max(max_end, end)
        else:
            errors.append(f"{hf_name}: data_offsets must have length 2, got {offsets}")

    if not monotonic_offsets:
        errors.append("safetensors data_offsets are not monotonic by plan order")

    proposed = plan.get("proposed_output") or {}
    expected_count = int(proposed.get("tensor_count", len(expected)))
    expected_payload_bytes = int(proposed.get("tensor_payload_bytes", max_end))
    if len(expected) != expected_count:
        errors.append(f"plan tensor_count mismatch: metadata={expected_count} tensors={len(expected)}")
    if max_end != expected_payload_bytes:
        errors.append(f"payload byte floor mismatch: header_max_end={max_end} plan={expected_payload_bytes}")

    return {
        "ok": not errors,
        "errors": errors,
        "expected_tensor_count": expected_count,
        "observed_tensor_count": len(observed_names),
        "dtype_set": sorted(dtype_set),
        "payload_bytes": max_end,
        "expected_payload_bytes": expected_payload_bytes,
        "offsets_match_safetensors_header": not errors,
        "monotonic_offsets": monotonic_offsets,
    }


def build_payload_loader_plan(plan: dict[str, Any], header: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_header_against_plan(header, plan)
    tensors: list[dict[str, Any]] = []
    gguf_offsets = expected_gguf_tensor_offsets(plan)

    if validation["ok"]:
        for tensor in _plan_tensors(plan):
            hf_name = str(tensor["hf_name"])
            gguf_name = str(tensor["gguf_name"])
            info = header[hf_name]
            tensors.append(
                {
                    "gguf_name": gguf_name,
                    "hf_name": hf_name,
                    "shape": [int(x) for x in tensor["shape"]],
                    "ggml_type": GGML_TYPE_BF16,
                    "nbytes": int(tensor["nbytes"]),
                    "offset": int(gguf_offsets[gguf_name]),
                    "source_data_offsets": [int(x) for x in info["data_offsets"]],
                    "copy_policy": COPY_POLICY,
                }
            )

    payload_contract = {
        "tensor_count": validation["observed_tensor_count"],
        "expected_tensor_count": validation["expected_tensor_count"],
        "payload_bytes": validation["payload_bytes"],
        "expected_payload_bytes": validation["expected_payload_bytes"],
        "dtype": "BF16" if validation["dtype_set"] == ["BF16"] else validation["dtype_set"],
        "ggml_type": GGML_TYPE_BF16,
        "copy_policy": COPY_POLICY,
        "quantization": "none",
        "packing": "none",
        "dtype_conversion": "none",
        "offsets_match_safetensors_header": validation["offsets_match_safetensors_header"],
        "gguf_offsets_monotonic": list(gguf_offsets.values()) == sorted(gguf_offsets.values()),
    }
    return {
        "ok": validation["ok"],
        "errors": validation["errors"],
        "payload_contract": payload_contract,
        "loader_tensors": tensors,
        "validation": validation,
    }


def compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    tensors = result.get("loader_tensors") or []
    samples = []
    if tensors:
        samples = [tensors[0], tensors[-1]]
    return {
        "ok": result["ok"],
        "errors": result["errors"],
        "payload_contract": result["payload_contract"],
        "sample_tensors": samples,
    }


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    plan_path = pathlib.Path(data.get("plan", DEFAULT_PLAN))
    if not plan_path.is_absolute():
        plan_path = HERE / plan_path
    plan = load_plan(plan_path)

    mode = data.get("mode", "synthetic_header_from_plan")
    if mode != "synthetic_header_from_plan":
        raise BF16PayloadParityError(f"unsupported fixture mode: {mode}")
    header = synthetic_safetensors_header(plan)
    mutations = data.get("mutations") or []
    for mutation in mutations:
        op = mutation.get("op")
        name = str(mutation.get("hf_name"))
        if op == "delete":
            header.pop(name, None)
        elif op == "set_shape":
            header[name]["shape"] = [int(x) for x in mutation["shape"]]
        elif op == "set_dtype":
            header[name]["dtype"] = str(mutation["dtype"])
        elif op == "set_offsets":
            header[name]["data_offsets"] = [int(x) for x in mutation["data_offsets"]]
        else:
            raise BF16PayloadParityError(f"unsupported mutation op: {op}")

    out = compact_summary(build_payload_loader_plan(plan, header))
    expected = data.get("expected")
    if expected is not None:
        for key_path, expected_value in expected.items():
            actual: Any = out
            for part in key_path.split("."):
                actual = actual[part]
            if actual != expected_value:
                raise BF16PayloadParityError(f"expected {key_path}={expected_value!r}, got {actual!r}")
    return out


def _write_sparse_safetensors(path: pathlib.Path, plan: dict[str, Any]) -> int:
    """Write a local sparse safetensors shell for converter header validation only."""

    header = synthetic_safetensors_header(plan)
    raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_bytes = int((plan.get("proposed_output") or {}).get("tensor_payload_bytes", 0))
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        data_start = f.tell()
        if payload_bytes > 0:
            f.seek(data_start + payload_bytes - 1)
            f.write(b"\0")
    return data_start


def self_test() -> dict[str, Any]:
    plan = load_plan(DEFAULT_PLAN)
    header = synthetic_safetensors_header(plan)
    good = build_payload_loader_plan(plan, header)
    if not good["ok"]:
        raise AssertionError(good["errors"])
    if len(good["loader_tensors"]) != 91:
        raise AssertionError(f"expected 91 loader tensors, got {len(good['loader_tensors'])}")
    if good["payload_contract"]["copy_policy"] != COPY_POLICY:
        raise AssertionError(good["payload_contract"])

    checks: list[tuple[str, dict[str, Any], str]] = []
    missing = dict(header)
    missing.pop("fc.weight")
    checks.append(("missing", missing, "missing tensors"))

    wrong_shape = json.loads(json.dumps(header))
    wrong_shape["fc.weight"]["shape"] = [2048, 10239]
    checks.append(("wrong_shape", wrong_shape, "shape mismatch"))

    wrong_dtype = json.loads(json.dumps(header))
    wrong_dtype["fc.weight"]["dtype"] = "F16"
    checks.append(("wrong_dtype", wrong_dtype, "expected BF16 dtype"))

    wrong_offsets = json.loads(json.dumps(header))
    wrong_offsets["fc.weight"]["data_offsets"] = [0, 4]
    checks.append(("wrong_offsets", wrong_offsets, "data_offsets mismatch"))

    rejected: dict[str, list[str]] = {}
    for label, bad_header, needle in checks:
        result = build_payload_loader_plan(plan, bad_header)
        if result["ok"]:
            raise AssertionError(f"{label} unexpectedly passed")
        if not any(needle in error for error in result["errors"]):
            raise AssertionError({"label": label, "needle": needle, "errors": result["errors"]})
        rejected[label] = result["errors"][:2]

    with tempfile.TemporaryDirectory(prefix="jetspec-bf16-payload-parity-") as tmp_s:
        sparse_path = pathlib.Path(tmp_s) / "synthetic.safetensors"
        data_start = _write_sparse_safetensors(sparse_path, plan)
        converter_data_start = converter._validate_safetensors_against_plan(sparse_path, plan)  # noqa: SLF001 - same inert fixture module
        if converter_data_start != data_start:
            raise AssertionError({"data_start": data_start, "converter_data_start": converter_data_start})

    summary = compact_summary(good)
    summary["rejected_cases"] = rejected
    summary["converter_sparse_header_validation"] = {
        "ok": True,
        "data_start": data_start,
        "file_size": 8 + len(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")) + good["payload_contract"]["payload_bytes"],
        "sparse_payload_written": False,
    }
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, default=DEFAULT_PLAN)
    parser.add_argument("--fixture", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--json", action="store_true", help="print default compact JSON summary")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            out = self_test()
        elif args.fixture:
            out = evaluate_fixture(json.loads(args.fixture.resolve().read_text(encoding="utf-8")))
        else:
            plan = load_plan(args.plan.resolve())
            out = compact_summary(build_payload_loader_plan(plan, synthetic_safetensors_header(plan)))
    except (OSError, KeyError, TypeError, ValueError, BF16PayloadParityError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if out.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
