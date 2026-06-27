#!/usr/bin/env python3
"""P1 standalone JetSpec draft-head loader prototype.

This is an inert, stdlib-only prototype under experiments/jetspec/. It is not a
llama.cpp model loader, is not wired into CMake, and does not make JetSpec GGUF
files runnable. It consumes the metadata-only GGUF preview, validates the staged
loader contract, maps metadata into the future `draft_head_metadata` shape from
`jetspec_runtime_contract.hpp`, and fails closed for runtime preparation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
from typing import Any

import convert_jetspec_head_to_gguf as converter
import parse_gguf_preview as preview


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_PLAN = HERE / "conversion_plans/JetSpec_jetspec-Qwen3.6-35B-A3B_gguf_plan.json"


class DraftHeadLoaderPrototypeError(RuntimeError):
    """Raised when the inert prototype cannot produce a valid loader plan."""


def _metadata_value(parsed: dict[str, Any], key: str) -> Any:
    try:
        return parsed["metadata"][key]["value"]
    except KeyError as exc:
        raise DraftHeadLoaderPrototypeError(f"missing required metadata key: {key}") from exc


def _int_list(value: Any, *, key: str) -> list[int]:
    if not isinstance(value, list):
        raise DraftHeadLoaderPrototypeError(f"{key} must be an array, got {type(value).__name__}")
    return [int(item) for item in value]


def map_draft_head_metadata(parsed: dict[str, Any]) -> dict[str, Any]:
    """Map parsed GGUF metadata to the inert C++ draft_head_metadata fields."""

    target_layer_ids = _int_list(_metadata_value(parsed, "jetspec.target_layer_ids"), key="jetspec.target_layer_ids")
    hidden_size = int(_metadata_value(parsed, "jetspec.embedding_length"))
    key_length = int(_metadata_value(parsed, "jetspec.attention.key_length"))
    value_length = int(_metadata_value(parsed, "jetspec.attention.value_length"))

    metadata = {
        "gguf_arch": str(_metadata_value(parsed, "general.architecture")),
        "head_arch": str(_metadata_value(parsed, "jetspec.architecture")),
        "source_arch": str(_metadata_value(parsed, "jetspec.source_architecture")),
        "block_size": int(_metadata_value(parsed, "jetspec.block_size")),
        "draft_depth": int(_metadata_value(parsed, "jetspec.draft_depth")),
        "mask_token_id": int(_metadata_value(parsed, "jetspec.mask_token_id")),
        "target_layers": int(_metadata_value(parsed, "jetspec.num_target_layers")),
        "target_layer_ids": target_layer_ids,
        "hidden_size": hidden_size,
        "concat_width": hidden_size * len(target_layer_ids),
        "draft_layers": int(_metadata_value(parsed, "jetspec.block_count")),
        "attention_heads": int(_metadata_value(parsed, "jetspec.attention.head_count")),
        "attention_heads_kv": int(_metadata_value(parsed, "jetspec.attention.head_count_kv")),
        "head_dim": key_length,
        "ffn_size": int(_metadata_value(parsed, "jetspec.feed_forward_length")),
        "vocab_size": int(_metadata_value(parsed, "jetspec.vocab_size")),
        "causal_head": bool(_metadata_value(parsed, "jetspec.causal_head")),
        "requires_target_embeddings": bool(_metadata_value(parsed, "jetspec.requires_target_embeddings")),
        "requires_target_lm_head": bool(_metadata_value(parsed, "jetspec.requires_target_lm_head")),
        "runtime_supported": bool(_metadata_value(parsed, "jetspec.experimental.runtime_supported")),
    }

    errors: list[str] = []
    if metadata["block_size"] != metadata["draft_depth"] + 1:
        errors.append("block_size must equal draft_depth + 1")
    if metadata["concat_width"] != 10240:
        errors.append(f"concat_width must be 10240, got {metadata['concat_width']}")
    if target_layer_ids != sorted(target_layer_ids) or len(target_layer_ids) != len(set(target_layer_ids)):
        errors.append(f"target_layer_ids must be sorted and unique, got {target_layer_ids}")
    if key_length != value_length:
        errors.append(f"key/value head_dim mismatch: key={key_length} value={value_length}")
    if metadata["runtime_supported"] is not False:
        errors.append("P1 preview metadata must preserve runtime_supported=false")
    if errors:
        raise DraftHeadLoaderPrototypeError("; ".join(errors))

    return metadata


def build_loader_plan_from_parsed(
    parsed: dict[str, Any],
    *,
    prepare_runtime: bool = False,
    allow_preview_runtime: bool = False,
) -> dict[str, Any]:
    """Build a P1 loader plan from a parsed GGUF preview.

    Default mode validates and maps metadata only. If `prepare_runtime` is true,
    the prototype intentionally applies runtime gates and returns ok=false for
    preview files unless the explicit experimental flag is supplied; even with
    the flag, current previews remain blocked because runtime_supported=false.
    """

    validation = preview.validate_jetspec_loader_contract(parsed)
    errors = list(validation["errors"])
    metadata: dict[str, Any] = {}

    if validation["ok"]:
        try:
            metadata = map_draft_head_metadata(parsed)
        except DraftHeadLoaderPrototypeError as exc:
            errors.append(str(exc))

    tensor_count = int(parsed.get("tensor_count", -1))
    preview_file = bool(parsed.get("metadata", {}).get("jetspec.experimental.preview", {}).get("value"))
    metadata_only = bool(parsed.get("metadata", {}).get("jetspec.experimental.metadata_only", {}).get("value"))
    payload_mode = "metadata_only" if tensor_count == 0 else "bf16_payload"

    if tensor_count != 0:
        errors.append("P1 loader prototype only accepts metadata-only previews; tensor payload validation belongs to P2")
    if tensor_count == 0 and metadata_only is not True:
        errors.append("metadata-only preview must set jetspec.experimental.metadata_only=true")

    runtime_failure: str | None = None
    runtime_errors: list[str] = []
    if prepare_runtime:
        if preview_file and not allow_preview_runtime:
            runtime_failure = "preview_not_allowed"
            runtime_errors.append("preview_not_allowed: pass --allow-preview-runtime only for explicit experimental inspection")
        elif not metadata.get("runtime_supported", False):
            runtime_failure = "unsupported_runtime"
            runtime_errors.append("unsupported_runtime: preview metadata preserves runtime_supported=false")
        elif tensor_count == 0:
            runtime_failure = "missing_tensor"
            runtime_errors.append("missing_tensor: metadata-only preview has no draft-head tensors")
    elif not metadata.get("runtime_supported", False):
        runtime_failure = "unsupported_runtime"

    errors.extend(runtime_errors)
    can_prepare_runtime = prepare_runtime and not errors and bool(metadata.get("runtime_supported")) and tensor_count != 0

    return {
        "ok": not errors,
        "errors": errors,
        "parsed_summary": {
            "path": parsed.get("path"),
            "version": parsed.get("version"),
            "kv_count": parsed.get("kv_count"),
            "tensor_count": tensor_count,
            "data_start": parsed.get("data_start"),
            "file_size": parsed.get("file_size"),
        },
        "loader_plan": {
            "metadata": metadata,
            "payload_mode": payload_mode,
            "tensors": [],
            "preview_file": preview_file,
            "metadata_only": metadata_only,
            "allow_preview_runtime": allow_preview_runtime,
        },
        "runtime_gate": {
            "prepare_runtime_requested": prepare_runtime,
            "allow_preview_runtime": allow_preview_runtime,
            "can_prepare_runtime": can_prepare_runtime,
            "failure": runtime_failure,
        },
        "validation": validation,
    }


def build_loader_plan(
    gguf_path: pathlib.Path,
    *,
    prepare_runtime: bool = False,
    allow_preview_runtime: bool = False,
) -> dict[str, Any]:
    parsed = preview.parse_gguf(gguf_path)
    return build_loader_plan_from_parsed(
        parsed,
        prepare_runtime=prepare_runtime,
        allow_preview_runtime=allow_preview_runtime,
    )


def _write_temp_preview(output: pathlib.Path) -> dict[str, Any]:
    plan = json.loads(DEFAULT_PLAN.read_text(encoding="utf-8"))
    return converter.write_gguf(output, plan, force=True)


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="jetspec-loader-p1-") as tmp_s:
        preview_path = pathlib.Path(tmp_s) / "preview.gguf"
        writer = _write_temp_preview(preview_path)

        inspect_plan = build_loader_plan(preview_path)
        if not inspect_plan["ok"]:
            raise AssertionError(inspect_plan["errors"])
        metadata = inspect_plan["loader_plan"]["metadata"]
        expected = {
            "gguf_arch": "jetspec_qwen3_draft_head",
            "head_arch": "qwen3_draft_head",
            "source_arch": "DFlashDraftModel",
            "block_size": 16,
            "draft_depth": 15,
            "mask_token_id": 248070,
            "target_layers": 40,
            "target_layer_ids": [1, 10, 19, 28, 37],
            "hidden_size": 2048,
            "concat_width": 10240,
            "draft_layers": 8,
            "attention_heads": 32,
            "attention_heads_kv": 4,
            "head_dim": 128,
            "ffn_size": 6144,
            "vocab_size": 248320,
            "causal_head": True,
            "requires_target_embeddings": True,
            "requires_target_lm_head": True,
            "runtime_supported": False,
        }
        if metadata != expected:
            raise AssertionError({"expected": expected, "got": metadata})

        blocked_no_flag = build_loader_plan(preview_path, prepare_runtime=True)
        if blocked_no_flag["ok"] or blocked_no_flag["runtime_gate"]["failure"] != "preview_not_allowed":
            raise AssertionError(blocked_no_flag)

        blocked_with_flag = build_loader_plan(preview_path, prepare_runtime=True, allow_preview_runtime=True)
        if blocked_with_flag["ok"] or blocked_with_flag["runtime_gate"]["failure"] != "unsupported_runtime":
            raise AssertionError(blocked_with_flag)

        return {
            "ok": True,
            "writer": writer,
            "metadata": metadata,
            "payload_mode": inspect_plan["loader_plan"]["payload_mode"],
            "runtime_gate_without_flag": blocked_no_flag["runtime_gate"],
            "runtime_gate_with_flag": blocked_with_flag["runtime_gate"],
        }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", nargs="?", type=pathlib.Path, help="metadata-only JetSpec GGUF preview")
    parser.add_argument("--prepare-runtime", action="store_true", help="apply runtime gates instead of metadata inspection only")
    parser.add_argument("--allow-preview-runtime", action="store_true", help="explicit experimental flag for preview runtime gate checks")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            print(json.dumps(self_test(), indent=2, sort_keys=True))
            return 0
        if args.gguf is None:
            print("error: GGUF path is required unless --self-test is used", file=sys.stderr)
            return 2
        result = build_loader_plan(
            args.gguf.resolve(),
            prepare_runtime=args.prepare_runtime,
            allow_preview_runtime=args.allow_preview_runtime,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    except (OSError, preview.GGUFParseError, DraftHeadLoaderPrototypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
