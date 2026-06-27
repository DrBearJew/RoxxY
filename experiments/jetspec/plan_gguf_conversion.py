#!/usr/bin/env python3
"""Plan a JetSpec draft-head GGUF conversion from inert manifest artifacts.

This is a validator/planner, not a runtime converter. It does not write GGUF and
it does not download model weights. It proves that the observed HF safetensors
manifest is internally consistent with the planned Qwen3.6 JetSpec draft-head
schema before any real converter/loader is wired into llama.cpp.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys
from typing import Any


DEFAULT_MANIFEST = "manifests/JetSpec_jetspec-Qwen3.6-35B-A3B_main.manifest.json"
DEFAULT_TENSOR_MAP = "tensor_map_qwen36_head.json"
DEFAULT_OUTPUT = "conversion_plans/JetSpec_jetspec-Qwen3.6-35B-A3B_gguf_plan.json"


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_entries(config_summary: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    """Provisional GGUF metadata keys for the draft head.

    These names intentionally use a `jetspec.` namespace until runtime loader names
    are reviewed. They should not be emitted by production converters yet.
    """

    rope = config_summary.get("rope_parameters") or {}
    return [
        {"key": "general.architecture", "type": "string", "value": "jetspec_qwen3_draft_head"},
        {"key": "general.name", "type": "string", "value": source.get("repo", "JetSpec draft head")},
        {"key": "general.source.huggingface.repository", "type": "string", "value": source.get("repo")},
        {"key": "general.source.huggingface.sha", "type": "string", "value": source.get("model_sha")},
        {"key": "jetspec.architecture", "type": "string", "value": "qwen3_draft_head"},
        {"key": "jetspec.source_architecture", "type": "string", "value": "DFlashDraftModel"},
        {"key": "jetspec.block_size", "type": "uint32", "value": config_summary.get("block_size")},
        {"key": "jetspec.draft_depth", "type": "uint32", "value": config_summary.get("draft_depth")},
        {"key": "jetspec.causal_head", "type": "bool", "value": config_summary.get("causal_head")},
        {"key": "jetspec.mask_token_id", "type": "uint32", "value": config_summary.get("mask_token_id")},
        {"key": "jetspec.target_layer_ids", "type": "array:uint32", "value": config_summary.get("target_layer_ids")},
        {"key": "jetspec.num_target_layers", "type": "uint32", "value": config_summary.get("num_target_layers")},
        {"key": "jetspec.requires_target_embeddings", "type": "bool", "value": True},
        {"key": "jetspec.requires_target_lm_head", "type": "bool", "value": True},
        {"key": "jetspec.embedding_length", "type": "uint32", "value": config_summary.get("hidden_size")},
        {"key": "jetspec.feed_forward_length", "type": "uint32", "value": config_summary.get("intermediate_size")},
        {"key": "jetspec.block_count", "type": "uint32", "value": config_summary.get("num_hidden_layers")},
        {"key": "jetspec.attention.head_count", "type": "uint32", "value": config_summary.get("num_attention_heads")},
        {"key": "jetspec.attention.head_count_kv", "type": "uint32", "value": config_summary.get("num_key_value_heads")},
        {"key": "jetspec.attention.key_length", "type": "uint32", "value": config_summary.get("head_dim")},
        {"key": "jetspec.attention.value_length", "type": "uint32", "value": config_summary.get("head_dim")},
        {"key": "jetspec.rope.freq_base", "type": "float32", "value": rope.get("rope_theta")},
        {"key": "jetspec.attention.layer_norm_rms_epsilon", "type": "float32", "value": config_summary.get("rms_norm_eps")},
        {"key": "jetspec.vocab_size", "type": "uint32", "value": config_summary.get("vocab_size")},
        {"key": "jetspec.tensor_data_dtype", "type": "string", "value": config_summary.get("dtype")},
    ]


def _expected_tensor_shapes(config_summary: dict[str, Any]) -> dict[str, list[int]]:
    hidden = int(config_summary["hidden_size"])
    intermediate = int(config_summary["intermediate_size"])
    n_layers = int(config_summary["num_hidden_layers"])
    n_heads = int(config_summary["num_attention_heads"])
    n_kv_heads = int(config_summary["num_key_value_heads"])
    head_dim = int(config_summary["head_dim"])
    target_layer_ids = list(config_summary["target_layer_ids"])

    expected: dict[str, list[int]] = {
        "fc.weight": [hidden, len(target_layer_ids) * hidden],
        "hidden_norm.weight": [hidden],
        "norm.weight": [hidden],
    }

    for il in range(n_layers):
        prefix = f"layers.{il}"
        expected[f"{prefix}.input_layernorm.weight"] = [hidden]
        expected[f"{prefix}.post_attention_layernorm.weight"] = [hidden]
        expected[f"{prefix}.self_attn.q_proj.weight"] = [n_heads * head_dim, hidden]
        expected[f"{prefix}.self_attn.k_proj.weight"] = [n_kv_heads * head_dim, hidden]
        expected[f"{prefix}.self_attn.v_proj.weight"] = [n_kv_heads * head_dim, hidden]
        expected[f"{prefix}.self_attn.o_proj.weight"] = [hidden, n_heads * head_dim]
        expected[f"{prefix}.self_attn.q_norm.weight"] = [head_dim]
        expected[f"{prefix}.self_attn.k_norm.weight"] = [head_dim]
        expected[f"{prefix}.mlp.gate_proj.weight"] = [intermediate, hidden]
        expected[f"{prefix}.mlp.up_proj.weight"] = [intermediate, hidden]
        expected[f"{prefix}.mlp.down_proj.weight"] = [hidden, intermediate]
    return expected


def _validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    summary = manifest.get("config_summary") or {}
    arch = summary.get("architectures") or []
    if "DFlashDraftModel" not in arch:
        errors.append(f"expected DFlashDraftModel architecture, got {arch}")
    if summary.get("causal_head") is not True:
        errors.append("expected causal_head=true")
    if summary.get("block_size") != 16:
        errors.append(f"expected block_size=16, got {summary.get('block_size')}")
    if summary.get("draft_depth") != 15:
        errors.append(f"expected draft_depth=15, got {summary.get('draft_depth')}")
    if summary.get("target_layer_ids") != [1, 10, 19, 28, 37]:
        errors.append(f"unexpected target_layer_ids={summary.get('target_layer_ids')}")
    safetensors = manifest.get("safetensors") or {}
    st_summary = safetensors.get("summary") or {}
    if st_summary.get("tensor_count") != 91:
        errors.append(f"expected 91 tensors, got {st_summary.get('tensor_count')}")
    if st_summary.get("param_count") != 473_995_264:
        errors.append(f"expected 473995264 params, got {st_summary.get('param_count')}")
    if st_summary.get("byte_mismatch_count") != 0:
        errors.append(f"safetensors byte mismatches: {st_summary.get('byte_mismatch_tensors')}")
    if st_summary.get("dtype_counts") != {"BF16": 91}:
        errors.append(f"expected all BF16 tensors, got {st_summary.get('dtype_counts')}")
    return errors


def _validate_tensor_map(manifest: dict[str, Any], tensor_map: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    tensors = manifest["safetensors"]["tensors"]
    manifest_by_name = {t["name"]: t for t in tensors}
    map_tensors = tensor_map.get("tensors") or []
    map_by_name = {t.get("hf_name"): t for t in map_tensors}

    missing_from_map = sorted(set(manifest_by_name) - set(map_by_name))
    extra_in_map = sorted(set(map_by_name) - set(manifest_by_name))
    if missing_from_map:
        errors.append(f"tensor map missing {len(missing_from_map)} tensors: {missing_from_map[:8]}")
    if extra_in_map:
        errors.append(f"tensor map has {len(extra_in_map)} unexpected tensors: {extra_in_map[:8]}")

    planned_tensors: list[dict[str, Any]] = []
    seen_gguf: set[str] = set()
    for name in sorted(manifest_by_name):
        src = manifest_by_name[name]
        mapped = map_by_name.get(name)
        if mapped is None:
            continue
        gguf_name = mapped.get("gguf_name")
        if not isinstance(gguf_name, str) or not gguf_name:
            errors.append(f"tensor {name} has invalid gguf_name={gguf_name!r}")
            continue
        if gguf_name in seen_gguf:
            errors.append(f"duplicate gguf tensor name: {gguf_name}")
        seen_gguf.add(gguf_name)
        if mapped.get("shape") != src.get("shape"):
            errors.append(f"shape drift for {name}: map={mapped.get('shape')} manifest={src.get('shape')}")
        if mapped.get("dtype") != src.get("dtype"):
            errors.append(f"dtype drift for {name}: map={mapped.get('dtype')} manifest={src.get('dtype')}")
        planned_tensors.append(
            {
                "hf_name": name,
                "gguf_name": gguf_name,
                "dtype": src["dtype"],
                "shape": src["shape"],
                "numel": src["numel"],
                "nbytes": src["nbytes"],
                "source_data_offsets": src["data_offsets"],
            }
        )
    return errors, planned_tensors


def _validate_expected_shapes(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    summary = manifest["config_summary"]
    expected = _expected_tensor_shapes(summary)
    actual = {t["name"]: t["shape"] for t in manifest["safetensors"]["tensors"]}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        errors.append(f"missing expected tensors: {missing[:12]} count={len(missing)}")
    if extra:
        errors.append(f"unexpected tensors: {extra[:12]} count={len(extra)}")
    for name in sorted(set(expected) & set(actual)):
        if list(expected[name]) != list(actual[name]):
            errors.append(f"shape mismatch {name}: expected {expected[name]} actual {actual[name]}")
    return errors


def build_plan(manifest_path: pathlib.Path, tensor_map_path: pathlib.Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    tensor_map = _load_json(tensor_map_path)

    errors: list[str] = []
    errors.extend(_validate_manifest(manifest))
    errors.extend(_validate_expected_shapes(manifest))
    map_errors, planned_tensors = _validate_tensor_map(manifest, tensor_map)
    errors.extend(map_errors)

    st_summary = manifest["safetensors"]["summary"]
    source = manifest["source"]
    config_summary = manifest["config_summary"]
    metadata = _metadata_entries(config_summary, source)

    plan = {
        "schema": "llama.cpp.experiments.jetspec.gguf_conversion_plan.v1",
        "status": "pass" if not errors else "fail",
        "generated_at": _utc_now(),
        "inputs": {
            "manifest": str(manifest_path),
            "tensor_map": str(tensor_map_path),
        },
        "source": {
            "repo": source.get("repo"),
            "model_sha": source.get("model_sha"),
            "revision": source.get("revision", "main"),
        },
        "proposed_output": {
            "gguf_arch": "jetspec_qwen3_draft_head",
            "runtime_supported": False,
            "reason": "planned metadata/tensor schema only; no llama.cpp loader is wired",
            "tensor_count": st_summary.get("tensor_count"),
            "param_count": st_summary.get("param_count"),
            "tensor_payload_bytes": st_summary.get("data_bytes"),
            "file_size_floor_bytes": st_summary.get("data_bytes"),
        },
        "metadata": metadata,
        "tensors": planned_tensors,
        "validation": {
            "ok": not errors,
            "errors": errors,
            "expected_tensor_count": 91,
            "observed_tensor_count": len(planned_tensors),
        },
        "promotion_blockers": [
            "Production GGUF metadata key names are not accepted or loaded yet.",
            "No llama.cpp DFlashDraftModel/JetSpec draft-head graph loader exists yet.",
            "Target hidden tap extraction for layers [1,10,19,28,37] is not wired to this head.",
            "Target embed_tokens and lm_head sharing contract is not implemented for this GGUF.",
        ],
    }
    return plan


def parse_args(argv: list[str]) -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=here / DEFAULT_MANIFEST)
    parser.add_argument("--tensor-map", type=pathlib.Path, default=here / DEFAULT_TENSOR_MAP)
    parser.add_argument("--output", type=pathlib.Path, default=here / DEFAULT_OUTPUT)
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    plan = build_plan(args.manifest.resolve(), args.tensor_map.resolve())
    text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.stdout:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
        print(f"status={plan['status']} tensors={plan['validation']['observed_tensor_count']}")
        if plan["validation"]["errors"]:
            for error in plan["validation"]["errors"]:
                print(f"error: {error}")
    return 0 if plan["validation"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
