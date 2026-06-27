#!/usr/bin/env python3
"""Validate P5F JetSpec binding facts against stored model artifacts.

This is an offline artifact check, not a runtime/model execution check. It ties
P5F source constants to the observed JetSpec draft-head safetensors manifest,
tensor map, and dry-run GGUF conversion plan. It intentionally does not create a
llama_context, execute the draft head, build a tree, or change CMake wiring.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

DEFAULT_MANIFEST = HERE / "manifests/JetSpec_jetspec-Qwen3.6-35B-A3B_main.manifest.json"
DEFAULT_TENSOR_MAP = HERE / "tensor_map_qwen36_head.json"
DEFAULT_PLAN = HERE / "conversion_plans/JetSpec_jetspec-Qwen3.6-35B-A3B_gguf_plan.json"
SOURCE_PATH = REPO_ROOT / "common/speculative.cpp"

EXPECTED_TARGET_LAYER_IDS = [1, 10, 19, 28, 37]
EXPECTED_ARTIFACT = {
    "repo": "JetSpec/jetspec-Qwen3.6-35B-A3B",
    "model_sha": "ffb38cf9917e0f426ab1b21d745e859f7788e467",
    "source_architecture": "DFlashDraftModel",
    "model_type": "qwen3",
    "dtype": "bfloat16",
    "block_size": 16,
    "draft_depth": 15,
    "mask_token_id": 248070,
    "target_hidden": 2048,
    "target_layer_count": 40,
    "target_tap_count": 5,
    "target_tap_width": 10240,
    "draft_layer_count": 8,
    "draft_head_count": 32,
    "draft_head_count_kv": 4,
    "head_dim": 128,
    "vocab_size": 248320,
    "tensor_count": 91,
    "param_count": 473_995_264,
    "tensor_payload_bytes": 947_990_528,
}

EXPECTED_SOURCE_CONSTANTS = {
    "JETSPEC_QWEN36_DRAFT_BLOCK_SIZE": EXPECTED_ARTIFACT["block_size"],
    "JETSPEC_QWEN36_TARGET_TAP_COUNT": EXPECTED_ARTIFACT["target_tap_count"],
    "JETSPEC_QWEN36_TARGET_HIDDEN": EXPECTED_ARTIFACT["target_hidden"],
    "JETSPEC_QWEN36_TARGET_LAYERS": EXPECTED_ARTIFACT["target_layer_count"],
    "JETSPEC_QWEN36_TARGET_TAP_WIDTH": EXPECTED_ARTIFACT["target_tap_width"],
    "JETSPEC_QWEN36_DRAFT_LAYERS": EXPECTED_ARTIFACT["draft_layer_count"],
    "JETSPEC_QWEN36_DRAFT_HEADS": EXPECTED_ARTIFACT["draft_head_count"],
    "JETSPEC_QWEN36_DRAFT_HEADS_KV": EXPECTED_ARTIFACT["draft_head_count_kv"],
    "JETSPEC_QWEN36_VOCAB_SIZE": EXPECTED_ARTIFACT["vocab_size"],
}

EXPECTED_METADATA = {
    "general.architecture": "jetspec_qwen3_draft_head",
    "jetspec.architecture": "qwen3_draft_head",
    "jetspec.source_architecture": "DFlashDraftModel",
    "jetspec.tensor_data_dtype": "bfloat16",
    "jetspec.block_size": 16,
    "jetspec.num_target_layers": 40,
    "jetspec.target_layer_ids": EXPECTED_TARGET_LAYER_IDS,
    "jetspec.embedding_length": 2048,
    "jetspec.block_count": 8,
    "jetspec.attention.head_count": 32,
    "jetspec.attention.head_count_kv": 4,
    "jetspec.vocab_size": 248320,
}


class P5FArtifactBindingError(ValueError):
    """Raised when offline P5F artifact-binding evidence is invalid."""


def _load_json(path: pathlib.Path) -> Any:
    if not path.exists():
        raise P5FArtifactBindingError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_dict(plan: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in plan.get("metadata") or []:
        key = item.get("key")
        if isinstance(key, str):
            out[key] = item.get("value")
    return out


def _parse_source_constants(source: str) -> tuple[dict[str, int], list[str]]:
    errors: list[str] = []
    constants: dict[str, int] = {}
    pattern = re.compile(r"static constexpr int32_t\s+(JETSPEC_QWEN36_[A-Z0-9_]+)\s*=\s*([^;]+);")
    for name, expr in pattern.findall(source):
        terms = [term.strip() for term in expr.split("*")]
        value = 1
        for term in terms:
            if term.isdigit():
                value *= int(term)
            elif term in constants:
                value *= constants[term]
            else:
                errors.append(f"cannot evaluate source constant {name}: unknown term {term!r} in {expr!r}")
                value = 0
                break
        constants[name] = value
    return constants, errors


def _expect(errors: list[str], what: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{what}: expected {expected!r}, got {actual!r}")


def _tensor_by_name(tensors: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(tensor.get("name")): tensor for tensor in tensors}


def _map_tensor_by_hf_name(tensors: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(tensor.get("hf_name")): tensor for tensor in tensors}


def validate_p5f_artifact_binding(
    manifest_path: pathlib.Path = DEFAULT_MANIFEST,
    tensor_map_path: pathlib.Path = DEFAULT_TENSOR_MAP,
    plan_path: pathlib.Path = DEFAULT_PLAN,
    source_path: pathlib.Path = SOURCE_PATH,
) -> dict[str, Any]:
    errors: list[str] = []

    try:
        manifest = _load_json(manifest_path)
        tensor_map = _load_json(tensor_map_path)
        plan = _load_json(plan_path)
    except (OSError, json.JSONDecodeError, P5FArtifactBindingError) as exc:
        return {
            "ok": False,
            "status": "artifact_binding_invalid",
            "errors": [str(exc)],
        }

    source = source_path.read_text(encoding="utf-8", errors="replace") if source_path.exists() else ""
    if not source:
        errors.append(f"missing P5F source file: {source_path}")

    config = manifest.get("config") or {}
    summary = manifest.get("config_summary") or {}
    st = (manifest.get("safetensors") or {}).get("summary") or {}
    tensors = (manifest.get("safetensors") or {}).get("tensors") or []
    source_info = manifest.get("source") or {}
    metadata = _metadata_dict(plan)
    planned = plan.get("proposed_output") or {}
    plan_validation = plan.get("validation") or {}
    map_runtime = tensor_map.get("required_runtime_target_features") or {}
    map_tensors = tensor_map.get("tensors") or []

    _expect(errors, "source repo", source_info.get("repo"), EXPECTED_ARTIFACT["repo"])
    _expect(errors, "source sha", source_info.get("model_sha"), EXPECTED_ARTIFACT["model_sha"])
    _expect(errors, "config architecture", summary.get("architectures"), [EXPECTED_ARTIFACT["source_architecture"]])
    _expect(errors, "model_type", summary.get("model_type"), EXPECTED_ARTIFACT["model_type"])
    _expect(errors, "dtype", summary.get("dtype"), EXPECTED_ARTIFACT["dtype"])
    _expect(errors, "causal_head", summary.get("causal_head"), True)
    _expect(errors, "block_size", summary.get("block_size"), EXPECTED_ARTIFACT["block_size"])
    _expect(errors, "draft_depth", summary.get("draft_depth"), EXPECTED_ARTIFACT["draft_depth"])
    _expect(errors, "mask_token_id", summary.get("mask_token_id"), EXPECTED_ARTIFACT["mask_token_id"])
    _expect(errors, "target_layer_ids", summary.get("target_layer_ids"), EXPECTED_TARGET_LAYER_IDS)
    _expect(errors, "num_target_layers", summary.get("num_target_layers"), EXPECTED_ARTIFACT["target_layer_count"])
    _expect(errors, "hidden_size", summary.get("hidden_size"), EXPECTED_ARTIFACT["target_hidden"])
    _expect(errors, "num_hidden_layers", summary.get("num_hidden_layers"), EXPECTED_ARTIFACT["draft_layer_count"])
    _expect(errors, "num_attention_heads", summary.get("num_attention_heads"), EXPECTED_ARTIFACT["draft_head_count"])
    _expect(errors, "num_key_value_heads", summary.get("num_key_value_heads"), EXPECTED_ARTIFACT["draft_head_count_kv"])
    _expect(errors, "head_dim", summary.get("head_dim"), EXPECTED_ARTIFACT["head_dim"])
    _expect(errors, "vocab_size", summary.get("vocab_size"), EXPECTED_ARTIFACT["vocab_size"])

    computed_tap_width = len(summary.get("target_layer_ids") or []) * int(summary.get("hidden_size") or 0)
    _expect(errors, "computed target tap width", computed_tap_width, EXPECTED_ARTIFACT["target_tap_width"])

    _expect(errors, "tensor_count", st.get("tensor_count"), EXPECTED_ARTIFACT["tensor_count"])
    _expect(errors, "param_count", st.get("param_count"), EXPECTED_ARTIFACT["param_count"])
    _expect(errors, "data_bytes", st.get("data_bytes"), EXPECTED_ARTIFACT["tensor_payload_bytes"])
    _expect(errors, "byte_mismatch_count", st.get("byte_mismatch_count"), 0)
    _expect(errors, "byte_mismatch_tensors", st.get("byte_mismatch_tensors"), [])
    _expect(errors, "dtype_counts", st.get("dtype_counts"), {"BF16": EXPECTED_ARTIFACT["tensor_count"]})

    by_name = _tensor_by_name(tensors)
    fc = by_name.get("fc.weight")
    if fc is None:
        errors.append("manifest missing fc.weight tensor")
    else:
        _expect(errors, "fc.weight dtype", fc.get("dtype"), "BF16")
        _expect(errors, "fc.weight shape", fc.get("shape"), [EXPECTED_ARTIFACT["target_hidden"], EXPECTED_ARTIFACT["target_tap_width"]])
        _expect(errors, "fc.weight nbytes", fc.get("nbytes"), 41_943_040)
    for forbidden in ["embed_tokens.weight", "lm_head.weight"]:
        if forbidden in by_name:
            errors.append(f"draft-head artifact must not include target-owned tensor {forbidden}")

    map_by_hf = _map_tensor_by_hf_name(map_tensors)
    mapped_fc = map_by_hf.get("fc.weight")
    if mapped_fc is None:
        errors.append("tensor map missing fc.weight")
    else:
        _expect(errors, "tensor-map fc.weight gguf_name", mapped_fc.get("gguf_name"), "draft.fc.weight")
        _expect(errors, "tensor-map fc.weight shape", mapped_fc.get("shape"), [EXPECTED_ARTIFACT["target_hidden"], EXPECTED_ARTIFACT["target_tap_width"]])

    _expect(errors, "runtime target requires embeddings", map_runtime.get("requires_target_embeddings"), True)
    _expect(errors, "runtime target requires lm_head", map_runtime.get("requires_target_lm_head"), True)
    _expect(errors, "runtime target layer ids", map_runtime.get("target_layer_ids"), EXPECTED_TARGET_LAYER_IDS)
    _expect(errors, "runtime block size", map_runtime.get("block_size"), EXPECTED_ARTIFACT["block_size"])

    _expect(errors, "plan validation ok", plan_validation.get("ok"), True)
    _expect(errors, "plan status", plan.get("status"), "pass")
    _expect(errors, "plan tensor_count", planned.get("tensor_count"), EXPECTED_ARTIFACT["tensor_count"])
    _expect(errors, "plan param_count", planned.get("param_count"), EXPECTED_ARTIFACT["param_count"])
    _expect(errors, "plan tensor_payload_bytes", planned.get("tensor_payload_bytes"), EXPECTED_ARTIFACT["tensor_payload_bytes"])
    _expect(errors, "plan runtime_supported", planned.get("runtime_supported"), False)

    for key, expected in EXPECTED_METADATA.items():
        _expect(errors, f"plan metadata {key}", metadata.get(key), expected)

    source_constants, constant_errors = _parse_source_constants(source)
    errors.extend(constant_errors)
    for key, expected in EXPECTED_SOURCE_CONSTANTS.items():
        _expect(errors, f"source constant {key}", source_constants.get(key), expected)

    for token in [
        "common_speculative_jetspec_preflight",
        "runtime_supported=false",
        "no draft tokens will be generated",
        "target_tap_width_vs_target",
        "llama_model_n_layer(model_tgt)",
        "llama_model_n_ctx_train(model_dft)",
    ]:
        if token not in source:
            errors.append(f"P5F source missing runtime-boundary token: {token}")

    # Keep this explicit: artifact checks are stronger than source-only contracts,
    # but they still do not prove live llama_context execution.
    not_runtime_executed = True
    artifact_verified = not errors
    status = "artifact_verified_not_runtime_executed" if artifact_verified else "artifact_binding_invalid"

    return {
        "ok": artifact_verified,
        "status": status,
        "errors": errors,
        "artifact_verified": artifact_verified,
        "runtime_executed": False,
        "not_runtime_executed": not_runtime_executed,
        "source_constants": {key: source_constants.get(key) for key in sorted(EXPECTED_SOURCE_CONSTANTS)},
        "artifact_binding": {
            "repo": source_info.get("repo"),
            "model_sha": source_info.get("model_sha"),
            "general_architecture": metadata.get("general.architecture"),
            "jetspec_architecture": metadata.get("jetspec.architecture"),
            "source_architecture": metadata.get("jetspec.source_architecture"),
            "tensor_dtype": metadata.get("jetspec.tensor_data_dtype"),
            "target_layer_ids": summary.get("target_layer_ids"),
            "target_hidden": summary.get("hidden_size"),
            "target_tap_count": len(summary.get("target_layer_ids") or []),
            "target_tap_width": computed_tap_width,
            "draft_block_size": summary.get("block_size"),
            "draft_layers": summary.get("num_hidden_layers"),
            "draft_heads": summary.get("num_attention_heads"),
            "draft_heads_kv": summary.get("num_key_value_heads"),
            "vocab_size": summary.get("vocab_size"),
            "tensor_count": st.get("tensor_count"),
            "bf16_tensor_count": (st.get("dtype_counts") or {}).get("BF16"),
            "byte_mismatch_count": st.get("byte_mismatch_count"),
            "fc_weight_shape": None if fc is None else fc.get("shape"),
            "requires_target_embeddings": map_runtime.get("requires_target_embeddings"),
            "requires_target_lm_head": map_runtime.get("requires_target_lm_head"),
            "runtime_supported": planned.get("runtime_supported"),
        },
        "limitations": [
            "does not instantiate llama_context",
            "does not load target GGUF into llama.cpp",
            "does not execute JetSpec draft-head graph",
            "does not build or verify a draft tree",
        ],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tensor-map", type=pathlib.Path, default=DEFAULT_TENSOR_MAP)
    parser.add_argument("--plan", type=pathlib.Path, default=DEFAULT_PLAN)
    parser.add_argument("--json", action="store_true", help="print machine-readable validation result")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = validate_p5f_artifact_binding(args.manifest.resolve(), args.tensor_map.resolve(), args.plan.resolve())
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        binding = result["artifact_binding"]
        print(
            "P5F artifact-binding validation passed "
            f"status={result['status']} tensors={binding['tensor_count']} "
            f"tap_width={binding['target_tap_width']} runtime_executed={result['runtime_executed']}"
        )
    else:
        print("P5F artifact-binding validation failed", file=sys.stderr)
        for error in result["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
