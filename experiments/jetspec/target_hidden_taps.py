#!/usr/bin/env python3
"""Inert target-hidden tap contract helpers for JetSpec staging.

These pure-Python helpers model the shape/order contract that a future llama.cpp
integration must satisfy when exposing target hidden states to the JetSpec draft
head. They do not call llama.cpp or load model weights.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_PLAN = "conversion_plans/JetSpec_jetspec-Qwen3.6-35B-A3B_gguf_plan.json"
DEFAULT_TARGET_LAYER_IDS = (1, 10, 19, 28, 37)
DEFAULT_TARGET_HIDDEN_SIZE = 2048
DEFAULT_NUM_TARGET_LAYERS = 40
DEFAULT_BLOCK_SIZE = 16
DEFAULT_MASK_TOKEN_ID = 248070


class TapContractError(ValueError):
    """Raised when a target-hidden tap fixture violates the JetSpec contract."""


@dataclasses.dataclass(frozen=True)
class TargetTapSpec:
    """Future loader-visible target-hidden capture contract.

    `target_layer_ids` are target decoder layer indices. A tap for layer L is the
    post-layer output matching HF `hidden_states[L + 1]`, then all taps are
    concatenated in this exact layer-id order along feature dim.
    """

    target_layer_ids: tuple[int, ...] = DEFAULT_TARGET_LAYER_IDS
    target_hidden_size: int = DEFAULT_TARGET_HIDDEN_SIZE
    num_target_layers: int = DEFAULT_NUM_TARGET_LAYERS
    block_size: int = DEFAULT_BLOCK_SIZE
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID
    draft_fc_input_size: int = len(DEFAULT_TARGET_LAYER_IDS) * DEFAULT_TARGET_HIDDEN_SIZE

    @property
    def draft_depth(self) -> int:
        return self.block_size - 1

    @property
    def concatenated_width(self) -> int:
        return len(self.target_layer_ids) * self.target_hidden_size

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_layer_ids": list(self.target_layer_ids),
            "target_hidden_size": self.target_hidden_size,
            "num_target_layers": self.num_target_layers,
            "block_size": self.block_size,
            "draft_depth": self.draft_depth,
            "mask_token_id": self.mask_token_id,
            "draft_fc_input_size": self.draft_fc_input_size,
            "concatenated_width": self.concatenated_width,
            "tap_semantics": "post-layer output, equivalent to HF hidden_states[layer_id + 1]",
            "concat_order": "target_layer_ids order",
        }


def _metadata_dict(plan: dict[str, Any]) -> dict[str, Any]:
    return {entry["key"]: entry["value"] for entry in plan.get("metadata", [])}


def _find_fc_input_width(plan: dict[str, Any]) -> int:
    for tensor in plan.get("tensors", []):
        if tensor.get("hf_name") == "fc.weight" or tensor.get("gguf_name") == "draft.fc.weight":
            shape = [int(x) for x in tensor.get("shape", [])]
            if len(shape) != 2:
                raise TapContractError(f"fc.weight must be rank-2, got shape={shape}")
            return shape[1]
    raise TapContractError("conversion plan does not contain fc.weight")


def spec_from_plan(plan: dict[str, Any]) -> TargetTapSpec:
    meta = _metadata_dict(plan)
    return TargetTapSpec(
        target_layer_ids=tuple(int(x) for x in meta["jetspec.target_layer_ids"]),
        target_hidden_size=int(meta["jetspec.embedding_length"]),
        num_target_layers=int(meta["jetspec.num_target_layers"]),
        block_size=int(meta["jetspec.block_size"]),
        mask_token_id=int(meta["jetspec.mask_token_id"]),
        draft_fc_input_size=_find_fc_input_width(plan),
    )


def spec_from_fixture(data: dict[str, Any]) -> TargetTapSpec:
    spec = data.get("spec") or {}
    return TargetTapSpec(
        target_layer_ids=tuple(int(x) for x in spec.get("target_layer_ids", DEFAULT_TARGET_LAYER_IDS)),
        target_hidden_size=int(spec.get("target_hidden_size", DEFAULT_TARGET_HIDDEN_SIZE)),
        num_target_layers=int(spec.get("num_target_layers", DEFAULT_NUM_TARGET_LAYERS)),
        block_size=int(spec.get("block_size", DEFAULT_BLOCK_SIZE)),
        mask_token_id=int(spec.get("mask_token_id", DEFAULT_MASK_TOKEN_ID)),
        draft_fc_input_size=int(
            spec.get(
                "draft_fc_input_size",
                len(spec.get("target_layer_ids", DEFAULT_TARGET_LAYER_IDS))
                * int(spec.get("target_hidden_size", DEFAULT_TARGET_HIDDEN_SIZE)),
            )
        ),
    )


def validate_tap_spec(spec: TargetTapSpec) -> list[str]:
    errors: list[str] = []
    ids = spec.target_layer_ids
    if not ids:
        errors.append("target_layer_ids must be non-empty")
    if len(set(ids)) != len(ids):
        errors.append(f"target_layer_ids must be unique: {ids}")
    if list(ids) != sorted(ids):
        errors.append(f"target_layer_ids must be sorted ascending: {ids}")
    if any(layer_id < 0 for layer_id in ids):
        errors.append(f"target_layer_ids must be non-negative: {ids}")
    if any(layer_id >= spec.num_target_layers for layer_id in ids):
        errors.append(f"target_layer_ids must be < num_target_layers={spec.num_target_layers}: {ids}")
    if spec.target_hidden_size <= 0:
        errors.append(f"target_hidden_size must be positive: {spec.target_hidden_size}")
    if spec.num_target_layers <= 0:
        errors.append(f"num_target_layers must be positive: {spec.num_target_layers}")
    if spec.block_size <= 1:
        errors.append(f"block_size must be > 1: {spec.block_size}")
    if spec.mask_token_id < 0:
        errors.append(f"mask_token_id must be non-negative: {spec.mask_token_id}")
    if spec.draft_fc_input_size != spec.concatenated_width:
        errors.append(
            "draft_fc_input_size must equal len(target_layer_ids)*target_hidden_size: "
            f"{spec.draft_fc_input_size} != {len(ids)}*{spec.target_hidden_size}"
        )
    return errors


def require_valid_tap_spec(spec: TargetTapSpec) -> None:
    errors = validate_tap_spec(spec)
    if errors:
        raise TapContractError("; ".join(errors))


def block_output_ids(anchor_token_id: int, spec: TargetTapSpec, fill_tokens: Sequence[int] | None = None) -> tuple[int, ...]:
    """Return `[anchor, fill..., mask...]` of length `block_size`.

    This mirrors upstream `DraftHeadForward.block_output_ids`: normal draft calls
    use all mask tokens after the anchor; conditioned/tree calls may provide a
    prefix path that is truncated/padded with masks.
    """

    require_valid_tap_spec(spec)
    anchor = int(anchor_token_id)
    if anchor < 0:
        raise TapContractError(f"anchor_token_id must be non-negative: {anchor_token_id}")
    max_fill = spec.block_size - 1
    fill = [int(x) for x in (fill_tokens or [])][:max_fill]
    if any(x < 0 for x in fill):
        raise TapContractError(f"fill_tokens must be non-negative: {fill}")
    fill.extend([spec.mask_token_id] * (max_fill - len(fill)))
    return (anchor, *fill)


def _coerce_vector(layer_id: int, vector: Sequence[Any], width: int) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes, bytearray)):
        raise TapContractError(f"layer {layer_id} vector must be a numeric sequence, got text")
    values = tuple(float(x) for x in vector)
    if len(values) != width:
        raise TapContractError(f"layer {layer_id} width mismatch: expected {width}, got {len(values)}")
    return values


def capture_hidden_taps(layer_outputs: Mapping[int | str, Sequence[Any]], spec: TargetTapSpec) -> tuple[tuple[int, tuple[float, ...]], ...]:
    """Capture requested target hidden vectors in `target_layer_ids` order.

    Extra layer outputs are ignored. Missing requested layers and wrong widths are
    fail-closed errors. Returned vectors are immutable tuples so the side-channel
    capture cannot mutate the caller's greedy-path hidden buffers.
    """

    require_valid_tap_spec(spec)
    normalized = {int(layer_id): vector for layer_id, vector in layer_outputs.items()}
    taps: list[tuple[int, tuple[float, ...]]] = []
    for layer_id in spec.target_layer_ids:
        if layer_id not in normalized:
            raise TapContractError(f"missing target hidden tap for layer {layer_id}")
        taps.append((layer_id, _coerce_vector(layer_id, normalized[layer_id], spec.target_hidden_size)))
    return tuple(taps)


def concatenate_taps(taps: Sequence[tuple[int, Sequence[Any]]], spec: TargetTapSpec) -> tuple[float, ...]:
    """Concatenate tap vectors exactly as the draft head's `fc.weight` expects."""

    require_valid_tap_spec(spec)
    ids = tuple(int(layer_id) for layer_id, _ in taps)
    if ids != spec.target_layer_ids:
        raise TapContractError(f"tap order mismatch: expected {spec.target_layer_ids}, got {ids}")
    out: list[float] = []
    for layer_id, vector in taps:
        out.extend(_coerce_vector(int(layer_id), vector, spec.target_hidden_size))
    if len(out) != spec.draft_fc_input_size:
        raise TapContractError(f"concatenated width mismatch: expected {spec.draft_fc_input_size}, got {len(out)}")
    return tuple(out)


def build_draft_fc_input(layer_outputs: Mapping[int | str, Sequence[Any]], spec: TargetTapSpec) -> tuple[float, ...]:
    return concatenate_taps(capture_hidden_taps(layer_outputs, spec), spec)


def _coerce_logits_matrix(logits: Sequence[Sequence[Any]], *, label: str) -> tuple[tuple[float, ...], ...]:
    if isinstance(logits, (str, bytes, bytearray)):
        raise TapContractError(f"{label} must be a 2-D numeric sequence, got text")
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for row_idx, row in enumerate(logits):
        if isinstance(row, (str, bytes, bytearray)):
            raise TapContractError(f"{label}[{row_idx}] must be numeric, got text")
        values = tuple(float(x) for x in row)
        if not values:
            raise TapContractError(f"{label}[{row_idx}] must be non-empty")
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise TapContractError(f"{label} row width mismatch: expected {width}, got {len(values)}")
        rows.append(values)
    if not rows:
        raise TapContractError(f"{label} must contain at least one row")
    return tuple(rows)


def greedy_token_ids(logits: Sequence[Sequence[Any]]) -> tuple[int, ...]:
    """Return deterministic argmax token IDs, matching sampler input invariance checks."""

    rows = _coerce_logits_matrix(logits, label="logits")
    return tuple(max(range(len(row)), key=lambda idx: row[idx]) for row in rows)


def _generated_hidden_states(data: dict[str, Any], spec: TargetTapSpec) -> tuple[tuple[float, ...], ...]:
    generator = data.get("hidden_state_generator") or {}
    base = float(generator.get("base", 0.0))
    state_stride = float(generator.get("state_stride", 100000.0))
    element_stride = float(generator.get("element_stride", 1.0))
    count = int(generator.get("state_count", spec.num_target_layers + 1))
    if count <= max(spec.target_layer_ids) + 1:
        raise TapContractError(
            f"generated hidden state count {count} cannot cover max target layer {max(spec.target_layer_ids)}"
        )
    return tuple(
        tuple(base + state_idx * state_stride + element_idx * element_stride for element_idx in range(spec.target_hidden_size))
        for state_idx in range(count)
    )


def hidden_states_from_fixture(data: dict[str, Any], spec: TargetTapSpec) -> tuple[tuple[float, ...], ...]:
    """Return HF-style hidden_states where index 0 is embedding output."""

    if "hf_hidden_states" not in data:
        return _generated_hidden_states(data, spec)
    raw_states = data["hf_hidden_states"]
    if isinstance(raw_states, (str, bytes, bytearray)):
        raise TapContractError("hf_hidden_states must be a sequence, got text")
    return tuple(_coerce_vector(idx, row, spec.target_hidden_size) for idx, row in enumerate(raw_states))


def extract_context_feature_from_hidden_states(
    hidden_states: Sequence[Sequence[Any]],
    spec: TargetTapSpec,
) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...]]:
    """Fallback/HF contract: target layer L maps to hidden_states[L + 1]."""

    require_valid_tap_spec(spec)
    rows = tuple(_coerce_vector(idx, row, spec.target_hidden_size) for idx, row in enumerate(hidden_states))
    taps: list[tuple[int, tuple[float, ...]]] = []
    indices: list[tuple[int, int]] = []
    for layer_id in spec.target_layer_ids:
        hidden_state_index = layer_id + 1
        if hidden_state_index >= len(rows):
            raise TapContractError(
                f"missing HF hidden_states[{hidden_state_index}] for target layer {layer_id}"
            )
        taps.append((layer_id, rows[hidden_state_index]))
        indices.append((layer_id, hidden_state_index))
    return tuple(indices), concatenate_taps(taps, spec)


def _summary(values: Sequence[float]) -> dict[str, Any]:
    vals = tuple(float(x) for x in values)
    if not vals:
        raise TapContractError("cannot summarize an empty vector")
    return {
        "width": len(vals),
        "first": vals[0],
        "last": vals[-1],
        "sum": sum(vals),
        "head": list(vals[:3]),
        "tail": list(vals[-3:]),
    }


def _validate_target_binding(data: dict[str, Any], spec: TargetTapSpec) -> list[str]:
    target = data.get("target_model") or {}
    errors: list[str] = []
    if not target:
        return errors
    if bool(target.get("has_token_embeddings", True)) is not True:
        errors.append("target embeddings are unavailable")
    if bool(target.get("has_lm_head", True)) is not True:
        errors.append("target lm_head is unavailable")
    if int(target.get("hidden_size", spec.target_hidden_size)) != spec.target_hidden_size:
        errors.append(
            f"target hidden_size mismatch: {target.get('hidden_size')} != {spec.target_hidden_size}"
        )
    layer_count = int(target.get("layer_count", target.get("num_target_layers", spec.num_target_layers)))
    if layer_count < spec.num_target_layers or max(spec.target_layer_ids) >= layer_count:
        errors.append(
            f"target layer_count mismatch: layer_count={layer_count}, required_layers={list(spec.target_layer_ids)}"
        )
    return errors


def build_parity_result(data: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the P3 A/B tap-capture parity fixture without touching runtime code."""

    spec = spec_from_fixture(data)
    errors = validate_tap_spec(spec) + _validate_target_binding(data, spec)
    hf_indices: tuple[tuple[int, int], ...] = tuple()
    fallback_concat: tuple[float, ...] = tuple()
    hook_concat: tuple[float, ...] = tuple()
    hook_layer_ids: list[int] = []

    if not errors:
        try:
            hidden_states = hidden_states_from_fixture(data, spec)
            hf_indices, fallback_concat = extract_context_feature_from_hidden_states(hidden_states, spec)
            hook_outputs = data.get("hook_layer_outputs")
            if hook_outputs is None:
                hook_outputs = {layer_id: hidden_states[layer_id + 1] for layer_id in spec.target_layer_ids}
            hook_taps = capture_hidden_taps(hook_outputs, spec)
            hook_layer_ids = [layer_id for layer_id, _ in hook_taps]
            hook_concat = concatenate_taps(hook_taps, spec)
            if hook_concat != fallback_concat:
                errors.append("hook capture differs from HF hidden_states[layer_id + 1] fallback")
        except (KeyError, TypeError, ValueError, TapContractError) as exc:
            errors.append(str(exc))

    baseline_logits = _coerce_logits_matrix(data["baseline_logits"], label="baseline_logits")
    capture_logits = _coerce_logits_matrix(data.get("capture_logits", data["baseline_logits"]), label="capture_logits")
    baseline_greedy = tuple(int(x) for x in data.get("baseline_greedy_token_ids", greedy_token_ids(baseline_logits)))
    capture_greedy = tuple(int(x) for x in data.get("capture_greedy_token_ids", greedy_token_ids(capture_logits)))
    logits_match = baseline_logits == capture_logits
    greedy_match = baseline_greedy == capture_greedy
    if not logits_match:
        errors.append("hidden tap capture changed normal target logits")
    if not greedy_match:
        errors.append("hidden tap capture changed greedy output")

    target_hidden_width = len(fallback_concat) if fallback_concat else spec.concatenated_width
    return {
        "ok": not errors,
        "errors": errors,
        "spec": spec.as_dict(),
        "target_binding": {
            "has_token_embeddings": bool((data.get("target_model") or {}).get("has_token_embeddings", True)),
            "has_lm_head": bool((data.get("target_model") or {}).get("has_lm_head", True)),
            "hidden_size": int((data.get("target_model") or {}).get("hidden_size", spec.target_hidden_size)),
            "layer_count": int((data.get("target_model") or {}).get("layer_count", spec.num_target_layers)),
        },
        "hf_hidden_state_indices": [
            {"layer_id": layer_id, "hidden_states_index": hidden_state_index}
            for layer_id, hidden_state_index in hf_indices
        ],
        "capture_layer_ids": hook_layer_ids,
        "concat_order": list(spec.target_layer_ids),
        "target_hidden_shape": [1, 1, target_hidden_width],
        "target_hidden_summary": _summary(fallback_concat) if fallback_concat else None,
        "fallback_matches_hook_capture": bool(fallback_concat and hook_concat == fallback_concat),
        "baseline_greedy_token_ids": list(baseline_greedy),
        "capture_greedy_token_ids": list(capture_greedy),
        "logits_match": logits_match,
        "greedy_output_match": greedy_match,
        "capture_is_side_channel_only": logits_match and greedy_match,
    }


def evaluate_parity_fixture(data: dict[str, Any]) -> dict[str, Any]:
    result = build_parity_result(data)
    if not result["ok"]:
        raise TapContractError("; ".join(result["errors"]))
    return result


def evaluate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("mode") == "target_hidden_tap_parity":
        return evaluate_parity_fixture(data)

    spec = spec_from_fixture(data)
    require_valid_tap_spec(spec)
    taps = capture_hidden_taps(data["layer_outputs"], spec)
    fc_input = concatenate_taps(taps, spec)
    block_ids = block_output_ids(
        int(data["anchor_token_id"]),
        spec,
        fill_tokens=data.get("fill_tokens"),
    )
    return {
        "spec": spec.as_dict(),
        "capture_layer_ids": [layer_id for layer_id, _ in taps],
        "block_output_ids": list(block_ids),
        "target_hidden_concat": list(fc_input),
        "target_hidden_shape": [1, 1, len(fc_input)],
    }


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: list[str]) -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, default=here / DEFAULT_PLAN)
    parser.add_argument("--fixture", type=pathlib.Path, help="evaluate a target-hidden tap fixture JSON")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.fixture:
            result = evaluate_fixture(_load_json(args.fixture.resolve()))
        else:
            plan = _load_json(args.plan.resolve())
            spec = spec_from_plan(plan)
            errors = validate_tap_spec(spec)
            result = {
                "ok": not errors,
                "errors": errors,
                "spec": spec.as_dict(),
                "capture_contract": {
                    "tap_source": "post-layer output equivalent to HF hidden_states[layer_id + 1]",
                    "concat_order": "target_layer_ids order",
                    "target_hidden_shape": ["batch", "tokens", spec.concatenated_width],
                    "block_output_ids": "[anchor_token_id] + fill_tokens[:15] + mask_token_id padding",
                    "greedy_path_mutation": "forbidden; tap capture is a side-channel only",
                },
            }
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            for key, value in result.items():
                print(f"{key}: {value}")
        return 0 if result.get("ok", True) else 1
    except (OSError, KeyError, TypeError, ValueError, TapContractError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
