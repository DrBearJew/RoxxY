#!/usr/bin/env python3
"""Header-only probe for P5F JetSpec target tensor binding requirements.

This probe reads GGUF tensor-info headers only. It does not load model weights,
create a llama_context, execute JetSpec preflight, run a draft-head graph, build a
tree, mutate KV, or emit draft tokens.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

from parse_gguf_preview import GGUFParseError, parse_gguf

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_TARGET = pathlib.Path("/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-IQ4_XS-00001-of-00002.gguf")

# parse_gguf() reports user-facing shapes with GGUF dims reversed. The linker in
# src/models/jetspec_qwen3_draft_head.cpp checks raw ggml_tensor::ne order.
EXPECTED_TARGET_TENSORS: dict[str, dict[str, list[int]]] = {
    "token_embd.weight": {
        "gguf_shape": [248320, 2048],
        "linker_shape": [2048, 248320],
    },
    "output.weight": {
        "gguf_shape": [248320, 2048],
        "linker_shape": [2048, 248320],
    },
    "output_norm.weight": {
        "gguf_shape": [2048],
        "linker_shape": [2048],
    },
}

_SPLIT_RE = re.compile(r"^(?P<prefix>.*-)(?P<idx>\d{5})-of-(?P<count>\d{5})(?P<suffix>\.gguf)$")


class P5FTargetTensorProbeError(ValueError):
    """Raised when header-only target tensor binding validation fails."""


def _derive_split_paths(path: pathlib.Path) -> list[pathlib.Path]:
    match = _SPLIT_RE.match(path.name)
    if match is None:
        return [path]

    count = int(match.group("count"))
    prefix = match.group("prefix")
    suffix = match.group("suffix")
    return [path.with_name(f"{prefix}{i:05d}-of-{count:05d}{suffix}") for i in range(1, count + 1)]


def _target_paths_from_args(paths: list[pathlib.Path], *, default_target: pathlib.Path = DEFAULT_TARGET) -> list[pathlib.Path]:
    if not paths:
        paths = [default_target]

    expanded: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for path in paths:
        for split in _derive_split_paths(path):
            resolved = split.expanduser()
            if resolved not in seen:
                expanded.append(resolved)
                seen.add(resolved)
    return expanded


def _collect_tensors(parsed_headers: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    tensors: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for parsed in parsed_headers:
        source = str(parsed.get("path", "<unknown>"))
        for tensor in parsed.get("tensors") or []:
            name = tensor.get("name")
            if not isinstance(name, str):
                errors.append(f"invalid tensor name in {source}: {name!r}")
                continue
            if name in tensors:
                errors.append(f"duplicate tensor {name} across target split headers")
                continue
            enriched = dict(tensor)
            enriched["source"] = source
            tensors[name] = enriched
    return tensors, errors


def validate_target_tensor_headers(parsed_headers: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate already-parsed GGUF headers against the P5F target tensor contract."""

    errors: list[str] = []
    if not parsed_headers:
        errors.append("no target GGUF headers parsed")

    tensors, tensor_errors = _collect_tensors(parsed_headers)
    errors.extend(tensor_errors)

    checked: dict[str, dict[str, Any]] = {}
    for name, expected in EXPECTED_TARGET_TENSORS.items():
        actual = tensors.get(name)
        if actual is None:
            errors.append(f"missing target tensor {name}")
            continue
        actual_shape = [int(x) for x in actual.get("shape") or []]
        if actual_shape != expected["gguf_shape"]:
            errors.append(f"target tensor {name} shape {actual_shape} expected {expected['gguf_shape']}")
        checked[name] = {
            "source": actual.get("source"),
            "gguf_shape": actual_shape,
            "linker_shape": expected["linker_shape"],
            "ggml_type": actual.get("ggml_type"),
            "offset": actual.get("offset"),
        }

    split_tensor_count = sum(int(parsed.get("tensor_count") or 0) for parsed in parsed_headers)
    split_declared_total = None
    for parsed in parsed_headers:
        metadata = parsed.get("metadata") or {}
        value = (metadata.get("split.tensors.count") or {}).get("value")
        if value is not None:
            split_declared_total = int(value)
            break
    if split_declared_total is not None and split_tensor_count != split_declared_total:
        errors.append(f"split tensor count {split_tensor_count} expected declared total {split_declared_total}")

    ok = not errors
    return {
        "ok": ok,
        "status": "target_tensor_headers_verified_not_loaded" if ok else "target_tensor_headers_invalid",
        "errors": errors,
        "runtime_executed": False,
        "model_loaded": False,
        "context_created": False,
        "draft_tokens_emitted": False,
        "headers_checked": len(parsed_headers),
        "tensor_infos_seen": len(tensors),
        "split_tensor_count": split_tensor_count,
        "split_declared_total": split_declared_total,
        "checked_tensors": checked,
        "limitations": [
            "reads GGUF headers only",
            "does not load or mmap target weights",
            "does not instantiate target/draft llama_context",
            "does not execute common_speculative_jetspec_preflight",
            "does not execute draft-head graph/tree/rollback runtime",
        ],
    }


def probe_target_tensor_headers(paths: list[pathlib.Path], *, require_target: bool = True) -> dict[str, Any]:
    expanded = _target_paths_from_args(paths)
    missing = [str(path) for path in expanded if not path.exists()]
    if missing:
        result = {
            "ok": not require_target,
            "status": "target_tensor_headers_missing_target",
            "errors": [f"missing target GGUF split: {path}" for path in missing],
            "target_paths": [str(path) for path in expanded],
            "runtime_executed": False,
            "model_loaded": False,
            "context_created": False,
            "draft_tokens_emitted": False,
        }
        return result

    parsed_headers: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in expanded:
        try:
            parsed_headers.append(parse_gguf(path))
        except (OSError, GGUFParseError) as exc:
            errors.append(f"failed to parse {path}: {exc}")
    if errors:
        return {
            "ok": False,
            "status": "target_tensor_headers_parse_failed",
            "errors": errors,
            "target_paths": [str(path) for path in expanded],
            "runtime_executed": False,
            "model_loaded": False,
            "context_created": False,
            "draft_tokens_emitted": False,
        }

    result = validate_target_tensor_headers(parsed_headers)
    result["target_paths"] = [str(path) for path in expanded]
    return result


def _self_test() -> dict[str, Any]:
    parsed_headers = [
        {
            "path": "target-00001-of-00002.gguf",
            "tensor_count": 0,
            "metadata": {"split.tensors.count": {"value": 3}},
            "tensors": [],
        },
        {
            "path": "target-00002-of-00002.gguf",
            "tensor_count": 3,
            "metadata": {},
            "tensors": [
                {"name": "output.weight", "shape": [248320, 2048], "ggml_type": 8, "offset": 0},
                {"name": "output_norm.weight", "shape": [2048], "ggml_type": 0, "offset": 1},
                {"name": "token_embd.weight", "shape": [248320, 2048], "ggml_type": 8, "offset": 2},
            ],
        },
    ]
    return validate_target_tensor_headers(parsed_headers)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", type=pathlib.Path, default=[], help="target GGUF path or split member; may be repeated")
    parser.add_argument("--no-require-target", action="store_true", help="return ok if the default local target path is absent")
    parser.add_argument("--self-test", action="store_true", help="run an in-memory no-file smoke")
    parser.add_argument("--json", action="store_true", help="print machine-readable result")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        result = _self_test()
    else:
        result = probe_target_tensor_headers(args.target, require_target=not args.no_require_target)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result.get("ok"):
        print(result.get("status", "target_tensor_headers_verified_not_loaded"))
    else:
        print("P5F target tensor header probe failed", file=sys.stderr)
        for error in result.get("errors") or []:
            print(f"- {error}", file=sys.stderr)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
