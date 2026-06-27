#!/usr/bin/env python3
"""Inspect a JetSpec HF draft-head repository and emit a JSON manifest.

This script is intentionally staged under experiments/jetspec/ and is not imported
or compiled by llama.cpp. It uses only Python stdlib and reads the safetensors
header via HTTP Range requests, so it can list tensor names/shapes without
pulling the multi-GB weight file.

Default target:
    JetSpec/jetspec-Qwen3.6-35B-A3B
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import pathlib
import re
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_REPO = "JetSpec/jetspec-Qwen3.6-35B-A3B"
DEFAULT_REVISION = "main"
USER_AGENT = "llama.cpp-jetspec-experiment/0.1"

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _quote_repo(repo: str) -> str:
    return urllib.parse.quote(repo.strip("/"), safe="/")


def _hf_url(repo: str, kind: str, revision: str, filename: str | None = None) -> str:
    repo_q = _quote_repo(repo)
    rev_q = urllib.parse.quote(revision, safe="")
    if kind == "api":
        return f"https://huggingface.co/api/models/{repo_q}"
    if filename is None:
        raise ValueError("filename is required for raw/resolve URLs")
    file_q = "/".join(urllib.parse.quote(part, safe="") for part in filename.split("/"))
    return f"https://huggingface.co/{repo_q}/{kind}/{rev_q}/{file_q}"


def _request(url: str, *, timeout: float, headers: dict[str, str] | None = None) -> urllib.response.addinfourl:
    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - explicit user-requested URL


def _fetch_bytes(url: str, *, timeout: float) -> bytes:
    with _request(url, timeout=timeout) as resp:
        return resp.read()


def _fetch_json(url: str, *, timeout: float) -> Any:
    return json.loads(_fetch_bytes(url, timeout=timeout).decode("utf-8"))


def _read_http_range(url: str, start: int, end: int, *, timeout: float) -> bytes:
    if start < 0 or end < start:
        raise ValueError(f"invalid range {start}-{end}")
    expected = end - start + 1
    headers = {"Range": f"bytes={start}-{end}"}
    with _request(url, timeout=timeout, headers=headers) as resp:
        status = getattr(resp, "status", None)
        if status == 206:
            data = resp.read(expected)
        else:
            # Some endpoints ignore Range. Read only the prefix needed to slice the
            # requested range, then close the stream; do not download the full file.
            prefix = resp.read(end + 1)
            data = prefix[start : end + 1]
    if len(data) != expected:
        raise RuntimeError(f"range read {start}-{end} returned {len(data)} bytes, expected {expected}")
    return data


def _read_safetensors_header_local(path: pathlib.Path, *, max_header_bytes: int) -> dict[str, Any]:
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"{path} is too small to be a safetensors file")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 0 or header_len > max_header_bytes:
            raise RuntimeError(f"unreasonable safetensors header length: {header_len}")
        header = f.read(header_len)
        if len(header) != header_len:
            raise RuntimeError(f"truncated safetensors header in {path}")
    return json.loads(header.decode("utf-8"))


def _read_safetensors_header_remote(url: str, *, timeout: float, max_header_bytes: int) -> dict[str, Any]:
    prefix = _read_http_range(url, 0, 7, timeout=timeout)
    header_len = struct.unpack("<Q", prefix)[0]
    if header_len <= 0 or header_len > max_header_bytes:
        raise RuntimeError(f"unreasonable safetensors header length: {header_len}")
    header = _read_http_range(url, 8, 8 + header_len - 1, timeout=timeout)
    return json.loads(header.decode("utf-8"))


def _numel(shape: list[int]) -> int:
    if not shape:
        return 1
    return math.prod(int(x) for x in shape)


def _safe_filename(repo: str, revision: str) -> str:
    text = f"{repo}@{revision}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _tensor_group(name: str) -> str:
    if name.startswith("layers."):
        parts = name.split(".")
        if len(parts) >= 4 and parts[0] == "layers" and parts[1].isdigit():
            return f"layers.{parts[1]}.{parts[2]}"
        return "layers"
    return name.split(".", 1)[0]


def _proposed_gguf_name(name: str) -> str:
    # Provisional only: keeps HF names recognizable while preventing collision
    # with target-model tensor namespaces.
    return f"draft.{name}"


def _summarize_tensors(header: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tensors: list[dict[str, Any]] = []
    dtype_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    param_count = 0
    data_bytes = 0
    byte_mismatch: list[str] = []

    for name in sorted(k for k in header.keys() if k != "__metadata__"):
        info = header[name]
        dtype = str(info.get("dtype", "UNKNOWN"))
        shape = [int(x) for x in info.get("shape", [])]
        offsets = info.get("data_offsets", [0, 0])
        begin, end = int(offsets[0]), int(offsets[1])
        nbytes = end - begin
        nparams = _numel(shape)
        group = _tensor_group(name)
        expected_nbytes = nparams * _DTYPE_BYTES.get(dtype, 0)
        if dtype in _DTYPE_BYTES and expected_nbytes != nbytes:
            byte_mismatch.append(name)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        group_counts[group] = group_counts.get(group, 0) + 1
        param_count += nparams
        data_bytes = max(data_bytes, end)
        tensors.append(
            {
                "name": name,
                "proposed_gguf_name": _proposed_gguf_name(name),
                "dtype": dtype,
                "shape": shape,
                "numel": nparams,
                "data_offsets": [begin, end],
                "nbytes": nbytes,
                "group": group,
            }
        )

    summary = {
        "tensor_count": len(tensors),
        "param_count": param_count,
        "data_bytes": data_bytes,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
        "byte_mismatch_count": len(byte_mismatch),
        "byte_mismatch_tensors": byte_mismatch[:50],
    }
    return tensors, summary


def _config_summary(config: dict[str, Any]) -> dict[str, Any]:
    dcfg = config.get("dflash_config") or {}
    return {
        "architectures": config.get("architectures", []),
        "model_type": config.get("model_type"),
        "dtype": config.get("dtype"),
        "block_size": config.get("block_size"),
        "draft_depth": int(config.get("block_size", 1)) - 1 if config.get("block_size") else None,
        "causal_head": dcfg.get("causal_head"),
        "mask_token_id": dcfg.get("mask_token_id"),
        "target_layer_ids": dcfg.get("target_layer_ids"),
        "num_target_layers": config.get("num_target_layers"),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "vocab_size": config.get("vocab_size"),
        "rope_parameters": config.get("rope_parameters"),
        "rms_norm_eps": config.get("rms_norm_eps"),
    }


def _validate_config(config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    arch = config.get("architectures") or []
    if "DFlashDraftModel" not in arch:
        warnings.append(f"architectures does not include DFlashDraftModel: {arch}")
    dcfg = config.get("dflash_config") or {}
    if dcfg.get("causal_head") is not True:
        warnings.append("dflash_config.causal_head is not true")
    if not dcfg.get("target_layer_ids"):
        warnings.append("dflash_config.target_layer_ids is empty/missing")
    if not dcfg.get("mask_token_id"):
        warnings.append("dflash_config.mask_token_id is empty/missing")
    if int(config.get("block_size", 0) or 0) <= 1:
        warnings.append("block_size is missing or <= 1")
    return warnings


def _api_file_list(api: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for sibling in api.get("siblings", []) or []:
        if "rfilename" in sibling:
            out.append({k: sibling[k] for k in sorted(sibling) if k in {"rfilename", "size", "blobId", "lfs"}})
    return out


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo
    revision = args.revision
    timeout = float(args.timeout)

    api_url = _hf_url(repo, "api", revision)
    config_url = _hf_url(repo, "raw", revision, "config.json")
    api = _fetch_json(api_url, timeout=timeout)
    config = _fetch_json(config_url, timeout=timeout)

    manifest: dict[str, Any] = {
        "schema": "llama.cpp.experiments.jetspec.hf_head_manifest.v1",
        "generated_at": _utc_now(),
        "source": {
            "repo": repo,
            "revision": revision,
            "api_url": api_url,
            "config_url": config_url,
            "model_sha": api.get("sha"),
            "last_modified": api.get("lastModified"),
            "siblings": _api_file_list(api),
        },
        "config": config,
        "config_summary": _config_summary(config),
        "validation_warnings": _validate_config(config),
    }

    if args.skip_safetensors:
        manifest["safetensors"] = {"skipped": True}
        return manifest

    if args.safetensors_path:
        st_path = pathlib.Path(args.safetensors_path).expanduser().resolve()
        header = _read_safetensors_header_local(st_path, max_header_bytes=args.max_header_bytes)
        st_source = {"kind": "local", "path": str(st_path)}
    else:
        st_url = _hf_url(repo, "resolve", revision, args.safetensors_file)
        header = _read_safetensors_header_remote(
            st_url,
            timeout=timeout,
            max_header_bytes=args.max_header_bytes,
        )
        st_source = {"kind": "remote_range", "url": st_url, "file": args.safetensors_file}

    tensors, tensor_summary = _summarize_tensors(header)
    manifest["safetensors"] = {
        "skipped": False,
        "source": st_source,
        "metadata": header.get("__metadata__", {}),
        "summary": tensor_summary,
        "tensors": tensors,
    }
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id (default: {DEFAULT_REPO})")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="HF revision/ref (default: main)")
    parser.add_argument("--safetensors-file", default="model.safetensors", help="safetensors filename in the repo")
    parser.add_argument("--safetensors-path", help="optional local safetensors path; avoids HTTP range reads")
    parser.add_argument("--skip-safetensors", action="store_true", help="only fetch API/config metadata")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    parser.add_argument("--max-header-bytes", type=int, default=64 * 1024 * 1024, help="safetensors header sanity limit")
    parser.add_argument("--output", help="manifest output path; default under experiments/jetspec/manifests")
    parser.add_argument("--stdout", action="store_true", help="write manifest to stdout instead of a file")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.stdout:
        sys.stdout.write(text)
        return 0

    if args.output:
        out = pathlib.Path(args.output).expanduser()
    else:
        here = pathlib.Path(__file__).resolve().parent
        out = here / "manifests" / f"{_safe_filename(args.repo, args.revision)}.manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    st = manifest.get("safetensors", {})
    tensor_count = st.get("summary", {}).get("tensor_count", "skipped") if isinstance(st, dict) else "unknown"
    print(f"wrote {out}")
    print(f"model_sha={manifest['source'].get('model_sha')} tensors={tensor_count}")
    warnings = manifest.get("validation_warnings") or []
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"inspect_hf_head.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
