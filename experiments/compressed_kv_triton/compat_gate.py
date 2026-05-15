#!/usr/bin/env python3
"""Triton compatibility gate for the compressed-KV experiment lane.

This script catches the vLLM #39664 class of failure before running heavier
prototypes: Triton API drift must not silently disable the fast path. It is
outside production CMake and only validates the isolated experiment lane.
"""

from __future__ import annotations

import importlib
import json
import platform
import py_compile
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

EXPERIMENT_DIR = Path(__file__).resolve().parent


_constexpr_function = getattr(triton, "constexpr_function", None)
if _constexpr_function is None:
    def _constexpr_function(fn):  # type: ignore[no-redef]
        return fn


@_constexpr_function
def _times_two(x):
    return x * 2


@triton.jit
def _constexpr_smoke_kernel(x, out, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    factor: tl.constexpr = _times_two(2)
    val = tl.load(x + offs, mask=mask, other=0.0).to(tl.float32) * factor
    tl.store(out + offs, val, mask=mask)


def _scan_sources() -> list[dict[str, object]]:
    forbidden = ["tl." + "constexpr_function", "triton.language." + "constexpr_function"]
    violations: list[dict[str, object]] = []
    for path in sorted(EXPERIMENT_DIR.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text()
        for pattern in forbidden:
            if pattern in text:
                violations.append({"file": str(path.relative_to(EXPERIMENT_DIR)), "pattern": pattern})
    return violations


EXPERIMENT_MODULES = (
    "compressed_kv_tl",
    "llama_cpp_kv_layout",
    "llama_cpp_tensor_layout_parity",
    "llama_cpp_block_table_parity",
    "dispatch_policy_contract",
    "materializers",
    "paged_materializers",
    "qk_only",
    "qk_2d_tiled",
    "online_softmax",
    "full_qkv",
    "qkv_2d_tiled",
    "varlen_qkv",
    "mask_semantics",
    "segmented_qkv",
    "compare_2d_segmented",
    "tbq4_domain_parity",
    "planar_iso_domain_parity",
    "autotune_metadata",
    "run_all_json",
)


def _compile_sources() -> list[dict[str, str]]:
    compiled: list[dict[str, str]] = []
    for path in sorted(EXPERIMENT_DIR.glob("*.py")):
        py_compile.compile(str(path), doraise=True)
        compiled.append({"file": path.name, "status": "py_compile_ok"})
    return compiled


def _import_experiment_modules() -> list[dict[str, str]]:
    imported: list[dict[str, str]] = []
    for name in EXPERIMENT_MODULES:
        importlib.import_module(name)
        imported.append({"module": name, "status": "imported"})
    return imported


def _compile_constexpr_smoke() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("torch reports no HIP/CUDA device")
    n = 256
    block = 256
    device = torch.device("cuda")
    x = torch.arange(n, device=device, dtype=torch.float32)
    out = torch.empty_like(x)
    _constexpr_smoke_kernel[(1,)](x, out, n, BLOCK=block)
    torch.cuda.synchronize()
    max_abs = torch.max(torch.abs(out - x * 4.0)).item()
    return {"max_abs_err": float(max_abs), "passed": max_abs == 0.0}


def main() -> int:
    if str(EXPERIMENT_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_DIR))

    info: dict[str, object] = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "triton_version": triton.__version__,
        "has_tl_constexpr": hasattr(tl, "constexpr"),
        "has_tl_constexpr_function": hasattr(tl, "constexpr_function"),
        "has_triton_constexpr_function": hasattr(triton, "constexpr_function"),
        "cuda_available": torch.cuda.is_available(),
    }

    failures: list[str] = []
    if not info["has_tl_constexpr"]:
        failures.append("triton.language.constexpr is missing")
    if not info["has_triton_constexpr_function"]:
        failures.append("triton.constexpr_function is missing")

    violations = _scan_sources()
    info["forbidden_source_references"] = violations
    if violations:
        failures.append("experiment sources reference tl.constexpr_function/triton.language.constexpr_function")

    try:
        info["compiled_sources"] = _compile_sources()
    except Exception as exc:  # fail loudly; no fallback path here
        failures.append(f"experiment source compile failed: {type(exc).__name__}: {exc}")

    try:
        info["imported_modules"] = _import_experiment_modules()
    except Exception as exc:  # fail loudly; no fallback path here
        failures.append(f"experiment module import failed: {type(exc).__name__}: {exc}")

    if torch.cuda.is_available() and info["has_triton_constexpr_function"]:
        try:
            info["constexpr_function_compile_smoke"] = _compile_constexpr_smoke()
            if not info["constexpr_function_compile_smoke"]["passed"]:  # type: ignore[index]
                failures.append("triton.constexpr_function compile smoke produced wrong result")
        except Exception as exc:
            failures.append(f"triton.constexpr_function compile smoke failed: {type(exc).__name__}: {exc}")
    else:
        failures.append("cannot run Triton compile smoke without HIP/CUDA and triton.constexpr_function")

    info["result"] = "PASS" if not failures else "FAIL"
    if failures:
        info["failures"] = failures

    print(json.dumps(info, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
