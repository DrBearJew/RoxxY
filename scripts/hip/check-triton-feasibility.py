#!/usr/bin/env python3
"""ROCm/Triton feasibility checkpoint for the isolated compressed-KV FA lane.

This script intentionally lives outside the llama.cpp runtime path. It verifies
that the local conda LLM environment can compile and launch a tiny Triton HIP
kernel before any C++ refactor depends on Triton-derived layout conclusions.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x, y, z, n: tl.constexpr, block: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    xv = tl.load(x + offs, mask=mask, other=0.0)
    yv = tl.load(y + offs, mask=mask, other=0.0)
    tl.store(z + offs, xv + yv, mask=mask)


def _rocminfo_gfx() -> str | None:
    try:
        out = subprocess.check_output(["rocminfo"], stderr=subprocess.DEVNULL, text=True, timeout=10)
    except Exception:
        return None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Name:") and "gfx" in line:
            return line.split("Name:", 1)[1].strip()
    return None


def main() -> int:
    info: dict[str, object] = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "rocminfo_gfx": _rocminfo_gfx(),
    }

    if not torch.cuda.is_available():
        print(json.dumps(info, indent=2, sort_keys=True))
        print("FAIL: torch reports no HIP/CUDA device", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    info["device_name"] = torch.cuda.get_device_name(0)
    info["device_capability"] = torch.cuda.get_device_capability(0)

    n = 4096
    block = 256
    x = torch.arange(n, device=device, dtype=torch.float32)
    y = torch.arange(n, device=device, dtype=torch.float32) * 0.5
    z = torch.empty_like(x)

    grid = (triton.cdiv(n, block),)
    _vector_add_kernel[grid](x, y, z, n, block=block)
    torch.cuda.synchronize()

    err = torch.max(torch.abs(z - (x + y))).item()
    info["vector_add_max_abs_err"] = err
    info["result"] = "PASS" if err == 0.0 else "FAIL"
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0 if err == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
