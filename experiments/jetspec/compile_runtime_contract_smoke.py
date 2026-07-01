#!/usr/bin/env python3
"""Compile-only JetSpec runtime contract smoke outside CMake."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
SMOKE = HERE / "jetspec_runtime_contract_compile_smoke.cpp"


def compile_smoke(*, compiler: str | None = None) -> dict[str, Any]:
    cxx = compiler or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        return {"ok": False, "status": "cpp_compiler_missing", "errors": ["no C++ compiler found"], "runtime_executed": False}
    cmd = [cxx, "-std=c++17", "-I", str(HERE), "-fsyntax-only", str(SMOKE)]
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, check=False)
    return {
        "ok": proc.returncode == 0,
        "status": "runtime_contract_compile_smoke_passed" if proc.returncode == 0 else "runtime_contract_compile_smoke_failed",
        "errors": [] if proc.returncode == 0 else [proc.stderr.strip() or proc.stdout.strip()],
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "runtime_executed": False,
        "cmake_wired": False,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out = compile_smoke(compiler=args.compiler)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
