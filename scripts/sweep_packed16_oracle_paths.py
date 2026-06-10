#!/usr/bin/env python3
"""Sweep packed16 decode/verify routes with the Python oracle as correctness gate.

This script runs tests/test-packed16-decode-variants for a matrix of shapes and
route variants, asks the harness to dump a fixture, compares each dumped GPU
output against scripts/packed16_fattn_oracle.py, and ranks passing candidates by
harness compute_ms.

Use from the repo root, preferably inside the LLM conda env because the oracle
uses NumPy:

  conda run -n LLM python scripts/sweep_packed16_oracle_paths.py \
    --variants small_verify_fa2,small_verify_fa4_pvwmma \
    --nq 2,3,4 --nk 4096,14336 --hk 4 --gqa 6 --repeat 3 --warmup 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Import sibling oracle module.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import packed16_fattn_oracle as oracle  # noqa: E402

RESULT_RE = re.compile(
    r"variant=(?P<variant>\S+)\s+"
    r"max_abs=(?P<harness_max_abs>[-+0-9.eE]+)\s+"
    r"rms=(?P<harness_rms>[-+0-9.eE]+)\s+"
    r"finite=(?P<harness_finite>\d+)\s+"
    r"compute_ms=(?P<compute_ms>[-+0-9.eE]+)\s+"
    r"repeat=(?P<repeat>\d+)\s+"
    r"RESULT=(?P<harness_result>\S+)"
)


def parse_csv_list_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_list_str(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_env(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        if not k:
            raise ValueError(f"empty env key in {item!r}")
        out[k] = v
    return out


def parse_candidates(items: list[str], variants: list[str]) -> list[dict[str, Any]]:
    """Parse candidate specs.

    Format: LABEL|VARIANT|KEY=VALUE,KEY=VALUE
    The env section may be empty. If no specs are provided, variants become
    candidates with empty per-candidate env.
    """
    if not items:
        return [{"label": v, "variant": v, "env": {}} for v in variants]

    candidates: list[dict[str, Any]] = []
    for item in items:
        parts = item.split("|", 2)
        if len(parts) != 3:
            raise ValueError(f"--candidate expects LABEL|VARIANT|KEY=VALUE,KEY=VALUE, got {item!r}")
        label, variant, env_text = (p.strip() for p in parts)
        if not label or not variant:
            raise ValueError(f"--candidate requires non-empty label and variant, got {item!r}")
        env = parse_env([x.strip() for x in env_text.split(",") if x.strip()]) if env_text else {}
        candidates.append({"label": label, "variant": variant, "env": env})
    return candidates


def run_one(
    *,
    binary: Path,
    out_dir: Path,
    variant: str,
    nq: int,
    nk: int,
    hk: int,
    gqa: int,
    warmup: int,
    repeat: int,
    block_size: int,
    atol: float,
    rms_tol: float,
    extra_env: dict[str, str],
    timeout: int,
    tag: str,
    candidate_label: str,
    candidate_env: dict[str, str],
) -> dict[str, Any]:
    safe_label = candidate_label.replace("/", "_").replace(" ", "_")
    case_name = f"{tag}candidate-{safe_label}_variant-{variant}_nq-{nq}_nk-{nk}_hk-{hk}_gqa-{gqa}"
    case_name = case_name.replace("/", "_")
    case_dir = out_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = case_dir / "fixture"
    log_path = case_dir / "harness.log"

    env = os.environ.copy()
    env.update(extra_env)
    env.update(
        {
            "PACKED16_DECODE_TEST_VARIANTS": variant,
            "PACKED16_DECODE_TEST_NQ": str(nq),
            "PACKED16_DECODE_TEST_NK": str(nk),
            "PACKED16_DECODE_TEST_HK": str(hk),
            "PACKED16_DECODE_TEST_GQA": str(gqa),
            "PACKED16_DECODE_TEST_WARMUP": str(warmup),
            "PACKED16_DECODE_TEST_REPEAT": str(repeat),
            "PACKED16_DECODE_TEST_DUMP_DIR": str(dump_dir),
        }
    )

    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(binary)],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    elapsed_s = time.perf_counter() - t0
    log_path.write_text(proc.stdout)

    parsed: dict[str, Any] = {}
    for m in RESULT_RE.finditer(proc.stdout):
        if m.group("variant") == variant:
            parsed = m.groupdict()
    if parsed:
        for key in ("harness_max_abs", "harness_rms", "compute_ms"):
            parsed[key] = float(parsed[key])
        parsed["harness_finite"] = bool(int(parsed["harness_finite"]))
        parsed["repeat"] = int(parsed["repeat"])
        parsed["harness_pass"] = parsed["harness_result"] == "PASS"
    else:
        parsed = {
            "variant": variant,
            "harness_max_abs": None,
            "harness_rms": None,
            "compute_ms": None,
            "harness_finite": False,
            "repeat": repeat,
            "harness_result": "MISSING",
            "harness_pass": False,
        }

    oracle_result: dict[str, Any] | None = None
    oracle_pass = False
    oracle_max_abs = None
    oracle_rms = None
    oracle_err = None
    if (dump_dir / "meta.json").exists():
        try:
            oracle_args = SimpleNamespace(
                compare_fixture=str(dump_dir),
                block_size=block_size,
                variant=[variant],
                atol=atol,
                rms_tol=rms_tol,
            )
            oracle_result = oracle.compare_harness_fixture(oracle_args)
            out_info = oracle_result.get("outputs", {}).get(variant, {})
            oracle_pass = bool(out_info.get("pass", False)) and bool(oracle_result.get("pass", False))
            oracle_max_abs = out_info.get("max_abs")
            oracle_rms = out_info.get("rms")
            (case_dir / "oracle.json").write_text(json.dumps(oracle_result, indent=2, sort_keys=True))
        except Exception as exc:  # keep sweep going
            oracle_err = repr(exc)
    else:
        oracle_err = "fixture meta.json missing"

    row: dict[str, Any] = {
        "candidate": candidate_label,
        "candidate_env": dict(candidate_env),
        "variant": variant,
        "nq": nq,
        "nk": nk,
        "hk": hk,
        "gqa": gqa,
        "hq": hk * gqa,
        "warmup": warmup,
        "repeat": repeat,
        "block_size": block_size,
        "returncode": proc.returncode,
        "elapsed_s": elapsed_s,
        "case_dir": str(case_dir),
        "log": str(log_path),
        "fixture": str(dump_dir),
        "oracle_pass": oracle_pass,
        "oracle_max_abs": oracle_max_abs,
        "oracle_rms": oracle_rms,
        "oracle_error": oracle_err,
        **parsed,
    }
    row["pass"] = bool(row.get("harness_pass")) and bool(row.get("oracle_pass")) and proc.returncode == 0
    return row


def summarize_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_shape: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"nq={row['nq']} nk={row['nk']} hk={row['hk']} gqa={row['gqa']}"
        by_shape.setdefault(key, []).append(row)

    best: dict[str, Any] = {}
    for key, items in by_shape.items():
        passing = [r for r in items if r.get("pass") and r.get("compute_ms") is not None]
        passing.sort(key=lambda r: float(r["compute_ms"]))
        best[key] = passing[0] if passing else None
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="Rank packed16 attention route variants using oracle-gated harness runs")
    ap.add_argument("--binary", type=Path, default=REPO_ROOT / "build-rocm-ninja/bin/test-packed16-decode-variants")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / ".harness/artifacts/packed16-oracle-sweep")
    ap.add_argument("--variants", default="small_verify_fa2,small_verify_fa4,small_verify_fa4_pvwmma,bm_dot4_pages_intflash_vfrag_wmma")
    ap.add_argument("--nq", default="2,3,4")
    ap.add_argument("--nk", default="256,4096,14336")
    ap.add_argument("--hk", type=int, default=4)
    ap.add_argument("--gqa", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--block-size", type=int, default=224, help="oracle online block size, not a kernel knob")
    ap.add_argument("--atol", type=float, default=2.5e-2)
    ap.add_argument("--rms-tol", type=float, default=5.0e-3)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--env", action="append", default=[], help="extra env KEY=VALUE for every harness run")
    ap.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="repeatable LABEL|VARIANT|KEY=VALUE,KEY=VALUE candidate specs; overrides --variants as the candidate list",
    )
    ap.add_argument("--split-sizes", default="", help="optional CSV; sets GGML_CUDA_ROCM_SMALL_VERIFY_FA2_SPLIT_SIZE per run")
    args = ap.parse_args()

    variants = parse_csv_list_str(args.variants)
    candidates = parse_candidates(args.candidate, variants)
    nqs = parse_csv_list_int(args.nq)
    nks = parse_csv_list_int(args.nk)
    extra_env_base = parse_env(args.env)
    split_sizes = parse_csv_list_int(args.split_sizes) if args.split_sizes else [None]

    if not args.binary.exists():
        raise SystemExit(f"missing binary: {args.binary}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for nk in nks:
        for nq in nqs:
            for candidate in candidates:
                variant = candidate["variant"]
                for split in split_sizes:
                    extra_env = dict(extra_env_base)
                    extra_env.update(candidate["env"])
                    tag = ""
                    if split is not None:
                        extra_env["GGML_CUDA_ROCM_SMALL_VERIFY_FA2_SPLIT_SIZE"] = str(split)
                        tag = f"split-{split}_"
                    print(
                        f"RUN candidate={candidate['label']} variant={variant} nq={nq} nk={nk} "
                        f"hk={args.hk} gqa={args.gqa} split={split} env={candidate['env']}",
                        flush=True,
                    )
                    row = run_one(
                        binary=args.binary,
                        out_dir=args.out_dir,
                        variant=variant,
                        nq=nq,
                        nk=nk,
                        hk=args.hk,
                        gqa=args.gqa,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        block_size=args.block_size,
                        atol=args.atol,
                        rms_tol=args.rms_tol,
                        extra_env=extra_env,
                        timeout=args.timeout,
                        tag=tag,
                        candidate_label=candidate["label"],
                        candidate_env=candidate["env"],
                    )
                    rows.append(row)
                    print(
                        f"  pass={row['pass']} compute_ms={row.get('compute_ms')} "
                        f"oracle=({row.get('oracle_max_abs')},{row.get('oracle_rms')}) "
                        f"harness={row.get('harness_result')} log={row['log']}",
                        flush=True,
                    )

    summary = {
        "rows": rows,
        "best_by_shape": summarize_best(rows),
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    csv_path = args.out_dir / "summary.csv"
    fields = [
        "pass", "candidate", "variant", "nq", "nk", "hk", "gqa", "compute_ms",
        "oracle_max_abs", "oracle_rms", "harness_max_abs", "harness_rms",
        "harness_result", "returncode", "case_dir",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print("\nBest passing candidate by shape:")
    for key, best in summary["best_by_shape"].items():
        if best is None:
            print(f"  {key}: none")
        else:
            print(f"  {key}: {best.get('candidate', best['variant'])} variant={best['variant']} compute_ms={best['compute_ms']:.6f} oracle_max_abs={best['oracle_max_abs']}")
    print(f"\nWrote {args.out_dir / 'summary.json'}")
    print(f"Wrote {csv_path}")

    return 0 if any(r.get("pass") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
