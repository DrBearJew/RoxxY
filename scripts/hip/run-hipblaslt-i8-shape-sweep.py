#!/usr/bin/env python3
"""Universal hipBLASLt dense-I8 shape sweep runner.

Runs shapes sequentially under a GPU lock and writes per-shape logs plus CSV summaries.
This benchmarks dense I8 x I8 -> I32 hipBLASLt kernels only; it does not prove q4/qK exactness.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

RE_RESULT = re.compile(
    r"^\[hipblaslt-i8-result\]\s+"
    r"(?P<kv>.*)$"
)
RE_BEST = re.compile(
    r"^\[hipblaslt-i8-best\]\s+"
    r"(?P<kv>.*)$"
)
RE_LEGACY_ALGO = re.compile(
    r"^\[hipblaslt-i8\]\s+algo=(?P<algo>\d+)\s+workspace=(?P<workspace>\d+)\s+"
    r"avg_ms=(?P<avg_ms>[0-9.]+)\s+tmac_s=(?P<tmac_s>[0-9.]+)"
)
RE_LEGACY_BEST = re.compile(
    r"^\[hipblaslt-i8\]\s+best_algo=(?P<algo>-?\d+)\s+best_avg_ms=(?P<avg_ms>[0-9.]+)\s+"
    r"best_tmac_s=(?P<tmac_s>[0-9.]+)"
)

SUMMARY_FIELDS = [
    "seq", "label", "model", "priority", "kind", "M", "N", "K", "status",
    "best_avg_ms", "best_tmac_s", "best_algo_ord", "best_solution_index", "best_solution_name",
    "heuristic_returned", "duration_sec", "log_dir", "notes",
]
RESULT_FIELDS = [
    "seq", "label", "model", "priority", "kind", "M", "N", "K",
    "algo_ord", "solution_index", "solution_name", "kernel_name", "workspace",
    "avg_ms", "tmac_s", "status", "source",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_out_dir() -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path.home() / ".harness" / "artifacts" / f"hipblaslt-i8-shape-sweep-{stamp}"


def sanitize(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.+-]+", "_", s.strip())
    return s.strip("_") or "shape"


def parse_kv_blob(blob: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tok in blob.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def priority_rank(priority: str) -> int:
    m = re.match(r"^[Pp]?(\d+)$", str(priority).strip())
    return int(m.group(1)) if m else 999


def priority_allowed(priority: str, filt: str) -> bool:
    if not filt:
        return True
    filt = filt.strip()
    p = str(priority).strip().upper()
    # "P1" means include P0..P1. "P0,P2" means exact set.
    if "," in filt:
        allowed = {x.strip().upper() for x in filt.split(",") if x.strip()}
        return p in allowed
    if filt.upper().startswith("P") or filt.isdigit():
        return priority_rank(p) <= priority_rank(filt)
    return p == filt.upper()


def load_shapes(path: Path, filter_priority: str = "", max_shapes: int = 0) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"label", "model", "priority", "M", "N", "K", "kind", "notes"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"shape CSV {path} missing columns: {sorted(missing)}")
        for row in reader:
            if not row.get("label") or row.get("label", "").startswith("#"):
                continue
            if not priority_allowed(row.get("priority", ""), filter_priority):
                continue
            for key in ("M", "N", "K"):
                int(row[key])
            rows.append(row)
            if max_shapes and len(rows) >= max_shapes:
                break
    return rows


def find_hipcc() -> str:
    for cand in (os.environ.get("HIPCC"), shutil.which("hipcc"), "/opt/rocm/bin/hipcc", "/opt/rocm-7.2.3/bin/hipcc"):
        if cand and Path(cand).exists():
            return cand
    raise SystemExit("hipcc not found; set HIPCC=/path/to/hipcc")


def build_bench(repo: Path, bench_out: Path, arch: str) -> None:
    src = repo / "scripts" / "hip" / "hipblaslt-i8-gemm-bench.hip"
    bench_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [find_hipcc(), "-O3", f"--offload-arch={arch}", str(src), "-L/opt/rocm-7.2.3/lib", "-lhipblaslt", "-o", str(bench_out)]
    print("[build]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=repo)


def run_one(
    seq: int,
    row: Dict[str, str],
    bench: Path,
    out: Path,
    warmup: int,
    reps: int,
    algos: int,
    timeout_sec: int,
    algo_source: str,
) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    m, n, k = int(row["M"]), int(row["N"]), int(row["K"])
    slug = f"{seq:04d}_{sanitize(row['priority'])}_{sanitize(row['model'])}_{sanitize(row['label'])}_M{m}_N{n}_K{k}"
    log_dir = out / "logs" / slug
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(bench), "--m", str(m), "--n", str(n), "--k", str(k),
        "--warmup", str(warmup), "--reps", str(reps), "--algos", str(algos),
        "--algo-source", algo_source,
    ]
    env = os.environ.copy()
    env.setdefault("HIP_VISIBLE_DEVICES", "0")
    # HIPBLASLT_LOG_MASK=32 commonly emits hipblaslt-bench command lines; harmless if ignored.
    env.setdefault("HIPBLASLT_LOG_MASK", "32")

    start = time.time()
    stdout_path = log_dir / "run.log"
    stderr_path = log_dir / "stderr.log"
    status = "ok"
    timed_out = False
    proc = None
    try:
        proc = subprocess.run(
            cmd,
            cwd=bench.parent,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        timed_out = True
        status = "timeout"
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        rc = 124
    else:
        stdout = proc.stdout
        stderr = proc.stderr
        rc = proc.returncode
        if rc != 0:
            status = f"exit_{rc}"

    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    bench_lines = "\n".join(line for line in (stdout + "\n" + stderr).splitlines() if "hipblaslt-bench" in line or "solution_index" in line)
    (log_dir / "hipblaslt-bench-capture.log").write_text(bench_lines + ("\n" if bench_lines else ""))

    results: List[Dict[str, str]] = []
    best: Optional[Dict[str, str]] = None
    heuristic_returned = ""
    for line in stdout.splitlines() + stderr.splitlines():
        if "heuristic_returned=" in line:
            m_heur = re.search(r"heuristic_returned=(\d+)", line)
            if m_heur:
                heuristic_returned = m_heur.group(1)
        m_res = RE_RESULT.match(line)
        if m_res:
            kv = parse_kv_blob(m_res.group("kv"))
            rec = {
                "seq": str(seq), "label": row["label"], "model": row["model"],
                "priority": row["priority"], "kind": row["kind"],
                "M": str(m), "N": str(n), "K": str(k),
                "algo_ord": kv.get("algo_ord", kv.get("algo", "")),
                "solution_index": kv.get("solution_index", ""),
                "solution_name": kv.get("solution_name", ""),
                "kernel_name": kv.get("kernel_name", ""),
                "workspace": kv.get("workspace", ""),
                "avg_ms": kv.get("avg_ms", ""),
                "tmac_s": kv.get("tmac_s", ""),
                "status": kv.get("status", "ok"),
                "source": "result",
            }
            results.append(rec)
            continue
        m_best = RE_BEST.match(line)
        if m_best:
            best = parse_kv_blob(m_best.group("kv"))
            continue
        m_old = RE_LEGACY_ALGO.match(line)
        if m_old:
            rec = {
                "seq": str(seq), "label": row["label"], "model": row["model"],
                "priority": row["priority"], "kind": row["kind"],
                "M": str(m), "N": str(n), "K": str(k),
                "algo_ord": m_old.group("algo"), "solution_index": "", "solution_name": "", "kernel_name": "",
                "workspace": m_old.group("workspace"), "avg_ms": m_old.group("avg_ms"), "tmac_s": m_old.group("tmac_s"),
                "status": "ok", "source": "legacy",
            }
            results.append(rec)
            continue
        m_old_best = RE_LEGACY_BEST.match(line)
        if m_old_best:
            best = {"algo_ord": m_old_best.group("algo"), "avg_ms": m_old_best.group("avg_ms"), "tmac_s": m_old_best.group("tmac_s")}

    result_csv = log_dir / "result.csv"
    with result_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    if status == "ok" and not best and results:
        best_rec = min(results, key=lambda r: float(r["avg_ms"]) if r.get("avg_ms") else math.inf)
        best = {
            "algo_ord": best_rec.get("algo_ord", ""),
            "solution_index": best_rec.get("solution_index", ""),
            "solution_name": best_rec.get("solution_name", ""),
            "avg_ms": best_rec.get("avg_ms", ""),
            "tmac_s": best_rec.get("tmac_s", ""),
        }
    if status == "ok" and not best:
        status = "no_best"

    duration = time.time() - start
    summary = {
        "seq": str(seq), "label": row["label"], "model": row["model"], "priority": row["priority"], "kind": row["kind"],
        "M": str(m), "N": str(n), "K": str(k), "status": status,
        "best_avg_ms": (best or {}).get("avg_ms", (best or {}).get("best_avg_ms", "")),
        "best_tmac_s": (best or {}).get("tmac_s", (best or {}).get("best_tmac_s", "")),
        "best_algo_ord": (best or {}).get("algo_ord", (best or {}).get("algo", "")),
        "best_solution_index": (best or {}).get("solution_index", ""),
        "best_solution_name": (best or {}).get("solution_name", ""),
        "heuristic_returned": heuristic_returned,
        "duration_sec": f"{duration:.3f}",
        "log_dir": str(log_dir),
        "notes": row.get("notes", ""),
    }
    return summary, results


def write_csv(path: Path, fields: List[str], rows: Iterable[Dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path, default=repo_root() / "scripts" / "hip" / "hipblaslt-i8-shapes.csv")
    parser.add_argument("--out", type=Path, default=default_out_dir())
    parser.add_argument("--bench", type=Path, default=None, help="Existing bench binary. Default: <out>/bin/hipblaslt_i8_gemm_bench")
    parser.add_argument("--build", action="store_true", help="Build bench before running")
    parser.add_argument("--arch", default="gfx1100")
    parser.add_argument("--filter-priority", default="", help="P1 means include P0..P1; comma list means exact priorities")
    parser.add_argument("--max-shapes", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--algos", type=int, default=64)
    parser.add_argument("--algo-source", default="heuristic", choices=["heuristic", "all"])
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--lock", type=Path, default=Path("/tmp/hipblaslt-i8-shape-sweep.gpu.lock"))
    args = parser.parse_args()

    repo = repo_root()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bench = args.bench.resolve() if args.bench else out / "bin" / "hipblaslt_i8_gemm_bench"

    shapes = load_shapes(args.shapes, args.filter_priority, args.max_shapes)
    if not shapes:
        raise SystemExit("no shapes selected")

    manifest = {
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "repo": str(repo),
        "shapes": str(args.shapes.resolve()),
        "out": str(out),
        "bench": str(bench),
        "build": args.build,
        "arch": args.arch,
        "filter_priority": args.filter_priority,
        "max_shapes": args.max_shapes,
        "warmup": args.warmup,
        "reps": args.reps,
        "algos": args.algos,
        "algo_source": args.algo_source,
        "timeout_sec": args.timeout_sec,
        "note": "dense I8 hipBLASLt timing only; not q4/qK exactness or model quality",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(args.shapes, out / "shapes.csv")

    if args.build or not bench.exists():
        build_bench(repo, bench, args.arch)
    if not bench.exists():
        raise SystemExit(f"bench binary not found: {bench}")

    summaries: List[Dict[str, str]] = []
    all_results: List[Dict[str, str]] = []
    failures: List[Dict[str, str]] = []

    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("w") as lockf:
        print(f"[lock] waiting for {args.lock}", flush=True)
        fcntl.flock(lockf, fcntl.LOCK_EX)
        print(f"[lock] acquired {args.lock}", flush=True)
        for seq, row in enumerate(shapes, 1):
            print(f"[shape {seq}/{len(shapes)}] {row['priority']} {row['model']} {row['label']} M={row['M']} N={row['N']} K={row['K']}", flush=True)
            summary, results = run_one(seq, row, bench, out, args.warmup, args.reps, args.algos, args.timeout_sec, args.algo_source)
            summaries.append(summary)
            all_results.extend(results)
            if summary["status"] != "ok":
                failures.append(summary)
            write_csv(out / "summary.csv", SUMMARY_FIELDS, summaries)
            write_csv(out / "all_results.csv", RESULT_FIELDS, all_results)
            write_csv(out / "failures.csv", SUMMARY_FIELDS, failures)
        fcntl.flock(lockf, fcntl.LOCK_UN)

    ok = [r for r in summaries if r["status"] == "ok"]
    write_csv(out / "best_by_shape.csv", SUMMARY_FIELDS, ok)
    print(f"[done] out={out} shapes={len(summaries)} ok={len(ok)} failures={len(failures)}", flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
