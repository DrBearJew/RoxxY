#!/usr/bin/env python3
"""
Packed16 PWMMA context-growth estimator.

This is a lightweight calculator/simulator for the RDNA3 packed16 PWMMA
prefill path.  It consumes logs produced with:

  GGML_CUDA_PWMMA_PROFILE=1 ... llama-bench ... 2> stderr.log

or the summary.json emitted by local profile sweep scripts.  It fits the
measured BM64 i8/PV-WMMA kernel phase costs as a function of effective nk and
then lets us ask "what if" questions for levers such as faster PV, faster QK,
softmax improvements, larger effective BN, or causal-tile reduction.

It is deliberately empirical: DOT4/matrix calculators validate correctness and
register mapping; this script estimates throughput impact from measured phase
ratios rather than pretending to model ROCm scheduling perfectly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as stats
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

PROFILE_RE = re.compile(
    r"PWMMA PROFILE: impl=(\S+) nq=(\d+) nk=(\d+) kernel_ms=([0-9.]+) "
    r"qk_cycles=(\d+) softmax_cycles=(\d+) pv_cycles=(\d+)"
)


@dataclass(frozen=True)
class ProfileRow:
    impl: str
    nq: int
    nk: int
    kernel_ms: float
    qk_cycles: int
    softmax_cycles: int
    pv_cycles: int

    @property
    def total_cycles(self) -> int:
        return self.qk_cycles + self.softmax_cycles + self.pv_cycles


@dataclass(frozen=True)
class PhasePoint:
    nk: int
    samples: int
    kernel_ms: float
    qk_cycles: float
    softmax_cycles: float
    pv_cycles: float
    total_cycles: float
    qk_pct: float
    softmax_pct: float
    pv_pct: float


def median(xs: Iterable[float]) -> float:
    return float(stats.median(list(xs)))


def parse_profiles(path: Path) -> list[ProfileRow]:
    if path.name == "summary.json":
        data = json.loads(path.read_text())
        rows = data.get("profiles", [])
        return [ProfileRow(
            impl=str(r["impl"]), nq=int(r["nq"]), nk=int(r["nk"]),
            kernel_ms=float(r["kernel_ms"]), qk_cycles=int(r["qk_cycles"]),
            softmax_cycles=int(r["softmax_cycles"]), pv_cycles=int(r["pv_cycles"]),
        ) for r in rows]

    text = path.read_text(errors="replace")
    out: list[ProfileRow] = []
    for m in PROFILE_RE.finditer(text):
        impl, nq, nk, ms, qk, sm, pv = m.groups()
        out.append(ProfileRow(impl, int(nq), int(nk), float(ms), int(qk), int(sm), int(pv)))
    return out


def summarize(rows: list[ProfileRow]) -> list[PhasePoint]:
    by: dict[int, list[ProfileRow]] = {}
    for r in rows:
        by.setdefault(r.nk, []).append(r)
    pts: list[PhasePoint] = []
    for nk, rs in sorted(by.items()):
        qk = median(r.qk_cycles for r in rs)
        sm = median(r.softmax_cycles for r in rs)
        pv = median(r.pv_cycles for r in rs)
        total = qk + sm + pv
        pts.append(PhasePoint(
            nk=nk,
            samples=len(rs),
            kernel_ms=median(r.kernel_ms for r in rs),
            qk_cycles=qk,
            softmax_cycles=sm,
            pv_cycles=pv,
            total_cycles=total,
            qk_pct=100.0 * qk / total if total else 0.0,
            softmax_pct=100.0 * sm / total if total else 0.0,
            pv_pct=100.0 * pv / total if total else 0.0,
        ))
    return pts


def interp_phase(pts: list[PhasePoint], nk: int) -> tuple[float, float, float, float]:
    """Linear interpolation/extrapolation of qk, softmax, pv, ms by nk."""
    if not pts:
        raise SystemExit("no profile points")
    if len(pts) == 1:
        p = pts[0]
        scale = nk / p.nk
        return p.qk_cycles * scale, p.softmax_cycles * scale, p.pv_cycles * scale, p.kernel_ms * scale
    xs = [p.nk for p in pts]
    if nk <= xs[0]:
        a, b = pts[0], pts[1]
    elif nk >= xs[-1]:
        a, b = pts[-2], pts[-1]
    else:
        for i in range(len(pts) - 1):
            if pts[i].nk <= nk <= pts[i + 1].nk:
                a, b = pts[i], pts[i + 1]
                break
    t = (nk - a.nk) / (b.nk - a.nk)
    lerp = lambda av, bv: av + t * (bv - av)
    return (
        lerp(a.qk_cycles, b.qk_cycles),
        lerp(a.softmax_cycles, b.softmax_cycles),
        lerp(a.pv_cycles, b.pv_cycles),
        lerp(a.kernel_ms, b.kernel_ms),
    )


def predict_prompt(pts: list[PhasePoint], prompt: int, ubatch: int, qk_speedup: float, softmax_speedup: float,
                   pv_speedup: float, active_tile_factor: float) -> dict:
    chunks = math.ceil(prompt / ubatch)
    base_ms = new_ms = 0.0
    base_cycles = new_cycles = 0.0
    chunk_rows = []
    for i in range(chunks):
        nq = min(ubatch, prompt - i * ubatch)
        nk = min(prompt, (i + 1) * ubatch)
        qk, sm, pv, ms = interp_phase(pts, nk)
        # Approximate partial final chunk by query-token fraction.
        frac = nq / ubatch
        qk *= frac; sm *= frac; pv *= frac; ms *= frac
        base = qk + sm + pv
        new = (qk * active_tile_factor / qk_speedup +
               sm * active_tile_factor / softmax_speedup +
               pv * active_tile_factor / pv_speedup)
        base_ms += ms
        new_ms += ms * (new / base if base else 1.0)
        base_cycles += base
        new_cycles += new
        chunk_rows.append({"chunk": i + 1, "nq": nq, "nk": nk, "base_ms": ms, "estimated_ms": ms * (new / base if base else 1.0)})
    return {
        "prompt": prompt,
        "ubatch": ubatch,
        "chunks": chunk_rows,
        "base_kernel_ms_sum": base_ms,
        "estimated_kernel_ms_sum": new_ms,
        "estimated_kernel_speedup": base_ms / new_ms if new_ms else 0.0,
        "base_phase_cycles": base_cycles,
        "estimated_phase_cycles": new_cycles,
        "estimated_phase_speedup": base_cycles / new_cycles if new_cycles else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate packed16 PWMMA context-growth and optimization levers from GGML_CUDA_PWMMA_PROFILE logs.")
    ap.add_argument("profile", type=Path, help="stderr.log or summary.json containing PWMMA PROFILE lines")
    ap.add_argument("--prompt", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--ubatch", type=int, default=1024)
    ap.add_argument("--qk-speedup", type=float, default=1.0, help="hypothetical QK phase speedup, e.g. 1.10")
    ap.add_argument("--softmax-speedup", type=float, default=1.0, help="hypothetical softmax phase speedup")
    ap.add_argument("--pv-speedup", type=float, default=1.0, help="hypothetical PV phase speedup")
    ap.add_argument("--active-tile-factor", type=float, default=1.0, help="fraction of active work after scheduling/causal improvements")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = parse_profiles(args.profile)
    if not rows:
        raise SystemExit(f"no PWMMA PROFILE rows found in {args.profile}")
    pts = summarize(rows)
    preds = [predict_prompt(pts, p, args.ubatch, args.qk_speedup, args.softmax_speedup, args.pv_speedup, args.active_tile_factor)
             for p in args.prompt]
    out = {
        "source": str(args.profile),
        "phase_points": [asdict(p) for p in pts],
        "levers": {
            "qk_speedup": args.qk_speedup,
            "softmax_speedup": args.softmax_speedup,
            "pv_speedup": args.pv_speedup,
            "active_tile_factor": args.active_tile_factor,
        },
        "predictions": preds,
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"source: {args.profile}")
    print("\nmeasured phase medians:")
    print("nk\tsamples\tkernel_ms\tQK%\tSM%\tPV%")
    for p in pts:
        print(f"{p.nk}\t{p.samples}\t{p.kernel_ms:.3f}\t{p.qk_pct:.2f}\t{p.softmax_pct:.2f}\t{p.pv_pct:.2f}")
    print("\nwhat-if prediction:")
    print("prompt\tbase_kernel_ms\test_kernel_ms\tspeedup")
    for r in preds:
        print(f"{r['prompt']}\t{r['base_kernel_ms_sum']:.3f}\t{r['estimated_kernel_ms_sum']:.3f}\t{r['estimated_kernel_speedup']:.3f}x")


if __name__ == "__main__":
    main()
