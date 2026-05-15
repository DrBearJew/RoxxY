#!/usr/bin/env python3
"""Build a safe RDNA3 MMQ selector policy from static math + llama-bench data.

This is the "shorter path" control-plane tool: Python scores the available
C++ runtime paths, rejects unsafe/slow ones, and emits a tiny policy artifact
that wrappers/docs/C++ dispatch can consume.

Typical use:
  scripts/hip/rdna3-mmq-policy.py \
    --summary benches/rocm-rdna3/qwen35b-pp128-256-512-20260516-005350/summary.variants.clean.json \
    --out-dir benches/rocm-rdna3/qwen35b-pp128-256-512-20260516-005350/policy

The default policy is conservative: variants that exceed the current known-good
F32 accumulator pressure budget are listed but not selected unless
--allow-over-budget is passed. For stabilization work, run a cap sweep across
maxx32/maxx48/maxx64/maxx128 and then rerun this tool with --allow-over-budget;
only promote a larger cap if it beats the current safe cap and passes canaries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DTYPE_BITS = {"f32": 32}


KNOWN_VARIANT_ENVS: dict[str, dict[str, str]] = {
    "baseline": {},
    "rdna2_opt": {"RDNA2_MATMUL_OPT_V1": "1"},
    "maxx32": {"RDNA2_MATMUL_OPT_V1": "1", "GGML_CUDA_MMQ_MAX_X": "32"},
    "maxx48": {"RDNA2_MATMUL_OPT_V1": "1", "GGML_CUDA_MMQ_MAX_X": "48"},
    "maxx64": {"RDNA2_MATMUL_OPT_V1": "1", "GGML_CUDA_MMQ_MAX_X": "64"},
    "maxx96": {"RDNA2_MATMUL_OPT_V1": "1", "GGML_CUDA_MMQ_MAX_X": "96"},
    "maxx128": {"RDNA2_MATMUL_OPT_V1": "1", "GGML_CUDA_MMQ_MAX_X": "128"},
    "scratch16k": {"RDNA2_MATMUL_OPT_V1": "1", "GGML_CUDA_IQ4_XS_MMQ_SCRATCH16K": "1"},
}

# The scratch16k prototype globally caps the selector to x64 in the current
# branch, so model it with the same accumulator pressure as maxx64.
KNOWN_VARIANT_MMQ_X: dict[str, int | None] = {
    "baseline": None,
    "rdna2_opt": None,
    "maxx32": 32,
    "maxx48": 48,
    "maxx64": 64,
    "maxx96": 96,
    "maxx128": 128,
    "scratch16k": 64,
}


@dataclass(frozen=True)
class HardwareBudget:
    smpbo: int = 65_536
    warp_size: int = 32
    nwarps: int = 8
    sizeof_int: int = 4
    accum_soft_budget_bytes_per_thread: int = 96

    @property
    def threads_per_block(self) -> int:
        return self.warp_size * self.nwarps

    @property
    def pad_align_bytes(self) -> int:
        return self.threads_per_block * self.sizeof_int


@dataclass(frozen=True)
class CurrentMMQConstants:
    mmq_tile_ne_k: int = 32
    qi8_0: int = 8
    qk8_1: int = 32
    sizeof_half2: int = 4

    @property
    def mmq_mma_tile_x_k_q8_0(self) -> int:
        return 2 * self.mmq_tile_ne_k + 2 * self.mmq_tile_ne_k // self.qi8_0 + 4

    @property
    def sizeof_block_q8_1_mmq(self) -> int:
        return 4 * self.qk8_1 + 4 * self.sizeof_half2


@dataclass
class StaticShape:
    mmq_x: int
    mmq_y: int
    ncols_max: int
    tiles_x: int
    accum_floats_per_thread: float
    accum_bytes_per_thread: float
    nbs_ids: int
    nbs_x: int
    nbs_y_raw: int
    nbs_y_padded: int
    coherent_shared: int
    explicit_double_buffer_shared: int
    coherent_fits: bool
    explicit_double_buffer_fits: bool
    valid_granularity: bool
    safe_by_accum_budget: bool
    score: float
    reason: str

    @property
    def viable(self) -> bool:
        return self.coherent_fits and self.valid_granularity and self.safe_by_accum_budget


@dataclass
class PerfScore:
    throughput_tok_s: float
    weighted_seconds: float
    total_weighted_tokens: float
    prompts: dict[str, dict[str, float]]
    missing_prompts: list[int]
    speedup_vs_baseline: float | None = None


@dataclass
class VariantPolicy:
    name: str
    env: dict[str, str]
    static: dict[str, Any] | None
    perf: dict[str, Any] | None
    eligible: bool
    reject_reasons: list[str]


def pad(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def amd_granularity(mmq_x: int) -> int:
    return 32 if mmq_x >= 128 else 16


def current_iq4xs_q8_f32_shape(mmq_x: int, mmq_y: int, ncols_max: int, hw: HardwareBudget) -> StaticShape:
    c = CurrentMMQConstants()
    threads = hw.threads_per_block
    acc_floats = mmq_x * mmq_y / threads
    acc_bytes = acc_floats * DTYPE_BITS["f32"] / 8
    nbs_ids = mmq_x * hw.sizeof_int
    nbs_x = mmq_y * c.mmq_mma_tile_x_k_q8_0 * hw.sizeof_int
    nbs_y_raw = mmq_x * c.sizeof_block_q8_1_mmq
    nbs_y_padded = pad(nbs_y_raw, hw.pad_align_bytes)
    coherent = nbs_ids + nbs_x + nbs_y_padded
    explicit_double = coherent + nbs_x + hw.sizeof_int
    valid_granularity = mmq_x % amd_granularity(mmq_x) == 0
    safe_acc = acc_bytes <= hw.accum_soft_budget_bytes_per_thread
    tiles_x = ceil_div(ncols_max, mmq_x)

    reasons = []
    if not valid_granularity:
        reasons.append("bad AMD MMA granularity")
    if coherent > hw.smpbo:
        reasons.append("coherent LDS exceeds smpbo")
    if acc_bytes > hw.accum_soft_budget_bytes_per_thread:
        reasons.append("over soft accumulator/register budget")
    if not reasons:
        reasons.append("fits coherent LDS and soft accumulator budget")

    # Lower is better. This keeps the model inspectable rather than clever.
    score = tiles_x
    score += max(0.0, acc_bytes - hw.accum_soft_budget_bytes_per_thread) / 8.0
    score += coherent / hw.smpbo * 0.25

    return StaticShape(
        mmq_x=mmq_x,
        mmq_y=mmq_y,
        ncols_max=ncols_max,
        tiles_x=tiles_x,
        accum_floats_per_thread=acc_floats,
        accum_bytes_per_thread=acc_bytes,
        nbs_ids=nbs_ids,
        nbs_x=nbs_x,
        nbs_y_raw=nbs_y_raw,
        nbs_y_padded=nbs_y_padded,
        coherent_shared=coherent,
        explicit_double_buffer_shared=explicit_double,
        coherent_fits=coherent <= hw.smpbo,
        explicit_double_buffer_fits=explicit_double <= hw.smpbo,
        valid_granularity=valid_granularity,
        safe_by_accum_budget=safe_acc,
        score=score,
        reason="; ".join(reasons),
    )


def parse_weights(text: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"bad --pp-weights item {part!r}; expected prompt=weight")
        k, v = part.split("=", 1)
        out[int(k)] = float(v)
    if not out:
        raise ValueError("--pp-weights produced no prompt weights")
    return out


def load_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def score_case(rows: list[dict[str, Any]], weights: dict[int, float]) -> PerfScore:
    by_prompt: dict[int, dict[str, Any]] = {}
    for row in rows:
        pp = int(row.get("n_prompt", -1))
        if pp in weights:
            by_prompt[pp] = row

    weighted_seconds = 0.0
    total_weighted_tokens = 0.0
    prompts: dict[str, dict[str, float]] = {}
    missing: list[int] = []
    for pp, weight in sorted(weights.items()):
        row = by_prompt.get(pp)
        if row is None:
            missing.append(pp)
            continue
        avg_ts = float(row["avg_ts"])
        stddev_ts = float(row.get("stddev_ts", 0.0))
        weighted_seconds += weight * pp / avg_ts
        total_weighted_tokens += weight * pp
        prompts[str(pp)] = {"avg_ts": avg_ts, "stddev_ts": stddev_ts}

    throughput = total_weighted_tokens / weighted_seconds if weighted_seconds > 0 else 0.0
    return PerfScore(
        throughput_tok_s=throughput,
        weighted_seconds=weighted_seconds,
        total_weighted_tokens=total_weighted_tokens,
        prompts=prompts,
        missing_prompts=missing,
    )


def repo_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def make_shape_candidates(args: argparse.Namespace, hw: HardwareBudget) -> list[StaticShape]:
    return [
        current_iq4xs_q8_f32_shape(x, args.mmq_y, args.ncols_max, hw)
        for x in args.mmq_x_candidates
    ]


def build_policy(args: argparse.Namespace) -> dict[str, Any]:
    hw = HardwareBudget(
        smpbo=args.smpbo,
        warp_size=args.warp_size,
        nwarps=args.nwarps,
        accum_soft_budget_bytes_per_thread=args.accum_budget,
    )
    weights = parse_weights(args.pp_weights)
    summary = load_summary(Path(args.summary) if args.summary else None)
    cases: dict[str, list[dict[str, Any]]] = {}
    variant_rc: dict[str, int] = {}
    if summary:
        cases = {str(k): list(v) for k, v in summary.get("cases", {}).items()}
        variant_rc = {str(k): int(v) for k, v in summary.get("variant_rc", {}).items()}

    shape_candidates = make_shape_candidates(args, hw)
    baseline_perf: PerfScore | None = None
    if "baseline" in cases:
        baseline_perf = score_case(cases["baseline"], weights)

    variants: list[VariantPolicy] = []
    names = list(KNOWN_VARIANT_ENVS)
    for extra_name in sorted(set(cases) - set(names)):
        names.append(extra_name)

    for name in names:
        env = KNOWN_VARIANT_ENVS.get(name, {})
        mmq_x = KNOWN_VARIANT_MMQ_X.get(name)
        static_shape = current_iq4xs_q8_f32_shape(mmq_x, args.mmq_y, args.ncols_max, hw) if mmq_x else None
        perf = score_case(cases.get(name, []), weights) if cases else None
        if perf and baseline_perf and baseline_perf.throughput_tok_s > 0:
            perf.speedup_vs_baseline = perf.throughput_tok_s / baseline_perf.throughput_tok_s

        reject: list[str] = []
        if name in variant_rc and variant_rc[name] != 0:
            reject.append(f"benchmark rc={variant_rc[name]}")
        if perf and perf.missing_prompts:
            reject.append(f"missing prompt rows: {perf.missing_prompts}")
        if static_shape and not static_shape.viable and not args.allow_over_budget:
            reject.append(static_shape.reason)
        if args.require_summary and not perf:
            reject.append("no benchmark data")

        variants.append(VariantPolicy(
            name=name,
            env=env,
            static=asdict(static_shape) | {"viable": static_shape.viable} if static_shape else None,
            perf=asdict(perf) if perf else None,
            eligible=not reject,
            reject_reasons=reject,
        ))

    eligible = [v for v in variants if v.eligible]
    selected: VariantPolicy | None = None
    basis = "none"
    perf_eligible = [v for v in eligible if v.perf and v.perf.get("throughput_tok_s", 0) > 0]
    if perf_eligible:
        selected = max(perf_eligible, key=lambda v: float(v.perf["throughput_tok_s"]))
        basis = "fastest eligible weighted llama-bench throughput"
    else:
        viable_shapes = [s for s in shape_candidates if s.viable]
        if viable_shapes:
            best_shape = min(viable_shapes, key=lambda s: (s.score, -s.mmq_x))
            for v in eligible:
                if KNOWN_VARIANT_MMQ_X.get(v.name) == best_shape.mmq_x:
                    selected = v
                    break
            basis = "static shape score; no benchmark summary available"

    if selected is None and eligible:
        selected = eligible[0]
        basis = "first eligible fallback"

    policy = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": repo_commit(),
        "model_label": args.model_label,
        "gpu_label": args.gpu_label,
        "backend": args.backend,
        "summary": str(Path(args.summary).resolve()) if args.summary else None,
        "weights": {str(k): v for k, v in sorted(weights.items())},
        "hardware_budget": asdict(hw),
        "selection_basis": basis,
        "selected_variant": selected.name if selected else None,
        "selected_env": selected.env if selected else {},
        "shape_candidates": [asdict(s) | {"viable": s.viable} for s in shape_candidates],
        "variants": [asdict(v) for v in variants],
    }
    return policy


def env_to_inline(env: dict[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in env.items()) or "baseline/no extra env"


def render_markdown(policy: dict[str, Any]) -> str:
    selected = policy.get("selected_variant") or "none"
    selected_env = policy.get("selected_env") or {}
    lines = [
        f"# RDNA3 MMQ selector policy: {policy['model_label']}",
        "",
        f"GPU/backend: `{policy['gpu_label']}` / `{policy['backend']}`",
        f"Selected: `{selected}`",
        f"Env: `{env_to_inline(selected_env)}`",
        f"Basis: {policy['selection_basis']}",
        "",
        "## Variant scores",
        "",
        "| variant | eligible | env | weighted tok/s | speedup vs baseline | static | notes |",
        "|---|:---:|---|---:|---:|---|---|",
    ]
    for variant in policy["variants"]:
        perf = variant.get("perf") or {}
        static = variant.get("static") or {}
        tok_s = perf.get("throughput_tok_s")
        speedup = perf.get("speedup_vs_baseline")
        tok_s_s = f"{tok_s:.1f}" if isinstance(tok_s, (int, float)) and tok_s > 0 else "—"
        speed_s = f"{speedup:.3f}x" if isinstance(speedup, (int, float)) and speedup > 0 else "—"
        if static:
            static_s = f"x{static['mmq_x']}: {static['reason']}"
        else:
            static_s = "no fixed MAX_X"
        notes = "; ".join(variant.get("reject_reasons") or [])
        if not notes and variant["name"] == selected:
            notes = "selected"
        lines.append(
            f"| `{variant['name']}` | {'yes' if variant['eligible'] else 'NO'} | "
            f"`{env_to_inline(variant['env'])}` | {tok_s_s} | {speed_s} | {static_s} | {notes} |"
        )

    lines += [
        "",
        "## Static shape candidates",
        "",
        "| mmq_x | tiles | acc B/thread | coherent LDS | double LDS | viable | reason |",
        "|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for shape in policy["shape_candidates"]:
        lines.append(
            f"| {shape['mmq_x']} | {shape['tiles_x']} | {shape['accum_bytes_per_thread']:.0f} | "
            f"{shape['coherent_shared']} | {shape['explicit_double_buffer_shared']} | "
            f"{'yes' if shape['viable'] else 'NO'} | {shape['reason']} |"
        )
    lines += [
        "",
        "## Shell activation",
        "",
        "```bash",
    ]
    if selected_env:
        for k, v in selected_env.items():
            lines.append(f"export {k}={v}")
    else:
        lines.append("# no extra selector env")
    lines += ["```", ""]
    return "\n".join(lines)


def render_env(policy: dict[str, Any]) -> str:
    lines = [
        "# Generated by scripts/hip/rdna3-mmq-policy.py",
        f"# selected_variant={policy.get('selected_variant')}",
    ]
    for k, v in (policy.get("selected_env") or {}).items():
        lines.append(f"export {k}={v}")
    return "\n".join(lines) + "\n"


def write_outputs(policy: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "policy.md").write_text(render_markdown(policy), encoding="utf-8")
    env_path = out_dir / "env.sh"
    env_path.write_text(render_env(policy), encoding="utf-8")
    env_path.chmod(env_path.stat().st_mode | 0o111)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", help="summary.variants.clean.json from llama-bench runs")
    p.add_argument("--out-dir", help="write policy.json, policy.md, and env.sh here")
    p.add_argument("--emit-env", action="store_true", help="print shell exports for the selected variant")
    p.add_argument("--json", action="store_true", help="print policy JSON to stdout")
    p.add_argument("--model-label", default="Qwen3.6-35B-A3B-IQ4_XS")
    p.add_argument("--gpu-label", default="gfx1100/RX 7900 XTX")
    p.add_argument("--backend", default="ROCm/HIP")
    p.add_argument("--pp-weights", default="128=1,256=1,512=1", help="comma list of prompt=weight")
    p.add_argument("--require-summary", action="store_true", help="reject variants without benchmark data")
    p.add_argument("--allow-over-budget", action="store_true", help="allow variants over the soft accumulator budget")
    p.add_argument("--smpbo", type=int, default=65_536)
    p.add_argument("--warp-size", type=int, default=32)
    p.add_argument("--nwarps", type=int, default=8)
    p.add_argument("--accum-budget", type=int, default=96, help="soft accumulator bytes/thread budget; x48 at y128 is 96")
    p.add_argument("--mmq-y", type=int, default=128)
    p.add_argument("--ncols-max", type=int, default=512)
    p.add_argument("--mmq-x-candidates", type=int, nargs="+", default=[16, 32, 48, 64, 80, 96, 112, 128])
    args = p.parse_args()

    policy = build_policy(args)
    if args.out_dir:
        write_outputs(policy, Path(args.out_dir))
    if args.emit_env:
        print(render_env(policy), end="")
    elif args.json or not args.out_dir:
        print(json.dumps(policy, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
