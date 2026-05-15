#!/usr/bin/env python3
"""Build a conservative RDNA3 FlashAttention path policy.

This is the FlashAttention companion to rdna3-mmq-policy.py. It treats Python
as the control plane: read smoke/benchmark artifacts, score candidate C++
attention routes, and emit a small policy artifact for humans/wrappers.

Default stance for compressed KV on ROCm/HIP RDNA3:
  - select the VEC FlashAttention path for TBQ4/Planar/Iso compressed KV;
  - keep rocWMMA compressed-KV FA paths visible but experimental until explicit
    smoke/perf evidence is supplied;
  - avoid TILE/WMMA/MMA routes that materialize full f16 K/V temp buffers for
    long-context compressed KV.

Typical use:
  scripts/hip/rdna3-fattn-policy.py \
    --summary benches/rocm-rdna3/ctx-fit-quant-sweep-20260516-004216/summary.json \
    --out-dir benches/rocm-rdna3/ctx-fit-quant-sweep-20260516-004216/fattn-policy
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STABLE_COMPRESSED_KV_TYPES = ["tbq4_0", "planar3_0", "iso3_0"]
EXPERIMENTAL_WMMA_TYPES = ["tbq4_0", "planar3_0", "iso3_0"]


@dataclass
class QuantEvidence:
    quant: str
    contexts: list[int] = field(default_factory=list)
    prompt_tok_s: dict[str, float] = field(default_factory=dict)
    decode_tok_s: dict[str, float] = field(default_factory=dict)
    peak_vram_gib: dict[str, float] = field(default_factory=dict)
    kernel_counts: dict[str, int] = field(default_factory=lambda: {"vec": 0, "wmma": 0, "other": 0})
    failures: list[str] = field(default_factory=list)
    selected_kernel_examples: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures and bool(self.contexts) and self.kernel_counts.get("vec", 0) > 0


@dataclass
class CandidatePolicy:
    name: str
    route: str
    env: dict[str, str]
    server_flags: list[str]
    cache_types: list[str]
    status: str
    eligible: bool
    reject_reasons: list[str]
    notes: str


def repo_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def gib(nbytes: int | float | None) -> float | None:
    if nbytes is None:
        return None
    return float(nbytes) / (1024 ** 3)


def load_summary(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"expected list summary, got {type(data).__name__}")
    return data


def kernel_kind(line: str) -> str:
    if "kernel=vec" in line:
        return "vec"
    if "kernel=wmma" in line:
        return "wmma"
    return "other"


def summarize_evidence(rows: list[dict[str, Any]]) -> dict[str, QuantEvidence]:
    evidence: dict[str, QuantEvidence] = {}
    for row in rows:
        quant = str(row.get("quant") or row.get("name") or "unknown")
        ev = evidence.setdefault(quant, QuantEvidence(quant=quant))
        ctx = int(row.get("ctx") or 0)
        if ctx:
            ev.contexts.append(ctx)

        name = str(row.get("name") or f"ctx{ctx}")
        request = row.get("request") or {}
        timings = request.get("timings") or {}
        if timings.get("prompt_per_second") is not None:
            ev.prompt_tok_s[name] = float(timings["prompt_per_second"])
        if timings.get("predicted_per_second") is not None:
            ev.decode_tok_s[name] = float(timings["predicted_per_second"])
        peak = gib(((request.get("vram") or {}).get("peak_b")))
        if peak is not None:
            ev.peak_vram_gib[name] = peak

        health = row.get("health") or {}
        log_counts = row.get("log_counts") or {}
        if row.get("server_rc") not in (0, None):
            ev.failures.append(f"{name}: server_rc={row.get('server_rc')}")
        if health.get("ok") is False:
            ev.failures.append(f"{name}: health failed")
        if request.get("ok") is False or request.get("http_status") not in (None, 200):
            ev.failures.append(f"{name}: request failed/http={request.get('http_status')}")
        for key in ["rocm_oom", "cublas_stack", "double_free"]:
            if int(log_counts.get(key, 0) or 0) != 0:
                ev.failures.append(f"{name}: {key}={log_counts.get(key)}")

        selected = [str(x) for x in row.get("selected_kernels") or []]
        if selected:
            ev.selected_kernel_examples.extend(selected[:3])
            for line in selected:
                ev.kernel_counts[kernel_kind(line)] += 1
        else:
            # Fallback to aggregate counts from the runner.
            ev.kernel_counts["vec"] += int(log_counts.get("kernel_vec", 0) or 0)
            ev.kernel_counts["wmma"] += int(log_counts.get("kernel_wmma", 0) or 0)

    for ev in evidence.values():
        ev.contexts = sorted(set(ev.contexts))
        ev.selected_kernel_examples = ev.selected_kernel_examples[:8]
    return evidence


def make_candidates(args: argparse.Namespace, evidence: dict[str, QuantEvidence]) -> list[CandidatePolicy]:
    missing_stable = [q for q in STABLE_COMPRESSED_KV_TYPES if q not in evidence]
    failed_stable = [q for q in STABLE_COMPRESSED_KV_TYPES if q in evidence and not evidence[q].ok]

    vec_reject: list[str] = []
    if args.require_summary and missing_stable:
        vec_reject.append(f"missing smoke evidence for {missing_stable}")
    if failed_stable:
        vec_reject.append(f"failed smoke evidence for {failed_stable}")

    wmma_base_reject = [
        "experimental opt-in path; needs standalone coherence/perf smokes before promotion",
    ]
    if not args.allow_experimental_wmma:
        wmma_base_reject.append("pass --allow-experimental-wmma only after fresh smokes")

    return [
        CandidatePolicy(
            name="vec_compressed_kv",
            route="kernel=vec",
            env={"COMPRESSED_KV_FATTN_LOG": "1"} if args.include_log_env else {},
            server_flags=["--flash-attn", "on"],
            cache_types=STABLE_COMPRESSED_KV_TYPES,
            status="stable default",
            eligible=not vec_reject,
            reject_reasons=vec_reject,
            notes="Preserves compressed KV in the FA loop; no full-cache f16 K/V temp materialization.",
        ),
        CandidatePolicy(
            name="wmma_tbq4",
            route="kernel=wmma_tbq4",
            env={"TBQ4_WMMA_FATTN": "1", "COMPRESSED_KV_FATTN_LOG": "1"},
            server_flags=["--flash-attn", "on"],
            cache_types=["tbq4_0"],
            status="experimental opt-in",
            eligible=False,
            reject_reasons=wmma_base_reject,
            notes="New direct TBQ4 rocWMMA FA route for D=128/256, nq>2, K=V=tbq4_0.",
        ),
        CandidatePolicy(
            name="wmma_planar_iso",
            route="kernel=wmma_compressed_kv",
            env={"COMPRESSED_KV_WMMA_FATTN": "1", "COMPRESSED_KV_FATTN_LOG": "1"},
            server_flags=["--flash-attn", "on"],
            cache_types=["planar3_0", "iso3_0"],
            status="experimental opt-in",
            eligible=False,
            reject_reasons=wmma_base_reject,
            notes="New original-domain Planar/Iso rocWMMA FA route for D=128/256, nq>2, K=V.",
        ),
        CandidatePolicy(
            name="tile_or_full_temp_quantized",
            route="kernel=tile/wmma/mma with f16 temp",
            env={},
            server_flags=["--flash-attn", "on"],
            cache_types=STABLE_COMPRESSED_KV_TYPES,
            status="rejected for long-context compressed KV",
            eligible=False,
            reject_reasons=["can materialize full f16 K/V temp buffers and erase compressed-KV memory savings"],
            notes="Useful for non-quant/f16 cases, not the 24 GB long-context compressed-KV target.",
        ),
    ]


def select_candidate(candidates: list[CandidatePolicy]) -> CandidatePolicy | None:
    for c in candidates:
        if c.name == "vec_compressed_kv" and c.eligible:
            return c
    for c in candidates:
        if c.eligible:
            return c
    return None


def build_policy(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_summary(Path(args.summary) if args.summary else None)
    evidence = summarize_evidence(rows)
    candidates = make_candidates(args, evidence)
    selected = select_candidate(candidates)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": repo_commit(),
        "model_label": args.model_label,
        "gpu_label": args.gpu_label,
        "backend": args.backend,
        "summary": str(Path(args.summary).resolve()) if args.summary else None,
        "selected_candidate": selected.name if selected else None,
        "selected_env": selected.env if selected else {},
        "selected_server_flags": selected.server_flags if selected else [],
        "selected_cache_types": selected.cache_types if selected else [],
        "selection_basis": "stable compressed-KV VEC FA route with smoke evidence; rocWMMA remains opt-in",
        "evidence": {q: asdict(ev) | {"ok": ev.ok} for q, ev in sorted(evidence.items())},
        "candidates": [asdict(c) for c in candidates],
    }


def env_inline(env: dict[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in env.items()) or "none"


def render_markdown(policy: dict[str, Any]) -> str:
    lines = [
        f"# RDNA3 FlashAttention policy: {policy['model_label']}",
        "",
        f"GPU/backend: `{policy['gpu_label']}` / `{policy['backend']}`",
        f"Selected: `{policy.get('selected_candidate') or 'none'}`",
        f"Server flags: `{' '.join(policy.get('selected_server_flags') or [])}`",
        f"Cache types: `{', '.join(policy.get('selected_cache_types') or [])}`",
        f"Env: `{env_inline(policy.get('selected_env') or {})}`",
        "",
        "## Candidates",
        "",
        "| candidate | eligible | route | cache types | env | status | notes |",
        "|---|:---:|---|---|---|---|---|",
    ]
    for c in policy["candidates"]:
        notes = "; ".join(c["reject_reasons"]) if c["reject_reasons"] else c["notes"]
        lines.append(
            f"| `{c['name']}` | {'yes' if c['eligible'] else 'NO'} | `{c['route']}` | "
            f"`{', '.join(c['cache_types'])}` | `{env_inline(c['env'])}` | {c['status']} | {notes} |"
        )

    lines += [
        "",
        "## Smoke evidence",
        "",
        "| quant | ok | contexts | kernels vec/wmma/other | prompt tok/s | decode tok/s | peak VRAM GiB | failures |",
        "|---|:---:|---:|---:|---|---|---|---|",
    ]
    for q, ev in policy["evidence"].items():
        prompt = ", ".join(f"{k}:{v:.1f}" for k, v in ev["prompt_tok_s"].items()) or "—"
        decode = ", ".join(f"{k}:{v:.1f}" for k, v in ev["decode_tok_s"].items()) or "—"
        peak = ", ".join(f"{k}:{v:.2f}" for k, v in ev["peak_vram_gib"].items()) or "—"
        kc = ev["kernel_counts"]
        lines.append(
            f"| `{q}` | {'yes' if ev['ok'] else 'NO'} | {', '.join(map(str, ev['contexts']))} | "
            f"{kc.get('vec', 0)}/{kc.get('wmma', 0)}/{kc.get('other', 0)} | {prompt} | {decode} | {peak} | "
            f"{'; '.join(ev['failures']) or '—'} |"
        )

    lines += [
        "",
        "## Activation",
        "",
        "```bash",
        "# server flags",
        "--flash-attn on --cache-type-k tbq4_0 --cache-type-v tbq4_0",
    ]
    for k, v in (policy.get("selected_env") or {}).items():
        lines.append(f"export {k}={v}")
    lines += ["```", ""]
    return "\n".join(lines)


def render_env(policy: dict[str, Any]) -> str:
    lines = [
        "# Generated by scripts/hip/rdna3-fattn-policy.py",
        f"# selected_candidate={policy.get('selected_candidate')}",
        "# server flags: " + " ".join(policy.get("selected_server_flags") or []),
        "# cache types: " + ",".join(policy.get("selected_cache_types") or []),
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
    p.add_argument("--summary", help="ctx-fit compressed-KV server smoke summary.json")
    p.add_argument("--out-dir", help="write policy.json, policy.md, and env.sh here")
    p.add_argument("--json", action="store_true", help="print policy JSON to stdout")
    p.add_argument("--emit-env", action="store_true", help="print shell exports/comments for selected path")
    p.add_argument("--require-summary", action="store_true", help="require smoke evidence for all stable compressed-KV types")
    p.add_argument("--include-log-env", action="store_true", help="include COMPRESSED_KV_FATTN_LOG=1 in selected env for route verification")
    p.add_argument("--allow-experimental-wmma", action="store_true", help="document opt-in WMMA without the extra warning; still not promoted without smoke/perf evidence")
    p.add_argument("--model-label", default="Qwen3.6-27B MTP compressed KV")
    p.add_argument("--gpu-label", default="gfx1100/RX 7900 XTX")
    p.add_argument("--backend", default="ROCm/HIP")
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
