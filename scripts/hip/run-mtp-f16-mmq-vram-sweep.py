#!/usr/bin/env python3
"""ROCm MTP f16/MMQ VRAM-observed server sweep.

Default goal: isolate MTP draft-prefill chunking, forced MMQ, and the opt-in
ROCm quantized-KV f16 FlashAttention route on a smaller MTP model before any
27B deep-context run.

The script intentionally records raw evidence (command, env, route logs,
rocm-smi samples, timings) and invalidates cases that do not use MTP or whose
LLAMA_MTP_PREFILL_CHUNK does not match --ubatch-size.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path("/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github")
MODEL_9B_DEFAULT = "/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.5-9B-Q5_K_M.gguf"


def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def run_text(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - evidence capture only
        return f"ERROR: {type(exc).__name__}: {exc}"


def vram_bytes() -> int | None:
    out = run_text(["rocm-smi", "--showmeminfo", "vram"], timeout=8.0)
    m = re.search(r"VRAM Total Used Memory \(B\):\s*(\d+)", out)
    return int(m.group(1)) if m else None


class VramSampler:
    def __init__(self, path: Path, interval: float) -> None:
        self.path = path
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.write()

    def mark(self, label: str) -> None:
        self.samples.append({"t": time.time(), "label": label, "vram_b": vram_bytes()})
        self.write()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append({"t": time.time(), "label": "sample", "vram_b": vram_bytes()})
            self._stop.wait(self.interval)

    def write(self) -> None:
        with self.path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t", "label", "vram_b"])
            w.writeheader()
            w.writerows(self.samples)

    def stats(self) -> dict[str, Any]:
        vals = [s["vram_b"] for s in self.samples if isinstance(s.get("vram_b"), int)]
        by_label = {s["label"]: s["vram_b"] for s in self.samples if s.get("label") != "sample"}
        return {
            "samples": len(self.samples),
            "min_b": min(vals) if vals else None,
            "max_b": max(vals) if vals else None,
            "first_b": vals[0] if vals else None,
            "last_b": vals[-1] if vals else None,
            "marks": by_label,
        }


@dataclass(frozen=True)
class Case:
    name: str
    chunk: int
    ubatch: int
    f16: bool
    mmq: bool

    @property
    def valid_contract(self) -> bool:
        return self.chunk == self.ubatch


def http_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    try:
        obj = json.loads(raw)
    except Exception:
        return {"ok": False, "raw": raw[:2000]}
    obj["_raw_preview"] = raw[:2000]
    return obj


def wait_health(base: str, proc: subprocess.Popen[bytes], timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: Any = None
    while time.time() < deadline:
        if proc.poll() is not None:
            return {"ok": False, "dead": True, "rc": proc.returncode, "last": last}
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                body = r.read().decode("utf-8", "replace")[:500]
                if r.status == 200:
                    return {"ok": True, "status": r.status, "body": body}
                last = {"status": r.status, "body": body}
        except Exception as exc:  # noqa: BLE001
            last = {"error_type": type(exc).__name__, "error": str(exc)[:500]}
        time.sleep(1)
    return {"ok": False, "timeout": True, "last": last}


def write_cmd(path: Path, env: dict[str, str], cmd: list[str]) -> None:
    keys = [
        "HIP_VISIBLE_DEVICES",
        "ROCM_VISIBLE_DEVICES",
        "LLAMA_MTP_PREFILL_CHUNK",
        "LLAMA_MTP_PREFILL_FORCE_MMQ",
        "GGML_CUDA_ROCM_QUANT_PREFILL_F16",
        "GGML_CUDA_ROCM_QUANT_PREFILL_F16_MAX_MIB",
        "GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_ALLOC",
        "GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_NKV",
        "TBQ4_COOP_SET_ROWS",
        "TBQ4_LAYER_ADAPTIVE",
        "COMPRESSED_KV_FATTN_LOG",
    ]
    parts = ["env"] + [f"{k}={env[k]}" for k in keys if k in env] + cmd
    path.write_text(" ".join(subprocess.list2cmdline([p]) for p in parts) + "\n")


def parse_routes(log_text: str) -> dict[str, Any]:
    lines = [ln for ln in log_text.splitlines() if "ggml_cuda_fattn_log_selection" in ln]
    counts: dict[str, int] = {}
    for ln in lines:
        m = re.search(r"kernel=([^\s]+) route=([^\s]+)", ln)
        key = f"{m.group(1)}:{m.group(2)}" if m else "unparsed"
        counts[key] = counts.get(key, 0) + 1
    return {"count": len(lines), "counts": counts, "tail": lines[-80:]}


def cleanup_proc(proc: subprocess.Popen[bytes] | None) -> int | None:
    if proc is None:
        return None
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=20)
    return proc.returncode


def build_cases(chunks: list[int], f16_values: list[int], mmq_values: list[int]) -> list[Case]:
    cases: list[Case] = []
    for chunk in chunks:
        for f16 in f16_values:
            for mmq in mmq_values:
                cases.append(Case(
                    name=f"chunk{chunk}_f16{f16}_mmq{mmq}",
                    chunk=chunk,
                    ubatch=chunk,
                    f16=bool(f16),
                    mmq=bool(mmq),
                ))
    return cases


def run_case(args: argparse.Namespace, case: Case, case_index: int, out_dir: Path) -> dict[str, Any]:
    cdir = out_dir / case.name
    cdir.mkdir(parents=True, exist_ok=True)
    port = args.base_port + case_index
    base = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env.update({
        "HIP_VISIBLE_DEVICES": "0",
        "ROCM_VISIBLE_DEVICES": "0",
        "LLAMA_MTP_PREFILL_CHUNK": str(case.chunk),
        "LLAMA_MTP_PREFILL_FORCE_MMQ": "1" if case.mmq else "0",
        "GGML_CUDA_ROCM_QUANT_PREFILL_F16": "1" if case.f16 else "0",
        "GGML_CUDA_ROCM_QUANT_PREFILL_F16_MAX_MIB": str(args.f16_max_mib),
        "TBQ4_COOP_SET_ROWS": "1",
        "TBQ4_LAYER_ADAPTIVE": "7",
        "COMPRESSED_KV_FATTN_LOG": "1",
    })

    cmd = [
        str(args.server_bin),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model", args.model,
        "-c", str(args.ctx_size),
        "--device", "ROCm0",
        "--flash-attn", "on",
        "--no-context-shift",
        "--no-webui",
        "--no-mmap",
        "--threads", str(args.threads),
        "--batch-size", str(case.ubatch),
        "--ubatch-size", str(case.ubatch),
        "--cache-type-k", "q8_0",
        "--cache-type-v", "tbq4_0",
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", str(args.spec_draft_n_max),
        "--parallel", "1",
        "--cache-prompt",
        "--fit", "off",
        "--no-warmup",
        "--log-verbosity", str(args.log_verbosity),
    ]
    write_cmd(cdir / "server.cmd.txt", env, cmd)

    summary: dict[str, Any] = {
        "case": case.__dict__,
        "contract_valid": case.valid_contract,
        "started": now_iso(),
        "cmd": cmd,
        "env_subset": {k: env[k] for k in sorted(env) if k.startswith(("LLAMA_MTP", "GGML_CUDA_ROCM", "TBQ4_", "COMPRESSED_KV", "HIP_VISIBLE", "ROCM_VISIBLE"))},
    }
    if not case.valid_contract:
        summary["invalid_reason"] = "LLAMA_MTP_PREFILL_CHUNK must equal --ubatch-size"
        (cdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    if args.dry_run:
        summary["dry_run"] = True
        (cdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    sampler = VramSampler(cdir / "vram.tsv", args.vram_interval)
    proc: subprocess.Popen[bytes] | None = None
    try:
        sampler.mark("baseline_before_launch")
        sampler.start()
        with (cdir / "server.stdout.log").open("wb") as stdout, (cdir / "server.stderr.log").open("wb") as stderr:
            proc = subprocess.Popen(cmd, cwd=args.repo, env=env, stdout=stdout, stderr=stderr, preexec_fn=os.setsid)
        summary["pid"] = proc.pid
        health = wait_health(base, proc, args.health_timeout)
        summary["health"] = health
        sampler.mark("after_health")
        if not health.get("ok"):
            return summary

        fill_prompt = " benchmark" * args.fill_tokens
        fill_payload = {"prompt": fill_prompt, "n_predict": 1, "temperature": 0.0, "top_k": 1, "cache_prompt": True, "seed": 1}
        (cdir / "fill.payload.json").write_text(json.dumps({"prompt_tokens_approx": args.fill_tokens, **{k: v for k, v in fill_payload.items() if k != "prompt"}}, indent=2) + "\n")
        sampler.mark("before_fill")
        t0 = time.time()
        try:
            fill = http_json(base + "/completion", fill_payload, args.request_timeout)
            summary["fill"] = {"ok": "timings" in fill, "wall_sec": round(time.time() - t0, 3), "timings": fill.get("timings"), "error": fill.get("error")}
            (cdir / "fill.raw.json").write_text(json.dumps(fill, indent=2) + "\n")
        except Exception as exc:  # noqa: BLE001
            summary["fill"] = {"ok": False, "wall_sec": round(time.time() - t0, 3), "error_type": type(exc).__name__, "error": str(exc)[:2000]}
        sampler.mark("after_fill")

        test_tokens = args.fill_tokens + args.pp_tokens
        test_prompt = " benchmark" * test_tokens
        test_payload = {"prompt": test_prompt, "n_predict": 1, "temperature": 0.0, "top_k": 1, "cache_prompt": True, "seed": 2}
        (cdir / "test.payload.json").write_text(json.dumps({"prompt_tokens_approx": test_tokens, **{k: v for k, v in test_payload.items() if k != "prompt"}}, indent=2) + "\n")
        sampler.mark("before_test")
        t0 = time.time()
        try:
            test = http_json(base + "/completion", test_payload, args.request_timeout)
            summary["test"] = {"ok": "timings" in test, "wall_sec": round(time.time() - t0, 3), "timings": test.get("timings"), "error": test.get("error")}
            (cdir / "test.raw.json").write_text(json.dumps(test, indent=2) + "\n")
        except Exception as exc:  # noqa: BLE001
            summary["test"] = {"ok": False, "wall_sec": round(time.time() - t0, 3), "error_type": type(exc).__name__, "error": str(exc)[:2000]}
        sampler.mark("after_test")
        return summary
    finally:
        summary["server_rc"] = cleanup_proc(proc)
        sampler.mark("after_shutdown")
        sampler.stop()
        summary["vram"] = sampler.stats()
        stderr_path = cdir / "server.stderr.log"
        log_text = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""
        routes = parse_routes(log_text)
        summary["routes"] = routes
        (cdir / "routes.txt").write_text("\n".join(routes["tail"]) + ("\n" if routes["tail"] else ""))
        summary["finished"] = now_iso()
        (cdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    ap.add_argument("--server-bin", type=Path, default=None)
    ap.add_argument("--model", default=MODEL_9B_DEFAULT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ctx-size", type=int, default=12288)
    ap.add_argument("--fill-tokens", type=int, default=8192)
    ap.add_argument("--pp-tokens", type=int, default=2048)
    ap.add_argument("--chunks", default="512,2048", help="Comma-separated chunk/ubatch sizes")
    ap.add_argument("--f16", default="0,1", help="Comma-separated GGML_CUDA_ROCM_QUANT_PREFILL_F16 values")
    ap.add_argument("--mmq", default="0,1", help="Comma-separated LLAMA_MTP_PREFILL_FORCE_MMQ values")
    ap.add_argument("--f16-max-mib", type=int, default=1024)
    ap.add_argument("--stable-f16-alloc", action="store_true", help="Set GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_ALLOC=1")
    ap.add_argument("--stable-nkv", type=int, default=0, help="Set GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_NKV when >0")
    ap.add_argument("--spec-draft-n-max", type=int, default=3)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--base-port", type=int, default=18450)
    ap.add_argument("--health-timeout", type=float, default=600)
    ap.add_argument("--request-timeout", type=float, default=1200)
    ap.add_argument("--vram-interval", type=float, default=1.0)
    ap.add_argument("--log-verbosity", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.repo = args.repo.resolve()
    if args.server_bin is None:
        args.server_bin = args.repo / "build-rocm-fa-fallback/bin/llama-server"
    else:
        args.server_bin = args.server_bin.resolve()
    if args.out is None:
        args.out = args.repo / "benches" / f"server-mtp-f16-mmq-vram-{now_tag()}"
    else:
        args.out = args.out.resolve()

    chunks = [int(x) for x in args.chunks.split(",") if x]
    f16_values = [int(x) for x in args.f16.split(",") if x]
    mmq_values = [int(x) for x in args.mmq.split(",") if x]
    cases = build_cases(chunks, f16_values, mmq_values)

    if args.stable_f16_alloc:
        os.environ["GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_ALLOC"] = "1"
        if args.stable_nkv > 0:
            os.environ["GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_NKV"] = str(args.stable_nkv)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started": now_iso(),
        "repo": str(args.repo),
        "server_bin": str(args.server_bin),
        "model": args.model,
        "ctx_size": args.ctx_size,
        "fill_tokens": args.fill_tokens,
        "pp_tokens": args.pp_tokens,
        "cases": [c.__dict__ for c in cases],
        "dry_run": args.dry_run,
        "stable_f16_alloc": args.stable_f16_alloc,
        "stable_nkv": args.stable_nkv,
        "contract": {
            "mtp_required": "--spec-type draft-mtp --spec-draft-n-max 3 --parallel 1",
            "chunk_equals_ubatch_required": True,
            "vram_sampling_required": True,
            "route_log_required": True,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        summary = run_case(args, case, i, args.out)
        fill_t = (summary.get("fill") or {}).get("timings") or {}
        test_t = (summary.get("test") or {}).get("timings") or {}
        vram = summary.get("vram") or {}
        marks = vram.get("marks") or {}
        loaded = marks.get("after_health")
        peak = vram.get("max_b")
        after_shutdown = marks.get("after_shutdown")
        rows.append({
            "case": case.name,
            "chunk": case.chunk,
            "ubatch": case.ubatch,
            "f16": int(case.f16),
            "mmq": int(case.mmq),
            "contract_valid": summary.get("contract_valid"),
            "health_ok": (summary.get("health") or {}).get("ok"),
            "fill_pps": fill_t.get("prompt_per_second"),
            "test_cache_n": test_t.get("cache_n"),
            "test_prompt_n": test_t.get("prompt_n"),
            "test_pps": test_t.get("prompt_per_second"),
            "loaded_gib": round(loaded / 1024**3, 3) if isinstance(loaded, int) else None,
            "peak_gib": round(peak / 1024**3, 3) if isinstance(peak, int) else None,
            "peak_delta_gib": round((peak - loaded) / 1024**3, 3) if isinstance(peak, int) and isinstance(loaded, int) else None,
            "after_shutdown_gib": round(after_shutdown / 1024**3, 3) if isinstance(after_shutdown, int) else None,
            "route_counts": json.dumps((summary.get("routes") or {}).get("counts", {}), sort_keys=True),
            "server_rc": summary.get("server_rc"),
            "dir": str(args.out / case.name),
        })

    fields = list(rows[0].keys()) if rows else []
    with (args.out / "summary.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(args.out)
    print((args.out / "summary.tsv").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
