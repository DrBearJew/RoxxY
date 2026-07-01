#!/usr/bin/env python3
"""Tensor-filtered I8 prefill smoke/correctness harness.

Runs a no-benchmark llama-server A/B:
  baseline: selected default-off I8 route disabled
  candidate: selected default-off I8 route enabled for one tensor substring

The harness captures logs/artifacts, checks route-specific candidate evidence,
and compares baseline vs candidate completion content hashes.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CANDIDATES = [
    # q8_0-heavy split model verified to initialize with llama-server in this tree.
    Path("/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-Q4_K_M-00001-of-00002.gguf"),
    # Fallbacks mirror existing local smoke scripts; they may not hit the q8_0 route.
    Path("/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.5-9B-Q5_K_M.gguf"),
    Path("/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf"),
]
COUNT_NEEDLES = {
    "hipblaslt_algo_log_lines": "hipBLASLt heuristic route=",
    "hipblaslt_k32_stage_lines": "route=hipblaslt_i8_q8_0_k32_stage",
    "hipblaslt_fullk_stage_lines": "route=hipblaslt_i8_q8_0_fullk_stage_k32_exact",
    "direct_hipblaslt_attr_lines": "direct_hipblaslt_i8_prefill",
    "direct_mmq_after_hipblaslt_fallback_lines": "direct_mmq_after_hipblaslt_i8_fallback",
    "hipblaslt_route_failed_lines": "q8_0 hipBLASLt-I8 route failed",
    "hipblaslt_scratch_rejected_lines": "q8_0 hipBLASLt-I8 scratch plan rejected",
    "hipblaslt_heuristic_failed_lines": "hipBLASLt heuristic failed",
    "hipblaslt_no_algo_lines": "hipBLASLt heuristic returned no successful algo",
    "rdna_q8_route_lines": "route=rdna_i8_q8_0",
    "rdna_q8_direct_attr_lines": "direct_rdna_i8_q8_0_prefill",
    "rdna_q8_after_hipblaslt_fallback_attr_lines": "direct_rdna_i8_q8_0_after_hipblaslt_i8_fallback",
    "rdna_q8_fallback_lines": "direct_mmq_after_rdna_i8_q8_0_fallback",
    "rdna_q8_rejected_lines": "rdna_i8_q8_0 route rejected",
    "rdna_q6_route_lines": "route=rdna_i8_q6_K_prefill",
    "rdna_q6_direct_attr_lines": "direct_rdna_i8_q6_K_prefill",
    "rdna_q6_fallback_lines": "direct_mmq_after_rdna_i8_q6_K_fallback",
    "assert_lines": "GGML_ASSERT",
    "fatal_lines": "FATAL",
}
ENV_PREFIXES = ("GGML_", "LLAMA_", "HIP", "ROCR", "HSA", "CUDA", "SPEC")
ROUTE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "hipblaslt-q8": {
        "enable_env": "GGML_CUDA_HIPBLASLT_I8_PREFILL",
        "log_env": "GGML_CUDA_HIPBLASLT_I8_PREFILL_ALGO_LOG",
        "tensor_env": "GGML_CUDA_HIPBLASLT_I8_PREFILL_TENSOR",
        "evidence_keys": ("hipblaslt_algo_log_lines",),
        "fallback_keys": (
            "direct_mmq_after_hipblaslt_fallback_lines",
            "hipblaslt_route_failed_lines",
            "hipblaslt_heuristic_failed_lines",
            "hipblaslt_no_algo_lines",
        ),
    },
    "rdna-q8": {
        "enable_env": "GGML_CUDA_RDNA_I8_Q8_0_PREFILL",
        "log_env": "GGML_CUDA_RDNA_I8_Q8_0_PREFILL_LOG",
        "tensor_env": "GGML_CUDA_RDNA_I8_Q8_0_PREFILL_TENSOR",
        "evidence_keys": ("rdna_q8_route_lines", "rdna_q8_direct_attr_lines"),
        "fallback_keys": ("rdna_q8_fallback_lines", "rdna_q8_rejected_lines"),
    },
    "rdna-q6": {
        "enable_env": "GGML_CUDA_RDNA_I8_Q6_K_PREFILL",
        "extra_enable_envs": ("GGML_CUDA_RDNA_I8_Q6_K_PREFILL_UNSAFE_HASH_DRIFT",),
        "log_env": "GGML_CUDA_RDNA_I8_Q6_K_PREFILL_LOG",
        "tensor_env": "GGML_CUDA_RDNA_I8_Q6_K_PREFILL_TENSOR",
        "evidence_keys": ("rdna_q6_route_lines", "rdna_q6_direct_attr_lines"),
        "fallback_keys": ("rdna_q6_fallback_lines",),
    },
}


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def pick_default_model() -> Optional[Path]:
    env_model = os.environ.get("LLAMA_MODEL") or os.environ.get("MODEL")
    if env_model:
        return Path(env_model)
    for path in DEFAULT_MODEL_CANDIDATES:
        if path.is_file():
            return path
    return None


def model_needs_mtp_source_without_args(args: argparse.Namespace) -> bool:
    name = args.model.name.lower()
    if "gemma-4" not in name or "assistant" not in name:
        return False
    return not any("mtp" in item.lower() or "draft" in item.lower() for item in args.extra_server_arg)


def free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def set_arg(args: List[str], opt: str, value: Any) -> List[str]:
    args = list(args)
    if opt in args:
        idx = args.index(opt)
        if idx + 1 >= len(args):
            raise ValueError(f"option {opt} lacks value slot")
        args[idx + 1] = str(value)
    else:
        args += [opt, str(value)]
    return args


def make_server_args(args: argparse.Namespace, port: int) -> List[str]:
    server = args.server_bin or (args.build / "bin" / "llama-server")
    cmd = [
        str(server),
        "--device", args.device,
        "--model", str(args.model),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-webui",
        "--flash-attn", args.flash_attn,
        "--ctx-size", str(args.ctx_size),
        "--batch-size", str(args.batch_size),
        "--ubatch-size", str(args.ubatch_size),
        "--parallel", "1",
        "--no-warmup",
        "--cache-type-k", args.cache_type_k,
        "--cache-type-v", args.cache_type_v,
    ]
    if args.no_mmap:
        cmd.append("--no-mmap")
    if args.kv_unified:
        cmd.append("--kv-unified")
    if args.extra_server_arg:
        cmd.extend(args.extra_server_arg)
    return cmd


def make_request(args: argparse.Namespace) -> Dict[str, Any]:
    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    return {
        "prompt": prompt,
        "n_predict": args.n_predict,
        "temperature": 0,
        "seed": args.seed,
        "stream": False,
        "cache_prompt": False,
        "ignore_eos": True,
    }


def visible_env(env: Dict[str, str]) -> Dict[str, str]:
    return {k: env[k] for k in sorted(env) if k.startswith(ENV_PREFIXES)}


def route_config(args: argparse.Namespace) -> Dict[str, Any]:
    return ROUTE_CONFIGS[args.route]


def make_env(case: str, args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HIP_VISIBLE_DEVICES", args.hip_visible_devices)
    env.setdefault("LLAMA_ARG_FIT", "off")

    for cfg in ROUTE_CONFIGS.values():
        env[str(cfg["enable_env"])] = "0"
        for extra in cfg.get("extra_enable_envs", ()):
            env[str(extra)] = "0"
        env[str(cfg["tensor_env"])] = args.tensor
        if cfg.get("log_env"):
            env[str(cfg["log_env"])] = "0"

    cfg = route_config(args)
    env[str(cfg["tensor_env"])] = args.tensor
    if cfg.get("log_env"):
        env[str(cfg["log_env"])] = "1"
    if case == "candidate":
        env[str(cfg["enable_env"])] = "1"
        for extra in cfg.get("extra_enable_envs", ()):
            env[str(extra)] = "1"

    if args.route == "hipblaslt-q8":
        if args.workspace_mb is not None:
            env["GGML_CUDA_HIPBLASLT_I8_WORKSPACE_MB"] = str(args.workspace_mb)
        if args.algo_ord is not None:
            env["GGML_CUDA_HIPBLASLT_I8_PREFILL_ALGO_ORD"] = str(args.algo_ord)
        if args.fullk_stage_k32_exact:
            env["GGML_CUDA_HIPBLASLT_I8_PREFILL_FULLK_STAGE_K32_EXACT"] = "1"
        else:
            env.pop("GGML_CUDA_HIPBLASLT_I8_PREFILL_FULLK_STAGE_K32_EXACT", None)
    else:
        env.pop("GGML_CUDA_HIPBLASLT_I8_PREFILL_FULLK_STAGE_K32_EXACT", None)

    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        env[key] = value
    return env


def http_json(port: int, method: str, path: str, obj: Optional[Dict[str, Any]], timeout_sec: int) -> Tuple[int, Dict[str, Any], str]:
    data = json.dumps(obj).encode("utf-8") if obj is not None else None
    headers = {"Content-Type": "application/json"} if obj is not None else {}
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        text = resp.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"raw": text}
        return int(resp.status), parsed, text


def wait_ready(port: int, proc: subprocess.Popen[bytes], timeout_sec: int) -> Tuple[bool, str]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, f"server_exit_{proc.returncode}"
        try:
            status, _obj, text = http_json(port, "GET", "/health", None, 2)
            if status == 200 and ("ok" in text.lower() or text.strip()):
                return True, "health"
        except Exception:
            time.sleep(1)
    return False, "health_timeout"


def terminate(proc: subprocess.Popen[bytes]) -> int:
    if proc.poll() is not None:
        return int(proc.returncode)
    try:
        proc.send_signal(signal.SIGINT)
        return int(proc.wait(timeout=20))
    except Exception:
        pass
    if proc.poll() is None:
        proc.terminate()
    try:
        return int(proc.wait(timeout=10))
    except Exception:
        proc.kill()
        return int(proc.wait(timeout=5))


def summarize_log(log: str) -> Dict[str, Any]:
    counts = {k: log.count(v) for k, v in COUNT_NEEDLES.items()}
    samples: List[str] = []
    for line in log.splitlines():
        if any(token in line for token in ("hipBLASLt", "hipblaslt_i8", "rdna_i8", "direct_rdna", "direct_mmq_after_rdna")):
            samples.append(line[:800])
            if len(samples) >= 30:
                break
    counts["route_samples"] = samples
    return counts


def content_from_response(resp: Dict[str, Any]) -> str:
    if "content" in resp:
        return str(resp.get("content") or "")
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        choice0 = choices[0]
        if isinstance(choice0, dict):
            return str(choice0.get("text") or choice0.get("content") or "")
    return ""


def run_case(root: Path, case: str, args: argparse.Namespace) -> Dict[str, Any]:
    case_dir = root / case
    case_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    cmd = make_server_args(args, port)
    env = make_env(case, args)
    request = make_request(args)
    (case_dir / "cmd.json").write_text(json.dumps({"args": cmd, "env": visible_env(env)}, indent=2) + "\n", encoding="utf-8")
    (case_dir / "request.json").write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.dry_run:
        summary = {
            "case": case,
            "case_dir": str(case_dir),
            "dry_run": True,
            "cmd": cmd,
            "env": visible_env(env),
            "route_contract_ok": None,
            "http_ok": None,
            "sha8": None,
        }
        (case_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    log_path = case_dir / "server.log"
    response_path = case_dir / "response.json"
    logf = log_path.open("wb")
    proc = subprocess.Popen(cmd, cwd=str(args.repo), env=env, stdout=logf, stderr=subprocess.STDOUT)
    err = ""
    status: Optional[int] = None
    resp_obj: Dict[str, Any] = {}
    wall_s: Optional[float] = None
    rc: Optional[int] = None
    try:
        ready, why = wait_ready(port, proc, args.start_timeout)
        if not ready:
            err = why
        else:
            t0 = time.time()
            status, resp_obj, _raw = http_json(port, "POST", "/completion", request, args.timeout)
            wall_s = time.time() - t0
            response_path.write_text(json.dumps(resp_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    finally:
        time.sleep(1)
        rc = terminate(proc)
        logf.close()

    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    content = content_from_response(resp_obj)
    timings = resp_obj.get("timings") if isinstance(resp_obj.get("timings"), dict) else {}
    counts = summarize_log(log)
    cfg = route_config(args)
    evidence_keys = tuple(str(k) for k in cfg["evidence_keys"])
    fallback_keys = tuple(str(k) for k in cfg["fallback_keys"])
    route_evidence_lines = sum(int(counts.get(k, 0) or 0) for k in evidence_keys)
    fallback_hits = [k for k in fallback_keys if counts.get(k, 0)]
    route_contract_ok = True
    route_failures: List[str] = []
    if case == "candidate":
        if route_evidence_lines <= 0:
            route_contract_ok = False
            route_failures.append(f"missing_{args.route}_route_evidence")
        if fallback_hits:
            route_contract_ok = False
            route_failures.append(f"{args.route}_fallback_or_route_failed:{','.join(fallback_hits)}")
    else:
        if route_evidence_lines > 0:
            route_contract_ok = False
            route_failures.append(f"baseline_unexpected_{args.route}_route_evidence")
    if counts["assert_lines"] or counts["fatal_lines"]:
        route_contract_ok = False
        route_failures.append("assert_or_fatal_in_log")

    http_ok = status == 200 and not err and bool(content)
    summary: Dict[str, Any] = {
        "case": case,
        "case_dir": str(case_dir),
        "dry_run": False,
        "http_status": status,
        "http_ok": http_ok,
        "err": err,
        "rc": rc,
        "wall_s": wall_s,
        "sha8": hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:8] if content else None,
        "content_len": len(content),
        "tokens_evaluated": resp_obj.get("tokens_evaluated") or timings.get("prompt_n"),
        "tokens_predicted": resp_obj.get("tokens_predicted") or timings.get("predicted_n"),
        "route_contract_ok": route_contract_ok,
        "route_failures": route_failures,
        "server_log_bytes": len(log.encode("utf-8", errors="replace")),
        **counts,
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def write_root_summary(root: Path, cases: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    case_list = list(cases)
    base = next((c for c in case_list if c.get("case") == "baseline"), {})
    cand = next((c for c in case_list if c.get("case") == "candidate"), {})
    dry_run = any(c.get("dry_run") for c in case_list)
    if dry_run:
        same_hash = None
        same_tokens_predicted = None
        all_http_ok = None
        all_route_ok = None
        ok = None
    else:
        same_hash = bool(base.get("sha8")) and base.get("sha8") == cand.get("sha8")
        same_tokens_predicted = base.get("tokens_predicted") == cand.get("tokens_predicted")
        all_http_ok = all(c.get("http_ok") for c in case_list) if case_list else None
        all_route_ok = all(c.get("route_contract_ok") for c in case_list) if case_list else None
        ok = bool(all_http_ok and all_route_ok and same_hash and same_tokens_predicted)
    summary = {
        "ok": ok,
        "dry_run": dry_run,
        "root": str(root),
        "same_hash": same_hash,
        "same_tokens_predicted": same_tokens_predicted,
        "all_http_ok": all_http_ok,
        "all_route_ok": all_route_ok,
        "baseline_sha8": base.get("sha8"),
        "candidate_sha8": cand.get("sha8"),
        "cases": case_list,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: List[str]) -> argparse.Namespace:
    default_model = pick_default_model()
    default_tensor = (
        os.environ.get("GGML_CUDA_HIPBLASLT_I8_PREFILL_TENSOR")
        or os.environ.get("GGML_CUDA_RDNA_I8_Q8_0_PREFILL_TENSOR")
        or os.environ.get("GGML_CUDA_RDNA_I8_Q6_K_PREFILL_TENSOR")
        or "ffn_down"
    )
    parser = argparse.ArgumentParser(
        description="Run a no-benchmark baseline-vs-candidate I8 prefill smoke with tensor filter and hash comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", type=Path, default=REPO, help="llama.cpp checkout root")
    parser.add_argument("--build", type=Path, default=Path(os.environ.get("BUILD", str(REPO / "build-rocm-qwen35-dev"))), help="build dir containing bin/llama-server")
    parser.add_argument("--server-bin", type=Path, default=Path(os.environ["LLAMA_SERVER_BIN"]) if os.environ.get("LLAMA_SERVER_BIN") else None, help="override llama-server executable")
    parser.add_argument("--model", type=Path, default=default_model, help="GGUF model; selected route must support at least one matched tensor")
    parser.add_argument("--route", choices=sorted(ROUTE_CONFIGS), default=os.environ.get("I8_PREFILL_SMOKE_ROUTE", "hipblaslt-q8"), help="default-off I8 route to enable for the candidate case")
    parser.add_argument("--tensor", default=default_tensor, help="substring filter passed to the selected route tensor env")
    parser.add_argument("--root", type=Path, default=None, help="artifact root; default: <repo>/.harness/tmp/hipblaslt-i8-prefill-smoke-<timestamp>")
    parser.add_argument("--prompt", default=("Summarize in one short paragraph why deterministic GPU correctness smokes compare content hashes. " * 12).strip(), help="completion prompt")
    parser.add_argument("--prompt-file", type=Path, default=None, help="optional prompt file")
    parser.add_argument("--n-predict", type=int, default=16, help="tokens to generate")
    parser.add_argument("--seed", type=int, default=1234, help="deterministic generation seed")
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q8_0")
    parser.add_argument("--flash-attn", default="on", choices=["on", "off", "auto"])
    parser.add_argument("--device", default="ROCm0")
    parser.add_argument("--hip-visible-devices", default=os.environ.get("HIP_VISIBLE_DEVICES", "0"))
    parser.add_argument("--workspace-mb", type=int, default=None, help="sets GGML_CUDA_HIPBLASLT_I8_WORKSPACE_MB")
    parser.add_argument("--algo-ord", type=int, default=None, help="sets GGML_CUDA_HIPBLASLT_I8_PREFILL_ALGO_ORD")
    parser.add_argument("--fullk-stage-k32-exact", action="store_true", help="hipblaslt-q8 only: sets GGML_CUDA_HIPBLASLT_I8_PREFILL_FULLK_STAGE_K32_EXACT=1")
    parser.add_argument("--kv-unified", action="store_true", help="pass --kv-unified to server")
    parser.add_argument("--no-mmap", action="store_true", help="pass --no-mmap to server")
    parser.add_argument("--env", action="append", default=[], help="extra environment KEY=VALUE; repeatable")
    parser.add_argument("--extra-server-arg", action="append", default=[], help="append one raw server arg; repeatable")
    parser.add_argument("--start-timeout", type=int, default=240)
    parser.add_argument("--timeout", type=int, default=600, help="completion request timeout seconds")
    parser.add_argument("--dry-run", action="store_true", help="write planned commands/envs without starting the server")
    args = parser.parse_args(argv)
    if args.model is None:
        parser.error("no default model found; pass --model /path/to/q8_0.gguf or set LLAMA_MODEL")
    args.repo = args.repo.resolve()
    args.build = args.build.resolve()
    args.model = args.model.resolve()
    if args.server_bin is not None:
        args.server_bin = args.server_bin.resolve()
    return args


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    server = args.server_bin or (args.build / "bin" / "llama-server")
    root = args.root or (args.repo / ".harness" / "tmp" / f"{args.route}-prefill-smoke-{timestamp()}")
    root.mkdir(parents=True, exist_ok=True)
    preflight_failures = []
    if not server.is_file() or not os.access(server, os.X_OK):
        preflight_failures.append(f"missing executable: {server}")
    if not args.model.is_file():
        preflight_failures.append(f"missing model: {args.model}")
    if model_needs_mtp_source_without_args(args):
        preflight_failures.append(
            "Gemma 4 assistant q8_0 model aborts without an MTP/draft source; pass the required source via --extra-server-arg or choose a standalone q8_0 model")
    preflight = {
        "root": str(root),
        "repo": str(args.repo),
        "server": str(server),
        "model": str(args.model),
        "route": args.route,
        "tensor": args.tensor,
        "fullk_stage_k32_exact": args.fullk_stage_k32_exact,
        "dry_run": args.dry_run,
        "preflight_failures": preflight_failures,
    }
    (root / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    if preflight_failures and not args.dry_run:
        print(json.dumps(preflight, indent=2), file=sys.stderr)
        return 2

    print(f"artifact_root={root}", flush=True)
    cases = [run_case(root, "baseline", args), run_case(root, "candidate", args)]
    summary = write_root_summary(root, cases)
    cfg = route_config(args)
    candidate = next((c for c in cases if c.get("case") == "candidate"), {})
    candidate_route_lines = sum(int(candidate.get(str(k), 0) or 0) for k in cfg["evidence_keys"])
    print(json.dumps({
        "ok": summary["ok"],
        "artifact_root": str(root),
        "route": args.route,
        "same_hash": summary["same_hash"],
        "same_tokens_predicted": summary["same_tokens_predicted"],
        "baseline_sha8": summary["baseline_sha8"],
        "candidate_sha8": summary["candidate_sha8"],
        "candidate_route_lines": candidate_route_lines,
        "candidate_failures": candidate.get("route_failures"),
    }, indent=2), flush=True)
    if args.dry_run:
        return 0
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
