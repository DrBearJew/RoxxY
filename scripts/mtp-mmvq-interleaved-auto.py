#!/usr/bin/env python3
"""
Smart wrapper for RoxxY RDNA3 MTP MMVQ interleaved-activation policies.

The low-level LLAMA_MTP_MMVQ_* knobs are intentionally experimental.  This
script gives users one safe entry point:

  scripts/mtp-mmvq-interleaved-auto.py --model model.gguf -- ./build-rocm/bin/llama-server ...

Behavior:
  * known/cached GGUF layout: apply the measured policy
  * unknown GGUF layout: leave interleaved routes off unless a policy is forced
  * always clears stale managed MMVQ/PDMQ env before applying a policy

This script does not benchmark unknown models by itself.  It is a safe policy
applier/cache layer; use --policy to force a developer-tested policy and
--trust-policy to cache it for the model fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple


MANAGED_VARS: Tuple[str, ...] = tuple(
    [
        "LLAMA_MTP_MMVQ_INTERLEAVED_ACT_MULTI_TYPE_NWARPS_UNSAFE",
        "GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_Q8V_N64",
        "GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_IMPL",
        "GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_VPATH",
        "GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE",
        "GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE_AUTO",
        "GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_GROUP",
        "GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_REQUIRE",
        "GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK",
        "GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK_ROOF_CAP",
        "GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK_COMPACT_EMPTY",
    ]
    + [
        f"LLAMA_MTP_MMVQ_{family}_INTERLEAVED_ACT{suffix}"
        for family in ("Q4K", "Q5K", "Q6K")
        for suffix in ("", "_LOG", "_FILTER", "_NCOLS", "_ROWS", "_NWARPS")
    ]
    + [
        f"LLAMA_MTP_MMVQ_{family}_INTERLEAVED_ACT{suffix}"
        for family in ("LEGACY", "LOWK")
        for suffix in ("", "_LOG", "_FILTER", "_TYPES", "_NCOLS", "_ROWS", "_NWARPS")
    ]
)

POLICIES: Mapping[str, Mapping[str, str]] = {
    "off": {},
    # Qwen3.6-27B-Q4_K_M-mtp.gguf q4 fast path is now a core default:
    # packed16 K, PDMQ, backend top-k, FFN/MMVQ, and q4/q6 interleaved-act
    # policy are selected by the runtime.  Keep the policy name for cache/backward
    # compatibility, but do not export q4 magic env from the launcher.
    "q4q6-27b-fast": {},
    # Same 27B Q4_K_M weight policy, but for users explicitly running q8_0 V.
    # q8_0 V baseline was ~52 tok/s; q8v-n64 was 56.4/57.2 tok/s with a
    # different prompt-x trajectory. Auto-selected only when the command/env
    # asks for q8_0 V on this known model.
    "q4q6-27b-q8v-fast": {
        "GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_Q8V_N64": "1",
    },
    # Measured on Qwen3.6-27B Heretic Native-MTP i1-Q6_K prompt-x n512:
    # baseline 84.66 tok/s; q6 nw4 89.62/90.94 tok/s, SHA-clean.
    "q6k-nw4": {
        "LLAMA_MTP_MMVQ_Q6K_INTERLEAVED_ACT": "1",
        "LLAMA_MTP_MMVQ_Q6K_INTERLEAVED_ACT_NWARPS": "4",
    },
    # Coverage/dev policy for route-smoke tests across GGUF quant families.
    # This is not a fastest-known production policy.
    "coverage": {
        "LLAMA_MTP_MMVQ_Q4K_INTERLEAVED_ACT": "1",
        "LLAMA_MTP_MMVQ_Q5K_INTERLEAVED_ACT": "1",
        "LLAMA_MTP_MMVQ_Q6K_INTERLEAVED_ACT": "1",
        "LLAMA_MTP_MMVQ_LEGACY_INTERLEAVED_ACT": "1",
        "LLAMA_MTP_MMVQ_LOWK_INTERLEAVED_ACT": "1",
    },
}

BUILTIN_NAME_MATCHES: Tuple[Tuple[str, str, str], ...] = (
    (
        "q4q6-27b-fast",
        "Qwen3.6 27B Q4_K_M MTP measured policy",
        "qwen3.6-27b-q4_k_m-mtp.gguf",
    ),
    (
        "q6k-nw4",
        "Qwen3.6 27B Heretic i1-Q6_K measured policy",
        "qwen3.6-27b-uncensored-heretic-v2-native-mtp-preserved.i1-q6_k.gguf",
    ),
)


class PolicyError(RuntimeError):
    pass


def cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "roxxxy" / "mtp-mmvq-interleaved-policies.json"
    return Path.home() / ".cache" / "roxxxy" / "mtp-mmvq-interleaved-policies.json"


def sample_file_hash(path: Path, sample_size: int = 1024 * 1024) -> str:
    """Hash file size plus first/middle/last samples; avoids reading huge GGUFs."""
    stat = path.stat()
    size = stat.st_size
    h = hashlib.sha256()
    h.update(f"size:{size}\n".encode())
    with path.open("rb") as f:
        offsets = [0]
        if size > sample_size:
            offsets.append(max(0, size // 2 - sample_size // 2))
            offsets.append(max(0, size - sample_size))
        seen = set()
        for off in offsets:
            if off in seen:
                continue
            seen.add(off)
            f.seek(off)
            chunk = f.read(sample_size)
            h.update(f"offset:{off}:len:{len(chunk)}\n".encode())
            h.update(chunk)
    return h.hexdigest()


def model_fingerprint(path: Path) -> Dict[str, object]:
    path = path.expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "size": stat.st_size,
        "sample_sha256": sample_file_hash(path),
    }


def load_cache(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"version": 1, "models": {}}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive CLI path
        raise PolicyError(f"failed to read cache {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PolicyError(f"cache {path} is not a JSON object")
    data.setdefault("version", 1)
    data.setdefault("models", {})
    if not isinstance(data["models"], dict):
        raise PolicyError(f"cache {path} has invalid models field")
    return data


def save_cache(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def builtin_policy_for_name(name: str) -> Tuple[str, str] | None:
    normalized = name.lower()
    for policy, reason, needle in BUILTIN_NAME_MATCHES:
        if needle in normalized:
            return policy, reason
    return None


def command_option_value(cmd: List[str], names: Tuple[str, ...]) -> str | None:
    for i, token in enumerate(cmd):
        for name in names:
            if token == name and i + 1 < len(cmd):
                return cmd[i + 1]
            prefix = name + "="
            if token.startswith(prefix):
                return token[len(prefix):]
    return None


def requested_v_cache_type(cmd: List[str], env: Mapping[str, str]) -> str | None:
    # Main V first, then draft V. The q8_0 policy only needs to know whether the
    # user is intentionally running q8_0 V; normal q4_0 launches stay on the q4
    # fast policy.
    value = command_option_value(cmd, ("--cache-type-v", "-ctv"))
    if value:
        return value.lower()
    value = command_option_value(cmd, ("--cache-type-v-draft", "--spec-draft-type-v", "-ctvd"))
    if value:
        return value.lower()
    value = env.get("LLAMA_ARG_CACHE_TYPE_V")
    if value:
        return value.lower()
    value = env.get("LLAMA_ARG_CACHE_TYPE_V_DRAFT")
    if value:
        return value.lower()
    return None


def resolve_policy(args: argparse.Namespace, fp: Mapping[str, object], cache: Mapping[str, object]) -> Tuple[str, str]:
    if args.policy != "auto":
        return args.policy, f"forced by --policy={args.policy}"

    if args.mode == "off":
        return "off", "mode=off"

    models = cache.get("models", {})
    key = str(fp["sample_sha256"])
    cached = models.get(key) if isinstance(models, dict) else None
    if isinstance(cached, dict):
        policy = cached.get("policy")
        if policy in POLICIES:
            return str(policy), f"cache hit {cache_path()}"

    builtin = builtin_policy_for_name(str(fp["name"]))
    if builtin is not None:
        return builtin

    if args.mode == "known":
        return "off", "unknown GGUF fingerprint/name; conservative baseline"

    raise PolicyError(f"unsupported mode {args.mode}")


def cleaned_env(base: Mapping[str, str], policy: Mapping[str, str]) -> Dict[str, str]:
    env = dict(base)
    for key in MANAGED_VARS:
        env.pop(key, None)
    env.update(policy)
    return env


def shell_lines(policy: Mapping[str, str], comment: str | None = None) -> str:
    lines: List[str] = []
    if comment:
        lines.append(f"# {comment}")
    for key in MANAGED_VARS:
        lines.append(f"unset {key}")
    for key, value in policy.items():
        lines.append(f"export {key}={shlex.quote(value)}")
    return "\n".join(lines) + "\n"


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply safe RoxxY MTP MMVQ interleaved-act env policy for a GGUF model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="GGUF model path used by the command")
    parser.add_argument(
        "--mode",
        choices=("known", "off"),
        default="known",
        help="known applies cached/builtin measured policies; unknown models stay baseline",
    )
    parser.add_argument(
        "--policy",
        choices=("auto",) + tuple(POLICIES.keys()),
        default="auto",
        help="force a specific policy instead of cache/builtin auto selection",
    )
    parser.add_argument(
        "--trust-policy",
        action="store_true",
        help="store the selected/forced policy in the local fingerprint cache",
    )
    parser.add_argument("--cache", type=Path, default=cache_path(), help="policy cache path")
    parser.add_argument("--json", action="store_true", help="print selected policy as JSON and exit")
    parser.add_argument("--shell", action="store_true", help="print shell exports/unsets and exit")
    parser.add_argument("--explain", action="store_true", help="print selected policy reason to stderr")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="optional command to run after --")
    args = parser.parse_args(list(argv))
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    return args


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(argv)
    model = Path(args.model).expanduser()
    if not model.exists():
        raise PolicyError(f"model does not exist: {model}")

    fp = model_fingerprint(model)
    cache = load_cache(args.cache)
    policy_name, reason = resolve_policy(args, fp, cache)

    if args.policy == "auto" and policy_name == "q4q6-27b-fast":
        v_cache_type = requested_v_cache_type(args.cmd, os.environ)
        if v_cache_type == "q8_0":
            policy_name = "q4q6-27b-q8v-fast"
            reason = f"{reason}; command/env requests q8_0 V"

    policy = dict(POLICIES[policy_name])

    if args.trust_policy:
        mutable_cache = dict(cache)
        models = dict(mutable_cache.get("models", {}))
        models[str(fp["sample_sha256"])] = {
            "policy": policy_name,
            "reason": reason,
            "model_name": fp["name"],
            "model_size": fp["size"],
        }
        mutable_cache["models"] = models
        save_cache(args.cache, mutable_cache)
        reason = f"{reason}; cached as {policy_name} in {args.cache}"

    if args.explain:
        print(f"mtp-mmvq policy={policy_name}: {reason}", file=sys.stderr)

    if args.json:
        print(json.dumps({"policy": policy_name, "reason": reason, "fingerprint": fp, "env": policy}, indent=2, sort_keys=True))
        return 0

    if args.shell or not args.cmd:
        print(shell_lines(policy, f"mtp-mmvq policy={policy_name}: {reason}"), end="")
        return 0

    env = cleaned_env(os.environ, policy)
    if args.explain:
        print("exec " + " ".join(shlex.quote(x) for x in args.cmd), file=sys.stderr)
    return subprocess.call(args.cmd, env=env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PolicyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
