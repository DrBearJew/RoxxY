#!/usr/bin/env python3
"""TBQ4 ROCm Coherence & Precision Gate Harness.

Runs the test layers defined in 07-coherence-precision-test.md:
  1. Deterministic smoke
  2. Next-token precision probes (q8_0 baseline vs TBQ4)
  3. Long-context needle retrieval
  4. Structured/tool-call canaries
  5. MTP token acceptance
  6. Cache and slot coherence

Usage:
  python3 harness.py [--quick] [--q8-ctx N] [--tbq4-ctx N] [--skip-needles] [--skip-mtp]
"""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

# ── configuration defaults ──────────────────────────────────────────────────

MODEL = os.environ.get(
    "HARNESS_MODEL",
    "/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf",
)
SERVER_BIN = os.environ.get(
    "HARNESS_SERVER_BIN",
    "/tmp/llama.cpp-mtp/build-rocm-tq/bin/llama-server",
)
CHAT_TEMPLATE = os.environ.get(
    "HARNESS_CHAT_TEMPLATE",
    "/home/mrtrent/.pi/agent/qwen36-merged-template.jinja",
)
Q8_CTX = 16384  # q8_0 64k OOMs; 16k fits
TBQ4_CTX = 16384  # full gate expects 65536; override for quick runs

# ── filler generation ───────────────────────────────────────────────────────

_FILLER = """Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do: once or twice she had peeped into the book her sister was reading,
but it had no pictures or conversations in it, "and what is the use of a book,"
thought Alice "without pictures or conversations?"
So she was considering in her own mind (as well as she could, for the hot day made her feel
very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble
of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.
There was nothing so very remarkable in that; nor did Alice think it so very much out of the way
to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over
afterwards, it occurred to her that she ought to have wondered at this, but at the time it all
seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket,
and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind
that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it,
and burning with curiosity, she ran across the field after it, and fortunately was just in time
to see it pop down a large rabbit-hole under the hedge.
In another moment down went Alice after it, never once considering how in the world she was to get out again.
The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down,
so suddenly that Alice had not a moment to think about stopping herself before she found herself
falling down a very deep well.
Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down
to look about her and to wonder what was going to happen next. First, she tried to look down and make out
what she was coming to, but it was too dark to see anything; then she looked at the sides of the well,
and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures
hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled
'ORANGE MARMALADE', but to her great disappointment it was empty: she did not like to drop the jar
for fear of killing somebody, so managed to put it into one of the cupboards as she fell past it.
"""


def _token_estimate(text: str) -> int:
    """Rough token count: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def make_filler(target_tokens: int) -> str:
    """Generate deterministic filler text of approximately target_tokens."""
    repeats = (target_tokens // _token_estimate(_FILLER)) + 2
    return (_FILLER * repeats)[: target_tokens * 5]  # generous slice


def make_needle_prompt(
    context_tokens: int, depth_pct: float, secret: str
) -> str:
    """Build a prompt with a secret needle at depth_pct through filler."""
    prefix_tokens = int(context_tokens * depth_pct / 100)
    suffix_tokens = context_tokens - prefix_tokens - _token_estimate(secret) - 20

    prefix = make_filler(prefix_tokens)
    suffix = make_filler(suffix_tokens)

    needle = f"\n\nIMPORTANT NEEDLE: The secret verification code is {secret}.\n\n"
    question = f"\n\nQuestion: What is the exact secret verification code? Answer with only the code.\n"

    return prefix + needle + suffix + question


# ── precision probes ────────────────────────────────────────────────────────

_PRECISION_PROBES: List[Tuple[str, str, str]] = [
    # (name, prompt, class)
    ("factual_capitals", "The capital of France is", "short factual/math/code"),
    ("math_2plus2", "2+2=", "short factual/math/code"),
    ("math_7x8", "7 * 8 =", "short factual/math/code"),
    ("code_python_func", "def fibonacci(n):\n    ", "short factual/math/code"),
    ("factual_water", "Water boils at", "short factual/math/code"),
    ("factual_earth", "The Earth revolves around the", "short factual/math/code"),
    ("code_html", "<html>\n<head>\n<title>", "short factual/math/code"),
    ("code_sql", "SELECT * FROM users WHERE", "short factual/math/code"),
    ("factual_largest_ocean", "The largest ocean on Earth is the", "short factual/math/code"),
    ("factual_light_speed", "The speed of light in vacuum is approximately", "short factual/math/code"),
]


def make_mid_context_probe(
    context_tokens: int, prompt: str
) -> Tuple[str, str]:
    """Wrap a probe in filler for mid-context testing."""
    filler = make_filler(context_tokens)
    return f"{filler}\n\n---\n\n{prompt}", prompt


def _extract_top_logprobs(resp: Dict) -> Tuple[str, int, List[Dict]]:
    """Extract chosen token text, chosen id, and top_logprobs list from response.

    llama-server /v1/completions with n_probs>0 returns:
      choices[0].logprobs.content[0] = {id, token, logprob, top_logprobs: [{id, token, logprob}, ...]}
    Returns (chosen_token, chosen_id, top_logprobs_list).
    """
    content = resp["choices"][0]["logprobs"]["content"][0]
    chosen_token = content["token"]
    chosen_id = content["id"]
    top_logprobs = content["top_logprobs"]
    return chosen_token, chosen_id, top_logprobs


def compute_probe_metrics(
    q8_resp: Dict, tbq4_resp: Dict
) -> Dict[str, Any]:
    """Compare two single-token probe responses."""
    q8_token, q8_top1_id, q8_top = _extract_top_logprobs(q8_resp)
    tbq4_token, tbq4_top1_id, tbq4_top = _extract_top_logprobs(tbq4_resp)
    top1_match = q8_top1_id == tbq4_top1_id

    # Build token→(id, probability) maps
    def _build_map(top_list):
        m = {}
        for p in top_list:
            m[p["token"]] = (p["id"], math.exp(p["logprob"]))
        return m

    q8_probs = _build_map(q8_top)
    tbq4_probs = _build_map(tbq4_top)

    # Top-10 Jaccard by token string
    q8_top10 = set(p["token"] for p in q8_top[:10])
    tbq4_top10 = set(p["token"] for p in tbq4_top[:10])
    intersection = q8_top10 & tbq4_top10
    union = q8_top10 | tbq4_top10
    jaccard = len(intersection) / len(union) if union else 0.0

    # Rank of q8 top-1 in TBQ4 top-64
    q8_top1_token = q8_token
    tbq4_ranked = [p["token"] for p in tbq4_top]
    try:
        q8_top1_rank_in_tbq4 = tbq4_ranked.index(q8_top1_token) + 1
    except ValueError:
        q8_top1_rank_in_tbq4 = 99

    # Shared logprob MAE (over shared tokens)
    shared_mae = None
    shared_tokens = set(q8_probs.keys()) & set(tbq4_probs.keys())
    if shared_tokens:
        deltas = []
        for tok in shared_tokens:
            _, q8_prob = q8_probs[tok]
            _, tbq4_prob = tbq4_probs[tok]
            q8_lp = math.log(max(q8_prob, 1e-12))
            tbq4_lp = math.log(max(tbq4_prob, 1e-12))
            deltas.append(abs(q8_lp - tbq4_lp))
        if deltas:
            shared_mae = sum(deltas) / len(deltas)

    # Approximate JSD over union of top-64
    approx_jsd = None
    all_tokens = set(q8_probs.keys()) | set(tbq4_probs.keys())
    if all_tokens:
        p_dist = {}
        q_dist = {}
        for tok in all_tokens:
            _, p = q8_probs.get(tok, (0, 1e-12))
            _, q = tbq4_probs.get(tok, (0, 1e-12))
            p_dist[tok] = p
            q_dist[tok] = q
        p_sum = sum(p_dist.values()) or 1.0
        q_sum = sum(q_dist.values()) or 1.0
        m_dist = {}
        for tok in all_tokens:
            m_dist[tok] = 0.5 * (p_dist[tok] / p_sum + q_dist[tok] / q_sum)
        kl_pm = 0.0
        kl_qm = 0.0
        for tok in all_tokens:
            p_val = p_dist[tok] / p_sum
            q_val = q_dist[tok] / q_sum
            m_val = m_dist[tok]
            if p_val > 0 and m_val > 0:
                kl_pm += p_val * math.log(p_val / m_val)
            if q_val > 0 and m_val > 0:
                kl_qm += q_val * math.log(q_val / m_val)
        approx_jsd = 0.5 * (kl_pm + kl_qm)

    return {
        "q8_token": q8_token,
        "tbq4_token": tbq4_token,
        "top1_match": top1_match,
        "q8_top1_rank_in_tbq4": q8_top1_rank_in_tbq4,
        "top10_jaccard": jaccard,
        "shared_logprob_mae": shared_mae,
        "approx_jsd": approx_jsd,
    }


# ── server management ───────────────────────────────────────────────────────


class Server:
    """Manage a llama-server process."""

    def __init__(
        self,
        port: int,
        ctx: int,
        cache_k: str,
        cache_v: str,
        spec_type: Optional[str] = None,
        parallel: int = 1,
    ):
        self.port = port
        self.ctx = ctx
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.spec_type = spec_type
        self.parallel = parallel
        self.process: Optional[subprocess.Popen] = None
        self.log_path = f"/tmp/harness-server-{port}.log"

    def start(self, timeout: int = 300) -> None:
        """Launch server and wait for ready."""
        cmd = [
            SERVER_BIN,
            "-m", MODEL,
            "--cache-type-k", self.cache_k,
            "--cache-type-v", self.cache_v,
            "-c", str(self.ctx),
            "--port", str(self.port),
            "--no-webui",
            "--no-warmup",
            "--parallel", str(self.parallel),
        ]
        # Qwen 3.6 merged chat template: enables <|think_off|> thinking suppression
        if os.path.exists(CHAT_TEMPLATE):
            cmd.extend(["--jinja", "--chat-template-file", CHAT_TEMPLATE])
        if self.spec_type:
            cmd.extend(["--spec-type", self.spec_type])
            # PR #22673: optimal MTP draft depth is 3; default 16 severely degrades acceptance
            cmd.extend(["--spec-draft-n-max", "3"])

        logf = open(self.log_path, "w")
        self.process = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                self._die(f"server exited early (code {self.process.returncode})")
            # Check log for "server is listening"
            try:
                with open(self.log_path) as f:
                    if "server is listening" in f.read():
                        # Verify /health
                        try:
                            urllib.request.urlopen(
                                f"http://localhost:{self.port}/health",
                                timeout=5,
                            ).read()
                            return
                        except Exception:
                            pass
            except Exception:
                pass
            time.sleep(1)
        self._die("server start timed out")

    def stop(self) -> None:
        """Kill the server process group."""
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.process = None

    def _die(self, msg: str) -> None:
        """Report failure and raise."""
        tail = ""
        try:
            with open(self.log_path) as f:
                lines = f.readlines()
                tail = "".join(lines[-40:])
        except Exception:
            pass
        self.stop()
        raise RuntimeError(f"{msg}\n--- log tail ---\n{tail}")

    def get_vram(self) -> int:
        """Extract VRAM used from server log (if logged)."""
        try:
            with open(self.log_path) as f:
                for line in f:
                    if "total VRAM used" in line:
                        m = re.search(r"(\d+)", line)
                        if m:
                            return int(m.group(1))
        except Exception:
            pass
        return 0


def fetch_health(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
        return True
    except Exception:
        return False


def fetch_chat(
    port: int,
    messages: List[Dict[str, str]],
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Call /v1/chat/completions with the merged template."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def fetch_completion(
    port: int,
    prompt: str,
    n_predict: int = 64,
    temperature: float = 0.0,
    cache_prompt: bool = False,
    n_probs: int = 0,
) -> Dict[str, Any]:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "cache_prompt": cache_prompt,
    }
    if n_probs:
        payload["n_probs"] = n_probs
        payload["post_sampling_probs"] = False

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


# ── test layers ─────────────────────────────────────────────────────────────


class GateResult:
    def __init__(self):
        self.critical_failures: List[str] = []
        self.warnings: List[str] = []
        self.passes: List[str] = []
        self.sections: Dict[str, Any] = OrderedDict()


class GateRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.q8_port = 16100
        self.tbq4_port = 16101
        self.result = GateResult()

    def run(self) -> GateResult:
        """Run all enabled test layers and return the result."""
        print("=" * 60)
        print("TBQ4 ROCm Coherence & Precision Gate")
        print(f"Model: {MODEL}")
        print(f"Binary: {SERVER_BIN}")
        print(f"q8_0 context: {self.args.q8_ctx}")
        print(f"TBQ4 context: {self.args.tbq4_ctx}")
        print(f"Quick mode: {self.args.quick}")
        print("=" * 60)

        # Layer 1: TBQ4 smoke
        self._run_smoke()

        if not self.args.skip_precision:
            self._run_precision_probes()

        if not self.args.skip_needles:
            self._run_needles()

        self._run_canaries()

        if not self.args.skip_mtp:
            self._run_mtp_acceptance()

        self._run_cache_coherence()

        self._write_summary()
        return self.result

    # ── layer 1: smoke ────────────────────────────────────────────────────

    def _run_smoke(self) -> None:
        print("\n── Layer 1: Deterministic Smoke ──")
        section: Dict[str, Any] = {"tests": [], "critical_failures": []}

        srv = Server(self.tbq4_port, self.args.tbq4_ctx, "tbq4_0", "tbq4_0")
        try:
            srv.start()
            section["vram_used"] = srv.get_vram()

            # Test 1: France → Paris
            for attempt in range(3):
                resp = fetch_completion(self.tbq4_port, "The capital of France is", 4)
                text = resp["choices"][0]["text"].strip()
                ok = "Paris" in text
                section["tests"].append(
                    {
                        "name": f"france_paris_attempt_{attempt+1}",
                        "passed": ok,
                        "output": text[:80],
                    }
                )
                if not ok:
                    section["critical_failures"].append(
                        f"France→Paris attempt {attempt+1}: got '{text[:60]}'"
                    )

            # Test 2: 2+2= → 4
            for attempt in range(3):
                resp = fetch_completion(self.tbq4_port, "2+2=", 4)
                text = resp["choices"][0]["text"].strip()
                ok = "4" in text[:10]
                section["tests"].append(
                    {
                        "name": f"math_2plus2_attempt_{attempt+1}",
                        "passed": ok,
                        "output": text[:80],
                    }
                )
                if not ok:
                    section["critical_failures"].append(
                        f"2+2→4 attempt {attempt+1}: got '{text[:60]}'"
                    )

            # Test 3: Repeat France 3x — identical output
            france_outputs = []
            for attempt in range(3):
                resp = fetch_completion(
                    self.tbq4_port,
                    "The capital of France is",
                    16,
                )
                text = resp["choices"][0]["text"]
                france_outputs.append(text)

            all_same = (
                france_outputs[0] == france_outputs[1] == france_outputs[2]
            )
            section["tests"].append(
                {
                    "name": "deterministic_repeat_3x",
                    "passed": all_same,
                    "outputs": [o[:80] for o in france_outputs],
                }
            )
            if not all_same:
                section["critical_failures"].append(
                    "Output not deterministic: repeated France prompt gave different results"
                )

            # Smoke summary
            all_ok = all(t["passed"] for t in section["tests"])
            if all_ok:
                self.result.passes.append("smoke: all deterministic tests passed")
                print("  ✅ Smoke tests passed")
            else:
                for f in section["critical_failures"]:
                    self.result.critical_failures.append(f"smoke: {f}")
                print(f"  ❌ Smoke failures: {section['critical_failures']}")

        finally:
            srv.stop()

        self.result.sections["smoke"] = section

    # ── layer 2: precision probes ─────────────────────────────────────────

    def _run_precision_probes(self) -> None:
        print("\n── Layer 2: Next-Token Precision Probes ──")
        section: Dict[str, Any] = {
            "probes": [],
            "class_summary": {},
            "critical_failures": [],
        }

        # Start q8_0 baseline
        q8 = Server(self.q8_port, self.args.q8_ctx, "q8_0", "q8_0")
        tbq4 = Server(self.tbq4_port, self.args.tbq4_ctx, "tbq4_0", "tbq4_0")

        try:
            print("  Starting q8_0 baseline server...")
            q8.start()

            # Run short precision probes against q8_0 first
            probes_to_run = list(_PRECISION_PROBES)

            # Add mid-context probes if not quick mode
            if not self.args.quick:
                ctxs = [2000, 8000]
                for ctx in ctxs:
                    if ctx <= self.args.q8_ctx:
                        for _, prompt, _ in _PRECISION_PROBES[:5]:
                            filler_prompt, _ = make_mid_context_probe(ctx, prompt)
                            probes_to_run.append(
                                (
                                    f"{prompt[:30].strip()}_ctx{ctx}",
                                    filler_prompt,
                                    f"medium {ctx//1000}k context",
                                )
                            )

            # Collect q8_0 responses
            q8_results: Dict[str, Dict] = {}
            print(f"  Running {len(probes_to_run)} probes against q8_0...")
            for name, prompt, pclass in probes_to_run:
                try:
                    resp = fetch_completion(
                        self.q8_port, prompt, n_predict=1, n_probs=64
                    )
                    q8_results[name] = resp
                except Exception as e:
                    section["critical_failures"].append(
                        f"q8_0 probe '{name}' failed: {e}"
                    )
                    print(f"    ❌ q8_0 probe '{name}' error: {e}")

            q8.stop()
            print("  q8_0 server stopped.")

            # Start TBQ4
            print("  Starting TBQ4 server...")
            tbq4.start()

            # Run same probes against TBQ4
            print(f"  Running probes against TBQ4...")
            tbq4_results: Dict[str, Dict] = {}
            for name, prompt, pclass in probes_to_run:
                if name not in q8_results:
                    continue
                try:
                    resp = fetch_completion(
                        self.tbq4_port,
                        prompt,
                        n_predict=1,
                        n_probs=64,
                    )
                    tbq4_results[name] = resp
                except Exception as e:
                    section["critical_failures"].append(
                        f"TBQ4 probe '{name}' failed: {e}"
                    )
                    print(f"    ❌ TBQ4 probe '{name}' error: {e}")

            # Compute metrics
            class_metrics: Dict[str, Dict[str, List[float]]] = {}
            for name, prompt, pclass in probes_to_run:
                if name not in q8_results or name not in tbq4_results:
                    continue

                metrics = compute_probe_metrics(
                    q8_results[name], tbq4_results[name]
                )
                probe_entry = {
                    "name": name,
                    "class": pclass,
                    **metrics,
                }
                section["probes"].append(probe_entry)

                # Accumulate class metrics
                if pclass not in class_metrics:
                    class_metrics[pclass] = {
                        "top1_match": [],
                        "top10_jaccard": [],
                        "shared_logprob_mae": [],
                        "approx_jsd": [],
                    }
                if metrics["top1_match"] is not None:
                    class_metrics[pclass]["top1_match"].append(
                        1.0 if metrics["top1_match"] else 0.0
                    )
                if metrics["top10_jaccard"] is not None:
                    class_metrics[pclass]["top10_jaccard"].append(
                        metrics["top10_jaccard"]
                    )
                if metrics["shared_logprob_mae"] is not None:
                    class_metrics[pclass]["shared_logprob_mae"].append(
                        metrics["shared_logprob_mae"]
                    )
                if metrics["approx_jsd"] is not None:
                    class_metrics[pclass]["approx_jsd"].append(
                        metrics["approx_jsd"]
                    )

            # Summarize by class
            for pclass, m in class_metrics.items():
                summary = {}
                if m["top1_match"]:
                    summary["top1_match_pct"] = round(
                        100.0 * sum(m["top1_match"]) / len(m["top1_match"]), 1
                    )
                if m["top10_jaccard"]:
                    summary["top10_jaccard_avg"] = round(
                        sum(m["top10_jaccard"]) / len(m["top10_jaccard"]), 3
                    )
                if m["shared_logprob_mae"]:
                    summary["shared_logprob_mae_avg"] = round(
                        sum(m["shared_logprob_mae"])
                        / len(m["shared_logprob_mae"]),
                        4,
                    )
                if m["approx_jsd"]:
                    summary["approx_jsd_avg"] = round(
                        sum(m["approx_jsd"]) / len(m["approx_jsd"]), 4
                    )
                section["class_summary"][pclass] = summary
                print(
                    f"  {pclass}: top1_match={summary.get('top1_match_pct','?')}%, "
                    f"jaccard={summary.get('top10_jaccard_avg','?')}, "
                    f"mae={summary.get('shared_logprob_mae_avg','?')}, "
                    f"jsd={summary.get('approx_jsd_avg','?')}"
                )

            # Check against acceptance criteria
            thresholds = {
                "short factual/math/code": {
                    "top1_match": 90,
                    "top10_jaccard": 0.70,
                    "shared_logprob_mae": 0.20,
                    "approx_jsd": 0.05,
                },
                "medium 2k-8k context": {
                    "top1_match": 85,
                    "top10_jaccard": 0.60,
                    "shared_logprob_mae": 0.30,
                    "approx_jsd": 0.08,
                },
            }

            for pclass, summary in section["class_summary"].items():
                thresh = thresholds.get(pclass)
                if not thresh:
                    continue
                if summary.get("top1_match_pct", 100) < thresh["top1_match"]:
                    self.result.critical_failures.append(
                        f"precision {pclass}: top1_match {summary['top1_match_pct']}% < {thresh['top1_match']}%"
                    )
                if summary.get("top10_jaccard_avg", 1.0) < thresh["top10_jaccard"]:
                    self.result.critical_failures.append(
                        f"precision {pclass}: jaccard {summary['top10_jaccard_avg']} < {thresh['top10_jaccard']}"
                    )

            prec_crit = len(section["critical_failures"]) == 0
            if prec_crit:
                self.result.passes.append("precision: all probe classes passed thresholds")
                print("  ✅ Precision probes passed")
            else:
                for f in section["critical_failures"]:
                    self.result.critical_failures.append(f)
                print(f"  ⚠️ Precision probe failures: {section['critical_failures']}")

        finally:
            q8.stop()
            tbq4.stop()

        self.result.sections["precision_probes"] = section

    # ── layer 3: needle retrieval ─────────────────────────────────────────

    def _run_needles(self) -> None:
        print("\n── Layer 3: Long-Context Needle Retrieval ──")
        section: Dict[str, Any] = {"tests": [], "critical_failures": []}

        # Only run needles up to TBQ4 context
        needle_contexts = [8192]
        if not self.args.quick and self.args.tbq4_ctx >= 16384:
            needle_contexts.append(16384)
        if not self.args.quick and self.args.tbq4_ctx >= 32768:
            needle_contexts.append(32768)

        depths = [10, 50, 90]

        srv = Server(self.tbq4_port, self.args.tbq4_ctx, "tbq4_0", "tbq4_0")
        try:
            srv.start()

            for ctx_tokens in needle_contexts:
                for depth in depths:
                    secret = f"TBQ4-RDNA3-{hashlib.md5(f'{ctx_tokens}-{depth}'.encode()).hexdigest()[:8]}"
                    prompt = make_needle_prompt(ctx_tokens, depth, secret)
                    try:
                        resp = fetch_completion(
                            self.tbq4_port,
                            prompt,
                            n_predict=32,
                        )
                        text = resp["choices"][0]["text"]
                        passed = secret in text
                        entry = {
                            "context_tokens": ctx_tokens,
                            "depth_pct": depth,
                            "secret": secret,
                            "output": text[:200],
                            "passed": passed,
                        }
                        section["tests"].append(entry)
                        icon = "✅" if passed else "❌"
                        print(
                            f"  {icon} Needle {ctx_tokens//1024}k depth={depth}%: "
                            f"{'found' if passed else 'MISSING'}"
                        )
                        if not passed and ctx_tokens <= 16384:
                            section["critical_failures"].append(
                                f"Needle {ctx_tokens//1024}k depth={depth}% not retrieved"
                            )
                    except Exception as e:
                        section["tests"].append(
                            {
                                "context_tokens": ctx_tokens,
                                "depth_pct": depth,
                                "error": str(e),
                                "passed": False,
                            }
                        )
                        section["critical_failures"].append(
                            f"Needle {ctx_tokens//1024}k depth={depth}% error: {e}"
                        )

            all_needle_ok = len(section["critical_failures"]) == 0
            if all_needle_ok:
                self.result.passes.append("needles: all retrieved")
                print("  ✅ Needle tests passed")
            else:
                for f in section["critical_failures"]:
                    self.result.critical_failures.append(f"needle: {f}")

        finally:
            srv.stop()

        self.result.sections["needles"] = section

    # ── layer 4: structured canaries ──────────────────────────────────────

    def _run_canaries(self) -> None:
        print("\n── Layer 4: Structured Canaries ──")
        section: Dict[str, Any] = {"tests": [], "critical_failures": []}

        srv = Server(self.tbq4_port, self.args.tbq4_ctx, "tbq4_0", "tbq4_0")
        try:
            srv.start()

            # 1. JSON only canary — <|think_off|> disables thinking via merged template
            resp = fetch_chat(
                self.tbq4_port,
                [
                    {"role": "system", "content": "<|think_off|>You are a JSON-only assistant. Output valid JSON, nothing else. No markdown, no prose."},
                    {"role": "user", "content": 'Return exactly valid minified JSON with keys "answer" and "check". answer must be 4 and check must be "ok".'},
                ],
                128,
            )
            text = resp["choices"][0]["message"]["content"]
            json_ok = False
            try:
                m = re.search(r'\{[^}]+\}', text)
                if m:
                    parsed = json.loads(m.group())
                    if parsed.get("answer") == 4 and parsed.get("check") == "ok":
                        json_ok = True
            except Exception:
                pass
            section["tests"].append({"name": "json_canary", "passed": json_ok, "output": text[:300]})
            if not json_ok:
                section["critical_failures"].append(f"JSON canary failed: got '{text[:120]}'")

            # 2. ChatML leak guard
            resp2 = fetch_chat(
                self.tbq4_port,
                [
                    {"role": "system", "content": "<|think_off|>You are a concise assistant. Output exactly what the user asks, with no extra text, no markdown, no formatting."},
                    {"role": "user", "content": "Say exactly: SAFE_OUTPUT"},
                ],
                32,
            )
            text2 = resp2["choices"][0]["message"]["content"]
            safe_ok = "SAFE_OUTPUT" in text2
            leak = any(tok in text2 for tok in ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<｜end▁of▁thinking｜>"])
            section["tests"].append({"name": "chatml_leak_guard", "passed": safe_ok and not leak, "output": text2[:300]})
            if not safe_ok:
                section["critical_failures"].append(f"ChatML leak guard: SAFE_OUTPUT not found in '{text2[:120]}'")
            if leak:
                self.result.warnings.append(f"ChatML leak guard: possible token leak in '{text2[:120]}'")

            # 3. Tool-call schema
            resp3 = fetch_chat(
                self.tbq4_port,
                [
                    {"role": "system", "content": "<|think_off|>You are a tool-calling assistant. Output ONLY valid JSON objects, nothing else."},
                    {"role": "user", "content": 'Return exactly one JSON object: {"tool":"read_file","args":{"path":"/tmp/example.txt"}}'},
                ],
                128,
            )
            text3 = resp3["choices"][0]["message"]["content"]
            tool_ok = False
            try:
                # Find JSON objects with balanced braces (supports nesting)
                for start_m in re.finditer(r'\{', text3):
                    pos = start_m.start()
                    depth = 0
                    end = -1
                    for i in range(pos, len(text3)):
                        if text3[i] == '{':
                            depth += 1
                        elif text3[i] == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    if end > pos:
                        try:
                            parsed = json.loads(text3[pos:end])
                            if parsed.get("tool") == "read_file" and "args" in parsed and parsed["args"].get("path") == "/tmp/example.txt":
                                tool_ok = True
                                break
                        except Exception:
                            continue
            except Exception:
                pass
            section["tests"].append({"name": "tool_call_schema", "passed": tool_ok, "output": text3[:300]})
            if not tool_ok:
                section["critical_failures"].append(f"Tool-call schema canary failed: got '{text3[:120]}'")

            # 4. Code syntax — extract from output even if thinking tags leak
            resp4 = fetch_chat(
                self.tbq4_port,
                [
                    {"role": "system", "content": "<|think_off|>You are a code-only assistant. Output ONLY Python code, nothing else. No explanations."},
                    {"role": "user", "content": "Write a Python function add(a, b) that returns a+b."},
                ],
                128,
            )
            text4 = resp4["choices"][0]["message"]["content"]
            code_ok = False
            # Try full output; then after </think>; then code-fenced blocks
            candidates = [text4]
            m_think = re.search(r'</think>\s*(.*)', text4, re.DOTALL)
            if m_think:
                candidates.insert(0, m_think.group(1))
            m_fence = re.search(r"```(?:python)?\s*\n(.*?)```", text4, re.DOTALL)
            if m_fence:
                candidates.insert(0, m_fence.group(1))
            for candidate in candidates:
                try:
                    tree = ast.parse(candidate.strip())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name == "add":
                            for child in ast.walk(node):
                                if isinstance(child, ast.Return):
                                    code_ok = True
                                    break
                    if code_ok:
                        break
                except (SyntaxError, IndentationError):
                    continue
            section["tests"].append({"name": "code_syntax_add", "passed": code_ok, "output": text4[:300]})
            if not code_ok:
                section["critical_failures"].append(f"Code syntax canary failed: cannot parse add() from '{text4[:120]}'")

            canaries_ok = len(section["critical_failures"]) == 0
            if canaries_ok:
                self.result.passes.append("canaries: all structured tests passed")
                print("  ✅ Canaries passed")
            else:
                for f in section["critical_failures"]:
                    self.result.critical_failures.append(f"canary: {f}")
                print(f"  ❌ Canary failures: {section['critical_failures']}")

        finally:
            srv.stop()

        self.result.sections["canaries"] = section

    # ── layer 5: MTP acceptance ───────────────────────────────────────────

    def _run_mtp_acceptance(self) -> None:
        print("\n── Layer 5: MTP Token Acceptance ──")
        section: Dict[str, Any] = {"tests": [], "critical_failures": []}

        prompt_short = (
            "Write a concise paragraph about AMD GPUs and open-source inference."
        )
        ctx = min(self.args.tbq4_ctx, 16384)  # practical MTP context

        # Run q8_0 MTP baseline first (if ctx fits)
        q8_accept = None
        if ctx <= self.args.q8_ctx:
            print("  Starting q8_0 MTP baseline server...")
            srv_q8 = Server(
                self.q8_port, ctx, "q8_0", "q8_0", spec_type="mtp", parallel=1
            )
            try:
                srv_q8.start()
                resp_q8 = fetch_completion(self.q8_port, prompt_short, 64)
                t_q8 = resp_q8.get("timings", {})
                d_n = t_q8.get("draft_n", 0)
                d_a = t_q8.get("draft_n_accepted", 0)
                q8_accept = 100.0 * d_a / d_n if d_n else None
                section["tests"].append({
                    "name": "mtp_short_q8_0",
                    "cache": "q8_0",
                    "draft_n": d_n,
                    "draft_n_accepted": d_a,
                    "accept_pct": q8_accept,
                    "gen_tok_s": t_q8.get("predicted_per_second"),
                })
                print(f"  q8_0 MTP: draft_n={d_n}, accepted={d_a}, accept={q8_accept:.1f}%")
            except Exception as e:
                print(f"  ⚠️ q8_0 MTP error: {e}")
            finally:
                srv_q8.stop()

        # Run TBQ4 MTP
        print("  Starting TBQ4 MTP server...")
        srv = Server(
            self.tbq4_port, ctx, "tbq4_0", "tbq4_0", spec_type="mtp", parallel=1
        )
        try:
            srv.start()

            resp = fetch_completion(self.tbq4_port, prompt_short, 64)
            t = resp.get("timings", {})
            draft_n = t.get("draft_n", 0)
            draft_accepted = t.get("draft_n_accepted", 0)
            accept_pct = 100.0 * draft_accepted / draft_n if draft_n else None

            entry = {
                "name": "mtp_short_tbq4",
                "cache": "tbq4_0",
                "draft_n": draft_n,
                "draft_n_accepted": draft_accepted,
                "accept_pct": accept_pct,
                "gen_tok_s": t.get("predicted_per_second"),
                "text": resp["choices"][0]["text"][:120],
            }

            # Compare q8_0 vs TBQ4 acceptance
            if q8_accept is not None and accept_pct is not None:
                delta = accept_pct - q8_accept
                entry["accept_delta_vs_q8"] = delta
                if abs(delta) > 15:
                    section["critical_failures"].append(
                        f"MTP acceptance delta {delta:.1f}pp > 15pp threshold (q8={q8_accept:.1f}%, tbq4={accept_pct:.1f}%)"
                    )

            if draft_n == 0:
                entry["critical"] = "MTP not active: draft_n=0"
                section["critical_failures"].append("MTP not active")
            elif accept_pct is not None and accept_pct < 8:
                section["critical_failures"].append(
                    f"MTP acceptance {accept_pct:.1f}% below sanity floor 8%"
                )

            section["tests"].append(entry)
            print(
                f"  TBQ4 MTP: draft_n={draft_n}, accepted={draft_accepted}, "
                f"accept={accept_pct:.1f}%"
                + (f" (delta vs q8_0={delta:+.1f}pp)" if q8_accept is not None else "")
            )

            # Medium context continuation (2k)
            if not self.args.quick:
                filler = make_filler(2000)
                prompt_med = f"{filler}\n\nContinue the story in the same style:\n"
                try:
                    resp2 = fetch_completion(self.tbq4_port, prompt_med, 64)
                    t2 = resp2.get("timings", {})
                    d_n = t2.get("draft_n", 0)
                    d_a = t2.get("draft_n_accepted", 0)
                    a_pct = 100.0 * d_a / d_n if d_n else None
                    section["tests"].append({
                        "name": "mtp_medium_2k",
                        "draft_n": d_n,
                        "draft_n_accepted": d_a,
                        "accept_pct": a_pct,
                    })
                    print(f"  MTP 2k ctx: draft_n={d_n}, accepted={d_a}, accept={a_pct:.1f}%")
                except Exception as e:
                    print(f"  ⚠️ MTP medium context error: {e}")

            mtp_ok = len(section["critical_failures"]) == 0
            if mtp_ok:
                self.result.passes.append("mtp: acceptance measured, above sanity floor")
                print("  ✅ MTP acceptance tests passed")
            else:
                for f in section["critical_failures"]:
                    self.result.critical_failures.append(f"mtp: {f}")

        finally:
            srv.stop()

        self.result.sections["mtp_acceptance"] = section

    # ── layer 6: cache coherence ──────────────────────────────────────────

    def _run_cache_coherence(self) -> None:
        print("\n── Layer 6: Cache Coherence ──")
        section: Dict[str, Any] = {"tests": [], "critical_failures": []}

        srv = Server(self.tbq4_port, self.args.tbq4_ctx, "tbq4_0", "tbq4_0")
        try:
            srv.start()

            prompt = "The capital of France is"
            # Without cache
            resp1 = fetch_completion(
                self.tbq4_port, prompt, 16, cache_prompt=False
            )
            text1 = resp1["choices"][0]["text"]

            # With cache (first call caches, second reuses)
            resp2 = fetch_completion(
                self.tbq4_port, prompt, 16, cache_prompt=True
            )
            text2 = resp2["choices"][0]["text"]

            cache_match = text1 == text2
            section["tests"].append(
                {
                    "name": "cache_vs_nocache_match",
                    "passed": cache_match,
                    "no_cache": text1[:80],
                    "cached": text2[:80],
                }
            )

            if not cache_match:
                section["critical_failures"].append(
                    f"Cache coherence: no-cache='{text1[:60]}' vs cached='{text2[:60]}'"
                )

            cache_ok = len(section["critical_failures"]) == 0
            if cache_ok:
                self.result.passes.append("cache: output matches with/without cache_prompt")
                print("  ✅ Cache coherence passed")
            else:
                for f in section["critical_failures"]:
                    self.result.critical_failures.append(f"cache: {f}")
                print(f"  ❌ Cache failures: {section['critical_failures']}")

        finally:
            srv.stop()

        self.result.sections["cache_coherence"] = section

    # ── write summary ─────────────────────────────────────────────────────

    def _write_summary(self) -> None:
        summary = OrderedDict(
            [
                ("gate", "TBQ4 ROCm Coherence & Precision"),
                ("model", MODEL),
                ("binary", SERVER_BIN),
                ("q8_ctx", self.args.q8_ctx),
                ("tbq4_ctx", self.args.tbq4_ctx),
                ("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                (
                    "overall",
                    "PASS" if not self.result.critical_failures else "FAIL",
                ),
                ("passes", self.result.passes),
                ("warnings", self.result.warnings),
                ("critical_failures", self.result.critical_failures),
                ("sections", self.result.sections),
            ]
        )

        path = os.path.join(
            os.path.dirname(__file__), "gate-summary.json"
        )
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(
            f"GATE RESULT: {'✅ PASS' if not self.result.critical_failures else '❌ FAIL'}"
        )
        print(f"Summary written to: {path}")
        print(f"Passes: {len(self.result.passes)}")
        print(f"Warnings: {len(self.result.warnings)}")
        print(f"Critical failures: {len(self.result.critical_failures)}")
        for f in self.result.critical_failures:
            print(f"  ❌ {f}")
        print(f"{'='*60}")


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TBQ4 ROCm Coherence & Precision Gate Harness"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick subset (skip mid-context+needles, small ctx)",
    )
    parser.add_argument(
        "--q8-ctx", type=int, default=Q8_CTX, help="q8_0 context length"
    )
    parser.add_argument(
        "--tbq4-ctx", type=int, default=TBQ4_CTX, help="TBQ4 context length"
    )
    parser.add_argument(
        "--skip-precision",
        action="store_true",
        help="Skip precision probe baseline comparison",
    )
    parser.add_argument(
        "--skip-needles", action="store_true", help="Skip needle retrieval tests"
    )
    parser.add_argument(
        "--skip-mtp", action="store_true", help="Skip MTP acceptance tests"
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.q8_ctx = min(args.q8_ctx, 4096)
        args.tbq4_ctx = min(args.tbq4_ctx, 4096)
        args.skip_needles = True

    runner = GateRunner(args)
    result = runner.run()

    if result.critical_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
