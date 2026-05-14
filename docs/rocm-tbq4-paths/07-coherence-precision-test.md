# TBQ4 ROCm Coherence + Precision Gate

Purpose: prove TBQ4 VEC FA is not only fast, but numerically stable and behaviorally coherent at long context with MTP.

## Reference / candidate

- Reference: same model/build with `--cache-type-k q8_0 --cache-type-v q8_0`.
- Candidate: same model/build with `--cache-type-k tbq4_0 --cache-type-v tbq4_0`.
- Use same context length, seed, sampler, prompt, and `temperature=0.0`.
- Prefer 64k for final acceptance. If q8_0 64k OOMs, use q8_0 32k for precision and TBQ4 64k for long-context coherence.

## Test layers

### 1. Startup and deterministic smoke

Run candidate at 64k:

- model loads without OOM/crash;
- VRAM headroom recorded;
- `The capital of France is` returns Paris;
- `2+2=` returns 4;
- same request repeated 3 times gives identical token text and token ids.

Critical failure:

- garbage repetition (`33333`, repeated punctuation, repeated same token > 8 times);
- ChatML/special-token leakage unless explicitly requested;
- NaN/Inf probability values;
- server crash/OOM.

### 2. Next-token precision probe

For each probe prompt, request **one token only** with top probabilities:

```json
{
  "prompt": "...",
  "n_predict": 1,
  "temperature": 0.0,
  "n_probs": 64,
  "post_sampling_probs": false,
  "cache_prompt": false
}
```

Collect from both q8_0 and TBQ4:

- generated token id/text;
- generated logprob;
- top-64 token ids/logprobs.

Metrics per probe:

- `top1_match`: same generated token id;
- `baseline_top1_rank_in_tbq4`: rank of q8_0 top token in TBQ4 top-64, or `>64`;
- `tbq4_top1_rank_in_baseline`: reverse rank;
- `top10_jaccard`: overlap of top-10 token ids;
- `shared_logprob_mae`: mean absolute logprob delta over shared top-64 tokens;
- `approx_jsd_top64`: Jensen-Shannon divergence over the union of reported top-64 probabilities, renormalized over the union.

Suggested acceptance:

| Probe class | top1 match | top10 Jaccard | shared logprob MAE | approx JSD |
|---|---:|---:|---:|---:|
| short factual/math/code | >= 90% | >= 0.70 | <= 0.20 nats | <= 0.05 |
| medium 2k-8k context | >= 85% | >= 0.60 | <= 0.30 nats | <= 0.08 |
| long 16k-56k context | >= 70% | >= 0.45 | <= 0.50 nats | <= 0.15 |

Critical failure:

- q8_0 top-1 missing from TBQ4 top-64 on more than 20% of probes;
- any probe has nonsensical top distribution, NaN, Inf, or repeated degenerate token as top-1;
- precision metrics sharply worsen after the `nthreads_KQ` patch versus the correctness-fixed baseline.

### 3. Long-context needle retrieval

Generate deterministic filler and insert a unique secret at several depths.

Contexts:

- 8k tokens;
- 16k tokens;
- 32k tokens;
- 56k-60k tokens under 64k context.

Needle depths:

- 10%;
- 50%;
- 90%.

Prompt shape:

```text
<large filler>
IMPORTANT NEEDLE: The secret verification code is TBQ4-RDNA3-<uuid>.
<large filler>

Question: What is the exact secret verification code? Answer with only the code.
```

Acceptance:

- exact secret appears in answer for all 8k/16k/32k tests;
- exact secret appears in at least 2/3 56k-60k depth tests;
- no special-token leakage or looping;
- compare q8_0 and TBQ4 if q8_0 fits, otherwise TBQ4-only long-context coherence is acceptable.

### 4. Structured/tool-call coherence canaries

Run deterministic prompts where corruption is obvious:

1. JSON only:

```text
Return exactly valid minified JSON with keys answer and check. answer must be 4 and check must be "ok". No prose.
```

Pass: parses as JSON and fields match.

2. ChatML leak guard:

```text
Say exactly: SAFE_OUTPUT
Do not output XML, markdown, special tokens, or thinking tags.
```

Pass: contains `SAFE_OUTPUT`; fails on `<|im_start|>`, `<think>`, repeated tokens, or unrelated text.

3. Tool-call-like schema:

```text
Return exactly one JSON object: {"tool":"read_file","args":{"path":"/tmp/example.txt"}}
```

Pass: parses and matches schema.

4. Code syntax:

```text
Write a Python function add(a, b) that returns a+b. Output code only.
```

Pass: code parses with `ast.parse` and function returns expected values.

### 5. MTP token acceptance percentage

Run a generation workload with MTP explicitly enabled at server startup:

```bash
llama-server ... --spec-type mtp --cache-type-k tbq4_0 --cache-type-v tbq4_0 -c 65536
```

For each request, require the server timing payload to contain:

```json
"timings": {
  "draft_n": 123,
  "draft_n_accepted": 98
}
```

Compute:

```text
mtp_accept_pct = 100 * draft_n_accepted / draft_n
```

Recommended workloads:

- short code completion, 128 predicted tokens;
- prose continuation after 2k context, 256 predicted tokens;
- long-context continuation after 16k-32k context, 256 predicted tokens;
- structured JSON/tool-call prompt, 128 predicted tokens.

Acceptance is mostly a regression/sanity metric, not an absolute quality metric. It varies heavily with prompt entropy, sampler settings, draft depth, and whether the model starts emitting `<think>` traces.

Suggested interpretation:

| Workload | Sanity floor | Main pass criterion |
|---|---:|---|
| short / easy continuation | >= 15% | within 10 percentage points or 25% relative of q8_0 |
| medium context | >= 10% | within 10 percentage points or 25% relative of q8_0 |
| long context 16k-32k | >= 8% | within 15 percentage points or 35% relative of q8_0 |
| structured/tool-call canary | no hard minimum | must remain coherent and schema-valid |

Compare q8_0 and TBQ4 when both fit:

- `tbq4_accept_pct` should remain close to `q8_0_accept_pct` for the same prompt; large acceptance drops are evidence that TBQ4 perturbed target logits enough to reject more draft tokens.
- If `draft_n` is absent while MTP was requested, fail the MTP gate: the server was not actually testing MTP acceptance.
- If acceptance is high but coherence fails, coherence wins: the candidate fails.

Observed TBQ4 smoke datum after the VEC fixes: 64-token MTP request at `-c 2048`, `--spec-type mtp --parallel 1 --spec-draft-n-max 3`, produced `draft_n=54`, `draft_n_accepted=45`, `mtp_accept_pct=83.3%`, coherent output, and 54.0 tok/s generation. q8_0 comparison at same settings: `draft_n=57`, `draft_n_accepted=44`, `mtp_accept_pct=77.2%`, 49.8 tok/s.

**Critical**: llama.cpp defaults `--spec-draft-n-max` to 16, which severely degrades aggregate acceptance (~36%). PR #22673 recommends n_max=3 for optimal acceptance (70-87%). The harness always sets `--spec-draft-n-max 3` when MTP is enabled.

### 6. Cache and slot coherence

- Same prompt with `cache_prompt=false` and then `cache_prompt=true`; output token ids should match.
- After a 28k+ prefill cached prompt, short prompts still pass smoke tests.
- Repeat the same long prompt twice; second run should reuse cache and produce the same answer.
- Optional: run 2 concurrent slots with independent prompts; no cross-contamination of answers.

### 7. Architecture support matrix

| Family | gfx targets | Status | Notes |
|---|---|---|---|
| RDNA3 | gfx1100 / gfx1101 / gfx1102 / gfx1103 | tested on gfx1100 | primary target; correctness + 64k MTP long-prompt perf passed |
| RDNA3.5 | gfx1150 / gfx1151 / gfx1152 | enabled, untested | covered by `GGML_CUDA_CC_IS_RDNA3`; should use same VEC path |
| RDNA4 | gfx1200 / gfx1201+ | enabled, untested | runtime dispatch allows `amd_wmma_available`; VEC path should avoid rocWMMA layout risk, but needs build/runtime validation |
| RDNA1/RDNA2 | gfx10xx | not enabled | no WMMA gate; can be investigated later as a pure VEC fallback if needed |
| CDNA/MI* | gfx9x/gfx94x | not enabled | separate MFMA path would need validation; not part of this acceptance gate |

Acceptance for a new GPU family:

1. build with `AMDGPU_TARGETS=<gfx>`;
2. run smoke + precision probes;
3. run at least 8k and 32k needle tests;
4. run MTP acceptance test if the model/server supports MTP;
5. record VRAM and throughput separately from correctness.

### 8. Performance is recorded, not the pass criterion

Record:

- prompt tokens;
- prompt ms / tok/s;
- generation tok/s;
- VRAM used;
- prompt cache MiB;
- context length;
- build commit/diff hash.

Performance regression fails only if correctness passes but speed drops below the previous accepted TBQ4 result by >15% on the same prompt length and context.

## Recommended final gate matrix

| Gate | Baseline | Candidate | Ctx | Prompt sizes |
|---|---|---|---:|---|
| smoke | none | TBQ4 | 64k | short |
| precision | q8_0 | TBQ4 | 32k or 64k | short, 2k, 8k, 16k, 28k |
| long needle | q8_0 if fits | TBQ4 | 64k | 8k, 16k, 32k, 56k |
| structured canaries | q8_0 | TBQ4 | 64k | short + after long prefill |
| MTP acceptance | q8_0 if fits | TBQ4 | 64k | 128-256 predicted tokens |
| cache coherency | none | TBQ4 | 64k | repeated 28k+ prompt |

## Minimal Python harness shape

The harness should:

1. start q8_0 server, wait for `/health` or `server is listening`;
2. run precision probes and write `q8_0.jsonl`;
3. stop q8_0;
4. start TBQ4 server;
5. run same probes, long needles, structured canaries, MTP acceptance, cache tests;
6. compute metrics into `summary.json`;
7. exit non-zero on any critical failure or threshold failure.

Use Python `urllib.request` or `requests`, not inline shell `curl -d`, to avoid argument-length limits for long prompts.

## Why this matches Hipfire-style screening

Hipfire's `mmq_screen` screens each static weight matrix by comparing WMMA vs MMQ on batch=16 synthetic activations and falling back per unsafe weight. TBQ4 FA corruption risk is different: it lives in KV-cache attention over prompt-dependent K/V rows, not static weights. Therefore the analogous safety gate is a prompt-set next-token distribution comparison plus long-context behavioral canaries, not a per-weight load-time sweep.

If TBQ4 ever gets an alternate fast path with selectable fallback, this gate can become a runtime/CI screen: fail the fast path if q8_0 top-token distributions drift beyond threshold on the calibration prompt set.
