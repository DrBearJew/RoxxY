# MTP / QBlock env knob registry

Last updated: 2026-06-18

Purpose: stop QBlock/PDMQ/WMMA env knob drift. If a new env variable affects QBlock, PDMQ, packed16 QBlock routing, WMMA probes, or strict verification, add it here and to `scripts/hip/qblock-env-audit.py` in the same patch.

## Packed16 K contract

Packed16 K is persistent quantized K storage, not f16-class VRAM.

For the active QBlock head dim `D=256`:

```text
payload = (256 / 4) I32 = 64 * 4 = 256 bytes
scales  = (256 / 32) F16 = 8 * 2 = 16 bytes
total   = 272 bytes per K row per KV head
```

That is q8_0-equivalent footprint (`8 * 34` bytes per row/head), versus f16 K at `256 * 2 = 512` bytes per row/head. Say “34 bytes” only when explicitly referring to one 32-dim block, not a full D=256 row.

## Persistent V4_K16D16 V cache candidate

Default-off, HIP-only storage bring-up for persistent V cache candidates. The 130 B format is lower-VRAM but shared-scale and not q4_0-exact; the 144 B format has no VRAM savings and exists to persist the oracle-proven q4_0-exact K16D16 layout. Neither is a promoted FA consumer route yet.

| Env | Default | Status | Scope | Notes |
| --- | --- | --- | --- | --- |
| `GGML_CUDA_ROCM_V4_K16D16_V_CACHE` | `0` | candidate | KV cache / CUDA pack op | Allocates experimental internal `GGML_TYPE_V4_K16D16` V storage for D=256 q4_0 V layers and packs/seals full K16 blocks with a small F16 tail buffer. FA-only and fail-closed for non-FA access. Not production evidence until native FA consumption, strict quality gates, and VRAM accounting pass. |
| `GGML_CUDA_ROCM_V4_K16D16_144_V_CACHE` | `0` | candidate | KV cache / CUDA pack op | Allocates internal `GGML_TYPE_V4_K16D16_144` storage for D=256 q4_0 V layers. This is intended to be q4_0-exact 144 B/row persistent K16D16 layout with no VRAM saving; current scalar PDMQ A/B still mismatches q4_0, so DOT4/PV promotion remains blocked. |

## Rules

1. **No silent knobs.** New QBlock/PDMQ/WMMA env variables require a registry entry before use.
2. **Default-safe.** Candidate and diagnostic knobs default off. Promoted standard knobs may default on only after strict correctness, profile, and speed gates; they must retain an explicit opt-out when practical.
3. **One canonical name per feature.** Prefer `GGML_CUDA_ROCM_MTP_QBLOCK_*` for backend QBlock route knobs. `LLAMA_MTP_*` aliases are allowed only for existing user-facing controls.
4. **Candidate lifecycle.** Candidate knobs must be named `*_CANDIDATE`, must stay default-off, must report correctness and speed gates before demotion or promotion.
5. **Diagnostics are not routes.** `*_TRACE`, `*_PARITY`, `*_DEBUG_*`, and profile knobs must not write accepted state or change consumed outputs unless explicitly listed as candidate/route.
6. **No prefix amnesty.** Every live QBlock/PDMQ/WMMA env knob must have an exact registry row. Broad legacy families are not considered registered.

## Canonical run profiles

Artifact bundle reconstruction lives in `.harness/research/qblock-bundle-archaeology-20260617.md`. Treat the rows below as bundles, not as independent knobs.

| Bundle | Artifact evidence | Decision |
| --- | --- | --- |
| Current no-env recurrent-prefix QBlock standard | Strict no-route-env n32/n128/n512 pass: `qprog-qblock-standard-strict-n32-20260617-093302`, `...n128-20260617-093344`, `...n512-20260617-093513`; no-diagnostic n512 `qprog-qblock-standard-no-diagnostics-n512-20260617-094126` = `11.540293317291123 tok/s`, `sha=df54c11d`, diagnostics off. | Standard runtime bundle. |
| Historical explicit GQA1/8x32/non-split serial-QKV | `qprog-qblock-serial-qkv-batched-fa-speed-ab-n512-20260616-160453/serial_qkv_batched_fa` = `11.515631401282532 tok/s`; PVBlockExact default-on artifact `qprog-qblock-pvblockexact-defaulton-vs-optout-speed-ab-n512-20260617-090423/02-default-pvblockexact` = `11.82711948333598 tok/s`; current explicit replay `qprog-qblock-explicit-canonical-gqa1-nondiag-n512-20260617-104717/run` = `11.374450119204015 tok/s`. | Equivalent/rollback recipe, not a hidden faster preset. |
| GQA6/split64 roweq-fastcap5 | Strict compare pass in `qprog-nmax6-roweq-fastcap5-working-20260616-014804/01-compare-n128`; valid ctx12288 speed `03-speed-n512-ctx12288` = `9.739972968911736 tok/s`; exact-only vs fastcap5 A/B neutral at ~`9.76 tok/s`. | Strict-passing experiment but slower than standard; do not promote. |
| prompt-`x` row-commit/spec-default smoke | Historical `step43-standard-prompt-x-qblock-default-n512-20260615-211055` = `55.27962529051494 tok/s`; current repro `qprog-promptx-step43-repro-current-n512-20260617-095822/run` = `62.28828673634196 tok/s`. | Different prompt/mode; not strict recurrent-prefix standard. |
| Direct graph side-effect/batch/group-row bundles | Strict n32 can pass, but n128 no-diagnostic speed loses to standard: side-effect `15.18000821380757`, batch-row `15.14943676642847`, group-row `15.243800774956723` tok/s vs standard n128 `15.824584481027806`. | Frozen legacy/debug; not promotion path. |
| Unsafe/stale shortcuts | accepted-row-only commit fails pre-repair state match; serial-QKV opt-out fails strict state despite faster no-compare n128; full-attn batch opt-out is slower. | Rejected or rollback-only. |

### Standard QBlock route, no diagnostics

Default standard route for Qwen35/Qwen3.6 MTP verifier blocks: QBlock recurrent-prefix backend, QBlock prefix full-attn batching, serial-QKV batched FA on validated non-terminal full-attn layers, packed16 DOT4/MMQ PDMQ, and PVBlockExact where supported. Do not enable compare/trace/profile knobs for a normal run.

Rollback/forcing knobs:

```bash
LLAMA_MTP_QBLOCK_RECURRENT_PREFIX=0                  # fall back from QBlock recurrent-prefix backend
LLAMA_MTP_QBLOCK_PREFIX_FULL_ATTN_BATCH=0            # fall back from QBlock prefix full-attn batching
LLAMA_MTP_QBLOCK_PREFIX_FULL_ATTN_SERIAL_QKV_BATCHED_FA=0  # fall back from serial-QKV batched FA
GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_PVBLOCK_EXACT=0    # fall back from PVBlockExact
```

### Strict compare sanity

Use this to prove strict scalar/QBlock parity after source changes. These are diagnostic knobs, not standard runtime defaults.

```bash
LLAMA_MTP_VERIFY_COMPARE=1
LLAMA_MTP_QBLOCK_TRACE=1
LLAMA_MTP_QBLOCK_RECURRENT_PREFIX=1
LLAMA_MTP_DISABLE_PACKED16_FA=0
LLAMA_MTP_ENABLE_FA=1
GGML_CUDA_ROCM_MTP_QBLOCK_GQA_GROUP=1
GGML_CUDA_ROCM_MTP_QBLOCK_SPLITK=1
GGML_CUDA_ROCM_MTP_QBLOCK_SPLITK_MIN_NK=0
# Optional shape override only when the test name says so:
# GGML_CUDA_ROCM_MTP_QBLOCK_SHAPE=8x32
```

Required result fields: HTTP 200, `rc=0`, token hash unchanged, `candidate_state_ok=1`, `oracle_state_ok=1`, pre/post state match, `state_match_all=true`.

### Whole-target QBlock verifier candidate

Experimental route candidate. Default-off. This is the non-DIRECT landing zone for a future Qwen35/Qwen3.6 verifier executor that owns target trunk layers `0..63` instead of building `LLM_GRAPH_TYPE_DECODER_PREFIX_VERIFY`.

| Env | Default | Status | Scope | Notes |
| --- | --- | --- | --- | --- |
| `LLAMA_MTP_QBLOCK_WHOLE_TARGET_CANDIDATE` | `0` | candidate | server/QBlock verifier | Sets `LLAMA_QBLOCK_VERIFY_FLAG_WHOLE_TARGET_CANDIDATE` on the QBlock verify plan. By itself this now fast-fallbacks; it is not a speed route. Does not use or add `LLAMA_MTP_QBLOCK_DIRECT_*` route knobs. |
| `LLAMA_MTP_QBLOCK_WHOLE_TARGET_CANDIDATE_FULL_LOGITS` | `0` | correctness-only | QBlock verifier | Opts into the current slow full-logits whole-target scaffold for strict correctness investigation. Materializes `n_tokens*vocab` logits and must not be used as speed evidence. Mutually exclusive with `LLAMA_MTP_QBLOCK_WHOLE_TARGET_SPEED_CANDIDATE`. |
| `LLAMA_MTP_QBLOCK_WHOLE_TARGET_SPEED_CANDIDATE` | `0` | speed-candidate | server/QBlock verifier | Sets `LLAMA_QBLOCK_VERIFY_FLAG_WHOLE_TARGET_SPEED_CANDIDATE` on the QBlock verify plan. Publishes sampled top1 output rows only via backend `ggml_lm_head_top_k`; must not allocate, compute, copy, or gate on `n_tokens*vocab` logits. Default-off and not promotion evidence until strict correctness plus speed A/B pass. Mutually exclusive with `LLAMA_MTP_QBLOCK_WHOLE_TARGET_CANDIDATE_FULL_LOGITS`. |

### State compare telemetry

Diagnostic-only state span comparison. These aliases reuse the existing Direct layer-compare implementation without enabling Direct graph execution.

| Env | Default | Status | Scope | Notes |
| --- | --- | --- | --- | --- |
| `LLAMA_MTP_QBLOCK_STATE_LAYER_COMPARE` | `0` | diagnostic | server trace | Neutral alias for state span compare telemetry. Emits `QBLOCK_STATE_LAYER_COMPARE` lines. |
| `LLAMA_MTP_QBLOCK_STATE_LAYER_COMPARE_LIMIT` | `-1` | diagnostic | server trace | Optional max layer index for neutral state compare. Falls back to old Direct limit when unset. |
| `LLAMA_MTP_QBLOCK_EXACT_COMMIT_LAYER_COMPARE` | `0` | diagnostic | server trace | Exact-commit bring-up alias for the same compare helper. Emits `QBLOCK_EXACT_COMMIT_LAYER_COMPARE` lines. |
| `LLAMA_MTP_QBLOCK_EXACT_COMMIT_LAYER_COMPARE_LIMIT` | `-1` | diagnostic | server trace | Optional max layer index for exact-commit compare. |
| `LLAMA_MTP_QBLOCK_RS_STAGE_TRACE` | `0` | diagnostic | recurrent trace | Neutral alias for existing R/S staging trace telemetry. Emits `QBLOCK_RS_STAGE_TRACE` lines. |
| `LLAMA_MTP_QBLOCK_RS_STAGE_TRACE_LAYER` | `-1` | diagnostic | recurrent trace | Optional layer filter for neutral R/S staging trace. |
| `LLAMA_MTP_QBLOCK_RS_STAGE_TRACE_KIND` | unset | diagnostic | recurrent trace | Optional component filter, usually `R` or `S`. |
| `LLAMA_MTP_QBLOCK_EXACT_COMMIT_RS_STAGE_TRACE` | `0` | diagnostic | recurrent trace | Exact-commit bring-up alias for R/S staging trace. Emits `QBLOCK_EXACT_COMMIT_RS_STAGE_TRACE` lines. |
| `LLAMA_MTP_QBLOCK_EXACT_COMMIT_RS_STAGE_TRACE_LAYER` | `-1` | diagnostic | recurrent trace | Optional layer filter for exact-commit R/S staging trace. |
| `LLAMA_MTP_QBLOCK_EXACT_COMMIT_RS_STAGE_TRACE_KIND` | unset | diagnostic | recurrent trace | Optional component filter for exact-commit R/S staging trace. |

Legacy compatibility remains available through `LLAMA_MTP_QBLOCK_DIRECT_LAYER_COMPARE`, `LLAMA_MTP_QBLOCK_DIRECT_LAYER_LIMIT`, and `LLAMA_MTP_QBLOCK_DIRECT_RS_STAGE_TRACE*`, but new strict-performance work should prefer the neutral names above.

Forced UBATCH1 verifier decode now regenerates recurrent rollback snapshot rows by copying the live R/S row to the matching rollback slot after each one-token verifier ubatch. This is not a new route knob; it is required so `LLAMA_MTP_TARGET_BATCH_VERIFY_UBATCH1=1` diagnostics can make partial/zero accepts strict-state exact instead of copying stale snapshot rows. Existing `LLAMA_MTP_RS_COMMIT_TRACE` / `LLAMA_MTP_RS_COMMIT_LAYER_TRACE` diagnostics also emit `LLAMA_RS_SNAPSHOT_TRACE` / `LLAMA_RS_SNAPSHOT_LAYER_TRACE` lines when this path snapshots rows.

## Standard route controls

| Env | Default | Status | Scope | Notes |
| --- | --- | --- | --- | --- |
| `LLAMA_MTP_ENABLE_FA` | launcher-defined | standard | MTP graph | Enables MTP FA route family. |
| `LLAMA_MTP_DISABLE_PACKED16_FA` | `0` in QBlock runs | standard | MTP graph | Must remain `0` for packed16 FA/QBlock route. |
| `LLAMA_MTP_QBLOCK_ACTIVE` | internal | standard | graph/backend | Signals active QBlock path. Do not use as a candidate gate. |
| `LLAMA_MTP_QBLOCK_RECURRENT_PREFIX` | `1` | standard | server scheduler | Selects QBlock recurrent-prefix verifier backend when recurrent state requires exact prefix verification; set `0` to fall back to serial/recurrent-prefix alternatives. |
| `LLAMA_MTP_QBLOCK_RECURRENT_PREFIX_TOKEN_MAJOR_CANDIDATE` | `0` | failed candidate | Qwen35 prefix graph | Default-off token-major recurrent-prefix batching candidate. Strict n32 failed pre-repair state parity; kept only for diagnostics, not promotion evidence. |
| `LLAMA_MTP_QBLOCK_RECURRENT_PREFIX_TOKEN_MAJOR_CANDIDATE_LAYER` | unset | failed candidate filter | Qwen35 prefix graph | Optional single-layer filter for token-major recurrent-prefix candidate. |
| `LLAMA_MTP_QBLOCK_RECURRENT_PREFIX_TOKEN_MAJOR_CANDIDATE_LAYERS` | unset | failed candidate filter | Qwen35 prefix graph | Optional comma/range layer filter for token-major recurrent-prefix candidate; overrides the single-layer filter. |
| `LLAMA_MTP_QBLOCK_RECURRENT_PREFIX_CONV_BATCH_CANDIDATE` | `0` | candidate | Qwen35 prefix graph | Default-off conv-batched recurrent-prefix candidate. Batches recurrent input conv/R snapshots but keeps GDN/S state updates row-serial AR; requires strict compare and speed A/B before promotion. |
| `LLAMA_MTP_QBLOCK_RECURRENT_PREFIX_CONV_BATCH_CANDIDATE_LAYER` | unset | candidate filter | Qwen35 prefix graph | Optional single-layer filter for conv-batched recurrent-prefix candidate. |
| `LLAMA_MTP_QBLOCK_RECURRENT_PREFIX_CONV_BATCH_CANDIDATE_LAYERS` | unset | candidate filter | Qwen35 prefix graph | Optional comma/range layer filter for conv-batched recurrent-prefix candidate; overrides the single-layer filter. |
| `LLAMA_MTP_QBLOCK_PREFIX_FULL_ATTN_BATCH` | `1` | standard | Qwen35 prefix graph | Enables QBlock full-attn prefix batching container; set `0` to fall back. Layer filters remain available for experiments. |
| `LLAMA_MTP_QBLOCK_PREFIX_FULL_ATTN_SERIAL_QKV_BATCHED_FA` | `1` | standard | Qwen35 prefix graph | Standard exact QBlock FA path. Unset default applies to validated non-terminal full-attn layers; explicit `=1` with no layer filter keeps legacy all-layer opt-in behavior; set `0` to fall back. |
| `LLAMA_MTP_QBLOCK_PREFIX_FULL_ATTN_SERIAL_QKV_BATCHED_FA_LAYERS` | unset | route override | Qwen35 prefix graph | Optional list/ranges; overrides the standard non-terminal layer policy for serial-QKV batched FA. |
| `LLAMA_MTP_QBLOCK_STRICT` | off unless requested | strict | graph/backend | Strict QBlock route gate. Diagnostic/assertion mode, not standard runtime default. |
| `LLAMA_MTP_VERIFY_COMPARE` | off unless requested | strict | verifier | Required for strict compare runs. Diagnostic; do not enable for normal standard route. |
| `LLAMA_MTP_QBLOCK_TRACE` | off | diagnostic | graph/backend | Route/state trace. High log volume. |
| `LLAMA_MTP_QBLOCK_DISABLE` | `0` | rollback | server scheduler | Disables server-side QBlock verifier scheduling and falls back to non-QBlock verifier behavior. |
| `LLAMA_MTP_QBLOCK_RECURRENT_REPAIR` | `0` | diagnostic/repair | server scheduler | Legacy repair experiment for recurrent QBlock state; not promotion evidence and not standard runtime. |
| `LLAMA_MTP_QBLOCK_REJECT_TRACE` | `0` | diagnostic | server scheduler | Emits QBlock rejection trace telemetry, including seq/position, planned/materialized rows, accepted rows, original/chunked draft size, commit mode, backend, and first reject row. Must not change accepted state. |
| `LLAMA_MTP_QBLOCK_VERIFY_CHUNK` | `1` | standard | server scheduler | Standard QBlock verifier chunk/fallback path. When a default speculative draft is larger than the chunk size and recurrent rollback is disabled, verifies the first chunk. If the chunk fully accepts while tail remains, it restores the checkpoint and falls back to full verification; otherwise it commits/re-drafts normally. Does not change ngram/spec knobs. Set `0` to disable. |
| `LLAMA_MTP_QBLOCK_VERIFY_CHUNK_SIZE` | `4` | standard | server scheduler | Draft-token chunk size for `LLAMA_MTP_QBLOCK_VERIFY_CHUNK`; clamped to at least 1. |
| `LLAMA_MTP_QBLOCK_VERIFY_CHUNK_TRACE` | `0` | diagnostic | server scheduler | Emits QBlock verifier chunk telemetry. Must not change accepted state. |
| `LLAMA_MTP_QBLOCK_VERIFY_CHUNK_STAGED_CANDIDATE` | `0` | candidate | server scheduler | Default-off staged-tail subcandidate for QBlock verifier chunking. If a chunk boundary matches the original tail, verifies subsequent tail chunks before committing instead of discarding the tail or immediately falling back to full verification. High risk; requires long output-hash A/B before promotion. |
| `LLAMA_MTP_QBLOCK_VERIFY_CHUNK_STAGED_MAX_CHUNKS` | `16` | candidate | server scheduler | Safety cap on staged tail chunks per QBlock block. |
| `LLAMA_MTP_QBLOCK_NQ` | unset | diagnostic/shape | backend | Test-only QBlock `nq` override/reporting aid. Do not use for promotion runs unless named in artifact. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_PDMQ` | route-dependent | standard | backend | Enables PDMQ QBlock backend path. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_VEC` | `0` | candidate | backend | Enables the QBlock verify VEC candidate for packed16 K + q4_0 V. Does not replace PDMQ; requires cols=nq vs cols=1 bit-exact smoke before promotion. |
| `GGML_CUDA_ROCM_PACKED16_FA2_VEC` | `0` | experimental/global | backend | Legacy/global packed16 FA2 VEC opt-in for packed16 K + q4_0 V. Do not use as QBlock promotion evidence; prefer `GGML_CUDA_ROCM_MTP_QBLOCK_VEC` for QBlock-scoped validation. |
| `GGML_CUDA_ROCM_PACKED16_FA2_VEC_COLS` | auto | candidate/smoke | backend | Packed16 VEC columns per block. For `MTP_QBLOCK_VERIFY_QK` with `GGML_CUDA_ROCM_MTP_QBLOCK_VEC=1`, `4/8/16` are QBlock-scoped candidates and `1` is the bit-exact smoke baseline; outside QBlock, multi-col values remain unsafe probe-only. |
| `GGML_CUDA_ROCM_MTP_VERIFY_SMALLQ_PDMQ` | route-dependent | standard | backend | Small-Q PDMQ verify selection. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_MAX_NQ` | launcher-defined | standard | backend | QBlock max FA rows. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_SHAPE` | auto | shape override | backend | Explicit compact shapes: `1x32`, `2x32`, `4x32`, `8x32`, `16x16`. Artifact name must include forced shape. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_SHAPE_POLICY` | `safe` | experimental | backend | `tetris`/fit policies. Not a promotion gate alone. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_GQA_GROUP` | auto/`1` in strict GQA1 route | shape/dataflow | backend | Allowed values currently `1`, `2`, `4`, `6`. Must be artifact-visible. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_SPLITK` | `1` for non-split | experimental | backend | Split-K route is not strict-state exact unless separately proven. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_SPLITK_MIN_NK` | route default | experimental | backend | Split-K threshold. |
| `GGML_CUDA_DP16_FA_QBLOCK_Q_PRECISION` | route default | qpack | backend | QBlock Q precision, e.g. `qpack`. |
| `LLAMA_MTP_QBLOCK_Q_PRECISION` | alias | qpack | backend | Alias for QBlock Q precision. |
| `GGML_CUDA_DP16_FA_QBLOCK_ROWMAP_MODE` | route default | qprog | backend | QBlock rowmap mode, e.g. `identity`. |
| `LLAMA_MTP_QBLOCK_ROWMAP_MODE` | alias | qprog | backend | Alias for QBlock rowmap mode. |
| `GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_PVBLOCK_EXACT` | `1` | standard | backend | Exact scalar PVBlock specialization for packed16 DOT4/MMQ. Default-on when supported (`q4_0` raw-LDS V, no staged V, no competing PV candidate); set `0` to force the old scalar PV accumulator path for A/B or rollback. |
| `GGML_CUDA_ROCM_PACKED16_K_SCALE_GROUP_QBLOCKS` | route default | scale | backend | K scale grouping. Contract-sensitive. |
| `LLAMA_MTP_PACKED16_K_SCALE_GROUP_QBLOCKS` | alias | scale | backend | Alias for K scale grouping. |

### QBlock VEC cols smoke gate

Before promoting `GGML_CUDA_ROCM_MTP_QBLOCK_VEC=1` for a QBlock shape, run a bit-exact cols gate against the same request/workload:

```bash
scripts/hip/qblock-vec-cols-smoke.py --nq 4 -- <command-that-emits-sha-or-json-summary>
```

The helper runs the command with `GGML_CUDA_ROCM_PACKED16_FA2_VEC_COLS=1` and then with `GGML_CUDA_ROCM_PACKED16_FA2_VEC_COLS=<nq>`, both under `GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_fa2_vec`, and forces generic unsafe VEC probes off so non-QBlock VEC calls remain on their validated cols path. It fails unless both hashes match and both merged logs show the `rocm_packed16_fa2_vec` route marker. Use `--no-require-route-log` only when route evidence is checked separately. This validates the QBlock-scoped `cols=nq` promotion independently from PDMQ.

## WMMA / PDMQ diagnostics and candidates

PDMQ phase profiling, WMMA oracle trace, WMMA QK parity, and low-level tail/state probes were removed from the live PDMQ route in the stage-1 cleanup. Use archived research notes or a new microbench/test harness for future layout/parity work.

## PWMMA reference/probe knobs

These are for the older packed16 PWMMA route and layout probes, not QBlock route promotion.

| Env | Default | Status | Notes |
| --- | --- | --- | --- |
| `GGML_CUDA_PWMMA_PROFILE` | `0` | diagnostic | PWMMA timing. |
| `GGML_CUDA_PWMMA_ABORT_AFTER_LAYOUT` | `0` | debug | Layout abort. |
| `GGML_CUDA_PWMMA_BM64_PROBE` | `0` | diagnostic | BM64 PWMMA probe. |
| `GGML_CUDA_PWMMA_DUMP_PACKED16` | `0` | diagnostic | Dump packed16 layout. |
| `GGML_CUDA_PWMMA_I8_DOT4_SHADOW_PROBE` | `0` | diagnostic | Synthetic i8 WMMA/DOT4 shadow probe. |
| `GGML_CUDA_PWMMA_I8_LIVE_DOT4_SHADOW` | `0` | diagnostic | Live tile i8 WMMA/DOT4 shadow. |
| `GGML_CUDA_PWMMA_PV_MULTI_PROBE` | `0` | diagnostic | PV multi probe. |
| `GGML_CUDA_PWMMA_PV_WMMA_SHADOW` | `0` | diagnostic | PV-WMMA shadow. |

## Legacy-prefix audit policy

Prefix registration is disabled. Existing legacy/debug families may remain in source only if their concrete env names appear as exact rows in this registry; otherwise `scripts/hip/qblock-env-audit.py` reports them as unregistered cleanup debt.

## Removed / do-not-revive knobs

| Env | Status | Why |
| --- | --- | --- |
| `LLAMA_MTP_HEAD_CACHE_ONLY_NO_OUTPUT` | removed/do not revive | Unsafe cache-only/no-output shortcut. |
| `LLAMA_MTP_EXPERIMENTAL_HEAD_KV_PREFILL_ONLY` | default-off legacy experiment | Must not be used to claim strict correctness. |
| `LLAMA_MTP_EXPERIMENTAL_HEAD_OUTPUT_ROWS_ONLY` | default-off legacy experiment | Must not be used to claim strict correctness. |
| `GGML_CUDA_ROCM_Q8Q4_DOT4_PREFILL*` | removed q8q4 FA prototype | Old q8_0 K + q4_0 V DOT4 prefill route could auto-select in 2026-06-02 DP16/q8 logs and had no recent QBlock/V4 hits; prototype header and dispatch removed. |
| `GGML_CUDA_ROCM_Q8Q4_WMMA_I8*` | removed q8q4 FA prototype | Old q8_0 K + q4_0 V WMMA-I8 route had no searched runtime hits; prototype header and dispatch removed. |
| `GGML_HIP_LEGACY_Q8Q4_FA` / `GGML_CUDA_LEGACY_Q8Q4_FA` | removed temporary build gate | Superseded by deleting the q8q4 FA prototype headers and route plumbing. |
| `GGML_HIP_MMVQ_COMPILE_LEGACY_INTERLEAVED_ACT` / `GGML_CUDA_MMVQ_COMPILE_LEGACY_INTERLEAVED_ACT` | removed legacy MMVQ build gate | Superseded by deleting the non-K legacy interleaved activation route from `mmvq.cu`; generic MMVQ and K-quant interleaved routes remain. |
| `LLAMA_MTP_MMVQ_LEGACY_INTERLEAVED_ACT*` | removed legacy MMVQ route | Runtime knobs removed with the non-K legacy interleaved activation route. |
| `GGML_CUDA_PDMQ_WMMA_QK_CANDIDATE*` / `GGML_CUDA_ROCM_MTP_QBLOCK_WMMA_QK_CANDIDATE` / `LLAMA_MTP_QBLOCK_WMMA_QK_CANDIDATE` | removed PDMQ candidate path | Private consumed WMMA-QK marker/candidate path was product-runtime clutter without a current acceptance route. |
| `GGML_CUDA_PDMQ_DEBUG_KSCALE_ONE` / `GGML_CUDA_PDMQ_DEBUG_VSCALE_ONE` | removed wrong-result debug bypass | Wrong-result scale bypass experiments removed from product runtime. |
| `GGML_CUDA_PDMQ_DEBUG_SELF_PROBES` / `GGML_CUDA_PDMQ_DEBUG_SKIP_PROBES` | removed runtime self-probes | PDMQ synthetic self-probes removed from launch path; future probes belong in tests/microbenches. |
| `GGML_CUDA_PDMQ_PROFILE` | removed runtime profiler | Phase cycle profiling was intertwined with hot kernels and removed from the live route; use external profiling or a dedicated microbench. |
| `GGML_CUDA_PDMQ_WMMA_ORACLE_TRACE` / `GGML_CUDA_ROCM_MTP_QBLOCK_WMMA_ORACLE_TRACE` / `LLAMA_MTP_QBLOCK_WMMA_ORACLE_TRACE` | removed oracle trace | Host-side PDMQ WMMA layout telemetry removed from product runtime. |
| `GGML_CUDA_PDMQ_WMMA_QK_PARITY*` / `GGML_CUDA_ROCM_MTP_QBLOCK_WMMA_QK_PARITY` / `LLAMA_MTP_QBLOCK_WMMA_QK_PARITY` | removed parity probe | Device parity probe and printf state removed from live PDMQ kernels. |
| `GGML_CUDA_PDMQ_DEBUG_TAIL_PROB*` / `GGML_CUDA_PDMQ_DEBUG_STATE*` | removed debug probes | Tail probability, scalar-ref, input-probe, and state-print debug blocks removed from live PDMQ kernels. |
| `GGML_CUDA_ROCM_V4_K16D16_144_EXACT_SPLITK_CANDIDATE*` | removed tombstone | Demoted split-K candidate was only an aborting tombstone after scalar split-K failed speed and q4-LDS split-K diverged. |
| `GGML_HIP_PDMQ_FULL_EXPERIMENTAL_MATRIX` | removed build gate | Stage-2 cleanup made PDMQ compact-only; full experimental rebuild plumbing is not maintained. |
| `GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_IMPL` | removed runtime selector | KSHARED/stagev/directv legacy variants were deleted from the live route surface. |
| `GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_Q8V_N64` | removed q8v experiment | Q8V_N64 was a demoted N64-shape experiment and no longer has launcher/planner policy. |
| `GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE=m1n64/m2n64/m4n64` | removed shape values | N64 PDMQ shapes were deleted; compact shape overrides are `1x32`, `2x32`, `4x32`, `8x32`, `16x16`. |
| `GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_VPATH=direct_pv_q4/direct_legacy` | removed VPATH values | Explicit direct legacy V aliases were deleted; compact VPATH remains `stage_f32`, raw LDS, and persistent V4 diagnostics. |

## Current known lesson

The old `WMMA_QK_CANDIDATE` marker path was removed rather than kept as a default-off runtime candidate. Future WMMA-QK work needs a fresh design note, likely logical/head-packed M fill, and must land as a tested implementation path rather than private product-runtime scaffolding.
