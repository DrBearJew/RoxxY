# MTP / QBlock env knob registry

Last updated: 2026-06-23

Purpose: stop QBlock/PDMQ/WMMA env knob drift. If a new env variable affects QBlock, PDMQ, packed8/packed16 QBlock routing, WMMA probes, or strict verification, add it here and to `scripts/hip/qblock-env-audit.py` in the same patch.

## PDMQ K sibling selector

`GGML_CUDA_ROCM_PDMQ_K_CACHE=0|1` controls the compressed-K sidecar allocation. The legacy alias `GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=0|1` is still honored.

`GGML_CUDA_ROCM_PDMQ_K_FORMAT=default|packed16_q8|packed8_q4|packed4_q2|none`. Default remains `packed16_q8` for normal PDMQ/canonical runs. When the user explicitly/effectively requests `q8_0` K cache, the default PDMQ format becomes `packed8_q4` so q8 means the packed8 replacement path, not the legacy raw `GGML_TYPE_Q8_0` VEC FA path. Packed8/packed4 replace packed16 K storage when selected; they are not sidecars. Packed4 is reserved/unsupported in the current mainline because there is no native int2 compute path. Forced route mode must abort on missing or wrong metadata.

Packed8 q4 is a lossy KV format. It is a perf/VRAM route, not a canonical token-hash route. Daily-tree `q8_0` K controls also miss the current packed16 canonical hash, so do not spend promotion time chasing `accabc9a` with global q4 scale tweaks.

## Packed16 standard and packed8 q8-request K contracts

Packed16 K is the standard/canonical persistent q8-like PDMQ storage, not f16-class VRAM. Packed8 K is the replacement selected for q8_0 K requests.

For the active QBlock head dim `D=256`, packed8 uses:

```text
payload = (256 / 8) I32 = 32 * 4 = 128 bytes
scales  = (256 / 32) F16 = 8 * 2 = 16 bytes
total   = 144 bytes per K row per KV head
```

Packed16 fallback uses:

```text
payload = (256 / 4) I32 = 64 * 4 = 256 bytes
scales  = (256 / 32) F16 = 8 * 2 = 16 bytes
total   = 272 bytes per K row per KV head
```

That is q8_0-equivalent footprint (`8 * 34` bytes per row/head), versus f16 K at `256 * 2 = 512` bytes per row/head. Say “34 bytes” only when explicitly referring to one 32-dim block, not a full D=256 row.

## Persistent V4_K16D16 V cache candidate

Default-on, HIP-only standard storage for the persistent V144/PV4 cache stack when it is compiled. The 130 B format is lower-VRAM but shared-scale and not q4_0-exact; the 144 B format has no VRAM savings and exists to persist the oracle-proven q4_0-exact K16D16 layout.

V144/PV4 is the standard scalar path. `GGML_CUDA_ROCM_V4_K16D16_144_PROFILE` is a separate explicit DP16-DOT/profile switch and must not be treated as the standard V144/PV4 activation knob. The legacy compatibility knob `GGML_CUDA_ROCM_V4_K16D16_144_PV4=0|1` controls the standard scalar V144/PV4 path; unset means enabled. Use `GGML_CUDA_ROCM_V4_K16D16_144_PV4_DISABLE=1` as a hard runtime rollback when a control run needs V144/PV4 inactive.

Fast active-RS route gates now default to batched target verification; set `LLAMA_MTP_TARGET_BATCH_VERIFY_UNSAFE=0` for strict serial controls. Explicit opt-outs remain available for rollback: `GGML_CUDA_ROCM_V4_K16D16_144_V_CACHE=0`, `GGML_CUDA_ROCM_V4_K16D16_144_MTP_DRAFT_V_CACHE=0`, and `GGML_CUDA_ROCM_V4_K16D16_144_PDMQ_DECODE_SPLITK=1`. DP16-DOT/profile-only controls such as `GGML_CUDA_DP16_FA_QPACK` stay explicit unless the profile is deliberately enabled.

| Env | Default | Status | Scope | Notes |
| --- | --- | --- | --- | --- |
| `GGML_CUDA_ROCM_V4_K16D16_V_CACHE` | `0` | candidate | KV cache / CUDA pack op | Allocates experimental internal `GGML_TYPE_V4_K16D16` V storage for D=256 q4_0 V layers and packs/seals full K16 blocks with a small F16 tail buffer. FA-only and fail-closed for non-FA access. Not production evidence until native FA consumption, strict quality gates, and VRAM accounting pass. |
| `GGML_CUDA_ROCM_V4_K16D16_144_V_CACHE` | `1` | standard candidate | KV cache / CUDA pack op | Allocates internal `GGML_TYPE_V4_K16D16_144` storage for D=256 q4_0 V layers. Enabled by the standard V144/PV4 path unless explicitly set to `0` or hard-disabled with `GGML_CUDA_ROCM_V4_K16D16_144_PV4_DISABLE=1`. |
| `GGML_CUDA_ROCM_V4_K16D16_144_PROFILE` | `0` | explicit DP16-DOT profile | DP16/QPack profile | Extra DP16-DOT/profile switch. Unset means off and must not disable or enable the standard scalar V144/PV4 path. |
| `GGML_CUDA_ROCM_V4_K16D16_144_PV_IMPL` | `scalar` | scalar selector | PDMQ | Selects the V144 P*V arithmetic implementation for the standard scalar V144/PV4 path. Mainline accepts only `scalar`. Experimental candidate names are kept out of this runtime option contract and live under the harness future-patches area. |
| `GGML_CUDA_ROCM_V4_K16D16_144_PV4` | `1` | standard scalar V144/PV4 | KV cache / FA selector / PDMQ | Standard scalar V144/PV4 path when compiled. Set `0` for runtime controls. |
| `GGML_CUDA_ROCM_V4_K16D16_144_PV4_DISABLE` | `0` | rollback/control | PDMQ | Hard runtime opt-out for V144/PV4 PDMQ selection even when legacy `PV4=1` or profile envs are present. Use for A/B controls and stale shell-profile cleanup. |

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
| Goal55 pure-MTP active-RS candidate | `goal55-puremtp-rs-gate-20260622-163506`: n128 `53.91264719100367 tok/s`, n512 `43.29012381482717 tok/s`, hashes `accabc9a`/`8d10ba2d`, target/draft `n_rs_seq=[4,0]`. Standard PV4 profile `goal55-standard-pv4-profile-final-20260622-185554`: n128 `53.501416951589576 tok/s`, n512 `42.720123781558655 tok/s`, same hashes, target/draft `n_rs_seq=[4,0]`. `goal55-puremtp-rs-n512-cycle-20260622-164515`: n512 draft acceptance `357/612`, 255 rejected drafts, worst 16-output windows at 0.25-0.42 acceptance. | Candidate speed profile. Must not use `--spec-default`; `--spec-default` mixes ngram with MTP, disables target `n_rs_seq`, and drops n512 to ~36 tok/s. Remaining gap is proposal cadence/waste, not MMQ cap. |
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

Removed from the live source tree during the 2026-06-23 desloppify pass. The orphan public `llama_decode_qblock_verify` API and uncompiled `src/llama-qblock-direct.inc` implementation were archived out-of-tree under the harness `archived-code/20260623-qblock-direct-orphan-api/` artifact. Do not document or use `LLAMA_MTP_QBLOCK_WHOLE_TARGET_*` envs unless a compiled verifier API is intentionally restored.

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

### Draft branch-candidate metadata

Default-off server-visible metadata channel for future tree/block verifiers. It records candidate token/logit/rank metadata next to `spec_draft` without changing sampler token selection. Normal draft paths capture real draft top-k rows; target-sidecar direct paths publish only rank0 selected-token metadata so they cannot masquerade as alternate branches.

Transport path: server maps the selected verifier rows to optional `llama_batch` / `llama_ubatch` QBlock row metadata arrays (`parent`, `branch_id`, `candidate_rank`, `output_policy`), `llama-graph.cpp` packs those rows into `FLASH_ATTN_EXT` op params, and the PDMQ/QBlock backend applies them to `dp16_fa_qblock_program` before route trace/kernel launch. Route logs expose `qprog_meta_rows`. The PDMQ/QBlock backend now treats local verifier-window K rows as tree-visible only when they are the current row or an ancestor under `row_parent`; past context remains visible. At the FA level, `ATTENTION_ONLY` still writes attention output, while server `logits=false` suppresses final logits for sibling sidecars.

| Env | Default | Status | Scope | Notes |
| --- | --- | --- | --- | --- |
| `LLAMA_MTP_DRAFT_BRANCH_CANDIDATES` | `0` | metadata | speculative draft/server | Enables per-draft-depth candidate capture in `common_speculative_draft_params::branch_candidates`. When enabled without an explicit `LLAMA_MTP_BACKEND_TOPK_K`, backend top-k width defaults to 16 for metadata only; sampled token selection remains unchanged. |
| `LLAMA_MTP_DRAFT_CANDIDATES_TRACE` | `0` | diagnostic | server trace | Enables branch-candidate capture and emits `MTP_DRAFT_CANDIDATES` lines from `server_slot::update_batch()`. Diagnostic only; do not use for clean speed gates. |
| `LLAMA_MTP_QBLOCK_SIBLING_ROWS_PROTOTYPE` | `0` | experimental | server/QBlock verifier | Single-slot-only prototype that appends non-selected candidate rows after the selected verifier path. Rows are not added to `spec_i_batch`, use `logits=false`, and carry `ATTENTION_ONLY` output policy. Token selection is unchanged. |
| `LLAMA_MTP_QBLOCK_SIBLING_ROWS_MAX` | `1` | experimental | server/QBlock verifier | Maximum sibling sidecar rows to append when the prototype is enabled; clamped to 8. |
| `LLAMA_MTP_QBLOCK_SIBLING_ROWS_TRACE` | `0` | diagnostic | server/QBlock verifier | Emits `MTP_QBLOCK_SIBLING_ROWS` summaries. Candidate trace also enables this summary. |
| `LLAMA_MTP_QBLOCK_SIBLING_LOGITS_PROBE` | `0` | diagnostic/probe | server/QBlock verifier | Safe parent-logit hit detector for sibling candidates. It enables branch-candidate capture, scopes raw verifier logits via `LLAMA_MTP_TOPK_VERIFY=1`, disables target fused-top1 sampled buffers for the verifier decode, and emits `MTP_QBLOCK_SIBLING_TOP1 source=parent_logits` with sibling candidate top1/rank. It deliberately suppresses same-batch sibling sidecar rows because those rows can perturb verifier trajectory and their logits predict the token after the sidecar, not the sidecar candidate itself. |
| `LLAMA_MTP_QBLOCK_SIBLING_BRANCH_PLAN_TRACE` | `0` | diagnostic/probe | server sampler | Emits `MTP_QBLOCK_BRANCH_PLAN` after target sampling. It reports whether the actual sampled token at the first rejected draft depth was present in captured sibling candidates. `safe_commit=0` is intentional until same-cycle branch descendants and transactional tree commit exist. The logits probe enables this trace automatically. |
| `LLAMA_MTP_QBLOCK_SIBLING_BRANCH_STATE_TRACE` | `0` | diagnostic/probe | server sampler/spec state | Emits `MTP_QBLOCK_BRANCH_STATE` for sibling-hit rejects. It distinguishes next-cycle readiness from same-cycle branch extension: the emitted sampled token and parent-row hidden seed are available for the normal next cycle, but same-cycle extension remains unavailable because draft branch descendants, target branch rows, and transactional tree commit are not captured yet. The logits probe enables this trace automatically. |
| `LLAMA_MTP_QBLOCK_SIBLING_BRANCH_DESC_TRACE` | `0` | diagnostic/probe | server sampler/spec state | Emits `MTP_QBLOCK_BRANCH_DESC` for sibling-hit rejects. It records selected-path rows already available after the rejected parent and explicitly reports that sampled-sibling branch descendants and target branch rows are not captured (`same_cycle_replayable=0`, `safe_commit=0`). Branch-state tracing enables this trace automatically. |
| `LLAMA_MTP_QBLOCK_SIBLING_BRANCH_SUBTREE_TRACE` | `0` | diagnostic/probe | server sampler/spec state | Emits `MTP_QBLOCK_BRANCH_SUBTREE` for sibling-hit rejects. It records the sampled sibling root candidate, selected-path row/token lists, and bounded candidate lists, but remains trace-only: sampled-sibling descendants and target branch rows are still uncaptured in this root marker, with `same_cycle_replayable=0` and `safe_commit=0`. Branch-desc tracing enables this trace automatically. |
| `LLAMA_MTP_QBLOCK_SIBLING_DESC_PROBE` | `0` | diagnostic/probe | MTP draft/server sampler | Emits `MTP_QBLOCK_BRANCH_DESC_PROBE` for sibling-hit rejects. It snapshots/restores the draft sequence state, rewinds the draft KV to the sampled-sibling position, temporarily seeds MTP `pending_h` from the target parent row, and drafts sampled-sibling descendants without touching target KV, production sampler, prompt, output, or commit state. Still proof-only: target branch rows are missing, so `same_cycle_replayable=0` and `safe_commit=0`. |
| `LLAMA_MTP_QBLOCK_SIBLING_DESC_MAX` | `1` | diagnostic/probe | MTP draft/server sampler | Max sampled-sibling descendants for `LLAMA_MTP_QBLOCK_SIBLING_DESC_PROBE`; clamped to `1..8`. The 9B proof runner uses `1` to prove one transaction-tail replacement draft token per sibling-hit opportunity. |
| `LLAMA_MTP_QBLOCK_SIBLING_TARGET_ROWS_PROBE` | `0` | diagnostic/probe | target verifier/server sampler | Emits `MTP_QBLOCK_BRANCH_TARGET_PROBE` for sibling-hit rejects after descendant capture. After normal sampling, it clears a reserved scratch sequence, copies the immutable pre-verifier attention prefix into it, remaps the saved pre-verifier recurrent checkpoint onto the scratch seq, then serially replays `sampled + accepted selected-prefix + sampled sibling + sampled-sibling descendants` there to capture target branch rows/logits. It does not mutate the production sequence, call `common_speculative_process()`, or touch draft state, production sampler, prompt, output, or commit state. Same-cycle commit stays disabled. |
| `LLAMA_MTP_QBLOCK_SIBLING_TARGET_ROWS_MAX` | `1` | diagnostic/probe | target verifier/server sampler | Max sampled-sibling descendants to include in `MTP_QBLOCK_BRANCH_TARGET_PROBE`; clamped to `1..8`. With the default descendant probe depth, the 9B proof captures two target branch rows per sibling hit: the sampled sibling row and one descendant row. |
| `LLAMA_MTP_QBLOCK_SIBLING_TARGET_ROWS_ORACLE_COMPARE` | `0` | diagnostic/probe | target verifier/server sampler | When target-row probing is active, reuses the reserved scratch seq for a second normal row-serial replay of the exact same branch token/position list, compares branch-row top1/top2/watch rank/logit and compact raw-logit digest against the first captured rows, and appends `oracle_*` fields to `MTP_QBLOCK_BRANCH_TARGET_PROBE`. Default off; clean production gates must leave it disabled. |
| `LLAMA_MTP_QBLOCK_SIBLING_TARGET_ROWS_SAMPLER_ORACLE` | `0` | diagnostic/probe | target verifier/server sampler | When target-row probing is active, clones the pre-accept production sampler (`smpl_save`), accepts the selected-prefix tokens plus sampled sibling on the clone only, and samples scratch target branch-row logits (`idx=0`) with `common_sampler_sample`/`common_sampler_accept`. It appends `sampler_oracle_*` fields and token lists to `MTP_QBLOCK_BRANCH_TARGET_PROBE`; mismatches are evidence about branch-tail viability, not production mutations. The same probe also replays the sampler-oracle token list on the scratch target seq and appends `sampler_replay_*` row/logit/state fields, proving the branch sampler tail can be materialized independently of the draft descendant tail. Default off; clean production gates must leave it disabled. |
| `LLAMA_MTP_QBLOCK_SIBLING_TXN_PROOF` | `0` | diagnostic/probe | server sampler/spec state | Emits `MTP_QBLOCK_BRANCH_TXN_PROOF` immediately after the final accepted bundle is materialized and before prompt rewrite, `slot.sampled` update, output emission, sampler replacement, or context cleanup. It compares the ordinary output tail after the sampled sibling with the staged sampled-sibling descendant tokens, checks that scratch target rows/logits, sampler-oracle comparisons, and sampler-oracle branch replay cover that tail, and prints pre-rewrite prompt-tail lists. It also emits `MTP_QBLOCK_BRANCH_TXN_REPLAY_COMPARE`, a metadata-only compare of the ordinary final bundle against the sampler-oracle branch bundle plus the already-captured branch replay state digest; extra post-commit scratch decodes are intentionally avoided because they can perturb recurrent commit state. Branch-token/state gaps are reported as blocker evidence while structural list/captured-branch-state failures remain fail-closed. Both markers hard-code `production_mutation=0`, `same_cycle_replayable=0`, and `safe_commit=0`. Default off; clean production gates must leave it disabled. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_PROOF` | `0` | diagnostic/probe | packed16/V4_144 FA backend | Emits `MTP_QBLOCK_TXN_TAIL_PAGE` from the packed16 DOT4/PDMQ QBlock route. It performs static fail-closed eligibility checks for the transactional packed16-K + V4_144 tail-page ABI, builds a one-page aligned tail descriptor, validates the current QBlock verifier positions as an implicit linear lineage, reports descriptor/lineage status, and also emits `MTP_QBLOCK_TXN_TAIL_WRITE` from packed16-K/V4_144 cache writes. The write marker reports the exact 16-token page base/end, payload slot range, whether the write spans pages, how many page slots are not supplied by the current write (`slots_before`/`slots_after`), and the per-kind row/page/payload/merge-copy byte counts so partial-page staging requirements are explicit. It also emits proof-only `MTP_QBLOCK_TXN_TAIL_STAGE` lines that validate a separate producer-side staging descriptor for one-page read/merge/write metadata, plus `MTP_QBLOCK_TXN_TAIL_BACKEND` lines from the actual packed16-K and V4_144 backend pack ops to record source/destination stride/layout geometry and, for tiny verifier writes, copy at most 8 device-side row indices to prove the backend page/slot range when the stream is not graph-capturing (`idx_capture_skipped=1` during CUDA graph capture). When indices are copied, backend pack ops also emit `MTP_QBLOCK_TXN_TAIL_ADDR` with producer-side payload/scale byte spans for packed16-K and V4_144 page slots. Proof mode is accounting-only: it does not allocate scratch, copy KV data, or alter dispatch. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE` | `0` | guarded prototype gate | packed16/V4_144 FA backend | Enables the first real producer-side tail-page data path for packed16 K and V4_144 V: eligible non-capture, contiguous, one-page partial writes (`2..8` rows, copied backend indices, nonzero merge slots) allocate a 16-token scratch page, copy the existing destination page using the active layout, pack incoming slots into scratch, then commit the full scratch page back to the KV cache. K activation currently requires F32 source rows and packed16-Q8 K; V activation supports F32/F16 source rows and V4_144. Ineligible writes, q8 shadow K source, packed8_q4 K, full-page writes, and graph-capture probes fall back to the existing writer path. This runtime gate is silent by default; set `GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_PROOF=1` as well when marker evidence is needed. FA-side proof can still report `reason=state_sampler_not_wired`, so this is not a full transactional activation until sampler/recurrent commit wiring is added and hash gates pass. |
| `LLAMA_MTP_QBLOCK_BRANCH_NEXTCYCLE_CACHE_COMPARE` | `0` | diagnostic/probe | server sampler/spec state | When transaction replay compare has captured a sampler-oracle branch replay bundle, caches that bundle until the next ordinary cycle and emits `MTP_QBLOCK_BRANCH_NEXTCYCLE_CACHE_COMPARE` before any next target decode. The marker checks whether the current production `id_last` matches the cached branch final token and, for token-matched bundles, compares the cached branch replay recurrent-state bytes with the live production prefix-state bytes. It logs raw hash/first-diff fields plus canonical hashes/first-diff fields that zero the top-level serialized `seq_id` header, and prints small hex windows around raw/canonical diffs. It is metadata/read-only, performs no scratch decode or seq copy, reports prefix/token/state mismatches as evidence, and hard-codes `production_mutation=0`, `same_cycle_replayable=0`, and `safe_commit=0`. Default off; clean production gates must leave it disabled. |
| `LLAMA_MTP_QBLOCK_BRANCH_STATE_SEGMENT_COMPARE` | `0` | diagnostic/probe | server sampler/spec state | When nextcycle cache compare has a prefix/token-matched sampled replay and live state, emits `MTP_QBLOCK_BRANCH_STATE_SEGMENT_COMPARE`. It parses the partial recurrent sequence-state stream, reports cell counts and first cell positions, computes a second canonical digest that zeroes both the top-level serialized `seq_id` and recurrent cell `pos` metadata, and reports the first mismatching recurrent payload component (`R`/`S` layer). It also captures a boundary prefix state just before the sampled replay's final token and compares that to live pre-target state, via `boundary_prefix_*` fields, to distinguish simple one-token staging mismatch from earlier replay/commit divergence. This is the layer64-scratch lesson applied to state bytes without mutating production. Default off; clean production gates must leave it disabled. |
| `LLAMA_MTP_QBLOCK_SIBLING_TARGET_ROWS_SCRATCH_N_SEQ` | unset | diagnostic/probe | target context init | Raises target-context `n_seq_max` above server slot count for scratch-row proofs. The 9B target-row proof runner uses `2` with `--parallel 1`, `--kv-unified`, and doubled `--ctx-size` so production remains slot/seq `0` while scratch replay uses seq `1` in the same KV stream. |
| `LLAMA_MTP_QBLOCK_SIBLING_BRANCH_REPLAY_TRACE` | `0` | diagnostic/probe | server sampler/spec state | Emits `MTP_QBLOCK_BRANCH_REPLAY` after a sibling-hit reject resolves through normal final commit, and if a next normal draft is armed. It records whether the sibling sampled token is present in the emitted commit bundle, `output_tail_after_sampled` token-list fields, and the fail-closed transaction contract for replacing that tail (`transactional_*_required`, target branch rows, draft descendants). Same-cycle replay stays disabled (`same_cycle_replayable=0`, `safe_commit=0`). Branch-desc tracing enables this trace automatically. |

Sibling-row proof runs should use `scripts/hip/run-mtp-qblock-sibling-proof.py`. The proof requires `tree_mask=1`, `ATTENTION_ONLY` sidecar route metadata, `sidecar_in_spec_i_batch=0`, sidecar trimming before `common_speculative_process()`, replay from checkpoint of only sampled + accepted selected-path tokens, and final output with `sidecar_in_ids=0` / `sidecar_in_output=0`. A clean production gate must still run separately with these diagnostic envs disabled. The safe first top1-hit probe uses the 9B runner `scripts/hip/run-mtp-qblock-sibling-logits-probe-9b.py`; it is diagnostic only, emits parent-logit ranks, branch-plan preconditions, branch-state readiness, sampled-sibling root/subtree capture status, proof-only sampled-sibling descendant capture status, proof-only target branch row capture status, and normal next-cycle branch replay status; it must not be used as a speed gate.

## Standard route controls

| Env | Default | Status | Scope | Notes |
| --- | --- | --- | --- | --- |
| `LLAMA_MTP_ENABLE_FA` | `1` | standard | MTP graph | MTP FA route family is enabled by default when flash-attn is active; set `LLAMA_MTP_ENABLE_FA=0`, `LLAMA_MTP_DISABLE_FA=1`, or `GGML_CUDA_ROCM_MTP_DISABLE_FA=1` for controls. |
| `LLAMA_MTP_DISABLE_PACKED16_FA` | `0` | standard | MTP graph | Packed16 FA/QBlock route is enabled by default; set `1` to fall back. |
| `LLAMA_MTP_QBLOCK_TARGET_VERIFY_ACTIVE` | server-scoped | internal/standard | server → graph/backend | Server RAII signal set only around target-verifier decode chunks that contain only verifier rows. This is the production QBlock activation signal; do not export manually or use as a candidate gate. |
| `LLAMA_MTP_QBLOCK_ACTIVE` | unset | legacy/probe-only | MTP graph/probes | Legacy global signal retained for MTP graph/probe paths only. Ordinary decoder graphs ignore it to avoid globally misclassifying unrelated prefill/decode work. Do not use for production speed gates. |
| `LLAMA_MTP_QBLOCK_FUSED_VERIFY` | `0` | proof/opt-in | Qwen35 verifier graph | Enables the first-class fused QBlock/PDMQ verifier route for the non-recurrent tail layer when paired with `LLAMA_MTP_QBLOCK_FUSED_VERIFY_TAIL_LAYER=63`. Replaces deprecated `LLAMA_MTP_QBLOCK_PREFIX_FUSED_VERIFY`. |
| `LLAMA_MTP_QBLOCK_FUSED_VERIFY_TAIL_LAYER` | unset | proof/opt-in | Qwen35 verifier graph | Tail layer selector for fused QBlock/PDMQ verifier proof runs; Qwen35 target trunk tail is `63`. Replaces deprecated `LLAMA_MTP_QBLOCK_PREFIX_FUSED_VERIFY_TAIL_LAYER`. |
| `LLAMA_MTP_QBLOCK_FUSED_VERIFY_LAYER_MIN/MAX` | unset | unsafe prototype filter | Qwen35 verifier graph | Layer-major prototype filter. Keep unset for tail-fused proof route; aliases deprecated `LLAMA_MTP_QBLOCK_PREFIX_FUSED_VERIFY_LAYER_MIN/MAX`. |
| `LLAMA_MTP_QBLOCK_FUSED_VERIFY_LAYER_MAJOR_UNSAFE` | `0` | unsafe prototype | Qwen35 verifier graph | Explicit gate for the old layer-major prototype. Do not use for promotion evidence. Alias: deprecated `LLAMA_MTP_QBLOCK_PREFIX_FUSED_VERIFY_LAYER_MAJOR_UNSAFE`. |
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
| `LLAMA_MTP_QBLOCK_STRICT` | off unless requested | strict | graph/backend | Strict QBlock route gate. Diagnostic/assertion mode, not standard runtime default. Allows row-serial pre-tail layers during fused-tail verifier proof runs; the fused block itself must still satisfy QBlock invariants. |
| `LLAMA_MTP_VERIFY_COMPARE` | off unless requested | strict | verifier | Required for strict compare runs. Diagnostic; do not enable for normal standard route. |
| `LLAMA_MTP_QBLOCK_BLOCK_VERIFY_TRACE` | `0` | diagnostic/proof | server scheduler | Emits compact block-verifier target-decode lines with backend, draft rows, batch rows, forced accepted cap, sampled token, and draft token list. Used by `scripts/hip/run-mtp-qblock-nmax-probe.py`. |
| `LLAMA_MTP_QBLOCK_BLOCK_VERIFY_FORCE_ACCEPTED_ROWS` | unset | diagnostic/proof | server scheduler | Caps accepted draft rows after target sampling and applies the same cap to token oracle. Use only to force partial-accept/rollback cases; not a production knob. Accepted-row state proof is `MTP_VERIFY_COMPARE_POST_PREFIX_COMMIT` when prefix commit is enabled; per-token serial replay is logged separately as diagnostic because multi-token recurrent prefix arithmetic is not byte-identical for accepted>0. |
| `LLAMA_MTP_SPEC_CKPT_HOST` | `0` | diagnostic/proof | server scheduler | Stores speculative checkpoints in host state buffers instead of ON_DEVICE checkpoint buffers. Slow isolation knob for checkpoint load/save investigations; default remains ON_DEVICE. |
| `LLAMA_MTP_HIDDEN_SHIFT_TRACE` / `LLAMA_MTP_HIDDEN_SHIFT_REQUIRE` | `0` | diagnostic/proof | MTP draft process | Traces and optionally aborts on MTP hidden-shift invariants: pending hidden becomes catch-up row0, target verify row0 becomes catch-up row1, and verifier last hidden becomes next pending hidden. |
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
| `GGML_CUDA_ROCM_MTP_QBLOCK_MAX_NQ` | `8` | standard | backend | QBlock max FA rows; falls back to `GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ` when set. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_SHAPE` | auto | shape override | backend | Explicit compact shapes: `1x32`, `2x32`, `4x32`, `8x32`, `16x16`. Artifact name must include forced shape. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_SHAPE_POLICY` | `safe` | experimental | backend | `tetris`/fit policies. Not a promotion gate alone. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_GQA_GROUP` | auto/`1` in strict GQA1 route | shape/dataflow | backend | Allowed values currently `1`, `2`, `4`, `6`. Must be artifact-visible. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_SPLITK` | `1` for non-split | experimental | backend | Split-K route is not strict-state exact unless separately proven. |
| `GGML_CUDA_ROCM_MTP_QBLOCK_SPLITK_MIN_NK` | route default | experimental | backend | Split-K threshold. |
| `GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_LOGICAL_K_STRIDE` | `0` | proof/probe only | backend | Forces PDMQ K sidecar indexing to compact logical rows (`head_stride=nk`). Required only for QBlock prefix/probe equivalence harnesses that compare row-local logical FA views. Do not enable for production forced-PDMQ or default MTP speed gates; production must use capacity-strided sidecar rows to preserve hash `4219d799` on the 44k step44 gate. |
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

These are for packed16/PV4 PWMMA routes and opt-in packed8 prefill candidates, not QBlock route promotion.

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
| `GGML_CUDA_ROCM_V4_K16D16_144_PWMMA_PREFILL` | `0` | legacy candidate | Older V4 PWMMA prefill opt-in; superseded by the PV4-named default candidate/disable pair. |
| `GGML_CUDA_ROCM_V4_K16D16_144_PV4_PWMMA_PREFILL_CANDIDATE` | `1` | main candidate | PV4 V144 PWMMA prefill candidate. Default-on for non-MTP/no-spec prefill when V is `GGML_TYPE_V4_K16D16_144`; still guarded by min/max nq/nk knobs. |
| `GGML_CUDA_ROCM_V4_K16D16_144_PV4_PWMMA_DISABLE` | `0` | rollback/control | Hard-disables PV4 V144 PWMMA selection even when the default candidate would select it. |
| `GGML_CUDA_ROCM_V4_K16D16_144_PV4_PWMMA_PREFILL_MIN_NQ` | `16` | diagnostic/candidate scope | Minimum `nq` for the PV4 PWMMA prefill candidate. Default avoids verifier-like n4 target batches. |
| `GGML_CUDA_ROCM_V4_K16D16_144_PV4_PWMMA_PREFILL_MAX_NQ` | `0` | diagnostic/candidate scope | Optional maximum `nq`; `0` means unlimited. Used to isolate tail/small-batch drift. |
| `GGML_CUDA_ROCM_V4_K16D16_144_PV4_PWMMA_PREFILL_MAX_NK` | `0` | diagnostic/candidate scope | Optional maximum `nk`; `0` means unlimited. Used to isolate prompt chunk drift. |
| `GGML_CUDA_ROCM_PACKED8_Q4_144_PWMMA_PREFILL_CANDIDATE` | `0` | experimental opt-in | Enables packed8-q4 K prefill on the BM64 i8QK/PV-WMMA DBV route by expanding packed int4 K into int8 shared memory once per K tile. Packed8 K storage is standard, but this PWMMA prefill implementation stays opt-in; decode/verify stay on PDMQ. |
| `GGML_CUDA_ROCM_PACKED8_Q4_PWMMA_PREFILL` | `0` | alias/experimental opt-in | Short alias for `GGML_CUDA_ROCM_PACKED8_Q4_144_PWMMA_PREFILL_CANDIDATE`. |
| `GGML_CUDA_ROCM_PACKED8_Q4_PWMMA_PREFILL_MIN_NQ` | `16` | experimental scope | Minimum query rows for packed8 PWMMA prefill. Keeps verifier/small-Q batches off this route. |
| `GGML_CUDA_ROCM_PACKED8_Q4_PWMMA_PREFILL_MIN_NK` | `512` | experimental scope | Minimum K rows for packed8 PWMMA prefill. Default matches 512-token prompt chunks. |
| `LLAMA_MTP_PACKED16_HOTCOLD_K` | `0` | deprecated experiment | Old hot/cold F16-K-shadow path. Ignored/fail-closed unless `LLAMA_MTP_PACKED16_HOTCOLD_K_EXPERIMENT=1` is also set. Current standard route uses packed16 PDMQ K with PV4/V144 V, not hot/cold. |
| `LLAMA_MTP_PACKED16_HOTCOLD_K_EXPERIMENT` | `0` | extra experiment opt-in | Required extra gate for the old hot/cold path. Keep unset for current PV4/V144 standard route work. |
| `LLAMA_MTP_PACKED16_HOTCOLD_K_WINDOW` | `256` | deprecated experiment scope | Maximum hot K window for the old hot/cold F16-K-shadow experiment. |
| `LLAMA_MTP_PACKED16_HOTCOLD_K_LEGACY_Q4V` | `0` | legacy opt-in | Allows old `V=q4_0` hot/cold eligibility, but only with `LLAMA_MTP_PACKED16_HOTCOLD_K_EXPERIMENT=1`. Keep unset for current PV4/V144 standard route work. |

## Legacy-prefix audit policy

Prefix registration is disabled. Existing legacy/debug families may remain in source only if their concrete env names appear as exact rows in this registry; otherwise `scripts/hip/qblock-env-audit.py` reports them as unregistered cleanup debt.

## Build gates

| CMake option | Default | Scope | Notes |
|---|---:|---|---|
| `GGML_HIP_PDMQ_COMPILE_LEGACY_Q4V` | `OFF` | PDMQ compile matrix | Compiles old `GGML_TYPE_Q4_0` V PDMQ debug/fallback cells. Keep OFF for the standard PV4/V144 build; enable only for legacy debugging/control builds. |
| `GGML_HIP_PDMQ_COMPILE_V4_144_PV4` | `ON` | PDMQ compile matrix | Compiles scalar PV4/V144 PDMQ cells for the standard PV4/V144 path. This is separate from the explicit DP16-DOT/profile switch and can be hard-disabled at runtime with `GGML_CUDA_ROCM_V4_K16D16_144_PV4_DISABLE=1`. |
| `GGML_HIP_COMPILE_STANDARD_Q8_FATTN` | `OFF` | HIP compile trim | Compiles legacy standard `GGML_TYPE_Q8_0` FlashAttention VEC instances (`q8_0/q8_0`, `q8_0/q4_0`). Keep OFF in the packed8-standard dev build; enable only for old q8 controls. |

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
