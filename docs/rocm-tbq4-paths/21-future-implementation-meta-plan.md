# Future implementation meta-plan: compressed-KV ROCm FA

## Objective

Prepare future compressed-KV FlashAttention implementation work so each slice is source-backed, test-first where practical, production-safe by default, and reversible without changing the stable VEC route.

## Frozen baseline

- Repo: `/home/mrtrent/llama.cpp-mtp-tbq4-rdna3`.
- Branch: `tbq4-rdna3-experiment`.
- Baseline HEAD: `173f92d00` (`test(triton): tighten llama.cpp parity contracts`), matching `fork/tbq4-rdna3-experiment`.
- Known-good tag: `known-good/compressed-kv-fa-wmma-20260515` resolves to `42ade94b7`.
- Baseline tracked diff: empty.
- Unrelated/untracked exclusion: `docs/needle-action-drafter/` remains untouched.

## Source-backed contract before coding

Implementation slices must cite the exact source/test contract they rely on before editing production code.

- Dispatch boundary: `ggml/src/ggml-cuda/fattn.cu` keeps compressed KV on VEC by default, rejects mixed K/V types by default except `TBQ4_0/Q8_0`, and gates experimental routes behind `TBQ4_WMMA_FATTN` or `COMPRESSED_KV_WMMA_FATTN`.
- Runtime defaults: `/home/mrtrent/.local/bin/llama-server-wrapper` exports TBQ4 set-rows/MMQ/layer-adaptive flags only; `/home/mrtrent/.pi/agent/llama-swap-config.yaml` does not export experimental WMMA FA gates.
- Layout boundary: `src/llama-kv-cache.cpp::get_k/get_v` separates active `n_kv` views from cache `get_size()` stream stride; `set_input_k_idxs/set_input_v_idxs` map rows as `stream * get_size() + slot` for non-transposed FA paths.
- Format-domain boundary: TBQ4 is FWHT-domain with explicit Q/O rotation; Planar/Iso are original-domain materializers.
- Triton boundary: `experiments/compressed_kv_triton/` is a correctness/design oracle only, not CMake, wrapper, or `llama-server` runtime.
- Promotion boundary: paged/block-table C++ mapping stays deferred until the parity lane proves the row mapping and a narrow adapter is designed.

## Perplexity research lane

Use the installed Pi Perplexity tool directly, with cached session-token/OAuth auth, for external evidence discovery only.

- Direct tool invocation confirmed through the installed `pi-perplexity` registered `perplexity_search` implementation.
- Request IDs captured during this planning pass:
  - `056fc1dd-0691-4b8c-996f-b85386bb1d87`
  - `bce1b6d4-4cb7-46d6-9183-c27f1a346579`
- Useful discovered themes: ROCm/RDNA3 FlashAttention support is uneven; Triton-on-ROCm can be useful for algorithm checks; paged KV/block-table details must be verified from primary framework/source code, not accepted from search summaries.
- Caveat: Perplexity returned some weak or misaligned citations. Treat its output as search lead generation, then verify against local source, primary docs, or exact upstream code before changing this branch.

Required future query pattern:

```text
Primary sources only: <feature> ROCm HIP Triton llama.cpp <specific files or APIs>. Return GitHub repo/docs/PR URLs and separate evidence from recommendations.
```

## Independent lanes for future slices

1. Scout lane: read-only source map, contract citations, affected files, rollback path.
2. Research lane: Perplexity/source research when external patterns matter; primary-source verification required.
3. Test lane: failing or protective contract checks before production edits when practical.
4. Implementation lane: one minimal reversible code/config/doc slice.
5. Review lane: adversarial check for unsupported claims, dispatch drift, layout/domain mistakes, and scope creep.
6. Docs lane: concise checkpoint updates only; do not touch `docs/needle-action-drafter/`.

## Test/contract rule

Before production edits, either add a failing/protective check or name the existing gate that proves the contract:

- `experiments/compressed_kv_triton/run_all.sh`
- `/home/mrtrent/miniconda3/envs/LLM/bin/python experiments/compressed_kv_triton/run_all_json.py`
- `scripts/hip/check-compressed-kv-fa-invariants.sh`
- `git diff --check`

For runtime dispatch/materialization changes, add the safe smoke harness only after static/parity gates pass.

## Production defaults rule

- Default compressed-KV FA route remains VEC.
- `TBQ4_WMMA_FATTN=1` and `COMPRESSED_KV_WMMA_FATTN=1` remain opt-in only.
- No Triton/Python production dependency.
- Mixed `TBQ4_0/Q8_0` remains VEC until a Q8 V loader/domain policy exists.
- Mixed Planar/Iso remains rejected by default unless the production dispatch contract intentionally changes and tests are updated first.

## Implementation rule

Each future slice should be one small reversible change:

1. freeze baseline;
2. write/update the contract check;
3. edit the minimum source/doc surface;
4. run the validation ladder;
5. review and converge only on failed lanes;
6. commit/push only the scoped files.

## Review and convergence

- Treat reviewer/search claims as hypotheses until verified with code/tests.
- Reassign or redo only the failing lane; do not rerun the whole plan blindly.
- Stop after repeated failures in the same class and re-scout instead of broadening scope.

## Commit and documentation rule

- Use focused commits such as `docs: add compressed-kv future implementation plan` or `test(triton): extend <contract>`.
- Do not stage unrelated files or `docs/needle-action-drafter/`.
- Documentation must state non-goals, env gates, rollback, and validation evidence.
- Push branch state to `fork`; no PR is required unless explicitly requested.

## Record rule

After each durable slice, update `/home/mrtrent/.harness/state/anchor.md` with branch HEAD, changed artifacts, validation, and blockers.
