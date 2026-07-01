# JetSpec P5AH draft-head top-k construction readiness descriptor

Status: inert readiness descriptor only. This is not production source approval,
not draft-head graph execution, not a logits walk, not token emission, not a KV
mutation, not a server route, and not compiled by CMake.

## Scope

P5AH stays entirely under `experiments/jetspec/`. It records the readiness
contract for the future transition from the P5AE/P5AF/P5AG synthetic top-k ABI
chain to real draft-head full-vocab logits and top-k construction. It does not
edit `common/speculative.cpp`, `docs/speculative.md`, `src/`, `include/`,
`tools/server/`, repository `tests`, `examples`, `pocs`, `ggml/src`, top-level
`CMakeLists.txt`, or production `.cmake` files.

P5AH keeps the current runtime boundary: `runtime_supported=false`, model-only
JetSpec draft-head binding with `ctx_dft=nullptr`, no draft context, no `llama_decode`, no draft-head graph execution, no logits buffer read, no sampler,
no target logits walk, no target accept walk, no token commit, no hidden/KV
commit, no rejected-branch discard, no KV mutation, no visible publish, no CUDA,
no server behavior, no public API, no performance claim, no promotion claim, and
no draft tokens.

## Chain prerequisite

Future real top-k construction must require the existing synthetic ABI chain:

```bash
LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1
LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1
LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1
LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1
```

The future source slice must fail closed if P5AG is not ready, if the P5AG hash
is missing, or if `topk_abi_root_tail_conflict()` would reject the mixed root-tail
configuration. P5AH itself performs no source wiring and therefore has no new
env gate in production code.

## Future top-k contract captured here

`jetspec_draft_head_topk_readiness.py` validates the descriptor fixture and
reports `draft_head_topk_readiness_verified_not_executed`. The fixture fixes the
future real-top-k contract without executing it:

- `future_logits_source=draft_head_full_vocab_logits`
- forbidden sources: `target_logits`, `sampler`, `synthetic_full_vocab_softmax`,
  and `topk_only_renormalization`
- `target_layer_ids=[1,10,19,28,37]`
- `target_tap_width=10240`
- `embedding_length=2048`
- `vocab_size=248320`
- `topk_width=2`
- `tree_depth=1`
- `planned_draft_head_logits_rows=1`
- `actual_verified_logits_rows=0`
- `candidate_nodes=2`
- `output_tree_nodes=3`
- `parent_logits_rows=[{logits_row:0,parent_node:0,candidate_nodes:[1,2]}]`
- `rank_semantics=rank_stable_descending_logprob`
- `actual_accepted_nodes=0`
- `correction_token_present=0`

The single planned row is a readiness contract for a future explicit approval:
row 0 would be the draft-head full-vocab logits row for parent node 0 and would
produce two child candidates. P5AH keeps that planned row non-executable and
requires `actual_verified_logits_rows=0`.

## Fail-closed modes

The descriptor names future fail-closed modes before any production code is
allowed:

- `missing_p5ag_accept_boundary`
- `missing_p5ag_hash`
- `runtime_supported_true`
- `ctx_dft_non_null`
- `draft_context_created`
- `draft_head_graph_execution_attempted`
- `llama_decode_attempted`
- `draft_logits_rows_nonzero_without_approval`
- `target_logits_source_attempted`
- `sampler_source_attempted`
- `synthetic_source_used_as_real_topk`

## Verification evidence

Representative commands:

```bash
python3 experiments/jetspec/jetspec_draft_head_topk_readiness.py \
  --fixture experiments/jetspec/fixtures/jetspec_draft_head_topk_readiness_smoke.json
python3 experiments/jetspec/validate_p5ah_draft_head_topk_readiness.py --json
python3 -m unittest experiments/jetspec/test_p5ah_draft_head_topk_readiness.py
python3 experiments/jetspec/run_all_jetspec_contracts.py --json
```

Expected outcome: the fixture reports
`draft_head_topk_readiness_verified_not_executed`, the validator passes,
aggregate contracts pass, and JetSpec remains explicit-opt-in/non-drafting.

## Blocked next work

Real draft-head logits/top-k production construction remains blocked until a
separate explicit production-source approval. The blocked surface includes
creating `ctx_dft`, executing the JetSpec draft-head graph, reading logits,
selecting real top-k children, target accept walks, token emission, KV/hidden
commit or rollback mutation, server behavior, public API, CMake wiring, kernels,
performance claims, and promotion claims.
