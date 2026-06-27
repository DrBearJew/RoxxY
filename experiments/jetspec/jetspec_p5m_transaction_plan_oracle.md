# P5M JetSpec transaction/failpoint plan oracle

Status: inert transaction/failpoint plan oracle only. This file is not included by CMake and no llama.cpp runtime reads it.

P5M consolidates P5G/P5H/P5I/P5K/P5L into an ordered transaction contract for a future JetSpec tree-runtime slice. It reports `transaction_plan_oracle_verified_not_executed` and keeps the boundary `transaction_plan_oracle_only`.

P5M does not approve implementation. It performs no llama.cpp runtime work:

- no llama_context creation
- no draft-head graph execution
- no draft tokens emitted
- no real KV mutation
- no server route
- runtime_supported=false
- no CUDA dispatch
- no performance claim
- no promotion claim

## Files

- `transaction_plan_oracle.py`
- `fixtures/transaction_plan_oracle_smoke.json`
- `fixtures/transaction_plan_oracle_smoke.out.json`
- `validate_p5m_transaction_plan_oracle.py`
- `test_p5m_transaction_plan_oracle.py`

## Required transaction phases

The oracle requires an exact ordered transaction plan:

1. `snapshot_pre_round`
2. `reserve_transient_tree_pages`
3. `build_tree`
4. `build_verify_mask`
5. `accept_path`
6. `commit_tokens`
7. `commit_hidden_kv_survivors`
8. `discard_rejected_branches`
9. `publish_post_commit_state`

Every phase before `publish_post_commit_state` must preserve `pre_publish_visible_state_unmodified=true`. Only `publish_post_commit_state` may set `publishes_visible_state=true`, and it must require token commit, hidden/KV survivor commit, and rejected branch discard.

## Required failpoints

P5M requires rollback points after each mutating/planning phase:

- `after_reserve`
- `after_build_tree`
- `after_verify_mask`
- `after_accept`
- `after_token_commit`
- `after_hidden_kv_commit`
- `after_rejected_discard`

Each failpoint must restore the pre-round snapshot, including committed tokens, hidden/KV pages, and other-sequence pages. Rejected pages must remain unreachable and no partial publish may be visible.

## Visibility contract

The fixture must prove:

- no committed token visibility before publish
- no hidden/KV visibility before publish
- no page-map visibility before publish
- publish after commit and discard only
- rollback clears transient state
- post-publish committed tokens are `[accepted draft tokens | correction]`
- post-publish hidden/KV survivors are `[root | accepted]`
- correction hidden deferred
- rejected branches unreachable
- other-sequence pages unchanged
- no duplicate mutable physical page ownership

## Candidate primitives remain design-only

P5M names only design-only missing implementation candidates:

- `llama_kv_cache_jetspec_validate_page_ownership_oracle_candidate`
- `llama_kv_cache_jetspec_reserve_transient_tree_pages_candidate`
- `llama_kv_cache_jetspec_commit_page_survivor_path_candidate`
- `llama_kv_cache_jetspec_discard_rejected_tree_pages_candidate`
- `llama_kv_cache_jetspec_rollback_tree_transaction_candidate`

All must remain `design_only_missing_implementation`, `implementation_approved=false`, `touches_production_now=false`, `runtime_executed_now=false`, `claims_performance=false`, and `claims_promotion=false`.

## Existing helpers remain non-exact

P5M keeps these helpers as audited non-exact helpers and forbids silent promotion into transaction commit/rollback primitives:

- `seq_cp`
- `seq_rm`
- `seq_import_physical`

Any production transaction mapping must name a real future `llama_kv_cache_jetspec_*` primitive or remain `design_only_missing_implementation`. P5M does not convert existing helpers into ordered commit, rollback, discard, or publish primitives.

## Source backing

P5M requires explicit source backing from:

- P5G tree-runtime readiness
- P5H KV/hidden commit readiness
- P5I tree-runtime approval packet
- P5K KV ownership primitive design
- P5L page-map ownership oracle

This connects tree ABI, abstract KV ownership, approval gates, missing primitive design, and page-map ownership into one transaction/failpoint gate.

## Run

```bash
python3 experiments/jetspec/transaction_plan_oracle.py \
  --fixture experiments/jetspec/fixtures/transaction_plan_oracle_smoke.json
python3 experiments/jetspec/validate_p5m_transaction_plan_oracle.py
python3 experiments/jetspec/test_p5m_transaction_plan_oracle.py
```

Expected:

- fixture reports `transaction_plan_oracle_verified_not_executed`
- transaction phases are exactly ordered
- rollback points are complete
- pre-publish visible state is unchanged
- publish only happens after commit/discard
- rejected branches remain unreachable
- duplicate mutable physical page ownership fails
- implicit `seq_cp` / `seq_rm` / `seq_import_physical` mapping fails
- runtime boundary remains false for runtime_supported=false, no llama_context creation, no draft-head graph execution, no draft tokens emitted, no real KV mutation, no server route, no performance claim, and no promotion claim

## Still blocked

P5M does not unblock production tree runtime. Real transaction state, draft-head graph execution, real KV mutation, CUDA dispatch, server behavior, repository tests, examples, pocs, `ggml/src`, production CMake wiring, and public API remain blocked until explicit tree-runtime approval.
