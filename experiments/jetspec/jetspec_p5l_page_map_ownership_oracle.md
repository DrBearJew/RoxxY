# P5L JetSpec page-map ownership oracle

Status: inert page-map ownership oracle only. This file is not included by CMake and no llama.cpp runtime reads it.

## Boundary

P5L translates the QBlock/PageAttention page-map descriptor work into a JetSpec ownership oracle for future tree KV mutation approval. It reports `page_map_ownership_oracle_verified_not_executed` and keeps the boundary `page_map_oracle_only`.

P5L does not approve implementation. It performs no llama.cpp runtime work:

- no llama_context creation
- no draft-head graph execution
- no draft tokens emitted
- no real KV mutation
- no server route
- runtime_supported=false
- no performance claim
- no promotion claim

All files stay under `experiments/jetspec/`:

- `page_map_ownership_oracle.py`
- `fixtures/page_map_ownership_oracle_smoke.json`
- `fixtures/page_map_ownership_oracle_smoke.out.json`
- `validate_p5l_page_map_ownership_oracle.py`
- `test_p5l_page_map_ownership_oracle.py`

## Oracle checks

The oracle is executable fixture validation only. It rejects the fixture unless all required cases pass.

### `hidden_kv_survivor_page_ownership`

Required invariants:

- accepted path pages map to [root | accepted] only
- correction hidden deferred
- accepted path physical gather/compact explicit

This is the page-map form of P5K `hidden_kv_survivor_commit`. The future implementation name remains design-only:

- `llama_kv_cache_jetspec_commit_page_survivor_path_candidate`

### `rejected_branch_page_unreachable`

Required invariants:

- rejected transient pages unreachable after commit
- accepted path cannot read rejected siblings or descendants
- rollback restores pre-round page snapshot

This is the page-map form of P5K `rejected_branch_discard`. The future implementation name remains design-only:

- `llama_kv_cache_jetspec_discard_rejected_tree_pages_candidate`

### `cross_sequence_page_isolation`

Required invariants:

- other-sequence pages unchanged
- no duplicate mutable physical page ownership
- rollback preserves other sequences

This is the page-map form of P5K `cross_sequence_isolation`. The future implementation names remain design-only:

- `llama_kv_cache_jetspec_validate_page_ownership_oracle_candidate`
- `llama_kv_cache_jetspec_reserve_transient_tree_pages_candidate`

## Existing helpers remain non-exact

P5L keeps these helpers as audited non-exact helpers and forbids silent promotion into exact page ownership mappings:

- `seq_cp`
- `seq_rm`
- `seq_import_physical`

Any production mapping must name a real future `llama_kv_cache_jetspec_*` primitive or remain `design_only_missing_implementation`. P5L does not convert existing helpers into a commit/discard/isolation primitive.

## QBlock/PageAttention safety lessons

P5L preserves the useful part of the QBlock PageAttention work as an oracle rather than a speed feature:

- identity maps are only parity/oracle cases
- visible noncanonical owned overlays fail closed until proven with a full current-K map and commit/rollback proof
- a full current-K map is required before non-identity ownership can be accepted
- canonical write-through remains required unless separately proven safe
- the oracle makes no performance or promotion claim

## Acceptance commands

```bash
python3 experiments/jetspec/page_map_ownership_oracle.py \
  --fixture experiments/jetspec/fixtures/page_map_ownership_oracle_smoke.json
python3 experiments/jetspec/validate_p5l_page_map_ownership_oracle.py
python3 experiments/jetspec/test_p5l_page_map_ownership_oracle.py
python3 experiments/jetspec/run_all_jetspec_contracts.py --json
```

Expected result:

- fixture reports `page_map_ownership_oracle_verified_not_executed`
- duplicate physical page ownership fails
- rejected branch reachability fails
- accepted path without root fails
- implicit `seq_cp` / `seq_rm` / `seq_import_physical` mapping fails
- cross-sequence mutation fails
- runtime boundary remains false for runtime_supported=false, no llama_context creation, no draft-head graph execution, no draft tokens emitted, no real KV mutation, no server route, no performance claim, and no promotion claim

## Still blocked

P5L does not unblock production tree runtime. Real KV page reservation, survivor commit, rejected-page discard, cross-sequence ownership assertions, CUDA dispatch, server behavior, repository tests, examples, pocs, `ggml/src`, and CMake wiring remain blocked until explicit runtime approval.
