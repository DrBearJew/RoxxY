# JetSpec P5O pre-round snapshot descriptor candidate

Status: approved bounded production-source slice, default-off and non-drafting. This file is not included by CMake and no llama.cpp runtime reads it.

P5O is the next tree-runtime-approved source slice after P5N. It wires only the first P5M transaction phase, `snapshot_pre_round`, into the existing explicit `draft-jetspec` route in `common/speculative.cpp` and documents it in `docs/speculative.md`.

P5O still performs no next-phase runtime work:

- no reserve_transient_tree_pages
- no build_tree
- no build_verify_mask
- no accept_path
- no commit_tokens
- no hidden/KV survivor commit
- no rejected branch discard
- no publish_post_commit_state
- no real rollback
- no draft-head graph execution
- no draft tokens emitted
- no real KV mutation
- no CUDA dispatch
- no server route
- no public API
- no CMake wiring
- no performance claim
- no promotion claim

## Approved touched files

- `common/speculative.cpp`
- `docs/speculative.md`

## Snapshot descriptor

`begin(seq_id, prompt)` now builds an immutable pre-round descriptor when the existing explicit `draft-jetspec` route is active:

- `pre_round_snapshot_ready`
- `pre_round_snapshot_hash_last`
- `pre_round_seq_id_last`
- `pre_round_prompt_tokens_last`
- `pre_round_prompt_hash_last`
- `n_pre_round_snapshots`
- `JETSPEC_PRE_ROUND_SNAPSHOT_PHASE = "snapshot_pre_round"`

The descriptor validates `seq_id >= 0 && seq_id < n_seq`, hashes the prompt tokens, hashes the descriptor, and records phase `pre_round_snapshot_ready`. It does not reserve pages, build a tree, build a verify mask, accept, commit, publish, mutate KV, or emit drafts.

## Fail-closed behavior

P5O adds `invalid_pre_round_snapshot`. Invalid `seq_id` disables JetSpec in `begin()`. The P5N transaction scaffold requires `pre_round_snapshot_ready` and includes `pre_round_snapshot_hash_last`, `pre_round_seq_id_last`, `pre_round_prompt_tokens_last`, and `pre_round_prompt_hash_last` in `transaction_plan_hash_last`.

Trace output reports:

- `pre_round_snapshot_ready`
- `pre_round_snapshot_hash`
- `pre_round_seq_id`
- `pre_round_prompt_tokens`
- `pre_round_prompt_hash`
- `transaction_phase=snapshot_pre_round`
- `no_reserve=1`
- `no_tree_build=1`
- `no_verify_mask=1`
- `no_kv_mutation=1`
- `no_publish=1`
- `no_draft_tokens=1`

## Still blocked

P5O does not implement `reserve_transient_tree_pages`, `build_tree`, `build_verify_mask`, `accept_path`, token commit, hidden/KV survivor commit, rejected branch discard, post-commit publish, real rollback, draft-head graph execution, draft token emission, real KV mutation, CUDA dispatch, server behavior, repository tests, examples, pocs, `ggml/src`, production CMake wiring, or public API.

## Validation

```bash
python3 experiments/jetspec/validate_p5o_pre_round_snapshot.py
python3 experiments/jetspec/test_p5o_pre_round_snapshot.py
python3 experiments/jetspec/run_all_jetspec_contracts.py --json
```
