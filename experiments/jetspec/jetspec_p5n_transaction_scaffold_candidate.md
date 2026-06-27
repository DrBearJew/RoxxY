# JetSpec P5N transaction-plan scaffold candidate

Status: approved bounded production-source slice, default-off and non-drafting. This file is not included by CMake and no llama.cpp runtime reads it.

P5N is the first tree-runtime-approved source slice after P5M. It wires only a transaction-plan scaffold into the existing explicit `draft-jetspec` route in `common/speculative.cpp` and documents it in `docs/speculative.md`.

P5N still performs no draft-head runtime work:

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

## Required default-off gates

P5N remains reachable only through the existing gates:

- `LLAMA_JETSPEC_EXPERIMENTAL=1`
- explicit `--spec-type draft-jetspec`
- explicit draft-head path through the existing draft model option
- P5F draft-head/target binding preflight
- P5B target hidden tap side channel

Metadata-only preview files still reject as `preview_not_allowed` by default and `unsupported_runtime` with `LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1`; `runtime_supported=false` is still not runnable.

## Transaction scaffold contract

The scaffold records the P5M transaction phase order in production state after target taps are captured:

1. `snapshot_pre_round`
2. `reserve_transient_tree_pages`
3. `build_tree`
4. `build_verify_mask`
5. `accept_path`
6. `commit_tokens`
7. `commit_hidden_kv_survivors`
8. `discard_rejected_branches`
9. `publish_post_commit_state`

It also records the required rollback failpoints:

- `after_reserve`
- `after_build_tree`
- `after_verify_mask`
- `after_accept`
- `after_token_commit`
- `after_hidden_kv_commit`
- `after_rejected_discard`

The source keeps `JETSPEC_TRANSACTION_PHASE_COUNT = 9`, `JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT = 7`, `JETSPEC_TRANSACTION_PHASE_ORDER`, `JETSPEC_TRANSACTION_ROLLBACK_POINTS`, `build_transaction_plan_scaffold`, `transaction_plan_scaffold_ready`, `transaction_plan_ready`, and `transaction_plan_hash_last` as diagnostic scaffold state only.

## Fail-closed behavior

The route disables JetSpec before drafting if target tap row state cannot seed the transaction scaffold. The failure reason is `invalid_transaction_plan`.

The trace line explicitly reports:

- `transaction_plan_ready`
- `transaction_plan_hash`
- `transaction_plan_phases`
- `rollback_points`
- `no_kv_mutation=1`
- `no_publish=1`
- `no_draft_tokens=1`

## Still blocked

P5N does not implement real tree build, verify, accept, token commit, hidden/KV survivor commit, rejected-branch discard, post-commit publish, real rollback, draft-head graph execution, CUDA dispatch, server behavior, repository tests, examples, pocs, `ggml/src`, production CMake wiring, or public API.

## Validation

```bash
python3 experiments/jetspec/validate_p5n_transaction_scaffold.py
python3 experiments/jetspec/test_p5n_transaction_scaffold.py
python3 experiments/jetspec/run_all_jetspec_contracts.py --json
```

Expected:

- P5N tokens appear only in `common/speculative.cpp`, `docs/speculative.md`, and `experiments/jetspec/`
- no CMake hits
- no `llama_decode`, `llama_graph`, `tree_accept`, `llama_kv_cache`, or draft-token emission in the `draft-jetspec` implementation slice
- docs keep the no-draft/no-KV/no-CUDA/no-server boundary
