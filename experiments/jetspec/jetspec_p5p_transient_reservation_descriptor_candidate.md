# JetSpec P5P transient-reservation descriptor candidate

Status: approved bounded production-source slice, default-off and non-drafting. This file is not included by CMake and no llama.cpp runtime reads it.

P5P is the next bounded source slice after P5O. It wires only a diagnostic transient-reservation descriptor for the P5M transaction phase `reserve_transient_tree_pages` into the existing explicit `draft-jetspec` route in `common/speculative.cpp` and documents it in `docs/speculative.md`.

P5P records intent only. It does not perform real reservation:

- actual_pages_reserved=0
- no real page reservation
- no llama_kv_cache primitive
- no llama_kv_cache_jetspec primitive
- no seq_cp, seq_rm, or seq_import_physical mapping
- no build_tree
- no build_verify_mask
- no accept_path
- no commit_tokens
- no hidden/KV survivor commit
- no rejected branch discard
- no publish_post_commit_state
- no real rollback mutation
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

## Transient-reservation descriptor

After P5O snapshot readiness and P5N transaction scaffold readiness, P5P builds a descriptor-only record:

- `JETSPEC_TRANSIENT_RESERVATION_PHASE = "reserve_transient_tree_pages"`
- `JETSPEC_TRANSIENT_RESERVATION_ROLLBACK_POINT = "after_reserve"`
- `JETSPEC_TRANSIENT_RESERVATION_DESCRIPTOR = "transient_reservation_descriptor_only"`
- `transient_reservation_descriptor_ready`
- `transient_reservation_ready`
- `transient_reservation_hash_last`
- `n_transient_reservation_descriptors`
- `transient_reservation_node_budget_last`
- `transient_reservation_actual_pages_last`
- `invalid_transient_reservation_descriptor`

The descriptor hash includes the pre-round snapshot hash, transaction plan hash, sequence id, prompt token count, target tap hash, target tap row count, Qwen3.6 draft block size, the phase string `reserve_transient_tree_pages`, and rollback point `after_reserve`. The descriptor bounds node budget to `JETSPEC_QWEN36_DRAFT_BLOCK_SIZE` and keeps `transient_reservation_actual_pages_last = 0`.

## Fail-closed behavior

P5P fails closed with `invalid_transient_reservation_descriptor` if any prerequisite descriptor is missing or inconsistent, if target tap row state does not match cached target tap rows, or if the node budget is empty or exceeds the draft block size.

Trace output reports:

- `transient_reservation_ready`
- `transient_reservation_hash`
- `transient_reservation_phase=reserve_transient_tree_pages`
- `rollback_point=after_reserve`
- `transient_tree_node_budget`
- `actual_pages_reserved=0`
- `pre_publish_visible_state_unmodified=1`
- `no_tree_build=1`
- `no_verify_mask=1`
- `no_kv_mutation=1`
- `no_publish=1`
- `no_draft_tokens=1`

## Still blocked

P5P does not implement real `reserve_transient_tree_pages`, real KV/page reservation, `llama_kv_cache_jetspec_*` primitives, tree construction, verify mask construction, accept path runtime, token commit, hidden/KV survivor commit, rejected branch discard, publish/post-commit state, rollback mutation, draft-head graph execution, draft token emission, CUDA dispatch, server behavior, repository tests, examples, pocs, `ggml/src`, production CMake wiring, public API, performance claims, or promotion claims.

## Validation

```bash
python3 experiments/jetspec/validate_p5p_transient_reservation_descriptor.py
python3 experiments/jetspec/test_p5p_transient_reservation_descriptor.py
python3 experiments/jetspec/run_all_jetspec_contracts.py --json
```
