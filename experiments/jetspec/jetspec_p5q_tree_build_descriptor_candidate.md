# JetSpec P5Q tree-build descriptor candidate

Status: approved bounded production-source slice, default-off and non-drafting. This file is not included by CMake and no llama.cpp runtime reads it.

P5Q is the next bounded source slice after P5P. It wires only a diagnostic tree-build descriptor for the P5M transaction phase `build_tree` into the existing explicit `draft-jetspec` route in `common/speculative.cpp` and documents it in `docs/speculative.md`.

P5Q records ABI intent only. It does not construct a tree:

- actual_tree_nodes=0
- no real tree build
- no tree arrays
- no token_ids / parent_indices / depth / cum_logprob production vectors
- no verify mask
- no accept path runtime
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

## Tree-build descriptor

After P5O snapshot readiness, P5N transaction scaffold readiness, and P5P transient-reservation descriptor readiness, P5Q builds a descriptor-only record:

- `JETSPEC_TREE_BUILD_PHASE = "build_tree"`
- `JETSPEC_TREE_BUILD_ROLLBACK_POINT = "after_build_tree"`
- `JETSPEC_TREE_BUILD_DESCRIPTOR = "tree_build_descriptor_only"`
- `JETSPEC_TREE_ROOT_PARENT = -1`
- `JETSPEC_TREE_ROOT_DEPTH = 0`
- `tree_build_descriptor_ready`
- `tree_build_descriptor_hash_last`
- `n_tree_build_descriptors`
- `tree_build_seq_id_last`
- `tree_build_node_budget_last`
- `tree_build_root_parent_last`
- `tree_build_root_depth_last`
- `tree_build_actual_nodes_last`
- `invalid_tree_build_descriptor`

The descriptor hash includes the pre-round snapshot hash, transaction plan hash, transient reservation hash, sequence id, prompt token count/hash, target tap hash, target tap row metadata, node budget, `actual_tree_nodes=0`, the phase string `build_tree`, and rollback point `after_build_tree`. The descriptor preserves the P5G parent-before-child ABI boundary by recording root parent `-1` and root depth `0`, but it does not allocate or populate any tree arrays.

## Fail-closed behavior

P5Q fails closed with `invalid_tree_build_descriptor` if any prerequisite descriptor is missing or inconsistent, if `actual_pages_reserved` is nonzero, if target tap row state does not match cached target tap rows, or if the planned node budget is empty or exceeds the draft block size.

Trace output reports:

- `tree_build_descriptor_ready`
- `tree_build_descriptor_hash`
- `tree_build_phase=build_tree`
- `rollback_point=after_build_tree`
- `planned_tree_node_budget`
- `actual_tree_nodes=0`
- `tree_build_descriptor_only=1`
- `root_parent=-1`
- `root_depth=0`
- `pre_publish_visible_state_unmodified=1`
- `no_real_tree_build=1`
- `no_tree_arrays=1`
- `no_verify_mask=1`
- `no_accept=1`
- `no_kv_mutation=1`
- `no_publish=1`
- `no_draft_tokens=1`

## Still blocked

P5Q does not implement real tree construction, top-k tree expansion, `token_ids`, `parent_indices`, `depth`, or `cum_logprob` production arrays, verify mask construction, accept path runtime, token commit, hidden/KV survivor commit, rejected branch discard, publish/post-commit state, rollback mutation, draft-head graph execution, draft token emission, CUDA dispatch, server behavior, repository tests, examples, pocs, `ggml/src`, production CMake wiring, public API, performance claims, or promotion claims.

## Validation

```bash
python3 experiments/jetspec/validate_p5q_tree_build_descriptor.py
python3 experiments/jetspec/test_p5q_tree_build_descriptor.py
python3 experiments/jetspec/run_all_jetspec_contracts.py --json
```
