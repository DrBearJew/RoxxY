# JetSpec P5Y root-only verify-mask runtime candidate

Status: approved bounded production-source slice. This is the first real
verify-mask materialization step, but it is default-off, root-only, non-drafting,
and stops before accept/commit/publish.

## Production hook set

Only these production files are in scope for this candidate:

- `common/speculative.cpp`
- `docs/speculative.md`

No public `include/llama.h`, `tools/server/`, repository `tests`, `examples`,
`pocs`, `ggml/src`, production CMake wiring, kernels, server behavior, or
draft-head graph path is approved by P5Y.

## Runtime behavior

P5Y is explicitly gated by `LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY=1` and requires
`LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1` (requires `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`). Without both gates, the existing
descriptor-only route and P5X root-tree-only route remain unchanged.

With both gates set, the route still requires the existing JetSpec gates and P5F
binding preflight. After target taps are captured, P5N/P5P/P5Q descriptor state
is built, and P5X materializes the one-node root tree, P5Y materializes exactly
one in-memory root verify-mask ABI entry:

- `actual_tree_nodes=1`;
- `actual_verify_mask_entries=1`;
- `verify_mask_rows=1` and `verify_mask_cols=1`;
- `root_attends_self=1`;
- `root_mask_row=0` and `root_mask_col=0`;
- `prefix_visible=1`;
- `ancestor_only=1`;
- `sibling_visible=0`;
- `descendant_visible=0`.

If P5Y is requested without P5X, if the P5X root tree is missing, if the root
node count is not exactly one, or if the root mask is not the single self edge,
the route fails closed and disables JetSpec.

After the root verify mask is materialized, P5Y returns before P5S. It does not
call the accept path, token commit, hidden/KV survivor commit, rejected-branch
discard, or publish gate.

## Hard boundary

P5Y performs:

- no draft-head graph execution;
- no non-root verify mask;
- no mask tensor;
- no accept path or target logits walk;
- no token commit;
- no hidden/KV survivor commit;
- no rejected-branch discard;
- no publish;
- no visible state change;
- no KV mutation;
- no draft tokens.

## Source guard

`validate_p5y_root_verify_mask_runtime.py` checks:

- the env gate `LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY` is required;
- P5Y requires the P5X root-tree gate and ready root tree;
- missing P5X/prior descriptors and invalid root mask state fail closed;
- `build_root_only_verify_mask_runtime` creates only one root self-mask entry;
- `verify_mask_runtime_ready` and `root_verify_mask_runtime_ready` are traced;
- P5Y returns before accept/commit/discard/publish descriptors;
- no forbidden runtime calls, public API, server route, CMake wiring, kernels,
  repository tests/examples/pocs, performance claim, or promotion claim are
  added.

`probe_p5y_root_verify_mask_trace.py` is the fast trace contract probe. By
default it is no-model and validates source plus an exact representative trace
line; with `--trace-log` it validates a separately captured live P5Y log for
`actual_verify_mask_entries=1`, `root_attends_self=1`, and all downstream `no_*`
boundaries.

## Verification evidence

Representative commands:

```bash
cmake --build build-rocm-qwen35-dev -j2 --target llama-server
python3 experiments/jetspec/validate_p5y_root_verify_mask_runtime.py
python3 experiments/jetspec/probe_p5y_root_verify_mask_trace.py
python3 -m unittest discover -s experiments/jetspec -p 'test_p5y_root_verify_mask*.py'
python3 experiments/jetspec/run_all_jetspec_contracts.py
```

Expected outcome: build passes, P5Y source guard passes, aggregate contracts
pass, default behavior remains descriptor-only unless the P5X and P5Y gates are
both set, and JetSpec still emits no draft tokens.

## Blocked next work

Non-root verify masks, accept path, target logits walk, token commit, hidden/KV
mutation, rejected-branch discard, publish, server request behavior, kernels,
benchmarks, and promotion remain blocked until separate approval.
