# JetSpec P5X root-only runtime tree candidate

Status: approved bounded production-source slice. This is the first real tree
materialization step, but it is default-off, root-only, non-drafting, and stops
before verify/accept/commit/publish.

## Production hook set

Only these production files are in scope for this candidate:

- `common/speculative.cpp`
- `docs/speculative.md`

No public `include/llama.h`, `tools/server/`, repository `tests`, `examples`,
`pocs`, `ggml/src`, production CMake wiring, kernels, or draft-head graph path is
approved by P5X.

## Runtime behavior

P5X is explicitly gated by `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`. Without the gate, the
existing P5N-P5W descriptor-only route remains unchanged.

With the gate set, the route still requires the existing JetSpec gates and P5F
binding preflight. After target taps are captured and P5N/P5P/P5Q descriptor
state is built, P5X materializes exactly one in-memory root tree node:

- `actual_tree_nodes=1`;
- `tree_token_ids=[root_token]`, where `root_token` is the last pre-round prompt
  token captured in `begin()`;
- `tree_parent_indices=[-1]`;
- `tree_depth=[0]`;
- `tree_rank=[-1]`;
- `tree_cum_logprob=[0.0]`;
- `root_parent=-1` and `root_depth=0`;
- `parent_before_child=1`;
- `num_nodes_lte_budget=1`.

If the prompt is empty, the sequence id is invalid, the P5Q/prior descriptors are
missing, or the node budget cannot hold the root, the route fails closed and
disables JetSpec.

After the root node is materialized, P5X returns before P5R. It does not call the
verify-mask descriptor, accept path, token commit, hidden/KV survivor commit,
rejected-branch discard, or publish gate.

## Hard boundary

P5X performs:

- no draft-head graph execution;
- no top-k or non-root tree expansion;
- no verify mask;
- no accept path or target logits walk;
- no token commit;
- no hidden/KV survivor commit;
- no rejected-branch discard;
- no publish;
- no visible state change;
- no KV mutation;
- no draft tokens.

## Source guard

`validate_p5x_root_tree_runtime.py` checks:

- the env gate `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY` is required;
- root token capture uses the pre-round prompt tail;
- empty prompt, missing P5Q/prior descriptors, and invalid root token paths fail closed;
- default descriptor flow rejects `tree_build_actual_nodes_last != 0` before P5R;
- `build_root_only_runtime_tree` creates only one root node;
- `tree_build_runtime_ready` and `root_tree_runtime_ready` are traced;
- P5X returns before verify/accept/commit/discard/publish descriptors;
- no forbidden runtime calls, public API, server route, CMake wiring, kernels,
  repository tests/examples/pocs, performance claim, or promotion claim are
  added.

`probe_p5x_root_tree_trace.py` is the fast trace contract probe. By default it is
no-model and validates source plus an exact representative trace line; with
`--trace-log` it validates a separately captured live P5X log for
`actual_tree_nodes=1`, `root_tree_runtime_ready=1`, and all `no_*` boundaries.

## Verification evidence

Representative commands:

```bash
cmake --build build-rocm-qwen35-dev -j2 --target llama-server
python3 experiments/jetspec/validate_p5x_root_tree_runtime.py
python3 experiments/jetspec/probe_p5x_root_tree_trace.py
python3 -m unittest discover -s experiments/jetspec -p 'test_p5x_root_tree*.py'
python3 experiments/jetspec/run_all_jetspec_contracts.py
```

Expected outcome: build passes, P5X source guard passes, aggregate contracts
pass, default behavior remains descriptor-only unless `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`,
and JetSpec still emits no draft tokens.

## Blocked next work

Non-root/top-k tree construction, draft-head logits, verify masks, accept,
commit, hidden/KV mutation, rejected-branch discard, publish, server request
behavior, kernels, benchmarks, and promotion remain blocked until separate
approval.
