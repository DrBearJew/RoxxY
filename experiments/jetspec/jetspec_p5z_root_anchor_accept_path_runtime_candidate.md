# JetSpec P5Z root-anchor accept-path runtime candidate

Status: approved bounded production-source slice. This is the first accept-path
ABI materialization step, but it is default-off, root-anchor-only,
non-drafting, and stops before token commit.

## Production hook set

Only these production files are in scope for this candidate:

- `common/speculative.cpp`
- `docs/speculative.md`

No public `include/llama.h`, `tools/server/`, repository `tests`, `examples`,
`pocs`, `ggml/src`, production CMake wiring, kernels, server behavior, target
logits walk, or draft-head graph path is approved by P5Z.

## Runtime behavior

P5Z is explicitly gated by `LLAMA_JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_ONLY=1` and
requires `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1` plus
`LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY=1`. Without all three gates, the existing
descriptor-only route, P5X root-tree-only route, and P5Y root-verify-mask-only
route remain unchanged.

With all gates set, the route still requires the existing JetSpec gates and P5F
binding preflight. After target taps are captured, P5N/P5P/P5Q descriptor state
is built, P5X materializes the one-node root tree, and P5Y materializes the root
self-mask, P5Z materializes exactly one root-anchor accept-path ABI object:

- `root_verified_anchor=1`;
- `accept_path_len=0`;
- `actual_accepted_nodes=0`;
- `correction_token_present=0`.

The root anchor records that the verified tree root is the already-visible
pre-round prompt tail. It does not accept draft tokens, does not create a
correction token, and does not commit or publish anything.

If P5Z is requested without P5X/P5Y, if the P5Y root self-mask is missing, if the
root tree is not exactly one node, or if any accepted/correction count would
become nonzero, the route fails closed and disables JetSpec.

After the root-anchor accept-path ABI object is materialized, P5Z returns before
P5T (returns before P5T). It does not call token commit, hidden/KV survivor commit, rejected-branch
discard, or publish gate.

## Hard boundary

P5Z performs:

- no target logits walk;
- no target accept walk;
- no accepted draft tokens;
- no correction token;
- no token commit;
- no hidden/KV survivor commit;
- no rejected-branch discard;
- no publish;
- no visible state change;
- no KV mutation;
- no draft-head graph execution;
- no draft tokens.

## Source guard

`validate_p5z_root_anchor_accept_path_runtime.py` checks:

- the env gate `LLAMA_JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_ONLY` is required;
- P5Z requires P5X root tree and P5Y root verify-mask readiness;
- missing P5X/P5Y/prior descriptors and invalid root-anchor state fail closed;
- `build_root_anchor_accept_path_runtime` creates only the root-anchor ABI object;
- `accept_path_runtime_ready` and `root_anchor_accept_path_runtime_ready` are traced;
- P5Z returns before token commit/discard/publish descriptors;
- no forbidden runtime calls, public API, server route, CMake wiring, kernels,
  repository tests/examples/pocs, performance claim, or promotion claim are
  added.

`probe_p5z_root_anchor_accept_path_trace.py` is the fast trace contract probe. By
default it is no-model and validates source plus an exact representative trace
line; with `--trace-log` it validates a separately captured live P5Z log for
`root_verified_anchor=1`, `accept_path_len=0`, `actual_accepted_nodes=0`,
`correction_token_present=0`, and all downstream `no_*` boundaries.

## Verification evidence

Representative commands:

```bash
cmake --build build-rocm-qwen35-dev -j2 --target llama-server
python3 experiments/jetspec/validate_p5z_root_anchor_accept_path_runtime.py
python3 experiments/jetspec/probe_p5z_root_anchor_accept_path_trace.py
python3 -m unittest discover -s experiments/jetspec -p 'test_p5z_root_anchor_accept_path*.py'
python3 experiments/jetspec/run_all_jetspec_contracts.py
```

Expected outcome: build passes, P5Z source guard passes, aggregate contracts
pass, default behavior remains descriptor-only unless the P5X, P5Y, and P5Z
gates are all set, and JetSpec still emits no draft tokens.

## Blocked next work

Accepting any draft token, correction-token selection, target logits walk, token
commit, hidden/KV mutation, rejected-branch discard, publish, server request
behavior, kernels, benchmarks, and promotion remain blocked until separate
approval.
