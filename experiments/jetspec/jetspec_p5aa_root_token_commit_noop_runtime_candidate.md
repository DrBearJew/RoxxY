# JetSpec P5AA root-token-commit no-op runtime candidate

P5AA is an approved bounded production-source slice. It is default-off and
non-drafting. The slice shapes the root-only token-commit ABI after P5Z without
performing a real token commit, visible token publish, hidden/KV commit, rejected
branch discard, publish, KV mutation, draft-head graph execution, or draft token
emission.

## Gate

P5AA is explicitly gated by `LLAMA_JETSPEC_ROOT_TOKEN_COMMIT_NOOP_ONLY=1` and
requires all prior root-only runtime gates:

- `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`
- `LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY=1`
- `LLAMA_JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_ONLY=1`

If P5AA is requested without P5X/P5Y/P5Z readiness, the runtime disables
fail-closed with `invalid_root_token_commit_noop_runtime`.

## Runtime object

After P5X materializes one root tree node, P5Y materializes one root self-mask,
and P5Z materializes the root-anchor accept object, P5AA materializes exactly one
root token-commit no-op ABI object:

- `JETSPEC_ROOT_TOKEN_COMMIT_NOOP_RUNTIME_PHASE = "root_token_commit_noop_runtime"`
- `token_commit_runtime_ready`
- `root_token_commit_noop_runtime_ready=1`
- `root_token_commit_noop_runtime_hash_last != 0`
- `root_token_commit_noop_runtime_seq_id_last == root_anchor_accept_path_runtime_seq_id_last`
- `root_verified_anchor=1`
- `accept_path_len=0`
- `actual_accepted_nodes=0`
- `correction_token_present=0`
- `actual_committed_tokens=0`

The root token-commit no-op runtime object does not set
`token_commit_descriptor_ready`. It does not call the P5T descriptor builder.

## Boundary

After the no-op token-commit ABI object is materialized, P5AA returns before P5U.
It does not call hidden/KV survivor commit, rejected-branch discard, publish, or
any KV mutation path.

P5AA performs:

- no real token commit
- no visible token publish
- no hidden/KV survivor commit
- no rejected-branch discard
- no publish
- no visible state change
- no KV mutation
- no draft-head graph execution
- no draft tokens

## Validation

`validate_p5aa_root_token_commit_noop_runtime.py` verifies the source slice,
allowlist, gate dependency, return-before-P5U boundary, and no forbidden runtime
calls. `probe_p5aa_root_token_commit_noop_trace.py` validates the fast no-model
trace contract and accepts `--trace-log` for a separately captured live P5AA log.

Expected outcome: build passes, P5AA source guard passes, aggregate contracts
pass, and default behavior remains unchanged unless the P5X, P5Y, P5Z, and P5AA
gates are all enabled.
