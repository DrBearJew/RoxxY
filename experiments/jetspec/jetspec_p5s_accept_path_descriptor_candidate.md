# JetSpec P5S accept-path descriptor candidate

Status: approved bounded production-source slice. This is default-off and non-drafting.

## Scope

- Production files: `common/speculative.cpp`, `docs/speculative.md`.
- Descriptor function: `build_accept_path_descriptor`.
- Ready flag: `accept_path_descriptor_ready`.
- Fail-closed reason: `invalid_accept_path_descriptor`.
- Phase token: `accept_path`.
- Rollback point token: `after_accept`.
- Descriptor token: `accept_path_descriptor_only`.

## Zero-runtime boundary

- `actual_accepted_nodes=0`
- `correction_token_present=0`
- `no_real_accept=1`
- `no_commit_tokens=1`
- `no_kv_mutation=1`
- `no_publish=1`
- `no_draft_tokens=1`

The slice records descriptor-only intent for `accept_path` while preserving `pre_publish_visible_state_unmodified=1` where applicable. It performs no real accept, no target logits walk, no token commit, no KV mutation, no publish, no draft tokens; it performs no draft-head graph execution, no draft tokens emitted, no real KV mutation, no CUDA dispatch, no server route, no public API, and no CMake wiring.

Full real tree construction, verify, accept, commit, discard, publish, rollback mutation, server behavior, repository tests/examples/pocs, `ggml/src`, and public API remain blocked until separate approval.
