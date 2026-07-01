# JetSpec P5U hidden/KV survivor commit descriptor candidate

Status: approved bounded production-source slice. This is default-off and non-drafting.

## Scope

- Production files: `common/speculative.cpp`, `docs/speculative.md`.
- Descriptor function: `build_hidden_kv_survivor_commit_descriptor`.
- Ready flag: `hidden_kv_survivor_commit_descriptor_ready`.
- Fail-closed reason: `invalid_hidden_kv_survivor_commit_descriptor`.
- Phase token: `commit_hidden_kv_survivors`.
- Rollback point token: `after_hidden_kv_commit`.
- Descriptor token: `hidden_kv_survivor_commit_descriptor_only`.

## Zero-runtime boundary

- `actual_survivor_pages_committed=0`
- `no_real_hidden_kv_commit=1`
- `no_kv_mutation=1`
- `no_publish=1`
- `no_draft_tokens=1`

The slice records descriptor-only intent for `commit_hidden_kv_survivors` while preserving `pre_publish_visible_state_unmodified=1` where applicable. It performs no real hidden/KV commit, no KV mutation, no publish, no draft tokens; it performs no draft-head graph execution, no draft tokens emitted, no real KV mutation, no CUDA dispatch, no server route, no public API, and no CMake wiring.

Full real tree construction, verify, accept, commit, discard, publish, rollback mutation, server behavior, repository tests/examples/pocs, `ggml/src`, and public API remain blocked until separate approval.
