# JetSpec P5T token-commit descriptor candidate

Status: approved bounded production-source slice. This is default-off and non-drafting.

## Scope

- Production files: `common/speculative.cpp`, `docs/speculative.md`.
- Descriptor function: `build_token_commit_descriptor`.
- Ready flag: `token_commit_descriptor_ready`.
- Fail-closed reason: `invalid_token_commit_descriptor`.
- Phase token: `commit_tokens`.
- Rollback point token: `after_token_commit`.
- Descriptor token: `token_commit_descriptor_only`.

## Zero-runtime boundary

- `actual_committed_tokens=0`
- `no_real_token_commit=1`
- `no_visible_token_publish=1`
- `no_kv_mutation=1`
- `no_publish=1`
- `no_draft_tokens=1`

The slice records descriptor-only intent for `commit_tokens` while preserving `pre_publish_visible_state_unmodified=1` where applicable. It performs no real token commit, no visible token publish, no KV mutation, no publish, no draft tokens; it performs no draft-head graph execution, no draft tokens emitted, no real KV mutation, no CUDA dispatch, no server route, no public API, and no CMake wiring.

Full real tree construction, verify, accept, commit, discard, publish, rollback mutation, server behavior, repository tests/examples/pocs, `ggml/src`, and public API remain blocked until separate approval.
