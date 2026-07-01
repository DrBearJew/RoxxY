# JetSpec P5W publish-gate descriptor candidate

Status: approved bounded production-source slice. This is default-off and non-drafting.

## Scope

- Production files: `common/speculative.cpp`, `docs/speculative.md`.
- Descriptor function: `build_publish_gate_descriptor`.
- Ready flag: `publish_gate_descriptor_ready`.
- Fail-closed reason: `invalid_publish_gate_descriptor`.
- Phase token: `publish_post_commit_state`.
- Publish has no post-publish rollback point in P5M; this is a gate descriptor, not a visible publish.
- Descriptor token: `publish_gate_descriptor_only`.

## Zero-runtime boundary

- `actual_publish_visible_state=0`
- `publish_after_commit_and_discard_only=1`
- `no_real_publish=1`
- `no_visible_state_change=1`
- `no_draft_tokens=1`

The slice records descriptor-only intent for `publish_post_commit_state` while preserving `pre_publish_visible_state_unmodified=1` where applicable. It performs no real publish, no visible state change, no draft tokens; it performs no draft-head graph execution, no draft tokens emitted, no real KV mutation, no CUDA dispatch, no server route, no public API, and no CMake wiring.

Full real tree construction, verify, accept, commit, discard, publish, rollback mutation, server behavior, repository tests/examples/pocs, `ggml/src`, and public API remain blocked until separate approval.
