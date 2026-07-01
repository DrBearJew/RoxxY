# JetSpec P5V rejected-branch discard descriptor candidate

Status: approved bounded production-source slice. This is default-off and non-drafting.

## Scope

- Production files: `common/speculative.cpp`, `docs/speculative.md`.
- Descriptor function: `build_rejected_branch_discard_descriptor`.
- Ready flag: `rejected_branch_discard_descriptor_ready`.
- Fail-closed reason: `invalid_rejected_branch_discard_descriptor`.
- Phase token: `discard_rejected_branches`.
- Rollback point token: `after_rejected_discard`.
- Descriptor token: `rejected_branch_discard_descriptor_only`.

## Zero-runtime boundary

- `actual_pages_discarded=0`
- `rejected_branch_pages_reachable_after_discard=0`
- `no_real_rejected_branch_discard=1`
- `no_kv_mutation=1`
- `no_publish=1`
- `no_draft_tokens=1`

The slice records descriptor-only intent for `discard_rejected_branches` while preserving `pre_publish_visible_state_unmodified=1` where applicable. It performs no real rejected-branch discard, rejected_branch_pages_reachable_after_discard=0, no KV mutation, no publish, no draft tokens; it performs no draft-head graph execution, no draft tokens emitted, no real KV mutation, no CUDA dispatch, no server route, no public API, and no CMake wiring.

Full real tree construction, verify, accept, commit, discard, publish, rollback mutation, server behavior, repository tests/examples/pocs, `ggml/src`, and public API remain blocked until separate approval.
