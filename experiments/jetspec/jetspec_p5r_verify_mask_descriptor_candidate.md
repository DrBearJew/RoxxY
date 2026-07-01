# JetSpec P5R verify-mask descriptor candidate

Status: approved bounded production-source slice. This is default-off and non-drafting.

## Scope

- Production files: `common/speculative.cpp`, `docs/speculative.md`.
- Descriptor function: `build_verify_mask_descriptor`.
- Ready flag: `verify_mask_descriptor_ready`.
- Fail-closed reason: `invalid_verify_mask_descriptor`.
- Phase token: `build_verify_mask`.
- Rollback point token: `after_verify_mask`.
- Descriptor token: `verify_mask_descriptor_only`.

## Zero-runtime boundary

- `actual_verify_mask_entries=0`
- `no_real_verify_mask=1`
- `no_verify_mask=1`
- `no_accept=1`
- `no_kv_mutation=1`
- `no_publish=1`
- `no_draft_tokens=1`

The slice records descriptor-only intent for `build_verify_mask` while preserving `pre_publish_visible_state_unmodified=1` where applicable. It performs no real verify mask, no mask tensor, no accept, no KV mutation, no publish, no draft tokens; it performs no draft-head graph execution, no draft tokens emitted, no real KV mutation, no CUDA dispatch, no server route, no public API, and no CMake wiring.

Full real tree construction, verify, accept, commit, discard, publish, rollback mutation, server behavior, repository tests/examples/pocs, `ggml/src`, and public API remain blocked until separate approval.
