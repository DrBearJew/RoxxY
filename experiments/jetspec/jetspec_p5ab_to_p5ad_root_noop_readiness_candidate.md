# JetSpec P5AB-P5AD root no-op readiness candidate

P5AB-P5AD are approved bounded production-source slices that finish the root-only
no-op transaction tail after P5AA. They make the codebase ready to start a
separately approved real root-only trace test, but they do not run that test.

## Gates

All gates are default-off and require the prior root-only gates:

- P5AB: `LLAMA_JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_ONLY=1`
- P5AC: `LLAMA_JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_ONLY=1`
- P5AD: `LLAMA_JETSPEC_ROOT_PUBLISH_GATE_NOOP_ONLY=1`

P5AB requires P5X/P5Y/P5Z/P5AA readiness. P5AC requires P5AB readiness. P5AD
requires P5AC readiness.

## Invariants

- `root_hidden_kv_commit_noop_runtime_ready=1`
- `root_rejected_branch_discard_noop_runtime_ready=1`
- `root_publish_gate_noop_runtime_ready=1`
- `actual_committed_tokens=0`
- `actual_survivor_pages_committed=0`
- `actual_pages_discarded=0`
- `rejected_branch_pages_reachable_after_discard=0`
- `actual_publish_visible_state=0`
- `publish_after_commit_and_discard_only=1`
- `root_runtime_ready_for_real_test=1`

## Boundary

The slices do not call the descriptor chain builders for P5U/P5V/P5W, do not set
legacy descriptor-ready flags, do not mutate KV, do not publish visible state, do
not execute the draft-head graph, and do not emit draft tokens.

## Stop condition

After validators, no-model probes, build, and aggregate contracts pass, the
implementation is ready to start a real/live root-only trace test under separate
approval. The real test should use all root-only gates through P5AD and should
still expect no draft tokens, no KV mutation, and no visible publish.
