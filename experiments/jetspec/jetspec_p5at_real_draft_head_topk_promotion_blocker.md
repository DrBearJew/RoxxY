# P5AT real draft-head top-k promotion blocker audit

Status: inert experiments-only audit packet. This is not production source approval.

Purpose: after P5AS proves the real draft-head top-k tail can reach the publish-gate no-op ABI with zero side effects, P5AT records the remaining promotion blockers before any target logits walk, target accept walk, real token commit, hidden/KV commit, rejected-branch discard, publish gate, visible-state publish, or draft-token emission can be implemented.

Boundary:
- `experiments/jetspec/` only
- no environment gate
- no `common/speculative.cpp` hook
- no `tools/server` hook
- no public API hook
- no `ggml` hook
- no CMake wiring
- no model load
- no context creation
- no `llama_decode`
- no target logits walk
- no target accept walk
- no accept
- no real token commit
- no visible token publish
- no real hidden/KV commit
- no hidden/KV commit
- no real rejected-branch discard
- no rejected-branch discard
- no real publish
- no publish
- no visible state change
- no KV mutation
- no draft tokens

Required predecessor evidence:
- P5AK real draft-head top-k candidate ABI metadata
- P5AL real draft-head top-k shadow tree ABI metadata
- P5AM real draft-head top-k verify-mask ABI metadata
- P5AN real draft-head top-k accept-boundary ABI metadata
- P5AO real draft-head top-k accept-path descriptor ABI metadata
- P5AP real draft-head top-k token-commit no-op ABI
- P5AQ real draft-head top-k hidden/KV commit no-op ABI
- P5AR real draft-head top-k rejected-branch discard no-op ABI
- P5AS real draft-head top-k publish-gate no-op ABI

Required P5AS terminal evidence:
- `p5as_real_draft_head_topk_publish_gate_noop_trace_contract_verified`
- `real_topk_publish_gate_noop_runtime_ready=1`
- `publish_gate_noop=1`
- `publish_after_commit_and_discard_only=1`
- `actual_committed_tokens=0`
- `actual_survivor_pages_committed=0`
- `actual_pages_discarded=0`
- `rejected_branch_pages_reachable_after_discard=0`
- `actual_publish_visible_state=0`
- `no_target_logits_walk=1`
- `no_target_accept_walk=1`
- `no_accept=1`
- `no_real_token_commit=1`
- `no_visible_token_publish=1`
- `no_real_hidden_kv_commit=1`
- `no_hidden_kv_commit=1`
- `no_real_rejected_branch_discard=1`
- `no_rejected_branch_discard=1`
- `no_real_publish=1`
- `no_publish=1`
- `no_visible_state_change=1`
- `no_kv_mutation=1`
- `no_draft_tokens=1`

Promotion blockers recorded by this audit:
- `target_logits_walk_approved=false`
- `target_accept_walk_approved=false`
- `real_token_commit_approved=false`
- `visible_token_publish_approved=false`
- `real_hidden_kv_commit_approved=false`
- `real_rejected_branch_discard_approved=false`
- `real_publish_approved=false`
- `product_runtime_hooks_approved=false`
- `draft_token_emission_approved=false`
- `performance_promotion_approved=false`

Explicitly blocked product/runtime hooks:
- P5T token-commit runtime/product hook
- P5U hidden/KV survivor commit runtime/product hook
- P5V rejected-branch discard runtime/product hook
- P5W publish-gate runtime/product hook
- P5AD root publish/no-op product hook reuse
- server draft-token emission hook
- public API route hook
- ggml/KV mutation hook

P5AT verifier status:
- `p5at_real_draft_head_topk_promotion_blocker_verified_not_executed`

P5AT is a blocker audit, not a promotion. It keeps the P5AS chain terminal and requires separate explicit architecture approval before any production-source hook, target-logits verification, accept walk, state mutation, visible publish, or draft-token generation is added.
