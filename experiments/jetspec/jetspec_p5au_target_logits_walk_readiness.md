# P5AU target-logits walk readiness

Status: inert experiments-only readiness packet. This is not production source approval and not a runtime hook.

Purpose: after P5AS proves the real draft-head top-k tail can reach the publish-gate no-op ABI and P5AT records the promotion blockers, P5AU defines the first future target-side verification contract: which target logits rows would be read to verify the P5AS real draft-head top-k tree. It still performs no target logits walk.

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
- no llama decode call
- no target logits walk
- no target accept walk
- no accept
- no token commit
- no hidden/KV commit
- no rejected-branch discard
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
- P5AT real draft-head top-k promotion blocker audit

P5AS source tree contract consumed by this readiness packet:
- `actual_tree_nodes=3`
- `candidate_nodes=2`
- `real_tree_token_ids=[root_token,top1,top2]`
- `candidate_ids=[top1,top2]`
- `real_tree_token_ids[1:] == candidate_ids`
- `actual_verify_mask_entries=5`
- `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`
- `accept_path_descriptor_len=0`
- `actual_accepted_nodes=0`
- `correction_token_present=0`

Future target-logits plan recorded but not executed:
- `planned_target_logits_rows=1`
- `planned_parent_nodes=[0]`
- `planned_candidate_nodes=[1,2]`
- `planned_candidate_ids=[top1,top2]`
- `planned_row_semantics=parent_position_scores_candidate_children`
- `planned_target_logits_source=target_model_full_vocab_logits`
- `planned_target_position_source=target_cache_position_for_parent_node_0`
- `planned_accept_decision_source=target_logits_not_yet_executed`
- `actual_target_logits_rows_walked=0`
- `actual_target_accept_steps=0`
- `actual_accepted_nodes=0`
- `correction_token_present=0`

Approval blockers that stay false:
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

Explicitly blocked runtime hooks:
- P5AU target-logits walk runtime hook
- P5T token-commit runtime/product hook
- P5U hidden/KV survivor commit runtime/product hook
- P5V rejected-branch discard runtime/product hook
- P5W publish-gate runtime/product hook
- P5AD root publish/no-op product hook reuse
- server draft-token emission hook
- public API route hook
- ggml/KV mutation hook

P5AU verifier status:
- `p5au_target_logits_walk_readiness_verified_not_executed`

P5AU is readiness only. The next runtime-capable step, if explicitly approved, would be a default-off target-logits walk canary that reads the planned target row while still performing no accept, no commit, no publish, no KV mutation, and no draft-token emission.
