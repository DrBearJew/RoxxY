# P5AW target-accept walk readiness

Status: inert experiments-only readiness packet. This is not production source approval and not a runtime hook.

Purpose: after P5AV proves a default-off target-logits walk canary can read exactly one existing target logits row, P5AW defines the next target-side verification contract: how a future target-accept walk canary would consume that P5AV row. It still performs no target accept walk and no accept.

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
- no additional target logits walk
- no target accept walk
- no accept
- no token commit
- no visible token publish
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
- P5AU target-logits walk readiness
- P5AV real draft-head top-k target-logits walk canary

P5AV source evidence consumed by this readiness packet:
- `p5av_target_logits_walk_canary_trace_contract_verified`
- `p5av_real_draft_head_topk_target_logits_walk_canary_contract_wiring_validated`
- `planned_target_logits_rows=1`
- `actual_target_logits_rows_walked=1`
- `target_logits_source=target_model_full_vocab_logits`
- `target_logits_width=248320`
- `planned_parent_nodes=[0]`
- `planned_candidate_nodes=[1,2]`
- `candidate_ids=[top1,top2]`
- `target_candidate_logits=[target_logit_top1,target_logit_top2]`
- `row_semantics=parent_position_scores_candidate_children`
- `accept_decision_source=target_logits_canary_only_no_accept`
- `target_logits_walk_canary_only=true`
- `actual_target_accept_steps=0`
- `actual_accepted_nodes=0`
- `correction_token_present=0`

Future target-accept plan recorded but not executed:
- `planned_target_accept_steps=1`
- `planned_accept_parent_nodes=[0]`
- `planned_accept_candidate_nodes=[1,2]`
- `planned_accept_candidate_ids=[top1,top2]`
- `planned_accept_score_source=target_candidate_logits_from_p5av_row`
- `planned_accept_rule=greedy_target_argmax_child_match_or_correction`
- `planned_correction_token_source=target_full_vocab_argmax_when_no_child_match`
- `planned_accept_output_visibility=metadata_only_until_explicit_runtime_approval`
- `actual_target_accept_steps=0`
- `actual_accepted_nodes=0`
- `correction_token_present=0`

Approval blockers that stay false:
- `target_accept_walk_approved=false`
- `real_accept_approved=false`
- `real_token_commit_approved=false`
- `visible_token_publish_approved=false`
- `real_hidden_kv_commit_approved=false`
- `real_rejected_branch_discard_approved=false`
- `real_publish_approved=false`
- `product_runtime_hooks_approved=false`
- `draft_token_emission_approved=false`
- `performance_promotion_approved=false`

Explicitly blocked runtime hooks:
- P5AW target-accept walk runtime hook
- P5T token-commit runtime/product hook
- P5U hidden/KV survivor commit runtime/product hook
- P5V rejected-branch discard runtime/product hook
- P5W publish-gate runtime/product hook
- P5AD root publish/no-op product hook reuse
- server draft-token emission hook
- public API route hook
- ggml/KV mutation hook

P5AW verifier status:
- `p5aw_target_accept_walk_readiness_verified_not_executed`

P5AW is readiness only. A future runtime-capable step, if explicitly approved, would be a default-off target-accept walk canary that computes one metadata-only accept/correction decision from the already-walked P5AV target row while still performing no token commit, no hidden/KV commit, no rejected-branch discard, no publish, no KV mutation, and no draft-token emission.
