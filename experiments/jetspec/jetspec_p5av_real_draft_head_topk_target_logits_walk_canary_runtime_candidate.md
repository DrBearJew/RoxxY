# P5AV real draft-head top-k target-logits walk canary

Status: default-off runtime hook plus no-model contract wiring for the first target-side verification canary after P5AS/P5AU.

Gate: `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TARGET_LOGITS_WALK_CANARY=1`.

Trace contract:
- `draft-jetspec p5av_real_draft_head_topk_target_logits_walk_canary_runtime`
- `phase=real_draft_head_topk_target_logits_walk_canary_ready`
- `invalid_real_draft_head_topk_target_logits_walk_canary_runtime` on failed prerequisites
- `target_logits_walk_canary_ready=1`
- `real_topk_publish_gate_noop_runtime_ready=1`
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
- `actual_target_accept_steps=0`
- `actual_accepted_nodes=0`
- `correction_token_present=0`

Required gate chain: P5AV requires P5AS plus P5AR, P5AQ, P5AP, P5AO, P5AN, P5AM, P5AL, P5AK, P5AJ, P5AG, P5AF, P5AE, and P5X readiness.
- `LLAMA_JETSPEC_EXPERIMENTAL=1`
- `LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1`
- `LLAMA_JETSPEC_DRAFT_HEAD_LOAD=1`
- `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`
- `LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1`
- `LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1`
- `LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_VERIFY_MASK_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_BOUNDARY_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_PATH_DESCRIPTOR_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TOKEN_COMMIT_NOOP_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_HIDDEN_KV_COMMIT_NOOP_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_REJECTED_BRANCH_DISCARD_NOOP_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_PUBLISH_GATE_NOOP_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TARGET_LOGITS_WALK_CANARY=1`

Runtime metadata requirements:
- reuses the existing target batch output row; no new target decode is issued
- `target_logits_batch_index` names the first logits-producing target batch index
- `target_logits_pos` records the target batch position for that row
- `target_logits_seq_id` records the target sequence id for that row when present
- candidate IDs are the P5AK/P5AS real draft-head top-k `candidate_ids`
- target candidate logits are read from the target model full-vocab row at those candidate IDs
- `target_logits_walk_canary_only=1`

Zero side-effect counters:
- `actual_target_accept_steps=0`
- `actual_accepted_nodes=0`
- `correction_token_present=0`
- `actual_committed_tokens=0`
- `actual_survivor_pages_committed=0`
- `actual_pages_discarded=0`
- `rejected_branch_pages_reachable_after_discard=0`
- `actual_publish_visible_state=0`

Negative boundary:
- root-tail conflict disabled
- not a P5T token-commit hook
- not a P5U hidden/KV commit hook
- not a P5V rejected-branch discard hook
- not a P5W publish hook
- not a P5AD root publish/no-op reuse path
- rejects `no_target_logits_walk=1` in the P5AV trace because this canary performs the one approved target logits row read
- rejects `actual_target_logits_rows_walked=0`
- rejects `actual_target_accept_steps=1`
- rejects `actual_accepted_nodes=1`
- rejects `actual_committed_tokens=1`
- rejects `actual_publish_visible_state=1`
- rejects `#gen drafts = 1`
- rejects `#gen tokens = 1`

This slice performs one target logits row read from already-computed target logits. It performs no target accept walk, no accept, no token commit, no visible token publish, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens.

Implementation boundary:
- source hook is limited to `common/speculative.cpp`
- no server/public API/ggml/CMake wiring
- validator/probe/unittest are no-model by default; live GPU/model execution remains separate
