# P5AR real draft-head top-k rejected-branch discard no-op ABI

Status: candidate no-model contract wiring for the next post-P5AQ real draft-head tail ABI slice.

Gate: `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_REJECTED_BRANCH_DISCARD_NOOP_ABI_ONLY=1`.

Trace contract:
- `draft-jetspec p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime`
- `phase=real_draft_head_topk_rejected_branch_discard_noop_ready`
- `invalid_real_draft_head_topk_rejected_branch_discard_noop_runtime` on failed prerequisites
- `real_topk_rejected_branch_discard_noop_runtime_ready=1`
- `real_topk_hidden_kv_commit_noop_runtime_ready=1`
- `real_topk_token_commit_noop_runtime_ready=1`
- `real_topk_accept_path_descriptor_runtime_ready=1`
- `real_topk_accept_boundary_runtime_ready=1`
- `real_topk_verify_mask_runtime_ready=1`
- `real_topk_tree_runtime_ready=1`
- `real_topk_candidate_runtime_ready=1`

Required gate chain: P5AR requires P5AQ plus P5AP, P5AO, P5AN, P5AM, P5AL, P5AK, P5AJ, P5AG, P5AF, P5AE, and P5X readiness.
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

Runtime metadata requirements:
- `logits_source=draft_head_full_vocab_logits`
- `ctx_dft_present=1`
- `decode_rc=0`
- `logits_rows=1`
- `logits_width=248320`
- `actual_verified_logits_rows=1`
- `topk_k=2`
- `actual_tree_nodes=3`
- `real_tree_token_ids=[root_token,top1,top2]`
- `candidate_nodes=2`
- `candidate_ids=[top1,top2]`
- `real_tree_token_ids[1:] == candidate_ids`
- `rank_semantics=rank_stable_descending_logit`
- `actual_verify_mask_entries=5`
- `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`
- `accept_boundary_candidate_nodes=2`
- `accept_boundary_verified_edges=5`
- `accept_path_descriptor_len=0`
- `actual_accepted_nodes=0`
- `correction_token_present=0`
- `accept_decision_source=none_no_target_logits`
- `token_commit_noop=1`
- `hidden_kv_commit_noop=1`
- `reuse_p5aq_hidden_kv_commit_noop=1`
- `rejected_branch_discard_noop=1`

Zero side-effect counters:
- `actual_committed_tokens=0`
- `actual_survivor_pages_committed=0`
- `actual_pages_discarded=0`
- `rejected_branch_pages_reachable_after_discard=0`
- `actual_publish_visible_state=0`

Negative boundary:
- root-tail conflict disabled
- not the generic P5V rejected-branch discard descriptor
- not the root P5AC rejected-branch discard no-op path
- returns before downstream publish
- rejects `rejected_branch_discard_descriptor_ready=1`
- rejects `rejected_branch_discard_runtime_ready=1`
- rejects `root_rejected_branch_discard_noop_runtime_ready=1`
- rejects `publish_gate_descriptor_ready=1`
- rejects `root_publish_gate_noop_runtime_ready=1`
- rejects `hidden_kv_survivor_commit_descriptor_ready=1`
- rejects `root_hidden_kv_commit_noop_runtime_ready=1`
- rejects `token_commit_descriptor_ready=1`
- rejects `root_token_commit_noop_runtime_ready=1`
- rejects `#gen drafts = 1`
- rejects `#gen tokens = 1`

This slice performs no target logits walk, no target accept walk, no accept, no real token commit, no visible token publish, no real hidden/KV commit, no hidden/KV commit, no real rejected-branch discard, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens.

Implementation boundary:
- source hook is limited to `common/speculative.cpp`
- no server/public API/ggml/CMake wiring
- validator/probe/unittest are no-model by default; live GPU/model execution remains separate
