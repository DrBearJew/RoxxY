# JetSpec P5AQ real draft-head top-k hidden/KV commit no-op ABI runtime candidate

Status: candidate, default-off runtime hook plus no-model contract wiring, hidden/KV commit no-op ABI boundary.

P5AQ defines the next shadow real hidden/KV commit no-op ABI after P5AP. It is gated by `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_HIDDEN_KV_COMMIT_NOOP_ABI_ONLY=1`, fails closed as `invalid_real_draft_head_topk_hidden_kv_commit_noop_runtime`, and requires the ready chain P5AP, P5AO, P5AN, P5AM, P5AL, and P5AK. The source hook is limited to `common/speculative.cpp` plus experiments/docs contracts; it does not edit server, ggml, public API, CMake, tests, examples, or pocs. It is not the generic P5U `hidden_kv_survivor_commit_descriptor_ready` path and not the root P5AB `root_hidden_kv_commit_noop_runtime_ready` path.

Required gates:

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

Trace contract:

- `draft-jetspec p5aq_real_draft_head_topk_hidden_kv_commit_noop_runtime`
- `phase=real_draft_head_topk_hidden_kv_commit_noop_ready`
- `real_topk_hidden_kv_commit_noop_runtime_ready=1`
- `real_topk_token_commit_noop_runtime_ready=1`
- `real_topk_accept_path_descriptor_runtime_ready=1`
- `real_topk_accept_boundary_runtime_ready=1`
- `real_topk_verify_mask_runtime_ready=1`
- `real_topk_tree_runtime_ready=1`
- `real_topk_candidate_runtime_ready=1`
- `logits_source=draft_head_full_vocab_logits`
- `ctx_dft_present=1`
- `decode_rc=0`
- `logits_rows=1`
- `logits_width=248320`
- `actual_verified_logits_rows=1`
- `topk_k=2`
- `actual_tree_nodes=3`
- `candidate_nodes=2`
- `real_tree_token_ids=[root_token,top1,top2]`
- `candidate_ids=[top1,top2]`
- `real_tree_token_ids[1:] == candidate_ids`
- `actual_verify_mask_entries=5`
- `accept_boundary_candidate_nodes=2`
- `accept_boundary_verified_edges=5`
- `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`
- `accept_path_descriptor_len=0`
- `actual_accepted_nodes=0`
- `correction_token_present=0`
- `accept_decision_source=none_no_target_logits`
- `token_commit_noop=1`
- `hidden_kv_commit_noop=1`
- `reuse_p5ap_token_commit_noop=1`
- `actual_committed_tokens=0`
- `actual_survivor_pages_committed=0`
- `actual_pages_discarded=0`
- `actual_publish_visible_state=0`
- `no_target_logits_walk=1`
- `no_target_accept_walk=1`
- `no_accept=1`
- `no_real_token_commit=1`
- `no_visible_token_publish=1`
- `no_real_hidden_kv_commit=1`
- `no_hidden_kv_commit=1`
- `no_rejected_branch_discard=1`
- `no_publish=1`
- `no_visible_state_change=1`
- `no_kv_mutation=1`
- `no_draft_tokens=1`

P5AQ converts the P5AP real token-commit no-op record into a hidden/KV commit no-op ABI record with `actual_survivor_pages_committed=0` and `hidden_kv_commit_noop=1`. It does not choose accepted nodes, does not create a correction token, does not commit survivor hidden/KV pages, does not discard rejected branches, and does not publish visible state. It performs no target logits walk, no target accept walk, no accept, no real token commit, no visible token publish, no real hidden/KV commit, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens. It returns before downstream rejected/discard/publish. It must reject generic P5U `hidden_kv_survivor_commit_descriptor_ready=1`, generic hidden KV `hidden_kv_commit_runtime_ready=1`, root P5AB `root_hidden_kv_commit_noop_runtime_ready=1`, rejected branch discard readiness `rejected_branch_discard_descriptor_ready=1`, publish readiness `publish_gate_descriptor_ready=1`, generic P5T `token_commit_descriptor_ready=1`, root P5AA `root_token_commit_noop_runtime_ready=1`, `#gen drafts = 1`, and `#gen tokens = 1` in traces.

Validation:

- `python3 experiments/jetspec/validate_p5aq_real_draft_head_topk_hidden_kv_commit_noop_runtime.py --json`
- `python3 experiments/jetspec/probe_p5aq_real_draft_head_topk_hidden_kv_commit_noop_trace.py --json`
- `python3 -m unittest experiments/jetspec/test_p5aq_real_draft_head_topk_hidden_kv_commit_noop_trace_probe.py`

Live GPU/model execution remains a separate explicit canary step and is not part of this hidden/KV commit no-op ABI contract.
