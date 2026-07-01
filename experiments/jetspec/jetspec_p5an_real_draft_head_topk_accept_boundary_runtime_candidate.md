# JetSpec P5AN real draft-head top-k accept-boundary ABI runtime candidate

Status: candidate, no-model contract wiring only, default-off ABI boundary.

P5AN defines the next shadow real accept-boundary ABI after P5AM. It is gated by `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_BOUNDARY_ABI_ONLY=1` and requires the ready chain P5AM, P5AL, P5AK, P5AJ, P5AG, P5AF, P5AE, and P5X. This note is intentionally test/docs wiring only; it does not edit `common/speculative.cpp`, server, runtime, ggml, or build files.

Required gates:

- `LLAMA_JETSPEC_EXPERIMENTAL=1`
- `LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1`
- `LLAMA_JETSPEC_DRAFT_HEAD_LOAD=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_VERIFY_MASK_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_BOUNDARY_ABI_ONLY=1`
- `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`
- `LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1`
- `LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1`
- `LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1`

Trace contract:

- `draft-jetspec p5an_real_draft_head_topk_accept_boundary_runtime`
- `phase=real_draft_head_topk_accept_boundary_ready`
- `real_topk_accept_boundary_runtime_ready=1`
- `real_topk_verify_mask_runtime_ready=1`
- `real_topk_tree_runtime_ready=1`
- `real_topk_candidate_runtime_ready=1`
- `topk_accept_boundary_runtime_ready=1`
- `topk_verify_mask_runtime_ready=1`
- `topk_tree_runtime_ready=1`
- `logits_source=draft_head_full_vocab_logits`
- `actual_verified_logits_rows=1`
- `topk_k=2`
- `actual_tree_nodes=3`
- `candidate_nodes=2`
- `real_tree_token_ids=[root_token,top1,top2]`
- `candidate_ids=[top1,top2]`
- `real_tree_token_ids[1:] == candidate_ids`
- `real_tree_parent_indices=[-1,0,0]`
- `real_tree_depth=[0,1,1]`
- `real_tree_rank=[-1,0,1]`
- `actual_verify_mask_entries=5`
- `accept_boundary_candidate_nodes=2`
- `accept_boundary_verified_edges=5`
- `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`
- `real_verify_mask_rows=[0,1,1,2,2]`
- `real_verify_mask_cols=[0,0,1,0,2]`
- `real_verify_mask_values=[1,1,1,1,1]`
- `accept_decision_source=none_no_target_logits`
- `accept_path_len=0`
- `actual_accepted_nodes=0`
- `correction_token_present=0`
- `no target logits walk`, `no target accept walk`, `no accept`, `no mask tensor`, `no token commit`, `no hidden/KV commit`, `no rejected-branch discard`, `no publish`, `no visible state change`, `no KV mutation`, and `no draft tokens`

P5AN binds the P5AM shadow real-tree verify-mask metadata to an accept-boundary record. It must preserve the real tree IDs and P5AK candidate IDs, require the P5AM ancestor/self mask edges `[0:0,1:0,1:1,2:0,2:2]`, and stop before any target-logits walk or accept decision. It is not a token-producing path.

Validation:

- `validate_p5an_real_draft_head_topk_accept_boundary_runtime.py --json`
- `probe_p5an_real_draft_head_topk_accept_boundary_trace.py --json`
- `test_p5an_real_draft_head_topk_accept_boundary_trace_probe.py`

Live model execution remains a separate explicit canary step.
