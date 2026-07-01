# JetSpec P5AM real draft-head top-k verify-mask runtime candidate

Status: candidate, default-off ABI-only runtime metadata.

P5AM adds `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_VERIFY_MASK_ABI_ONLY=1`, after P5AL, P5AK, P5AJ, and the P5AG/P5AF/P5AE/P5X chain. It records a **shadow real-tree verify-mask ABI** for the P5AL shadow real tree. It does not rewrite the existing P5AF/P5AG synthetic hashes or canonical mask arrays.

Required gates:

- `LLAMA_JETSPEC_EXPERIMENTAL=1`
- `LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1`
- `LLAMA_JETSPEC_DRAFT_HEAD_LOAD=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_ABI_ONLY=1`
- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_VERIFY_MASK_ABI_ONLY=1`
- `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`
- `LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1`
- `LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1`
- `LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1`

Trace contract:

- `draft-jetspec p5am_real_draft_head_topk_verify_mask_runtime`
- `phase=real_draft_head_topk_verify_mask_ready`
- `real_topk_verify_mask_runtime_ready=1`
- `real_topk_tree_runtime_ready=1`
- `real_topk_candidate_runtime_ready=1`
- `logits_source=draft_head_full_vocab_logits`
- `actual_tree_nodes=3`
- `real_tree_token_ids=[root_token,top1,top2]`
- `real_tree_parent_indices=[-1,0,0]`
- `real_tree_depth=[0,1,1]`
- `real_tree_rank=[-1,0,1]`
- `real_tree_logits[1:] == candidate_logits`
- `actual_verify_mask_entries=5`
- `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`
- `real_verify_mask_rows=[0,1,1,2,2]`
- `real_verify_mask_cols=[0,0,1,0,2]`
- `real_verify_mask_values=[1,1,1,1,1]`
- `prefix_visible=1`, `ancestor_only=1`, `sibling_visible=0`, `descendant_visible=0`
- `no mask tensor`, `no accept`, `no token commit`, `no hidden/KV commit`, `no rejected-branch discard`, `no publish`, `no KV mutation`, and `no draft tokens`

P5AM is topology-only CPU metadata. It binds the P5AL shadow real-tree topology to the same ancestor/self edge semantics as P5AF, while keeping canonical synthetic tree/mask metadata untouched.

Validation:

- `validate_p5am_real_draft_head_topk_verify_mask_runtime.py --json`
- `probe_p5am_real_draft_head_topk_verify_mask_trace.py --json`
- `test_p5am_real_draft_head_topk_verify_mask_trace_probe.py`

Live model execution remains a separate explicit canary step.
