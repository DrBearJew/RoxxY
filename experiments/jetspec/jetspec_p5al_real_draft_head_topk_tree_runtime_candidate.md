# JetSpec P5AL real draft-head top-k tree ABI runtime candidate

P5AL is the proposed next bounded production-source slice after P5AK. It is default-off under `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_ABI_ONLY=1` and records a shadow real-tree ABI using the real P5AK draft-head top-k candidate IDs. It does not mutate the canonical P5AE synthetic tree arrays and does not rewrite the existing P5AE/P5AF/P5AG synthetic tree hashes. It must not accept, commit, mutate KV, publish visible state, or emit draft tokens.

## Required gates

```bash
LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_ABI_ONLY=1
LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY=1
LLAMA_JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY=1
LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1
LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1
LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1
LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1
```

P5AL requires the full chain:

1. P5X root-tree readiness.
2. P5AE synthetic top-k tree ABI readiness.
3. P5AF top-k verify-mask ABI readiness.
4. P5AG top-k accept-boundary ABI readiness.
5. P5AJ real draft-head full-vocab logits/top-k canary readiness.
6. P5AK real draft-head top-k candidate ABI readiness.

It fails closed unless `real_topk_candidate_runtime_ready=1`, `topk_accept_boundary_runtime_ready=1`, `topk_verify_mask_runtime_ready=1`, and `topk_tree_runtime_ready=1` are already true. It remains mutually exclusive with the root verify/accept/commit/publish tail gates through the existing top-k ABI conflict guard.

## Runtime ABI record

P5AL should record:

- `real_topk_tree_runtime_ready=1`
- `real_topk_candidate_runtime_ready=1`
- `logits_source=draft_head_full_vocab_logits`
- `actual_verified_logits_rows=1`
- `topk_k=2`
- `actual_tree_nodes=3`
- `parent_node=0`
- `candidate_nodes=2`
- `candidate_ids=[top1,top2]` copied from P5AK
- `real_tree_token_ids=[root_token,top1,top2]`
- `real_tree_parent_indices=[-1,0,0]`
- `real_tree_depth=[0,1,1]`
- `real_tree_rank=[-1,0,1]`
- `real_tree_logits=[0,top1_logit,top2_logit]`
- `rank_semantics=rank_stable_descending_logit`
- `accept_path_len=0`
- `actual_accepted_nodes=0`
- `correction_token_present=0`

The critical ABI proof is that `real_tree_token_ids[1:]` exactly equals the P5AK `candidate_ids` while preserving the three-node tree shape and ancestor-only boundary from P5AE/P5AF/P5AG.

## Forbidden side effects

P5AL performs no external logits walk, no target logits walk, no target accept walk, no sampler, no accept, no token commit, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens.

## Current status

After inspecting the runtime coder result, the P5AL env gate, builder, process branch, and trace boundary are present in `common/speculative.cpp`. The runtime trace token names are present after the token-name fix: the runtime log exposes `real_tree_token_ids=[%d,%d,%d]`, `real_tree_parent_indices=[%d,%d,%d]`, `real_tree_depth=[%d,%d,%d]`, and `real_tree_rank=[%d,%d,%d]`, proving the real tree children come from P5AK candidate IDs. The no-model validator and trace probe verify the source and parser contract only; live GPU/model execution remains intentionally out of scope for this slice.
