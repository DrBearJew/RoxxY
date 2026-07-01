# JetSpec P5AE top-k tree ABI runtime candidate

P5AE is an approved bounded production-source slice after the P5AD live root-only proof. It is default-off and materializes a synthetic non-root top-k `DraftTree` ABI object only. It does not use draft-head logits, does not execute the draft-head graph, does not build a verify mask, does not accept, does not commit, does not mutate KV, does not publish visible state, and does not emit draft tokens.

## Gate

```bash
LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1
```

P5AE also requires `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1` so the P5X root tree is materialized first. When this gate is enabled, the descriptor-only tree budget is floored to the synthetic three-node ABI size while still bounded by `JETSPEC_QWEN36_DRAFT_BLOCK_SIZE`; this avoids requiring a real multi-row draft-head logits batch for the synthetic ABI object. It is mutually exclusive with the root verify/accept/commit/publish tail gates `LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY=1`, `LLAMA_JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_ONLY=1`, `LLAMA_JETSPEC_ROOT_TOKEN_COMMIT_NOOP_ONLY=1`, `LLAMA_JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_ONLY=1`, `LLAMA_JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_ONLY=1`, and `LLAMA_JETSPEC_ROOT_PUBLISH_GATE_NOOP_ONLY=1`.

## Runtime materialized ABI

P5AE records:

- `topk_tree_runtime_ready=1`
- `topk_logprob_source=synthetic_full_vocab_softmax`
- `topk_width=2`
- `topk_depth=1`
- `actual_tree_nodes=3`
- `tree_token_ids=[root_token, synthetic_child_0, synthetic_child_1]`
- `tree_parent_indices=[-1,0,0]`
- `tree_depth=[0,1,1]`
- `tree_rank=[-1,0,1]`
- `tree_cum_logprob=[0.0,-0.1,-0.3]`
- `parent_before_child=1`
- `num_nodes_lte_budget=1`
- `non_root_nodes=2`

The synthetic children are ABI sentinels derived from the root token and are never emitted. This is a bounded non-root tree-layout proof, not draft-head output.

## Stop boundary

P5AE returns before verify-mask, accept, token commit, hidden/KV survivor commit, rejected-branch discard, and publish. It performs no draft-head graph execution, no draft logits, no KV mutation, no visible state change, and no draft tokens.
