# JetSpec P5AF top-k verify-mask ABI runtime candidate

P5AF is an approved bounded production-source slice layered after P5AE. It is default-off and materializes CPU metadata for the verify-mask ABI of the synthetic three-node top-k tree only. It does not allocate or dispatch a mask tensor, does not accept, does not commit, does not mutate KV, does not publish visible state, does not execute the draft-head graph, and does not emit draft tokens.

## Gate

```bash
LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1
```

P5AF requires `LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1` and `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`. It is mutually exclusive with the root verify/accept/commit/publish tail gates.

## Runtime materialized ABI

P5AF records:

- `topk_verify_mask_runtime_ready=1`
- `topk_tree_runtime_ready=1`
- `actual_tree_nodes=3`
- `actual_verify_mask_entries=5`
- `verify_mask_rows=3`
- `verify_mask_cols=3`
- `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`
- `prefix_visible=1`
- `ancestor_only=1`
- `root_attends_self=1`
- `child_attends_root=1`
- `child_attends_self=1`
- `sibling_visible=0`
- `descendant_visible=0`

The allowed edges encode ancestor-only tree attention for the root plus two sibling children: each child attends the root and itself; siblings do not attend each other.

## Stop boundary

P5AF returns before accept, token commit, hidden/KV survivor commit, rejected-branch discard, and publish. It performs no mask tensor allocation, no draft-head graph execution, no KV mutation, no visible state change, and no draft tokens.
