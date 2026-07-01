# JetSpec P5AG top-k accept-boundary ABI runtime candidate

P5AG is an approved bounded production-source slice layered after P5AF. It is default-off and materializes CPU metadata for the accept boundary of the synthetic three-node top-k tree only. It does not inspect target logits, does not walk a target accept path, does not commit, does not mutate KV, does not publish visible state, does not execute the draft-head graph, and does not emit draft tokens.

## Gate

```bash
LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1
```

P5AG requires `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`, `LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1`, and `LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1`. It is mutually exclusive with the root verify/accept/commit/publish tail gates.

## Runtime materialized ABI

P5AG records:

- `topk_accept_boundary_runtime_ready=1`
- `topk_verify_mask_runtime_ready=1`
- `topk_tree_runtime_ready=1`
- `actual_tree_nodes=3`
- `actual_verify_mask_entries=5`
- `accept_boundary_candidate_nodes=2`
- `accept_boundary_verified_edges=5`
- `actual_verified_logits_rows=0`
- `accept_decision_source=none_no_logits`
- `accept_path_len=0`
- `actual_accepted_nodes=0`
- `correction_token_present=0`

The accept boundary records that the two synthetic child candidates exist behind a verified ancestor-mask ABI, but no logits or accept decision have been evaluated.

## Stop boundary

P5AG returns before token commit, hidden/KV survivor commit, rejected-branch discard, and publish. It performs no target logits walk, no target accept walk, no token commit, no hidden/KV commit, no rejected-branch discard, no publish, no KV mutation, no visible state change, no draft-head graph execution, and no draft tokens.
