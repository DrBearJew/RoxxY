# JetSpec P5AK real draft-head top-k candidate ABI runtime candidate

P5AK is an approved bounded production-source slice layered after P5AJ and P5AG. It is default-off and records real draft-head top-k candidate metadata from the P5AJ logits canary into the already-materialized P5AG accept-boundary ABI. It does not replace the synthetic tree yet, does not accept, does not commit, does not mutate KV, does not publish visible state, and does not emit draft tokens.

## Gate

```bash
LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY=1
```

P5AK requires the full explicit chain:

- `LLAMA_JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY=1`
- `LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1`
- `LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1`
- `LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1`
- `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`

It fails closed without P5AJ logits readiness, P5AG accept-boundary readiness, P5AF verify-mask readiness, P5AE top-k tree readiness, and P5X root-tree readiness. It remains mutually exclusive with the root verify/accept/commit/publish tail gates through the existing top-k ABI conflict guard.

## Runtime materialized ABI

P5AK records:

- `real_topk_candidate_runtime_ready=1`
- `topk_accept_boundary_runtime_ready=1`
- `topk_verify_mask_runtime_ready=1`
- `topk_tree_runtime_ready=1`
- `logits_source=draft_head_full_vocab_logits`
- `ctx_dft_present=1`
- `decode_rc=0`
- `logits_rows=1`
- `logits_width=248320`
- `actual_verified_logits_rows=1`
- `topk_k=2`
- `parent_node=0`
- `candidate_nodes=2`
- `candidate_ids=[top1,top2]` copied from P5AJ
- `candidate_logits=[top1_logit,top2_logit]` copied from P5AJ
- `rank_semantics=rank_stable_descending_logit`
- `accept_path_len=0`
- `actual_accepted_nodes=0`
- `correction_token_present=0`

The candidate IDs and logits are metadata only. They consume the P5AJ top1/top2 output and the P5AG readiness boundary. They do not trigger a sampler, target logits walk, target accept walk, or synthetic full-vocab softmax/top-k-only renormalization.

## Stop boundary

P5AK returns before token commit, hidden/KV survivor commit, rejected-branch discard, and publish. It performs no external logits walk, no target logits walk, no target accept walk, no sampler, no accept, no token commit, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens.

`probe_p5ak_real_draft_head_topk_candidate_trace.py` is the fast no-model trace contract probe. It validates source tokens and a synthetic trace line by default, and can validate a separately captured live log through `--trace-log` without making aggregate verification load a model.
