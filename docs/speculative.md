# Speculative Decoding

llama.cpp supports speculative decoding, a technique that can significantly accelerate token generation by predicting multiple tokens ahead of the main model.

[Speculative decoding](https://en.wikipedia.org/wiki/Transformer_(deep_learning)#Speculative_decoding) leverages the fact that computing n tokens in a batch (as in prompt processing) is more efficient than computing n sequentially (as in response generation). By generating draft tokens quickly and then verifying them with the target model in a single batch, this approach can achieve substantial speedups when the draft predictions are frequently correct.

## Implementations

The `llama-server` application supports several implementations of speculative decoding. An implementation with draft model can be mixed with an implementation without draft model.

### Draft Model (`draft`)

A much smaller model (called the _draft model_) generates drafts.
A draft model is the most used approach in speculative decoding.

### JetSpec draft head (`draft-jetspec`, experimental)

`draft-jetspec` is a staged, explicit-opt-in JetSpec integration route. It is
not enabled by default and currently fails closed before generating drafts unless
all JetSpec experimental gates and runtime contracts are satisfied.

Required gates:

```bash
LLAMA_JETSPEC_EXPERIMENTAL=1 LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1 LLAMA_JETSPEC_DRAFT_HEAD_LOAD=1 \
  llama-server [...] --spec-type draft-jetspec --spec-draft-model <draft-head.gguf>
```

Preview/metadata-only JetSpec GGUF files remain non-runnable; the loader keeps
`runtime_supported=false` and rejects them before graph execution. A GGUF that
claims true-valued `runtime_supported` metadata is rejected until JetSpec draft-head graph
execution exists. In the current bounded runtime slice, the server may load the
91-tensor draft-head GGUF as a model-only binding, with no draft context and no
draft-head graph execution. The route first runs a fail-closed draft-head/target
binding preflight, then ingests target tap rows from the private P5B side
channel, records a diagnostic hash, tracks private runtime-state/failure
bookkeeping, records a P5O pre-round snapshot descriptor in `begin()`, builds
a P5N transaction-plan scaffold for the next JetSpec round, records a P5P transient-reservation descriptor for `reserve_transient_tree_pages`, records a P5Q tree-build descriptor for `build_tree`, and then records P5R-P5W descriptor-only gates for the remaining P5M phases through `publish_post_commit_state`. The preflight
checks the target context plus either draft context or model-only draft-head
metadata/shape against the staged Qwen3.6 JetSpec draft-head contract, requires
the target tap count/width to match, and requires target tensor presence/shape
for `token_embd.weight` `[2048,248320]`, `output.weight` `[2048,248320]`, and
`output_norm.weight` `[2048]` before any future graph execution.
The pre-round snapshot descriptor records the sequence id, prompt token count,
prompt hash, and `snapshot_pre_round` hash; it still performs no reserve, no tree
build, and no verify mask work. The transaction scaffold records the P5M phase
order `snapshot_pre_round`, `reserve_transient_tree_pages`, `build_tree`,
`build_verify_mask`, `accept_path`, `commit_tokens`,
`commit_hidden_kv_survivors`, `discard_rejected_branches`, and
`publish_post_commit_state`, plus rollback failpoints, but it does not publish
state, mutate KV, dispatch CUDA, or execute the draft head. The P5P transient-reservation descriptor records descriptor-only reservation intent for `reserve_transient_tree_pages`, bounds the transient tree node budget by the staged draft block size, records `rollback_point=after_reserve`, and keeps `actual_pages_reserved=0`; it performs no real page reservation, no llama_kv_cache primitive, no tree build, no verify mask, no draft tokens, no CUDA, no server route, no public API, and no CMake wiring. The P5Q tree-build descriptor records descriptor-only tree ABI intent for `build_tree`, records `rollback_point=after_build_tree`, root parent `-1`, root depth `0`, planned tree node budget, and `actual_tree_nodes=0`; it performs no real tree build, no tree arrays, no verify mask, no accept path runtime, no draft tokens, no CUDA, no server route, no public API, and no CMake wiring. The P5R verify-mask descriptor records descriptor-only verify-mask ABI intent for `build_verify_mask`, records `rollback_point=after_verify_mask`, and keeps `actual_verify_mask_entries=0`; it performs no real verify mask, no mask tensor, no accept, no KV mutation, no publish, and no draft tokens. The P5S accept-path descriptor records descriptor-only accept ABI intent for `accept_path`, records `rollback_point=after_accept`, and keeps `actual_accepted_nodes=0` plus `correction_token_present=0`; it performs no real accept, no target logits walk, no token commit, no KV mutation, no publish, and no draft tokens. The P5T token-commit descriptor records descriptor-only commit intent for `commit_tokens`, records `rollback_point=after_token_commit`, and keeps `actual_committed_tokens=0`; it performs no real token commit, no visible token publish, no KV mutation, no publish, and no draft tokens. The P5U hidden/KV survivor commit descriptor records descriptor-only survivor commit intent for `commit_hidden_kv_survivors`, records `rollback_point=after_hidden_kv_commit`, and keeps `actual_survivor_pages_committed=0`; it performs no real hidden/KV commit, no KV mutation, no publish, and no draft tokens. The P5V rejected-branch discard descriptor records descriptor-only discard intent for `discard_rejected_branches`, records `rollback_point=after_rejected_discard`, and keeps `actual_pages_discarded=0` plus `rejected_branch_pages_reachable_after_discard=0`; it performs no real rejected-branch discard, no KV mutation, no publish, and no draft tokens. The P5W publish-gate descriptor records descriptor-only publish-gate intent for `publish_post_commit_state`, records `publish_after_commit_and_discard_only=1`, and keeps `actual_publish_visible_state=0`; it performs no real publish, no visible state change, and no draft tokens. The route still
emits no draft tokens and does not execute the draft head; it remains a no draft
tokens route. Set `LLAMA_JETSPEC_TRACE=1`,
`LLAMA_JETSPEC_TAP_TRACE=1`, or `LLAMA_JETSPEC_STATE_TRACE=1` to log captured
target tap row counts, hashes, phase, failure state, pre-round snapshot readiness, pre-round snapshot hash, transaction plan readiness, transaction plan hash, transient reservation readiness, transient reservation hash, transient tree node budget, `actual_pages_reserved=0`, tree build descriptor readiness, tree build descriptor hash, planned tree node budget, `actual_tree_nodes=0`, verify mask descriptor readiness, `actual_verify_mask_entries=0`, accept path descriptor readiness, `actual_accepted_nodes=0`, token commit descriptor readiness, `actual_committed_tokens=0`, hidden/KV survivor commit descriptor readiness, `actual_survivor_pages_committed=0`, rejected-branch discard descriptor readiness, `actual_pages_discarded=0`, publish gate descriptor readiness, and `actual_publish_visible_state=0`.
The current P5W/P5V/P5U/P5T/P5S/P5R/P5Q/P5P/P5O/P5N boundary is: no real reserve, no real tree build, no tree arrays, no real verify mask, no accept, no token commit, no hidden/KV commit, no rejected-branch discard, no KV mutation, no publish, no visible state change, no draft tokens.

P5X root-only runtime tree materialization is a separate default-off first runtime
tree slice gated by `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`. It stops after the P5Q tree
build descriptor and materializes only the root `DraftTree` ABI node from the
pre-round prompt tail: `actual_tree_nodes=1`, `tree_token_ids=[root_token]`,
`tree_parent_indices=[-1]`, `tree_depth=[0]`, `tree_rank=[-1]`, and
`tree_cum_logprob=[0.0]`. It records `tree_build_runtime_ready`,
`root_tree_runtime_ready=1`, `root_parent=-1`, `root_depth=0`,
`parent_before_child=1`, and `num_nodes_lte_budget=1`, then returns before P5R.
It performs no draft-head graph execution, no top-k/non-root tree expansion, no
verify mask, no accept, no token commit, no hidden/KV commit, no rejected-branch
discard, no publish, no visible state change, no KV mutation, and no draft
tokens. `probe_p5x_root_tree_trace.py` is the fast no-model trace contract probe:
it validates the exact `p5x_root_tree_runtime` trace boundary and can validate a
separately captured live trace via `--trace-log`, but its default aggregate mode
does not load a model or create a context.

P5Y root-only verify-mask materialization is a separate default-off first
verify-mask runtime slice gated by `LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY=1`. It
requires `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1` and a ready P5X root tree. It
materializes only the root self-attention verify-mask ABI entry:
`actual_verify_mask_entries=1`, `verify_mask_rows=1`, `verify_mask_cols=1`,
`root_attends_self=1`, `root_mask_row=0`, `root_mask_col=0`,
`prefix_visible=1`, `ancestor_only=1`, `sibling_visible=0`, and
`descendant_visible=0`, then returns before P5S. It performs no draft-head graph
execution, no non-root verify mask, no mask tensor, no accept, no token commit,
no hidden/KV commit, no rejected-branch discard, no publish, no visible state
change, no KV mutation, and no draft tokens. `probe_p5y_root_verify_mask_trace.py`
is the fast no-model trace contract probe and can validate a separately captured
live trace via `--trace-log` without making aggregate verification load a model.

P5Z root-anchor accept-path materialization is a separate default-off accept-path
ABI slice gated by `LLAMA_JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_ONLY=1`. It requires
`LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`, `LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY=1`,
a ready P5X root tree, and a ready P5Y root self-mask. It materializes only the
verified root anchor, not accepted draft tokens: `root_verified_anchor=1`,
`accept_path_len=0`, `actual_accepted_nodes=0`, and
`correction_token_present=0`, then returns before P5T. It performs no target
logits walk, no target accept walk, no token commit, no hidden/KV commit, no
rejected-branch discard, no publish, no visible state change, no KV mutation, no
draft-head graph execution, and no draft tokens. `probe_p5z_root_anchor_accept_path_trace.py`
is the fast no-model trace contract probe and can validate a separately captured
live trace via `--trace-log` without making aggregate verification load a model.

P5AA root-token-commit no-op materialization is a separate default-off token
commit ABI shaping slice gated by `LLAMA_JETSPEC_ROOT_TOKEN_COMMIT_NOOP_ONLY=1`.
It requires `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`,
`LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY=1`,
`LLAMA_JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_ONLY=1`, a ready P5X root tree, a ready
P5Y root self-mask, and a ready P5Z root-anchor accept object. It materializes
only the no-op root token-commit boundary: `root_token_commit_noop_runtime_ready=1`,
`actual_committed_tokens=0`, `root_verified_anchor=1`, `accept_path_len=0`,
`actual_accepted_nodes=0`, and `correction_token_present=0`, then returns before P5U.
It performs no real token commit, no visible token publish, no hidden/KV
commit, no rejected-branch discard, no publish, no visible state change, no KV
mutation, no draft-head graph execution, and no draft tokens.
`probe_p5aa_root_token_commit_noop_trace.py` is the fast no-model trace contract
probe and can validate a separately captured live trace via `--trace-log` without
making aggregate verification load a model.

P5AB/P5AC/P5AD complete the root-only no-op round tail needed before starting a
real root-only test. They are separate default-off gates layered after P5AA:
`LLAMA_JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_ONLY=1`,
`LLAMA_JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_ONLY=1`, and
`LLAMA_JETSPEC_ROOT_PUBLISH_GATE_NOOP_ONLY=1`. P5AB materializes
`root_hidden_kv_commit_noop_runtime_ready=1` with
`actual_survivor_pages_committed=0`; P5AC materializes
`root_rejected_branch_discard_noop_runtime_ready=1` with
`actual_pages_discarded=0` and `rejected_branch_pages_reachable_after_discard=0`;
P5AD materializes `root_publish_gate_noop_runtime_ready=1` with
`actual_publish_visible_state=0`, `publish_after_commit_and_discard_only=1`, and
`root_runtime_ready_for_real_test=1`. They perform no real hidden/KV commit, no
real rejected-branch discard, no real publish, no visible state change, no KV
mutation, no draft-head graph execution, and no draft tokens. P5AD is the first
source-only readiness marker and is ready to start a real root-only test under a separately approved real root-only
trace test; it does not itself run that test.

P5AE top-k tree ABI materialization is a separate default-off non-root tree
layout slice gated by `LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1`. It requires
`LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`, floors the descriptor-only tree budget to
the synthetic three-node ABI size while still staying within
`JETSPEC_QWEN36_DRAFT_BLOCK_SIZE`, and is mutually exclusive with the root
verify/accept/commit/publish tail gates. It materializes a synthetic ABI-only
three-node tree from the P5X root: `topk_logprob_source=synthetic_full_vocab_softmax`,
`topk_width=2`, `topk_depth=1`, `actual_tree_nodes=3`,
`tree_token_ids=[root_token,synthetic_child_0,synthetic_child_1]`,
`tree_parent_indices=[-1,0,0]`, `tree_depth=[0,1,1]`,
`tree_rank=[-1,0,1]`, `tree_cum_logprob=[0.0,-0.1,-0.3]`,
`parent_before_child=1`, `num_nodes_lte_budget=1`, and `non_root_nodes=2`, then
returns before verify-mask, accept, commit, hidden/KV commit, discard, and
publish. The synthetic children prove the widened DraftTree ABI capacity only;
they are not draft logits, are not emitted, and do not come from draft-head graph
execution. P5AE performs no draft-head graph execution, no draft logits, no
verify mask, no accept, no token commit, no KV mutation, no publish, no visible
state change, and no draft tokens.

P5AF top-k verify-mask ABI materialization is a separate default-off CPU-metadata
mask slice gated by `LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1`. It requires
P5AE and P5X, remains mutually exclusive with the root tail gates, and records
ancestor-only sparse mask metadata for the synthetic three-node tree:
`actual_tree_nodes=3`, `actual_verify_mask_entries=5`, `verify_mask_rows=3`,
`verify_mask_cols=3`, `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`,
`prefix_visible=1`, `ancestor_only=1`, `root_attends_self=1`,
`child_attends_root=1`, `child_attends_self=1`, `sibling_visible=0`, and
`descendant_visible=0`. It returns before accept, commit, hidden/KV commit,
discard, and publish. It allocates no mask tensor, dispatches no CUDA, executes
no draft-head graph, mutates no KV, publishes no visible state, and emits no
draft tokens.

P5AG top-k accept-boundary ABI materialization is a separate default-off CPU-metadata
accept-boundary slice gated by `LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1`. It
requires P5X, P5AE, and P5AF, remains mutually exclusive with the root tail gates,
and records only that the synthetic two child candidates sit behind the verified
ancestor-mask ABI: `topk_accept_boundary_runtime_ready=1`,
`actual_tree_nodes=3`, `actual_verify_mask_entries=5`,
`accept_boundary_candidate_nodes=2`, `accept_boundary_verified_edges=5`,
`actual_verified_logits_rows=0`, `accept_decision_source=none_no_logits`,
`accept_path_len=0`, `actual_accepted_nodes=0`, and
`correction_token_present=0`. It returns before token commit, hidden/KV commit,
discard, and publish. It performs no target logits walk, no target accept walk,
no token commit, no hidden/KV commit, no rejected-branch discard, no publish,
no visible state change, no KV mutation, no draft-head graph execution, and no
draft tokens.

P5AK real draft-head top-k candidate ABI metadata is a separate default-off
bridge slice gated by `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY=1`. It
requires `LLAMA_JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY=1`, P5AJ logits readiness,
and the P5AG/P5AF/P5AE/P5X top-k ABI chain. It consumes the P5AJ top1/top2
candidate IDs/logits and the P5AG accept-boundary readiness only, then records
`real_topk_candidate_runtime_ready=1`, `logits_source=draft_head_full_vocab_logits`,
`ctx_dft_present=1`, `decode_rc=0`, `logits_rows=1`, `logits_width=248320`,
`actual_verified_logits_rows=1`, `topk_k=2`, `parent_node=0`,
`candidate_nodes=2`, `candidate_ids=[top1,top2]`,
`candidate_logits=[top1_logit,top2_logit]`,
`rank_semantics=rank_stable_descending_logit`, `accept_path_len=0`,
`actual_accepted_nodes=0`, and `correction_token_present=0`. The records are ABI
metadata only and do not replace the synthetic tree yet. It performs no external
logits walk, no target logits walk, no target accept walk, no sampler, no accept,
no token commit, no hidden/KV commit, no rejected-branch discard, no publish,
no visible state change, no KV mutation, and no draft tokens.

P5AL real draft-head top-k tree ABI is a separate default-off shadow real-tree ABI
bridge slice gated by `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_ABI_ONLY=1`. It
requires `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY=1`, P5AK real draft-head
top-k candidate ABI readiness, P5AJ logits readiness, and the P5AG/P5AF/P5AE/P5X
top-k ABI chain. The runtime record is `real_topk_tree_runtime_ready=1`,
`real_topk_candidate_runtime_ready=1`, `logits_source=draft_head_full_vocab_logits`,
`actual_verified_logits_rows=1`, `topk_k=2`, `actual_tree_nodes=3`,
`parent_node=0`, `candidate_nodes=2`, `candidate_ids=[top1,top2]`,
`real_tree_token_ids=[root_token,top1,top2]`, `real_tree_parent_indices=[-1,0,0]`,
`real_tree_depth=[0,1,1]`, `real_tree_rank=[-1,0,1]`,
`real_tree_logits=[0,top1_logit,top2_logit]`,
`rank_semantics=rank_stable_descending_logit`, `accept_path_len=0`,
`actual_accepted_nodes=0`, and `correction_token_present=0`. The trace proof is
that `real_tree_token_ids[1:]` exactly matches the P5AK `candidate_ids` and
`real_tree_logits[1:]` exactly matches the P5AK `candidate_logits`. It does not rewrite the existing P5AE/P5AF/P5AG synthetic tree hashes. This slice performs
no accept, no target logits walk, no target accept walk, no sampler, no token
commit, no hidden/KV commit, no rejected-branch discard, no publish, no visible
state change, no KV mutation, and no draft tokens.

P5AM real draft-head top-k verify-mask ABI is a separate default-off shadow real-tree verify-mask ABI bridge slice gated by `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_VERIFY_MASK_ABI_ONLY=1`. It requires P5AL shadow real-tree readiness, P5AK real candidate readiness, P5AJ logits readiness, and the P5AG/P5AF/P5AE/P5X chain. The runtime record is `real_topk_verify_mask_runtime_ready=1`, `real_topk_tree_runtime_ready=1`, `actual_tree_nodes=3`, `real_tree_token_ids=[root_token,top1,top2]`, `actual_verify_mask_entries=5`, `verify_mask_rows=3`, `verify_mask_cols=3`, `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`, `real_verify_mask_rows=[0,1,1,2,2]`, `real_verify_mask_cols=[0,0,1,0,2]`, `real_verify_mask_values=[1,1,1,1,1]`, `prefix_visible=1`, `ancestor_only=1`, `sibling_visible=0`, and `descendant_visible=0`. It does not rewrite P5AF/P5AG synthetic hashes or canonical mask arrays. It performs no mask tensor allocation, no accept, no target logits walk, no target accept walk, no token commit, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens.

P5AN real draft-head top-k accept-boundary ABI is a no-model contract-wiring slice for the next default-off boundary gated by `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_BOUNDARY_ABI_ONLY=1`. It requires P5AM shadow real verify-mask readiness, P5AL shadow real-tree readiness, P5AK real candidate readiness, P5AJ logits readiness, and the P5AG/P5AF/P5AE/P5X chain. The expected trace is `draft-jetspec p5an_real_draft_head_topk_accept_boundary_runtime` with `phase=real_draft_head_topk_accept_boundary_ready`. The runtime record must include `real_topk_accept_boundary_runtime_ready=1`, `real_topk_verify_mask_runtime_ready=1`, `real_topk_tree_runtime_ready=1`, `real_topk_candidate_runtime_ready=1`, `actual_verified_logits_rows=1`, `actual_tree_nodes=3`, `real_tree_token_ids=[root_token,top1,top2]`, `candidate_ids=[top1,top2]`, `actual_verify_mask_entries=5`, `accept_boundary_candidate_nodes=2`, `accept_boundary_verified_edges=5`, `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`, `real_verify_mask_rows=[0,1,1,2,2]`, `real_verify_mask_cols=[0,0,1,0,2]`, `accept_decision_source=none_no_target_logits`, `accept_path_len=0`, `actual_accepted_nodes=0`, and `correction_token_present=0`. It performs no target logits walk, no target accept walk, no accept, no mask tensor allocation, no token commit, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens.

P5AO real draft-head top-k accept-path descriptor ABI is a default-off runtime hook plus no-model contract-wiring slice for the next descriptor boundary gated by `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_PATH_DESCRIPTOR_ABI_ONLY=1` and fails closed as `invalid_real_draft_head_topk_accept_path_descriptor_runtime`. It requires P5AN real accept-boundary readiness, P5AM shadow real verify-mask readiness, P5AL shadow real-tree readiness, P5AK real candidate readiness, P5AJ logits readiness, and the P5AG/P5AF/P5AE/P5X chain. The expected trace is `draft-jetspec p5ao_real_draft_head_topk_accept_path_descriptor_runtime` with `phase=real_draft_head_topk_accept_path_descriptor_ready`. The runtime record must include `real_topk_accept_path_descriptor_runtime_ready=1`, `real_topk_accept_boundary_runtime_ready=1`, `real_topk_verify_mask_runtime_ready=1`, `real_topk_tree_runtime_ready=1`, `real_topk_candidate_runtime_ready=1`, `logits_source=draft_head_full_vocab_logits`, `ctx_dft_present=1`, `decode_rc=0`, `logits_rows=1`, `logits_width=248320`, `actual_verified_logits_rows=1`, `topk_k=2`, `actual_tree_nodes=3`, `real_tree_token_ids=[root_token,top1,top2]`, `candidate_nodes=2`, `candidate_ids=[top1,top2]`, `rank_semantics=rank_stable_descending_logit`, `actual_verify_mask_entries=5`, `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`, `accept_boundary_candidate_nodes=2`, `accept_boundary_verified_edges=5`, `accept_path_descriptor_len=0`, `actual_accepted_nodes=0`, `correction_token_present=0`, `accept_decision_source=none_no_target_logits`, `descriptor_only=1`, and `reuse_p5an_accept_boundary_metadata=1`. It also requires `actual_committed_tokens=0`, `actual_survivor_pages_committed=0`, `actual_pages_discarded=0`, and `actual_publish_visible_state=0`. It performs no target logits walk, no target accept walk, no accept, no token commit, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens. The trace contract rejects token commit readiness, hidden KV readiness, rejected branch discard readiness, and publish readiness.

P5AP real draft-head top-k token-commit no-op ABI is a default-off runtime hook plus no-model contract-wiring slice for the next token-commit boundary gated by `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TOKEN_COMMIT_NOOP_ABI_ONLY=1` and fails closed as `invalid_real_draft_head_topk_token_commit_noop_runtime`. It is not the generic P5T token-commit descriptor and not the root P5AA token-commit no-op path. It requires P5AO real accept-path descriptor readiness, P5AN real accept-boundary readiness, P5AM shadow real verify-mask readiness, P5AL shadow real-tree readiness, P5AK real candidate readiness, P5AJ logits readiness, and the P5AG/P5AF/P5AE/P5X chain. The source hook is limited to `common/speculative.cpp` plus experiments/docs contracts and has no server, public API, ggml, or CMake wiring. The expected trace is `draft-jetspec p5ap_real_draft_head_topk_token_commit_noop_runtime` with `phase=real_draft_head_topk_token_commit_noop_ready`. The runtime record must include `real_topk_token_commit_noop_runtime_ready=1`, `real_topk_accept_path_descriptor_runtime_ready=1`, `real_topk_accept_boundary_runtime_ready=1`, `real_topk_verify_mask_runtime_ready=1`, `real_topk_tree_runtime_ready=1`, `real_topk_candidate_runtime_ready=1`, `logits_source=draft_head_full_vocab_logits`, `ctx_dft_present=1`, `decode_rc=0`, `logits_rows=1`, `logits_width=248320`, `actual_verified_logits_rows=1`, `topk_k=2`, `actual_tree_nodes=3`, `real_tree_token_ids=[root_token,top1,top2]`, `candidate_nodes=2`, `candidate_ids=[top1,top2]`, `rank_semantics=rank_stable_descending_logit`, `actual_verify_mask_entries=5`, `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`, `accept_boundary_candidate_nodes=2`, `accept_boundary_verified_edges=5`, `accept_path_descriptor_len=0`, `actual_accepted_nodes=0`, `correction_token_present=0`, `accept_decision_source=none_no_target_logits`, `token_commit_noop=1`, and `reuse_p5ao_accept_path_descriptor=1`. It also requires `actual_committed_tokens=0`, `actual_survivor_pages_committed=0`, `actual_pages_discarded=0`, and `actual_publish_visible_state=0`. It performs no target logits walk, no target accept walk, no accept, no real token commit, no visible token publish, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens, and returns before downstream hidden/KV/discard/publish. The trace contract rejects `token_commit_descriptor_ready=1`, `root_token_commit_noop_runtime_ready=1`, hidden KV readiness, rejected branch discard readiness, publish readiness, `#gen drafts = 1`, and `#gen tokens = 1`.

P5AQ real draft-head top-k hidden/KV commit no-op ABI is a default-off runtime hook plus no-model contract-wiring slice for the next hidden/KV boundary gated by `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_HIDDEN_KV_COMMIT_NOOP_ABI_ONLY=1` and fails closed as `invalid_real_draft_head_topk_hidden_kv_commit_noop_runtime`. It is not the generic P5U hidden/KV survivor commit descriptor and not the root P5AB hidden/KV commit no-op path. It requires P5AP real token-commit no-op readiness plus P5AO/P5AN/P5AM/P5AL/P5AK readiness. The source hook is limited to `common/speculative.cpp` plus experiments/docs contracts and has no server, public API, ggml, or CMake wiring. The expected trace is `draft-jetspec p5aq_real_draft_head_topk_hidden_kv_commit_noop_runtime` with `phase=real_draft_head_topk_hidden_kv_commit_noop_ready`. The runtime record must include `real_topk_hidden_kv_commit_noop_runtime_ready=1`, `real_topk_token_commit_noop_runtime_ready=1`, `real_topk_accept_path_descriptor_runtime_ready=1`, `real_topk_accept_boundary_runtime_ready=1`, `real_topk_verify_mask_runtime_ready=1`, `real_topk_tree_runtime_ready=1`, `real_topk_candidate_runtime_ready=1`, `logits_source=draft_head_full_vocab_logits`, `ctx_dft_present=1`, `decode_rc=0`, `logits_rows=1`, `logits_width=248320`, `actual_verified_logits_rows=1`, `topk_k=2`, `actual_tree_nodes=3`, `real_tree_token_ids=[root_token,top1,top2]`, `candidate_nodes=2`, `candidate_ids=[top1,top2]`, `rank_semantics=rank_stable_descending_logit`, `actual_verify_mask_entries=5`, `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`, `accept_boundary_candidate_nodes=2`, `accept_boundary_verified_edges=5`, `accept_path_descriptor_len=0`, `actual_accepted_nodes=0`, `correction_token_present=0`, `accept_decision_source=none_no_target_logits`, `token_commit_noop=1`, `hidden_kv_commit_noop=1`, and `reuse_p5ap_token_commit_noop=1`. It also requires `actual_committed_tokens=0`, `actual_survivor_pages_committed=0`, `actual_pages_discarded=0`, and `actual_publish_visible_state=0`. It performs no target logits walk, no target accept walk, no accept, no real token commit, no visible token publish, no real hidden/KV commit, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens, and returns before downstream rejected/discard/publish. The trace contract rejects `hidden_kv_survivor_commit_descriptor_ready=1`, `hidden_kv_commit_runtime_ready=1`, `root_hidden_kv_commit_noop_runtime_ready=1`, `rejected_branch_discard_descriptor_ready=1`, `publish_gate_descriptor_ready=1`, `token_commit_descriptor_ready=1`, `root_token_commit_noop_runtime_ready=1`, `#gen drafts = 1`, and `#gen tokens = 1`.

P5AR real draft-head top-k rejected-branch discard no-op ABI is a default-off runtime hook plus no-model contract-wiring slice for the next rejected-branch discard boundary gated by `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_REJECTED_BRANCH_DISCARD_NOOP_ABI_ONLY=1` and fails closed as `invalid_real_draft_head_topk_rejected_branch_discard_noop_runtime`. It is not the generic P5V rejected-branch discard descriptor and not the root P5AC rejected-branch discard no-op path. After P5AQ plus P5AP/P5AO/P5AN/P5AM/P5AL/P5AK/P5AJ/P5AG/P5AF/P5AE/P5X readiness and with the root-tail conflict disabled, the expected trace is `draft-jetspec p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime` with `phase=real_draft_head_topk_rejected_branch_discard_noop_ready`. The contract requires `real_topk_rejected_branch_discard_noop_runtime_ready=1`, `real_topk_hidden_kv_commit_noop_runtime_ready=1`, `real_topk_token_commit_noop_runtime_ready=1`, `real_topk_accept_path_descriptor_runtime_ready=1`, `real_topk_accept_boundary_runtime_ready=1`, `real_topk_verify_mask_runtime_ready=1`, `real_topk_tree_runtime_ready=1`, `real_topk_candidate_runtime_ready=1`, `logits_source=draft_head_full_vocab_logits`, `ctx_dft_present=1`, `decode_rc=0`, `logits_rows=1`, `logits_width=248320`, `actual_verified_logits_rows=1`, `topk_k=2`, `actual_tree_nodes=3`, `real_tree_token_ids=[root_token,top1,top2]`, `candidate_nodes=2`, `candidate_ids=[top1,top2]`, `rank_semantics=rank_stable_descending_logit`, `actual_verify_mask_entries=5`, `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`, `accept_boundary_candidate_nodes=2`, `accept_boundary_verified_edges=5`, `accept_path_descriptor_len=0`, `actual_accepted_nodes=0`, `correction_token_present=0`, `accept_decision_source=none_no_target_logits`, `token_commit_noop=1`, `hidden_kv_commit_noop=1`, `reuse_p5aq_hidden_kv_commit_noop=1`, and `rejected_branch_discard_noop=1`. It requires side-effect counters to remain zero: `actual_committed_tokens=0`, `actual_survivor_pages_committed=0`, `actual_pages_discarded=0`, `rejected_branch_pages_reachable_after_discard=0`, and `actual_publish_visible_state=0`. It performs no target logits walk, no target accept walk, no accept, no real token commit, no visible token publish, no real hidden/KV commit, no hidden/KV commit, no real rejected-branch discard, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens, and returns before downstream publish. The trace contract rejects `rejected_branch_discard_descriptor_ready=1`, `rejected_branch_discard_runtime_ready=1`, `root_rejected_branch_discard_noop_runtime_ready=1`, `publish_gate_descriptor_ready=1`, `root_publish_gate_noop_runtime_ready=1`, hidden KV readiness, token commit readiness, `#gen drafts = 1`, and `#gen tokens = 1`. `validate_p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime.py` and `probe_p5ar_real_draft_head_topk_rejected_branch_discard_noop_trace.py` verify the contract wiring; live GPU/model execution remains separate.

P5AS real draft-head top-k publish-gate no-op ABI is a default-off runtime hook plus no-model contract-wiring slice for the publish boundary gated by `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_PUBLISH_GATE_NOOP_ABI_ONLY=1` and fails closed as `invalid_real_draft_head_topk_publish_gate_noop_runtime`. It is not the generic P5W publish-gate descriptor and not the root P5AD publish-gate no-op path. After P5AR plus P5AQ/P5AP/P5AO/P5AN/P5AM/P5AL/P5AK/P5AJ/P5AG/P5AF/P5AE/P5X readiness and with the root-tail conflict disabled, the expected trace is `draft-jetspec p5as_real_draft_head_topk_publish_gate_noop_runtime` with `phase=real_draft_head_topk_publish_gate_noop_ready`. The contract requires `real_topk_publish_gate_noop_runtime_ready=1`, `real_topk_rejected_branch_discard_noop_runtime_ready=1`, `real_topk_hidden_kv_commit_noop_runtime_ready=1`, `real_topk_token_commit_noop_runtime_ready=1`, `real_topk_accept_path_descriptor_runtime_ready=1`, `real_topk_accept_boundary_runtime_ready=1`, `real_topk_verify_mask_runtime_ready=1`, `real_topk_tree_runtime_ready=1`, `real_topk_candidate_runtime_ready=1`, `logits_source=draft_head_full_vocab_logits`, `ctx_dft_present=1`, `decode_rc=0`, `logits_rows=1`, `logits_width=248320`, `actual_verified_logits_rows=1`, `topk_k=2`, `actual_tree_nodes=3`, `real_tree_token_ids=[root_token,top1,top2]`, `candidate_nodes=2`, `candidate_ids=[top1,top2]`, `rank_semantics=rank_stable_descending_logit`, `actual_verify_mask_entries=5`, `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`, `accept_boundary_candidate_nodes=2`, `accept_boundary_verified_edges=5`, `accept_path_descriptor_len=0`, `actual_accepted_nodes=0`, `correction_token_present=0`, `accept_decision_source=none_no_target_logits`, `token_commit_noop=1`, `hidden_kv_commit_noop=1`, `rejected_branch_discard_noop=1`, `reuse_p5ar_rejected_branch_discard_noop=1`, `publish_gate_noop=1`, and `publish_after_commit_and_discard_only=1`. It requires side-effect counters to remain zero: `actual_committed_tokens=0`, `actual_survivor_pages_committed=0`, `actual_pages_discarded=0`, `rejected_branch_pages_reachable_after_discard=0`, and `actual_publish_visible_state=0`. It performs no target logits walk, no target accept walk, no accept, no real token commit, no visible token publish, no real hidden/KV commit, no hidden/KV commit, no real rejected-branch discard, no rejected-branch discard, no real publish, no publish, no visible state change, no KV mutation, and no draft tokens. The trace contract rejects `publish_gate_descriptor_ready=1`, `publish_runtime_ready=1`, `root_publish_gate_noop_runtime_ready=1`, rejected branch discard readiness, hidden KV readiness, token commit readiness, `#gen drafts = 1`, and `#gen tokens = 1`. `validate_p5as_real_draft_head_topk_publish_gate_noop_runtime.py` and `probe_p5as_real_draft_head_topk_publish_gate_noop_trace.py` verify the contract wiring; live GPU/model execution remains separate.

P5AV real draft-head top-k target-logits walk canary is a default-off runtime hook plus no-model contract-wiring slice gated by `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TARGET_LOGITS_WALK_CANARY=1` and fails closed as `invalid_real_draft_head_topk_target_logits_walk_canary_runtime`. It is not a P5T token-commit descriptor, not a P5W publish-gate descriptor, and not a promotion of P5AS into visible state. After P5AS plus P5AR/P5AQ/P5AP/P5AO/P5AN/P5AM/P5AL/P5AK/P5AJ/P5AG/P5AF/P5AE/P5X readiness, the expected trace is `draft-jetspec p5av_real_draft_head_topk_target_logits_walk_canary_runtime` with `phase=real_draft_head_topk_target_logits_walk_canary_ready`. The contract requires `target_logits_walk_canary_ready=1`, `real_topk_publish_gate_noop_runtime_ready=1`, `planned_target_logits_rows=1`, `actual_target_logits_rows_walked=1`, `target_logits_source=target_model_full_vocab_logits`, `target_logits_width=248320`, `target_logits_batch_index`, `target_logits_pos`, `target_logits_seq_id`, `planned_parent_nodes=[0]`, `planned_candidate_nodes=[1,2]`, `row_semantics=parent_position_scores_candidate_children`, `candidate_ids=[top1,top2]`, `target_candidate_logits=[target_logit_top1,target_logit_top2]`, `accept_decision_source=target_logits_canary_only_no_accept`, and `target_logits_walk_canary_only=1`. It reuses one existing target logits row from the current target batch, issues no new target decode, and has source hook limited to `common/speculative.cpp` plus experiments/docs contracts with no server, public API, ggml, or CMake wiring. It requires side-effect counters to remain zero: `actual_target_accept_steps=0`, `actual_accepted_nodes=0`, `correction_token_present=0`, `actual_committed_tokens=0`, `actual_survivor_pages_committed=0`, `actual_pages_discarded=0`, `rejected_branch_pages_reachable_after_discard=0`, and `actual_publish_visible_state=0`. It performs no target accept walk, no accept, no token commit, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens. The trace contract intentionally rejects `no_target_logits_walk=1` because this canary walks one already-produced target logits row, and it rejects target accept, commit, publish, KV mutation, `#gen drafts = 1`, and `#gen tokens = 1`. `validate_p5av_real_draft_head_topk_target_logits_walk_canary_runtime.py` and `probe_p5av_real_draft_head_topk_target_logits_walk_canary_trace.py` verify the contract wiring; live GPU/model execution remains separate.

P5AW target-accept walk readiness is an inert experiments-only packet after P5AV and is not production source approval. It consumes the P5AV evidence that `actual_target_logits_rows_walked=1`, `target_logits_source=target_model_full_vocab_logits`, `target_logits_width=248320`, `planned_parent_nodes=[0]`, `planned_candidate_nodes=[1,2]`, and `target_candidate_logits=[target_logit_top1,target_logit_top2]` were captured with zero side effects. It records a future target-accept plan with `planned_target_accept_steps=1`, `planned_accept_parent_nodes=[0]`, `planned_accept_candidate_nodes=[1,2]`, `planned_accept_score_source=target_candidate_logits_from_p5av_row`, `planned_accept_rule=greedy_target_argmax_child_match_or_correction`, `planned_correction_token_source=target_full_vocab_argmax_when_no_child_match`, and `planned_accept_output_visibility=metadata_only_until_explicit_runtime_approval`. P5AW keeps `actual_target_accept_steps=0`, `actual_accepted_nodes=0`, `correction_token_present=0`, commit/publish/KV/draft-token counters at zero, and approval flags false, and reports `p5aw_target_accept_walk_readiness_verified_not_executed`. It adds no environment gate, no `common/speculative.cpp` hook, no server/public API/ggml/CMake wiring, no additional target logits walk, no target accept walk, no accept, no commit, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens. The future runtime-capable boundary would be a separate explicitly approved default-off target-accept walk canary, still with no token commit, no hidden/KV commit, no rejected-branch discard, no publish, no KV mutation, and no draft-token emission.

### n-gram Cache (`ngram-cache`)

An n-gram is a sequence of n tokens. The n-gram cache implementation maintains statistics about short n-gram sequences.
A draft is computed using probabilities derived from these statistics. External statistics can also be loaded from files for improved accuracy.

See:

- #5479, #6828, #6848

### n-gram Map (`ngram-simple`, `ngram-map-*`)

These implementations search the token history for patterns and use matching sequences as draft candidates.
They require no additional model but rely on patterns that have already appeared in the generated text.
An example to use this approach can be the rewriting of source code by a LLM.

#### n-gram Map (`ngram-simple`)

This implementation looks for the last n-gram in history that matches the current n-gram and creates a draft using the m tokens following the matched n-gram. It is the simplest self-speculative approach with minimal overhead.

```
llama-server [...] --spec-type ngram-simple --spec-draft-n-max 64
```

#### n-gram Map Key (`ngram-map-k`)

This implementation looks for the current n-gram of size n (called the _key_) in the token history. If the key n-gram is followed by the same m tokens (called the _mgram_) multiple times, it creates a draft using these m tokens. This approach requires a minimum number of occurrences (argument `--spec-ngram-map-k-min-hits`, default is 1) before generating drafts.

The number of accepted tokens is stored for each used n-gram.

**Example:**
```
llama-server [...] --spec-type ngram-map-k --spec-draft-n-max 64
```

#### n-gram Map Key-4-Values (`ngram-map-k4v`)

This experimental implementation looks for the current n-gram of size n (called the _key_) in the token history. For each key, up to four _values_ (n-grams of size m, called _mgrams_) are tracked. An internal statistic counts the occurrences of each mgram after the key n-gram. If one mgram is significantly more frequent than the others, it is used as the draft.

The number of accepted tokens is stored for each used n-gram.

**Example:** Server options to be used if there are a lot of longer repetitions.
```
llama-server [...] --spec-type ngram-map-k4v --spec-ngram-map-k4v-size-n 8 --spec-ngram-map-k4v-size-m 8 --spec-ngram-map-k4v-min-hits 2 --spec-draft-n-max 64
```

### n-gram Mod (`ngram-mod`)

Add basic ngram hasher for speculative decoding:

- For each ngram, compute a hash using LCG
- For each computed hash, store the next token
- During speculation, iteratively compute the rolling hash of the last n tokens and pick the next token from the storage

Some characteristics:

- Lightweight (~16 MB)
- Constant memory and complexity
- Can generate variable draft lengths (i.e. m is not fixed)

Currently, a single hash pool is shared across all server slots, so different requests can benefit from each other.

**Sample usage:**

```
# notes:
# - small `n` are not recommended
# - MoEs require long drafts
# - dense models: can reduce `--spec-ngram-mod-n-min` and `--spec-ngram-mod-n-max`

llama-server ... --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
```

Applications:

- Iterating over a block of text/code (e.g. in llama.vim)
- Reasoning models (when they have to repeat their thinking in the final answer)
- Summarization

Example Video:

- See #19164

### Differences between ngram-simple, ngram-map and ngram-mod

- ngram-simple looks for a previous matching n-gram and inserts the following m-gram.
- ngram-map-k looks for a previous matching n-gram and inserts the following m-gram but uses an internal hash-map of n-grams in the current context window.
- ngram-mod uses a hash pool which is shared across all server slots. The hash pool is a map from n-gram hash to the next token (not the next m-gram as in ngram-map).

## Command-Line Options

If a draft model is combined with a draftless decoding the draftless decoding has higher precedence.

### General Speculative Parameters

```
--spec-type [none|draft-jetspec|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]
                                        type of speculative decoding to use when no draft model is provided
                                        (default: none)
                                        (env: LLAMA_ARG_SPEC_TYPE)
--spec-default                          use default speculative decoding
```

### Draft Model Parameters

```
--spec-draft-model, -md, --model-draft  FNAME
                                        draft model for speculative decoding (default: unused)
                                        (env: LLAMA_ARG_SPEC_DRAFT_MODEL)
--spec-draft-hf, -hfd, -hfrd, --hf-repo-draft  <user>/<model>[:quant]
                                        HuggingFace repository for the draft model
--spec-draft-n-max                      N
                                        number of tokens to draft for speculative decoding (default: 16)
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_MAX)
--spec-draft-n-min                      N
                                        minimum number of draft tokens to use for speculative decoding (default: 0)
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_MIN)
--spec-draft-p-split, --draft-p-split   P
                                        speculative decoding split probability (default: 0.10)
                                        (env: LLAMA_ARG_SPEC_DRAFT_P_SPLIT)
--spec-draft-p-min, --draft-p-min       P
                                        minimum speculative decoding probability (greedy) (default: 0.00)
                                        (env: LLAMA_ARG_SPEC_DRAFT_P_MIN)
--spec-draft-ctx-size, -cd, --ctx-size-draft  N
                                        size of the prompt context for the draft model (default: 0, 0 = loaded from model)
                                        (env: LLAMA_ARG_SPEC_DRAFT_CTX_SIZE)
--spec-draft-ngl, -ngld, --gpu-layers-draft, --n-gpu-layers-draft  N
                                        max. number of draft model layers to store in VRAM, either an exact number, 'auto', or 'all' (default: auto)
                                        (env: LLAMA_ARG_N_GPU_LAYERS_DRAFT)
--spec-draft-device, -devd, --device-draft  <dev1,dev2,..>
                                        comma-separated list of devices to use for offloading the draft model
--spec-draft-replace, --spec-replace    TARGET  DRAFT
                                        translate the string in TARGET into DRAFT if the draft model and main model are not compatible
```

### n-gram Mod Parameters

```
--spec-ngram-mod-n-match                N
                                        ngram-mod lookup length (default: 24)
--spec-ngram-mod-n-min                  N
                                        minimum number of ngram tokens to use for ngram-based speculative decoding (default: 48)
--spec-ngram-mod-n-max                  N
                                        maximum number of ngram tokens to use for ngram-based speculative decoding (default: 64)
```

### n-gram Simple Parameters

```
--spec-ngram-simple-size-n              N
                                        ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-simple-size-m              N
                                        ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-simple-min-hits            N
                                        minimum hits for ngram-simple speculative decoding (default: 1)
```

### n-gram Map Key Parameters

```
--spec-ngram-map-k-size-n               N
                                        ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-map-k-size-m               N
                                        ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-map-k-min-hits             N
                                        minimum hits for ngram-map-k speculative decoding (default: 1)
```

### n-gram Map Key-4-Values Parameters

```
--spec-ngram-map-k4v-size-n             N
                                        ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-map-k4v-size-m             N
                                        ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-map-k4v-min-hits           N
                                        minimum hits for ngram-map-k4v speculative decoding (default: 1)
```

### `--spec-type TYPE`

Specifies a type of speculative decoding without draft model.

| Type | Description |
|------|-------------|
| `none` | No speculative decoding (default) |
| `draft-jetspec` | Experimental JetSpec draft-head route; explicit opt-in and fail-closed before runtime support |
| `ngram-cache` | Use n-gram cache lookup |
| `ngram-simple` | Use simple n-gram pattern matching |
| `ngram-map-k` | Use n-gram pattern matching with n-gram-keys |
| `ngram-map-k4v` | Use n-gram pattern matching with n-gram-keys and up to four m-gram values (experimental) |
| `ngram-mod` | Use basic ngram hasher for speculative decoding with shared pool |

**Example:** Server-instance used to refactor source code.
```bash
./llama-server [...] --spec-type ngram-simple
```

### `--spec-ngram-*-size-n N`

Sets the size N of the lookup n-gram for n-gram map based speculative decoding.
The n-gram size N determines how many tokens in a row to look back when searching for matching patterns.

Each n-gram implementation has its own parameter:

- `--spec-ngram-simple-size-n` for `ngram-simple`
- `--spec-ngram-map-k-size-n` for `ngram-map-k`
- `--spec-ngram-map-k4v-size-n` for `ngram-map-k4v`
- `--spec-ngram-mod-n-match` for `ngram-mod`

### `--spec-ngram-*-size-m M`

Sets the size M of the draft m-gram for n-gram map based speculative decoding.
The m-gram size determines how many tokens to draft when a match is found.
Larger values can provide more speedup but may reduce acceptance rate.

Each n-gram implementation has its own parameter:

- `--spec-ngram-simple-size-m` for `ngram-simple`
- `--spec-ngram-map-k-size-m` for `ngram-map-k`
- `--spec-ngram-map-k4v-size-m` for `ngram-map-k4v`

### `--spec-ngram-*-min-hits H`

This option defines how often a key has to appear in the token history to be used as a draft (default is 1).

Each n-gram implementation has its own parameter:

- `--spec-ngram-simple-min-hits` for `ngram-simple`
- `--spec-ngram-map-k-min-hits` for `ngram-map-k`
- `--spec-ngram-map-k4v-min-hits` for `ngram-map-k4v`

## Statistics
Each speculative decoding implementation prints statistics.

```
draft acceptance rate = 0.57576 (  171 accepted /   297 generated)
statistics ngram_simple: #calls = 15, #gen drafts = 5, #acc drafts = 5, #gen tokens = 187, #acc tokens = 73
statistics draft: #calls = 10, #gen drafts = 10, #acc drafts = 10, #gen tokens = 110, #acc tokens = 98
```

```
draft acceptance rate = 0.70312 (   90 accepted /   128 generated)
statistics ngram_mod: #calls = 810, #gen drafts = 15, #acc drafts = 15, #gen tokens = 960, #acc tokens = 730, dur(b,g,a) = 0.149, 0.347, 0.005 ms
```

```
statistics ngram_map_k: #calls(b,g,a) = 6 1690 26, #gen drafts = 26, #acc drafts = 26, #gen tokens = 1248, #acc tokens = 968, dur(b,g,a) = 2.234, 1.427, 0.016 ms
```


- `#calls(b,g,a)`: number of calls of begin (new prompt), generation and accumulation of this implementations
- `#gen drafts`: number of drafts generated by this implementation
- `#acc drafts`: number of drafts accepted (partially) by the main model
- `#gen tokens`: number of tokens generated by this implementation (including rejected tokens)
- `#acc tokens`: number of tokens accepted by the main model
- `dur(b,g,a): durations of begin (new prompt), generation and accumulation (process acceptance).
