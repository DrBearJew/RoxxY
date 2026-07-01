# JetSpec P5AO real draft-head top-k accept-path descriptor ABI runtime candidate

Status: candidate, default-off runtime hook plus no-model contract wiring, descriptor ABI boundary.

P5AO defines the next shadow real accept-path descriptor ABI after P5AN. It is gated by `LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_PATH_DESCRIPTOR_ABI_ONLY=1`, fails closed as `invalid_real_draft_head_topk_accept_path_descriptor_runtime`, and requires the ready chain P5AN, P5AM, P5AL, P5AK, P5AJ, P5AG, P5AF, P5AE, and P5X. The source hook is limited to `common/speculative.cpp` plus experiments/docs contracts; it does not edit server, ggml, public API, CMake, tests, examples, or pocs.

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

Trace contract:

- `draft-jetspec p5ao_real_draft_head_topk_accept_path_descriptor_runtime`
- `phase=real_draft_head_topk_accept_path_descriptor_ready`
- `real_topk_accept_path_descriptor_runtime_ready=1`
- `real_topk_accept_boundary_runtime_ready=1`
- `real_topk_verify_mask_runtime_ready=1`
- `real_topk_tree_runtime_ready=1`
- `real_topk_candidate_runtime_ready=1`
- `logits_source=draft_head_full_vocab_logits`
- `ctx_dft_present=1`
- `decode_rc=0`
- `logits_rows=1`
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
- `descriptor_only=1`
- `reuse_p5an_accept_boundary_metadata=1`
- `actual_committed_tokens=0`
- `actual_survivor_pages_committed=0`
- `actual_pages_discarded=0`
- `actual_publish_visible_state=0`
- `no target logits walk`, `no target accept walk`, `no accept`, `no token commit`, `no hidden/KV commit`, `no rejected-branch discard`, `no publish`, `no visible state change`, `no KV mutation`, and `no draft tokens`

P5AO converts the P5AN real accept-boundary metadata into an accept-path descriptor record with `accept_path_descriptor_len=0`. It does not choose or commit accepted nodes, does not create a correction token, and does not advance hidden/KV, discard, or publish state. It must reject token commit readiness, hidden KV readiness, rejected branch discard readiness, and publish readiness in traces.

Validation:

- `python3 experiments/jetspec/validate_p5ao_real_draft_head_topk_accept_path_descriptor_runtime.py --json`
- `python3 experiments/jetspec/probe_p5ao_real_draft_head_topk_accept_path_descriptor_trace.py --json`
- `python3 -m unittest experiments/jetspec/test_p5ao_real_draft_head_topk_accept_path_descriptor_trace_probe.py`

Live model execution remains a separate explicit canary step and is not part of this descriptor ABI contract.
