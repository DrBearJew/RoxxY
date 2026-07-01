# JetSpec promotion checklist / ADR

Status: inert governance artifact. This document is not included by CMake and no
llama.cpp runtime reads it.

Decision: JetSpec integration remains staged and default-off. P5A loader
registration, P5B target-hidden tap capture, P5C fail-closed speculative type
parsing, P5D target-tap ingestion, P5E runtime-state bookkeeping, P5F binding
preflight, P5N transaction-plan scaffold, P5O pre-round snapshot descriptor, P5P transient-reservation descriptor, P5Q tree-build descriptor, P5R-P5W descriptor gates, P5X root-only runtime tree materialization, P5Y root-only verify-mask materialization, P5Z root-anchor accept-path materialization, P5AA root-token-commit no-op materialization, P5AB-P5AD root no-op tail readiness, and P5AE/P5AF/P5AG synthetic top-k ABI materialization have explicit approval as bounded
production candidates; P5AH draft-head top-k construction readiness is inert under `experiments/jetspec/`; executable JetSpec tree drafting remains blocked until
its own approval.
P5N transaction-plan scaffold, P5O pre-round snapshot descriptor, P5P transient-reservation descriptor, P5Q tree-build descriptor, P5R-P5W descriptor gates, P5X root-only runtime tree materialization, P5Y root-only verify-mask materialization, P5Z root-anchor accept-path materialization, P5AA root-token-commit no-op materialization, P5AB-P5AD root no-op tail readiness, and P5AE/P5AF/P5AG synthetic top-k ABI materialization have explicit approval; P5AH draft-head top-k readiness has no production hook approval.

## Current hard boundary

Allowed now:

- `experiments/jetspec/*.md`
- `experiments/jetspec/*.py`
- `experiments/jetspec/*.hpp` that are not included from compiled paths
- `experiments/jetspec/fixtures/*.json`
- `experiments/jetspec/conversion_plans/*.json`
- `experiments/jetspec/manifests/*.json`
- P5A validation-only production hooks: `src/llama-arch.h`, `src/llama-arch.cpp`, `src/llama-model.cpp`, `src/models/models.h`, and `src/models/jetspec_qwen3_draft_head.cpp`
- P5B private side-channel hooks: `src/llama-cparams.h`, `src/llama-graph.h`, `src/llama-graph.cpp`, `src/llama-context.h`, `src/llama-context.cpp`, `src/llama-ext.h`, `src/models/qwen35.cpp`, and `src/models/qwen35moe.cpp`
- P5C fail-closed speculative type hooks: `common/common.h`, `common/speculative.cpp`, and `docs/speculative.md`
- P5D target-tap ingestion hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5E runtime-state bookkeeping hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5F binding-preflight hooks: `common/speculative.cpp` and `docs/speculative.md`
- JetSpec model-only server binding: `tools/server/server-context.cpp` with `ctx_dft=nullptr`, no draft context, and no graph execution
- P5N transaction-plan scaffold hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5O pre-round snapshot hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5P transient-reservation descriptor hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5Q tree-build descriptor hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5R verify-mask descriptor hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5S accept-path descriptor hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5T token-commit descriptor hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5U hidden/KV survivor commit descriptor hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5V rejected-branch discard descriptor hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5W publish-gate descriptor hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5X root-only runtime tree hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5Y root-only verify-mask runtime hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5Z root-anchor accept-path runtime hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5AA root-token-commit no-op runtime hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5AB-P5AD root no-op tail readiness hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5AE synthetic top-k tree ABI hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5AF synthetic top-k verify-mask ABI hooks: `common/speculative.cpp` and `docs/speculative.md`
- P5AG synthetic top-k accept-boundary ABI hooks: `common/speculative.cpp` and `docs/speculative.md`

Forbidden until promotion gates explicitly allow it:

- any non-P5A/P5B `src/models/*.cpp` or model registration edit
- `common/` files other than the approved P5C/P5D/P5E/P5F/P5N/P5O/P5P/P5Q/P5R/P5S/P5T/P5U/P5V/P5W/P5X/P5Y/P5Z/P5AA/P5AB-P5AD/P5AE/P5AF/P5AG hook set
- `tools/server/` except `tools/server/server-context.cpp` model-only JetSpec binding with `ctx_dft=nullptr`, no draft context, and no graph execution
- repository `tests/`
- `examples/`
- `pocs/`
- `ggml/src/`
- top-level `CMakeLists.txt` or any production `.cmake` wiring

The original `llama-server` build must remain unaffected. The aggregate verifier
must continue to report no CMake references to `experiments/jetspec`.

## Promotion phases

### P0: inert contract baseline

Required before any compiled-path work:

- `python3 run_all_jetspec_contracts.py` passes.
- GGUF plan validates 91 expected BF16 tensors.
- Metadata-only GGUF preview validates as `runtime_supported=false`.
- Runtime-state header validator passes with local-only includes.
- CMake isolation scan reports no references.

Current status: P4 tree verify and rollback parity evidence staged; P5 default-off plan drafted; P5A loader-registration candidate approved and implemented as validation-only fail-closed production hooks; P5B target-hidden tap capture approved and implemented as private side-channel hooks; P5C speculative type parsing approved and implemented as fail-closed non-executable route hooks; P5D target-tap ingestion approved and implemented as a non-drafting runtime slice; P5E runtime-state bookkeeping approved and implemented as a state-only runtime slice; P5F binding preflight and model-only server binding approved and implemented as a preflight-only/non-drafting runtime slice; P5N transaction-plan scaffold approved and implemented as a non-drafting transaction scaffold; P5O pre-round snapshot descriptor approved and implemented as a non-drafting snapshot descriptor; P5P transient-reservation descriptor approved and implemented as a non-drafting descriptor-only reservation slice; P5Q tree-build descriptor approved and implemented as a non-drafting descriptor-only tree-build slice; P5R verify-mask descriptor approved and implemented as a non-drafting descriptor-only verify-mask slice; P5S accept-path descriptor approved and implemented as a non-drafting descriptor-only accept slice; P5T token-commit descriptor approved and implemented as a non-drafting descriptor-only token-commit slice; P5U hidden/KV survivor commit descriptor approved and implemented as a non-drafting descriptor-only hidden/KV commit slice; P5V rejected-branch discard descriptor approved and implemented as a non-drafting descriptor-only discard slice; P5W publish-gate descriptor approved and implemented as a non-drafting descriptor-only publish gate; P5X root-only runtime tree materialization approved and implemented as a default-off one-node tree ABI slice; P5Y root-only verify-mask materialization approved and implemented as a default-off one-entry root self-mask ABI slice; P5Z root-anchor accept-path materialization approved and implemented as a default-off root-anchor accept ABI slice with no accepted draft tokens, correction token, commit, or publish; P5AA root-token-commit no-op materialization approved and implemented as a default-off root token-commit no-op ABI slice with actual_committed_tokens=0 and no visible publish or KV mutation; P5AB-P5AD root no-op tail readiness approved and implemented as default-off hidden/KV, discard, and publish no-op ABI slices with root_runtime_ready_for_real_test=1 and no KV mutation or visible publish; P5AE synthetic top-k tree ABI, P5AF synthetic top-k verify-mask ABI, and P5AG synthetic top-k accept-boundary ABI materialization approved and implemented as default-off synthetic non-root ABI slices with no draft logits, mask tensor, target logits walk, target accept walk, KV mutation, publish, or draft tokens; P5AH draft-head top-k readiness is staged as inert experiments-only evidence with `planned_draft_head_logits_rows=1`, `actual_verified_logits_rows=0`, `future_logits_source=draft_head_full_vocab_logits`, `runtime_supported=false`, and `ctx_dft=nullptr`; executable JetSpec drafting remains forbidden until separate explicit approval.

P5F artifact status: `validate_p5f_artifact_binding.py` now verifies the stored real JetSpec draft-head manifest/tensor map/conversion plan against P5F source constants and reports `artifact_verified_not_runtime_executed`. This covers 91 BF16 tensors, `fc.weight` shape `[2048, 10240]`, target tap layers `[1, 10, 19, 28, 37]`, target hidden size 2048, tap width 10240, draft block size 16, 8 draft layers, 32 attention heads, 4 KV heads, vocab size 248320, `requires_target_embeddings=true`, `requires_target_lm_head=true`, and `runtime_supported=false`. The server may bind the 91-tensor draft-head as model-only under explicit JetSpec gates, including `LLAMA_JETSPEC_DRAFT_HEAD_LOAD=1`, with `ctx_dft=nullptr`; graph execution remains blocked. `probe_p5f_loader_gate.py` writes the metadata-only preview and verifies the built `llama-cli` fails closed as `preview_not_allowed` by default and `unsupported_runtime` with `LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1` plus `LLAMA_JETSPEC_EXPERIMENTAL=1`, and source-asserts true-valued `runtime_supported` metadata is rejected before optional load gates. `probe_p5f_target_tensor_binding.py` reads target GGUF split headers only and reports `target_tensor_headers_verified_not_loaded` for `token_embd.weight`, `output.weight`, and `output_norm.weight` linker shapes without loading weights or creating contexts. It still does not execute the draft-head graph or build/verify a tree.

P5G readiness status: `jetspec_tree_runtime_readiness.py --fixture fixtures/jetspec_tree_runtime_readiness_smoke.json` now validates an inert tree-runtime readiness contract and reports `tree_runtime_readiness_verified_not_executed`. This covers DraftTree parent-before-child ABI, full-vocab top-k logprobs without top-k-only renormalization, accum-logp expected trees, ancestor-only masks, hidden other-sequence tree columns, root-inclusive `accepted_path`, `acceptance_length` excluding root, deterministic duplicate-child overwrite, `[accepted draft tokens | correction]`, `[root | accepted] only`, correction hidden not appended in the same round, `max_len + accepted_path` gather positions, and a no-runtime boundary with no target/draft `llama_context`, no draft-head graph execution, no draft tokens, no KV cache mutation, and no server route. `validate_p5g_tree_runtime_readiness.py` and `test_p5g_tree_runtime_readiness.py` keep this under `experiments/jetspec/`; production tree runtime remains blocked.

P5H readiness status: `jetspec_kv_commit_readiness.py --fixture fixtures/jetspec_kv_commit_readiness_smoke.json` now validates an inert KV/hidden commit ownership contract and reports `kv_commit_readiness_verified_not_executed`. This covers abstract transient tree slots, model hidden/KV rows trailing committed tokens by one, committed tokens `[accepted draft tokens | correction]`, hidden/KV survivors `[root | accepted]`, correction hidden not appended in the same round, rejected tree nodes unreachable after commit, `past_len + accepted_path` gather positions, duplicate/out-of-range accepted path rejection, cross-sequence slot isolation, explicit `missing primitive` ownership mapping for future llama.cpp KV operations, and a no-runtime boundary with no `llama_context`, no draft-head graph execution, no draft tokens, no real KV cache mutation, no server route, and `runtime_supported=false`. `validate_p5h_kv_commit_readiness.py` and `test_p5h_kv_commit_readiness.py` keep this under `experiments/jetspec/`; production tree runtime remains blocked.

P5I approval-packet status: `tree_runtime_approval_matrix.py --fixture fixtures/tree_runtime_approval_matrix_smoke.json` now validates an inert tree-runtime approval packet and reports `tree_runtime_approval_packet_verified_not_executed`. This maps future runtime actions to `validated_by_p5g`, `validated_by_p5h`, `missing_primitive`, or `blocked_pending_explicit_approval`; requires explicit primitive names for tree build, verify mask, accept path, token commit, hidden/KV survivor commit, rejected branch discard, cross-sequence isolation, and rollback/fail-closed disable; rejects production path touches, runtime execution claims, implicit primitives like `seq_cp`/`seq_rm`, performance claims, and promotion claims; and requires future gates for aggregate contracts, default build, disabled path, existing draft-MTP path, JetSpec opt-in fail-closed behavior, correctness matrix, and baseline benchmark comparison. `validate_p5i_tree_runtime_approval_packet.py` and `test_p5i_tree_runtime_approval_packet.py` keep this under `experiments/jetspec/`; production tree runtime remains blocked.

P5J primitive-audit status: `kv_primitive_audit.py --fixture fixtures/kv_primitive_audit_smoke.json` now validates a read-only KV/runtime primitive audit and reports `kv_primitive_audit_verified_not_executed`. It scans `src/llama-kv-cache.h` and `src/llama-kv-cache.cpp` for existing helpers `seq_rm`, `seq_cp`, `seq_import_physical`, `seq_keep`, `find_slot`, and `apply_ubatch`, then classifies hidden/KV survivor commit, rejected branch discard, and cross-sequence isolation as `exact_missing_primitive`. It rejects implicit mappings to `seq_cp`, `seq_rm`, or `seq_import_physical`, because those helpers are not exact JetSpec accepted-path tree gather/compact/discard ownership primitives. `validate_p5j_kv_primitive_audit.py` and `test_p5j_kv_primitive_audit.py` keep this under `experiments/jetspec/`; production tree runtime remains blocked.

P5K primitive-design status: `kv_ownership_primitive_design.py --fixture fixtures/kv_ownership_primitive_design_smoke.json` now validates an inert KV ownership primitive design packet and reports `kv_ownership_primitive_design_verified_not_executed`. Existing helpers remain audited non-exact helpers. The packet names design-only missing-implementation candidates `llama_kv_cache_jetspec_commit_survivor_path_candidate`, `llama_kv_cache_jetspec_discard_rejected_tree_candidate`, and `llama_kv_cache_jetspec_assert_cross_sequence_isolation_candidate`; requires accepted-path physical gather/compact, `[root | accepted] only`, correction hidden deferred, committed tail compact, rejected transient tree slots unreachable, not range removal only, rollback restores pre-round state, other-sequence slots unchanged, `seq_to_stream` isolation, no shared-slot corruption, and rollback preserving other sequences; and rejects implementation approval or implicit `seq_cp`/`seq_rm`/`seq_import_physical` mapping. `validate_p5k_kv_ownership_primitive_design.py` and `test_p5k_kv_ownership_primitive_design.py` keep this under `experiments/jetspec/`; production tree runtime remains blocked.

P5L page-map ownership oracle status: `page_map_ownership_oracle.py --fixture fixtures/page_map_ownership_oracle_smoke.json` now validates an inert page-map ownership oracle and reports `page_map_ownership_oracle_verified_not_executed`. The fixture is `page_map_oracle_only`: it models abstract page descriptors, accepted survivor pages, rejected branch pages, cross-sequence guard pages, and QBlock/PageAttention safety lessons without runtime execution. It requires accepted path pages map to `[root | accepted]` only, correction hidden deferred, accepted path physical gather/compact explicit, rejected transient pages unreachable after commit, accepted path cannot read rejected siblings or descendants, rollback restores pre-round page snapshot, other-sequence pages unchanged, no duplicate mutable physical page ownership, and rollback preserves other sequences. It keeps `seq_cp`/`seq_rm`/`seq_import_physical` as audited non-exact helpers; names design-only missing-implementation candidates `llama_kv_cache_jetspec_validate_page_ownership_oracle_candidate`, `llama_kv_cache_jetspec_reserve_transient_tree_pages_candidate`, `llama_kv_cache_jetspec_commit_page_survivor_path_candidate`, and `llama_kv_cache_jetspec_discard_rejected_tree_pages_candidate`; preserves identity maps are only parity/oracle cases, visible noncanonical owned overlays fail closed, a full current-K map requirement, and canonical write-through remains required; and rejects performance or promotion claims. `jetspec_p5l_page_map_ownership_oracle.md`, `validate_p5l_page_map_ownership_oracle.py`, and `test_p5l_page_map_ownership_oracle.py` keep this under `experiments/jetspec/`; production tree runtime remains blocked.

P5M transaction/failpoint plan oracle status: `transaction_plan_oracle.py --fixture fixtures/transaction_plan_oracle_smoke.json` now validates an inert transaction/failpoint plan oracle and reports `transaction_plan_oracle_verified_not_executed`. The fixture is `transaction_plan_oracle_only`: it orders `snapshot_pre_round`, `reserve_transient_tree_pages`, `build_tree`, `build_verify_mask`, `accept_path`, `commit_tokens`, `commit_hidden_kv_survivors`, `discard_rejected_branches`, and `publish_post_commit_state`; requires rollback points after reserve, tree build, verify mask, accept, token commit, hidden/KV commit, and rejected discard; keeps all committed token, hidden/KV, and page-map visibility hidden before publish; requires post-publish tokens `[accepted draft tokens | correction]`, hidden/KV survivors `[root | accepted]`, correction hidden deferred, rejected branches unreachable, other-sequence pages unchanged, no duplicate mutable physical page ownership, and pre-round snapshot restoration at every failpoint. It keeps `seq_cp`/`seq_rm`/`seq_import_physical` as audited non-exact helpers; names `llama_kv_cache_jetspec_rollback_tree_transaction_candidate` as design-only missing implementation; rejects implementation approval, runtime execution, production path touches, performance claims, and promotion claims. `jetspec_p5m_transaction_plan_oracle.md`, `validate_p5m_transaction_plan_oracle.py`, and `test_p5m_transaction_plan_oracle.py` keep this under `experiments/jetspec/`; production tree runtime remains blocked.

P5N transaction-plan scaffold status: `validate_p5n_transaction_scaffold.py` now validates the first approved bounded tree-runtime source slice. The slice is still default-off and non-drafting: after `draft-jetspec` passes P5F preflight and captures target taps, `common/speculative.cpp` records `JETSPEC_TRANSACTION_PHASE_ORDER`, `JETSPEC_TRANSACTION_ROLLBACK_POINTS`, `transaction_plan_scaffold_ready`, `transaction_plan_hash_last`, and `invalid_transaction_plan` fail-closed handling. Trace output reports `transaction_plan_ready`, `transaction_plan_hash`, `no_kv_mutation=1`, `no_publish=1`, and `no_draft_tokens=1`. `jetspec_p5n_transaction_scaffold_candidate.md`, `validate_p5n_transaction_scaffold.py`, and `test_p5n_transaction_scaffold.py` verify the source slice remains limited to `common/speculative.cpp` and `docs/speculative.md`, with no draft-head graph execution, no draft tokens emitted, no real KV mutation, no CUDA dispatch, no server route, no public API, no CMake wiring, no performance claim, and no promotion claim. Full tree build, verify, accept, commit, publish, real rollback, server behavior, repository tests/examples/pocs, `ggml/src`, and public API remain blocked.

P5O pre-round snapshot status: `validate_p5o_pre_round_snapshot.py` now validates the approved bounded pre-round snapshot descriptor source slice. The slice is still default-off and non-drafting: `common/speculative.cpp` records `JETSPEC_PRE_ROUND_SNAPSHOT_PHASE`, `pre_round_snapshot_ready`, `pre_round_snapshot_hash_last`, `pre_round_prompt_tokens_last`, `pre_round_seq_id_last`, and `invalid_pre_round_snapshot` fail-closed handling. Trace output reports `snapshot_ready`, `snapshot_hash`, `snapshot_prompt_tokens`, `transaction_phase=snapshot_pre_round`, `no_reserve=1`, `no_tree_build=1`, `no_verify_mask=1`, `no_kv_mutation=1`, `no_publish=1`, and `no_draft_tokens=1`. `jetspec_p5o_pre_round_snapshot_candidate.md`, `validate_p5o_pre_round_snapshot.py`, and `test_p5o_pre_round_snapshot.py` verify the pre-round snapshot descriptor remains limited to `common/speculative.cpp` and `docs/speculative.md`, with no reserve, tree build, verify mask, accept, commit, KV mutation, CUDA, server, public API, or CMake work. Full reserve, tree build, verify, accept, commit, publish, real rollback, server behavior, repository tests/examples/pocs, `ggml/src`, and public API remain blocked.

P5P transient-reservation descriptor status: `validate_p5p_transient_reservation_descriptor.py` now validates the approved bounded descriptor-only reservation source slice. The slice is still default-off and non-drafting: `common/speculative.cpp` records `JETSPEC_TRANSIENT_RESERVATION_PHASE`, `JETSPEC_TRANSIENT_RESERVATION_ROLLBACK_POINT`, `transient_reservation_descriptor_ready`, `transient_reservation_hash_last`, and `invalid_transient_reservation_descriptor` fail-closed handling. Trace output reports `transient_reservation_ready`, `transient_reservation_hash`, `transient_reservation_phase=reserve_transient_tree_pages`, `rollback_point=after_reserve`, `transient_tree_node_budget`, `actual_pages_reserved=0`, `pre_publish_visible_state_unmodified=1`, `no_tree_build=1`, `no_verify_mask=1`, `no_kv_mutation=1`, `no_publish=1`, and `no_draft_tokens=1`. `jetspec_p5p_transient_reservation_descriptor_candidate.md`, `validate_p5p_transient_reservation_descriptor.py`, and `test_p5p_transient_reservation_descriptor.py` verify the transient-reservation descriptor remains limited to `common/speculative.cpp` and `docs/speculative.md`, with no real page reservation, no llama_kv_cache primitive, no tree build, no verify mask, no draft tokens, no CUDA, no server route, no public API, and no CMake wiring. Full real reservation, tree build, verify, accept, commit, publish, real rollback, server behavior, repository tests/examples/pocs, `ggml/src`, and public API remain blocked.

P5Q tree-build descriptor status: `validate_p5q_tree_build_descriptor.py` now validates the approved bounded descriptor-only tree-build source slice. The slice is still default-off and non-drafting: `common/speculative.cpp` records `JETSPEC_TREE_BUILD_PHASE`, `JETSPEC_TREE_BUILD_ROLLBACK_POINT`, `tree_build_descriptor_ready`, `tree_build_descriptor_hash_last`, and `invalid_tree_build_descriptor` fail-closed handling. Trace output reports `tree_build_descriptor_ready`, `tree_build_descriptor_hash`, `tree_build_phase=build_tree`, `rollback_point=after_build_tree`, `planned_tree_node_budget`, `actual_tree_nodes=0`, `pre_publish_visible_state_unmodified=1`, `no_real_tree_build=1`, `no_tree_arrays=1`, `no_verify_mask=1`, `no_accept=1`, `no_kv_mutation=1`, `no_publish=1`, and `no_draft_tokens=1`. `jetspec_p5q_tree_build_descriptor_candidate.md`, `validate_p5q_tree_build_descriptor.py`, and `test_p5q_tree_build_descriptor.py` verify the tree-build descriptor remains limited to `common/speculative.cpp` and `docs/speculative.md`, with no real tree build, no tree arrays, no verify mask, no accept path runtime, no draft tokens, no CUDA, no server route, no public API, and no CMake wiring. Full real tree construction, verify, accept, commit, publish, real rollback, server behavior, repository tests/examples/pocs, `ggml/src`, and public API remain blocked.
P5R verify-mask descriptor status: `validate_p5r_verify_mask_descriptor.py` validates the approved bounded descriptor-only verify-mask source slice. The slice records `JETSPEC_VERIFY_MASK_PHASE`, `JETSPEC_VERIFY_MASK_ROLLBACK_POINT`, `verify_mask_descriptor_ready`, `verify_mask_descriptor_hash_last`, and `invalid_verify_mask_descriptor`; trace output reports `verify_mask_descriptor_hash`, `actual_verify_mask_entries=0`, `no_real_verify_mask=1`, `no_verify_mask=1`, `no_accept=1`, `no_kv_mutation=1`, `no_publish=1`, and `no_draft_tokens=1`.

P5S accept-path descriptor status: `validate_p5s_accept_path_descriptor.py` validates the approved bounded descriptor-only accept source slice. The slice records `JETSPEC_ACCEPT_PATH_PHASE`, `JETSPEC_ACCEPT_PATH_ROLLBACK_POINT`, `accept_path_descriptor_ready`, `accept_path_descriptor_hash_last`, and `invalid_accept_path_descriptor`; trace output reports `accept_path_descriptor_hash`, `actual_accepted_nodes=0`, `correction_token_present=0`, `no_real_accept=1`, `no_commit_tokens=1`, `no_kv_mutation=1`, `no_publish=1`, and `no_draft_tokens=1`.

P5T token-commit descriptor status: `validate_p5t_token_commit_descriptor.py` validates the approved bounded descriptor-only token-commit source slice. The slice records `JETSPEC_TOKEN_COMMIT_PHASE`, `JETSPEC_TOKEN_COMMIT_ROLLBACK_POINT`, `token_commit_descriptor_ready`, `token_commit_descriptor_hash_last`, and `invalid_token_commit_descriptor`; trace output reports `token_commit_descriptor_hash`, `actual_committed_tokens=0`, `no_real_token_commit=1`, `no_visible_token_publish=1`, `no_kv_mutation=1`, `no_publish=1`, and `no_draft_tokens=1`.

P5U hidden/KV survivor commit descriptor status: `validate_p5u_hidden_kv_survivor_commit_descriptor.py` validates the approved bounded descriptor-only hidden/KV survivor commit source slice. The slice records `JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_PHASE`, `JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_ROLLBACK_POINT`, `hidden_kv_survivor_commit_descriptor_ready`, `hidden_kv_survivor_commit_descriptor_hash_last`, and `invalid_hidden_kv_survivor_commit_descriptor`; trace output reports `hidden_kv_survivor_commit_descriptor_hash`, `actual_survivor_pages_committed=0`, `no_real_hidden_kv_commit=1`, `no_kv_mutation=1`, `no_publish=1`, and `no_draft_tokens=1`.

P5V rejected-branch discard descriptor status: `validate_p5v_rejected_branch_discard_descriptor.py` validates the approved bounded descriptor-only rejected-branch discard source slice. The slice records `JETSPEC_REJECTED_BRANCH_DISCARD_PHASE`, `JETSPEC_REJECTED_BRANCH_DISCARD_ROLLBACK_POINT`, `rejected_branch_discard_descriptor_ready`, `rejected_branch_discard_descriptor_hash_last`, and `invalid_rejected_branch_discard_descriptor`; trace output reports `rejected_branch_discard_descriptor_hash`, `actual_pages_discarded=0`, `rejected_branch_pages_reachable_after_discard=0`, `no_real_rejected_branch_discard=1`, `no_kv_mutation=1`, `no_publish=1`, and `no_draft_tokens=1`.

P5W publish-gate descriptor status: `validate_p5w_publish_gate_descriptor.py` validates the approved bounded descriptor-only publish gate source slice. The slice records `JETSPEC_PUBLISH_GATE_PHASE`, `publish_gate_descriptor_ready`, `publish_gate_descriptor_hash_last`, and `invalid_publish_gate_descriptor`; trace output reports `publish_gate_descriptor_hash`, `actual_publish_visible_state=0`, `publish_after_commit_and_discard_only=1`, `no_real_publish=1`, `no_visible_state_change=1`, and `no_draft_tokens=1`. Full real verify, accept, commit, discard, publish, rollback mutation, server behavior, repository tests/examples/pocs, `ggml/src`, and public API remain blocked.

P5X root-only runtime tree status: `validate_p5x_root_tree_runtime.py` validates the approved default-off one-node runtime tree source slice. With `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`, the slice records `JETSPEC_ROOT_TREE_RUNTIME_PHASE`, `tree_build_runtime_ready`, `root_tree_runtime_ready`, `root_tree_runtime_hash_last`, `actual_tree_nodes=1`, `tree_token_ids=[root_token]`, `tree_parent_indices=[-1]`, `tree_depth=[0]`, `tree_rank=[-1]`, `tree_cum_logprob=[0.0]`, `parent_before_child=1`, and `num_nodes_lte_budget=1`, then returns before P5R. `probe_p5x_root_tree_trace.py` validates the fast no-model trace contract, reports `p5x_root_tree_trace_contract_verified`, and can check a separately captured live log with `--trace-log`. It performs no draft-head graph execution, no top-k/non-root tree expansion, no verify mask, no accept, no token commit, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens.

P5Y root-only verify-mask status: `validate_p5y_root_verify_mask_runtime.py` validates the approved default-off one-entry root verify-mask runtime source slice. With `LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY=1` and `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`, the slice records `JETSPEC_ROOT_VERIFY_MASK_RUNTIME_PHASE`, `verify_mask_runtime_ready`, `root_verify_mask_runtime_ready`, `root_verify_mask_runtime_hash_last`, `actual_verify_mask_entries=1`, `verify_mask_rows=1`, `verify_mask_cols=1`, `root_attends_self=1`, `root_mask_row=0`, `root_mask_col=0`, `prefix_visible=1`, `ancestor_only=1`, `sibling_visible=0`, and `descendant_visible=0`, then returns before P5S. `probe_p5y_root_verify_mask_trace.py` validates the fast no-model trace contract, reports `p5y_root_verify_mask_trace_contract_verified`, and can check a separately captured live log with `--trace-log`. It performs no draft-head graph execution, no non-root verify mask, no mask tensor, no accept, no token commit, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens.

P5Z root-anchor accept-path status: `validate_p5z_root_anchor_accept_path_runtime.py` validates the approved default-off root-anchor accept-path runtime source slice. With `LLAMA_JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_ONLY=1`, `LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY=1`, and `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`, the slice records `JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_RUNTIME_PHASE`, `accept_path_runtime_ready`, `root_anchor_accept_path_runtime_ready`, `root_anchor_accept_path_runtime_hash_last`, `root_verified_anchor=1`, `accept_path_len=0`, `actual_accepted_nodes=0`, and `correction_token_present=0`, then returns before P5T. `probe_p5z_root_anchor_accept_path_trace.py` validates the fast no-model trace contract, reports `p5z_root_anchor_accept_path_trace_contract_verified`, and can check a separately captured live log with `--trace-log`. It performs no target logits walk, no target accept walk, no accepted draft tokens, no correction token, no token commit, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens.

P5AA root-token-commit no-op status: `validate_p5aa_root_token_commit_noop_runtime.py` validates the approved default-off root-token-commit no-op runtime source slice. With `LLAMA_JETSPEC_ROOT_TOKEN_COMMIT_NOOP_ONLY=1`, `LLAMA_JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_ONLY=1`, `LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY=1`, and `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`, the slice records `JETSPEC_ROOT_TOKEN_COMMIT_NOOP_RUNTIME_PHASE`, `token_commit_runtime_ready`, `root_token_commit_noop_runtime_ready`, `root_token_commit_noop_runtime_hash_last`, `root_verified_anchor=1`, `accept_path_len=0`, `actual_accepted_nodes=0`, `correction_token_present=0`, and `actual_committed_tokens=0`, then returns before P5U unless the later root no-op tail gates are enabled. `probe_p5aa_root_token_commit_noop_trace.py` validates the fast no-model trace contract, reports `p5aa_root_token_commit_noop_trace_contract_verified`, and can check a separately captured live log with `--trace-log`. It performs no real token commit, no visible token publish, no hidden/KV commit, no rejected-branch discard, no publish, no visible state change, no KV mutation, and no draft tokens.

P5AB-P5AD root no-op tail readiness status: `validate_p5ab_to_p5ad_root_noop_readiness.py` validates the approved default-off root no-op tail source slices. With `LLAMA_JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_ONLY=1`, `LLAMA_JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_ONLY=1`, and `LLAMA_JETSPEC_ROOT_PUBLISH_GATE_NOOP_ONLY=1`, the slices record `JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_RUNTIME_PHASE`, `JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_RUNTIME_PHASE`, `JETSPEC_ROOT_PUBLISH_GATE_NOOP_RUNTIME_PHASE`, `root_hidden_kv_commit_noop_runtime_ready`, `root_rejected_branch_discard_noop_runtime_ready`, `root_publish_gate_noop_runtime_ready`, `actual_survivor_pages_committed=0`, `actual_pages_discarded=0`, `rejected_branch_pages_reachable_after_discard=0`, `actual_publish_visible_state=0`, and `root_runtime_ready_for_real_test=1`. `probe_p5ad_root_ready_trace.py` validates the fast no-model terminal readiness trace contract, reports `p5ad_root_ready_trace_contract_verified`, and can check a separately captured live log with `--trace-log`. It is ready to start a real root-only test under separate approval and still performs no real hidden/KV commit, no real rejected-branch discard, no real publish, no visible state change, no KV mutation, and no draft tokens.

P5AE synthetic top-k tree ABI status: `validate_p5ae_topk_tree_runtime.py` validates the approved default-off synthetic non-root tree ABI source slice. With `LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1` and `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`, it floors the descriptor-only tree budget to the synthetic three-node ABI size and records `JETSPEC_TOPK_TREE_RUNTIME_PHASE`, `topk_tree_runtime_ready`, `topk_logprob_source=synthetic_full_vocab_softmax`, `topk_width=2`, `topk_depth=1`, `actual_tree_nodes=3`, `tree_parent_indices=[-1,0,0]`, `tree_depth=[0,1,1]`, `tree_rank=[-1,0,1]`, `tree_cum_logprob=[0.0,-0.1,-0.3]`, and `non_root_nodes=2`. `probe_p5ae_topk_tree_trace.py` reports `p5ae_topk_tree_trace_contract_verified`. It performs no draft logits, no verify mask, no accept, no KV mutation, no publish, and no draft tokens.

P5AF synthetic top-k verify-mask ABI status: `validate_p5af_topk_verify_mask_runtime.py` validates the approved default-off synthetic top-k verify-mask ABI source slice. With `LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1` after P5AE, it records `JETSPEC_TOPK_VERIFY_MASK_RUNTIME_PHASE`, `topk_verify_mask_runtime_ready`, `actual_tree_nodes=3`, `actual_verify_mask_entries=5`, `verify_mask_rows=3`, `verify_mask_cols=3`, `allowed_edges=[0:0,1:0,1:1,2:0,2:2]`, `ancestor_only=1`, `sibling_visible=0`, and `descendant_visible=0`. `probe_p5af_topk_verify_mask_trace.py` reports `p5af_topk_verify_mask_trace_contract_verified`. It performs no mask tensor allocation, no accept, no token commit, no KV mutation, no publish, and no draft tokens.

P5AG synthetic top-k accept-boundary ABI status: `validate_p5ag_topk_accept_boundary_runtime.py` validates the approved default-off synthetic top-k accept-boundary ABI source slice. With `LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1` after P5AF, it records `JETSPEC_TOPK_ACCEPT_BOUNDARY_RUNTIME_PHASE`, `topk_accept_boundary_runtime_ready`, `actual_tree_nodes=3`, `actual_verify_mask_entries=5`, `accept_boundary_candidate_nodes=2`, `accept_boundary_verified_edges=5`, `actual_verified_logits_rows=0`, `accept_decision_source=none_no_logits`, `accept_path_len=0`, `actual_accepted_nodes=0`, and `correction_token_present=0`. `probe_p5ag_topk_accept_boundary_trace.py` reports `p5ag_topk_accept_boundary_trace_contract_verified`. It performs no target logits walk, no target accept walk, no token commit, no hidden/KV commit, no rejected-branch discard, no KV mutation, no publish, and no draft tokens.

P5AH draft-head top-k readiness status: `jetspec_draft_head_topk_readiness.py --fixture fixtures/jetspec_draft_head_topk_readiness_smoke.json` validates an inert descriptor for the future transition from synthetic top-k ABI metadata to real draft-head logits and reports `draft_head_topk_readiness_verified_not_executed`. It requires the P5AG chain gates, keeps `actual_verified_logits_rows=0`, plans `planned_draft_head_logits_rows=1` for parent node 0 and candidate nodes `[1,2]`, records `future_logits_source=draft_head_full_vocab_logits`, forbids `target_logits`, `sampler`, `synthetic_full_vocab_softmax`, and `topk_only_renormalization`, preserves `runtime_supported=false` and `ctx_dft=nullptr`, and performs no `llama_decode`, no draft-head graph execution, no logits buffer read, no target logits walk, no target accept walk, no token commit, no hidden/KV commit, no rejected-branch discard, no KV mutation, no publish, no CUDA dispatch, no server route, no public API, no CMake wiring, no performance claim, no promotion claim, and no draft tokens. `validate_p5ah_draft_head_topk_readiness.py` and `test_p5ah_draft_head_topk_readiness.py` keep this under `experiments/jetspec/`; real draft-head logits/top-k production construction remains blocked pending separate explicit production-source approval.


### P1: isolated loader prototype, still outside production CMake

Allowed work:

- A standalone loader/parser prototype under `experiments/jetspec/`.
- No production `llama_model_loader` registration.
- No server flag.
- No top-level CMake wiring.

Required evidence:

- Reads metadata-only preview.
- Fails closed for preview runtime unless an explicit experimental flag is present.
- Maps metadata into the inert `draft_head_metadata` fields.
- Preserves `runtime_supported=false` for preview files.

Current P1 evidence:

- `draft_head_loader_prototype.py --self-test` reads the metadata-only GGUF preview and maps it to `draft_head_metadata`-shaped JSON.
- `test_draft_head_loader_prototype.py` covers target-layer fail-closed checks and preview-runtime gate behavior.
- Runtime preparation without `--allow-preview-runtime` fails closed as `preview_not_allowed`; with the flag it still fails closed as `unsupported_runtime` because `runtime_supported=false` is preserved.

### P2: BF16 payload converter/loader parity fixture

Allowed work:

- Local-only tensor payload read/write tests under `experiments/jetspec/`.
- A tiny synthetic BF16 fixture or explicitly local safetensors file.

Required evidence:

- All 91 tensor names and shapes match the conversion plan.
- BF16 payload offsets match safetensors header offsets.
- No quantization, packing, or dtype conversion yet.
- Loader rejects missing/wrong-shaped tensors.

Current P2 evidence:

- `bf16_payload_parity.py --fixture fixtures/bf16_payload_parity_smoke.json` validates the synthetic/local safetensors header against the 91-tensor conversion plan.
- `bf16_payload_parity.py --self-test` validates sparse converter-header compatibility without copying the 947,990,528 byte payload.
- `test_bf16_payload_parity.py` covers missing tensor, wrong shape, wrong dtype, wrong offset, and stable fixture output failure modes.
- Copy policy is `raw_bf16_no_transform`: no quantization, no packing, no dtype conversion.

### P3: target hidden tap parity fixture

Allowed work:

- An isolated target-hidden tap prototype outside server wiring.

Required evidence:

- Taps match HF `hidden_states[layer_id + 1]` semantics.
- Concatenation order is `[1, 10, 19, 28, 37]`.
- Width is exactly `10240`.
- A/B fixture proves enabling tap capture does not change normal target logits or greedy output.
- Missing tap, wrong width, or target mismatch fails closed.

Current P3 evidence:

- `target_hidden_taps.py --fixture fixtures/target_hidden_tap_parity_smoke.json --json` validates HF `hidden_states[layer_id + 1]` fallback against hook-style capture.
- `fixtures/target_hidden_tap_parity_smoke.out.json` records width `10240`, concat order `[1, 10, 19, 28, 37]`, unchanged logits, unchanged greedy token IDs, and `capture_is_side_channel_only=true`.
- `test_target_hidden_taps.py` covers hidden-state index mapping, logit drift rejection, greedy-output drift rejection, wrong hook index rejection, target hidden-size mismatch rejection, missing tap, wrong width, wrong order, and immutable capture.

### P4: tree verify and rollback parity fixture

Allowed work:

- Isolated tree verify prototype and rollback model outside production paths.

Required evidence:

- Verify mask gives every query full prefix visibility and ancestor-only tree visibility.
- Siblings, descendants, and rejected branches cannot attend each other.
- Hidden/KV commit appends `[root | accepted]` rows only.
- Token commit appends `[accepted draft tokens | correction]`.
- Rejected hidden/KV sentinels cannot be reached after commit.
- Greedy output matches the baseline target path on deterministic fixtures.

Current P4 evidence:

- `jetspec_round_contract.py --fixture fixtures/jetspec_round_parity_smoke.json` composes tree build, verify mask, greedy accept, hidden-cache commit, rollback checks, and baseline greedy-output parity.
- `fixtures/jetspec_round_parity_smoke.out.json` records accepted-path isolation from rejected tree nodes, absent rejected sentinels, `[root | accepted]` hidden append, `[accepted draft tokens | correction]` token append, and `greedy_output_matches_baseline=true`.
- `test_jetspec_round_contract.py` covers deterministic output, expected-accept mismatch, sentinel leakage, pre-round trail invariant, accepted-path isolation, and baseline greedy mismatch.

### P5: production-path candidate, default-off only

Allowed work only after P1-P4 evidence is saved:

- Minimal compiled-path candidate behind an explicit experimental build option or feature flag.
- Loader registration may be added only if preview files remain rejected by default.
- Server integration may be added only as explicit opt-in and must not change default behavior.

Required evidence:

- Default build/run with JetSpec disabled is unchanged.
- Existing MTP/speculative paths are not regressed.
- `llama-server` has no new behavior unless the explicit JetSpec flag is set.
- Failure paths cleanly disable JetSpec rather than partially running.

Current P5 evidence:

- `jetspec_p5_default_off_plan.md` remains the design record and explicitly did not itself approve production edits.
- `validate_p5_plan.py` requires default-off controls `LLAMA_JETSPEC_EXPERIMENTAL=1`, `LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1`, explicit `--spec-type draft-jetspec`, preview rejection, `runtime_supported=false`, no default behavior change, and rollback gates.
- P5A was explicitly approved after the plan and now registers `jetspec_qwen3_draft_head` only as a validation-only loader candidate.
- `jetspec_p5a_loader_candidate.md` records the approved P5A hook set, fail-closed loader outcomes, and blocked next work.
- `validate_p5a_loader_candidate.py` verifies the five P5A production files, preview rejection, `unsupported_runtime`, 0-or-91 tensor inventory, BF16 tensor checks, no graph construction, no P5C speculative type, and no explicit CMake wiring.
- P5B was explicitly approved after P5A and now captures Qwen35/Qwen35MoE post-layer target taps `[1, 10, 19, 28, 37]` as a private side channel with width `10240`.
- `jetspec_p5b_target_taps_candidate.md` records the approved P5B hook set, fixed tap layout, default-off API, source guard, and blocked next work.
- `validate_p5b_target_taps.py` verifies the P5B private API, fixed layer order, graph-reuse guard, width guard, no public `include/llama.h`, approved downstream connector only, and no explicit CMake wiring.
- P5C was explicitly approved after P5B and now accepts `--spec-type draft-jetspec` only as a fail-closed, non-executable route guarded by `LLAMA_JETSPEC_EXPERIMENTAL=1`.
- `jetspec_p5c_speculative_type_candidate.md` records the approved P5C hook set, parser route, fail-closed gates, no-draft placeholder, and blocked runtime work.
- `validate_p5c_speculative_type.py` verifies `COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC`, `draft-jetspec` parsing, the `jetspec` alias, the `runtime_supported=false` no-draft placeholder, no silent fallback to `draft-simple`, the model-only server binding reference, no public API/CMake route, and no kernel/test/example/poc wiring.
- P5D was explicitly approved after P5C and now ingests private P5B target tap rows inside the explicit `draft-jetspec` route while still emitting no draft tokens.
- `jetspec_p5d_target_tap_ingestion_candidate.md` records the approved P5D hook set, masked row copy, FNV-1a hash trace, fail-closed disable path, no-draft behavior, and blocked tree-runtime work.
- `validate_p5d_target_tap_ingestion.py` verifies target tap ingestion, `LLAMA_JETSPEC_TRACE=1`/`LLAMA_JETSPEC_TAP_TRACE=1`, row hashing, no draft tokens, no draft-head graph execution, and no public/server/CMake/kernel/test/example/poc wiring.
- P5E was explicitly approved after P5D and now records private runtime phase, failure state, row metadata, cached row counts, and draft-call counters inside the explicit `draft-jetspec` route while still emitting no draft tokens.
- `jetspec_p5e_runtime_state_candidate.md` records the approved P5E hook set, state-only runtime bookkeeping, `LLAMA_JETSPEC_STATE_TRACE=1`, fail-closed disable path, no-draft behavior, and blocked tree-runtime work.
- `validate_p5e_runtime_state.py` verifies runtime-state bookkeeping, failure state, row metadata, state trace, no draft tokens, no draft-head graph execution, and no public/server/CMake/kernel/test/example/poc wiring.
- P5F was explicitly approved after P5E and now validates explicit draft-head/target bindings before JetSpec instantiation while still emitting no draft tokens.
- `jetspec_p5f_binding_preflight_candidate.md` records the approved P5F hook set, target context plus draft context or model-only draft model checks, metadata/shape checks, Qwen3.6 shape constants, target tap count/width checks, model-only target tensor presence/shape checks, fail-closed disable path, no-draft behavior, and blocked tree-runtime work.
- `validate_p5f_binding_preflight.py` verifies `common_speculative_jetspec_preflight`, binding preflight, metadata/shape checks, target tap count/width checks, model-only draft-head path, target tensor presence/shape checks, no draft tokens, no draft-head graph execution, and no public API/CMake/kernel/test/example/poc wiring.
- `validate_p5f_artifact_binding.py` verifies the real stored draft-head artifact facts against P5F source constants and reports `artifact_verified_not_runtime_executed` rather than claiming live runtime execution.
- `probe_p5f_loader_gate.py` verifies the actual built loader rejects the metadata-only preview as `preview_not_allowed` by default and `unsupported_runtime` with `LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1` plus `LLAMA_JETSPEC_EXPERIMENTAL=1`, source-asserts true-valued `runtime_supported` metadata rejection, and reports `loader_gate_verified_preflight_still_blocked`.
- `probe_p5f_target_tensor_binding.py` verifies target GGUF split headers contain the model-only linker tensors and reports `target_tensor_headers_verified_not_loaded` without model load, context creation, graph execution, or draft tokens.
- `jetspec_p5g_tree_runtime_readiness.md` records an inert P5G readiness contract for the next tree-runtime slice without approving production runtime work.
- `jetspec_tree_runtime_readiness.py` validates the P5G smoke fixture and reports `tree_runtime_readiness_verified_not_executed` while covering DraftTree ABI, full-vocab top-k logprob input, ancestor-only mask, accept, commit, gather, duplicate-child overwrite, and no-runtime boundary.
- `validate_p5g_tree_runtime_readiness.py` verifies the P5G files, fixture output, forbidden production path boundary, and no CMake references.
- `test_p5g_tree_runtime_readiness.py` covers the P5G evaluator and validator failure modes.
- `jetspec_p5h_kv_commit_readiness.md` records an inert P5H KV/hidden commit ownership contract without approving production runtime work.
- `jetspec_kv_commit_readiness.py` validates the P5H smoke fixture and reports `kv_commit_readiness_verified_not_executed` while covering abstract slots, `past_len + accepted_path` gather, `[accepted draft tokens | correction]`, `[root | accepted]`, rejected-branch discard, cross-sequence isolation, explicit ownership primitive mapping, and no-runtime boundary.
- `validate_p5h_kv_commit_readiness.py` verifies the P5H files, fixture output, forbidden production path boundary, and no CMake references.
- `test_p5h_kv_commit_readiness.py` covers the P5H evaluator and validator failure modes.
- `jetspec_p5i_tree_runtime_approval_packet.md` records an inert P5I tree-runtime approval packet without approving production runtime work.
- `tree_runtime_approval_matrix.py` validates the P5I smoke fixture and reports `tree_runtime_approval_packet_verified_not_executed` while mapping future actions to P5G/P5H evidence, `missing_primitive`, or explicit-approval blockers.
- `validate_p5i_tree_runtime_approval_packet.py` verifies the P5I files, fixture output, forbidden production path boundary, and no CMake references.
- `test_p5i_tree_runtime_approval_packet.py` covers the P5I evaluator and validator failure modes.
- `jetspec_p5j_kv_primitive_audit.md` records an inert P5J KV/runtime primitive audit without approving production runtime work.
- `kv_primitive_audit.py` validates the P5J smoke fixture and reports `kv_primitive_audit_verified_not_executed` while scanning existing KV helpers and keeping P5I ownership gaps as `exact_missing_primitive`.
- `validate_p5j_kv_primitive_audit.py` verifies the P5J files, fixture output, forbidden production path boundary, and no CMake references.
- `test_p5j_kv_primitive_audit.py` covers the P5J evaluator and validator failure modes.
- `jetspec_p5k_kv_ownership_primitive_design.md` records an inert P5K KV ownership primitive design packet without approving production runtime work.
- `kv_ownership_primitive_design.py` validates the P5K smoke fixture and reports `kv_ownership_primitive_design_verified_not_executed` while naming design-only candidate primitives and keeping implementation blocked.
- `validate_p5k_kv_ownership_primitive_design.py` verifies the P5K files, fixture output, forbidden production path boundary, and no CMake references.
- `test_p5k_kv_ownership_primitive_design.py` covers the P5K evaluator and validator failure modes.
- `jetspec_p5l_page_map_ownership_oracle.md` records an inert P5L page-map ownership oracle without approving production runtime work.
- `page_map_ownership_oracle.py` validates the P5L smoke fixture and reports `page_map_ownership_oracle_verified_not_executed` while covering accepted survivor page ownership, rejected branch page unreachability, cross-sequence page isolation, QBlock/PageAttention descriptor lessons, non-exact helper mappings, and no-runtime boundary.
- `validate_p5l_page_map_ownership_oracle.py` verifies the P5L files, fixture output, forbidden production path boundary, and no CMake references.
- `test_p5l_page_map_ownership_oracle.py` covers the P5L evaluator and validator failure modes.
- `jetspec_p5m_transaction_plan_oracle.md` records an inert P5M transaction/failpoint plan oracle without approving production runtime work.
- `transaction_plan_oracle.py` validates the P5M smoke fixture and reports `transaction_plan_oracle_verified_not_executed` while covering ordered phases, rollback failpoints, pre-publish invisibility, `[accepted draft tokens | correction]`, `[root | accepted]`, rejected branch discard, cross-sequence page preservation, non-exact helper mappings, and no-runtime boundary.
- `validate_p5m_transaction_plan_oracle.py` verifies the P5M files, fixture output, forbidden production path boundary, and no CMake references.
- `test_p5m_transaction_plan_oracle.py` covers the P5M evaluator and validator failure modes.
- `jetspec_p5n_transaction_scaffold_candidate.md` records the approved P5N transaction-plan scaffold source slice while draft-token emission remains blocked.
- `validate_p5n_transaction_scaffold.py` verifies `common/speculative.cpp` and `docs/speculative.md` contain bounded transaction phase/failpoint scaffold state and no graph/KV/draft-token execution.
- `test_p5n_transaction_scaffold.py` covers the P5N source validator failure modes.
- `jetspec_p5o_pre_round_snapshot_candidate.md` records the approved P5O pre-round snapshot descriptor source slice while reserve/tree/verify/KV/publish/draft work remains blocked.
- `validate_p5o_pre_round_snapshot.py` verifies `common/speculative.cpp` and `docs/speculative.md` contain bounded snapshot descriptor state and no reserve/tree/verify/KV/draft-token execution.
- `test_p5o_pre_round_snapshot.py` covers the P5O source validator failure modes.
- `jetspec_p5p_transient_reservation_descriptor_candidate.md` records the approved P5P transient-reservation descriptor source slice while real reservation/KV/tree/verify/publish/draft work remains blocked.
- `validate_p5p_transient_reservation_descriptor.py` verifies `common/speculative.cpp` and `docs/speculative.md` contain bounded descriptor-only reservation state with `actual_pages_reserved=0` and no real reservation/KV/tree/verify/draft-token execution.
- `test_p5p_transient_reservation_descriptor.py` covers the P5P source validator failure modes.
- `jetspec_p5q_tree_build_descriptor_candidate.md` records the approved P5Q tree-build descriptor source slice while real tree/verify/accept/KV/publish/draft work remains blocked.
- `validate_p5q_tree_build_descriptor.py` verifies `common/speculative.cpp` and `docs/speculative.md` contain bounded descriptor-only tree-build state with `actual_tree_nodes=0` and no real tree/verify/accept/KV/draft-token execution.
- `test_p5q_tree_build_descriptor.py` covers the P5Q source validator failure modes.
- `jetspec_p5r_verify_mask_descriptor_candidate.md` records the approved P5R verify-mask descriptor source slice; `validate_p5r_verify_mask_descriptor.py` and `test_p5r_verify_mask_descriptor.py` verify `actual_verify_mask_entries=0`, no real verify mask, and no mask tensor.
- `jetspec_p5s_accept_path_descriptor_candidate.md` records the approved P5S accept-path descriptor source slice; `validate_p5s_accept_path_descriptor.py` and `test_p5s_accept_path_descriptor.py` verify `actual_accepted_nodes=0`, no real accept, and no target logits walk.
- `jetspec_p5t_token_commit_descriptor_candidate.md` records the approved P5T token-commit descriptor source slice; `validate_p5t_token_commit_descriptor.py` and `test_p5t_token_commit_descriptor.py` verify `actual_committed_tokens=0`, no real token commit, and no visible token publish.
- `jetspec_p5u_hidden_kv_survivor_commit_descriptor_candidate.md` records the approved P5U hidden/KV survivor commit descriptor source slice; `validate_p5u_hidden_kv_survivor_commit_descriptor.py` and `test_p5u_hidden_kv_survivor_commit_descriptor.py` verify `actual_survivor_pages_committed=0` and no real hidden/KV commit.
- `jetspec_p5v_rejected_branch_discard_descriptor_candidate.md` records the approved P5V rejected-branch discard descriptor source slice; `validate_p5v_rejected_branch_discard_descriptor.py` and `test_p5v_rejected_branch_discard_descriptor.py` verify `actual_pages_discarded=0` and no real rejected-branch discard.
- `jetspec_p5w_publish_gate_descriptor_candidate.md` records the approved P5W publish-gate descriptor source slice; `validate_p5w_publish_gate_descriptor.py` and `test_p5w_publish_gate_descriptor.py` verify `actual_publish_visible_state=0`, no real publish, and no visible state change.
- `jetspec_p5x_root_tree_runtime_candidate.md` records the approved P5X root-only runtime tree source slice; `validate_p5x_root_tree_runtime.py` and `test_p5x_root_tree_runtime.py` verify `actual_tree_nodes=1`, root-only arrays, return-before-P5R, and no verify/accept/commit/publish/draft tokens.
- `jetspec_p5y_root_verify_mask_runtime_candidate.md` records the approved P5Y root-only verify-mask runtime source slice; `validate_p5y_root_verify_mask_runtime.py`, `test_p5y_root_verify_mask_runtime.py`, `probe_p5y_root_verify_mask_trace.py`, and `test_p5y_root_verify_mask_trace_probe.py` verify `actual_verify_mask_entries=1`, root self-mask shape, return-before-P5S, and no accept/commit/publish/draft tokens.
- `jetspec_p5z_root_anchor_accept_path_runtime_candidate.md` records the approved P5Z root-anchor accept-path runtime source slice; `validate_p5z_root_anchor_accept_path_runtime.py`, `test_p5z_root_anchor_accept_path_runtime.py`, `probe_p5z_root_anchor_accept_path_trace.py`, and `test_p5z_root_anchor_accept_path_trace_probe.py` verify `root_verified_anchor=1`, `accept_path_len=0`, `actual_accepted_nodes=0`, `correction_token_present=0`, return-before-P5T, and no commit/publish/draft tokens.
- `jetspec_p5aa_root_token_commit_noop_runtime_candidate.md` records the approved P5AA root-token-commit no-op runtime source slice; `validate_p5aa_root_token_commit_noop_runtime.py`, `test_p5aa_root_token_commit_noop_runtime.py`, `probe_p5aa_root_token_commit_noop_trace.py`, and `test_p5aa_root_token_commit_noop_trace_probe.py` verify `actual_committed_tokens=0`, return-before-P5U, no visible token publish, no KV mutation, and no draft tokens.
- `jetspec_p5ab_to_p5ad_root_noop_readiness_candidate.md` records the approved P5AB-P5AD root no-op tail readiness source slices; `validate_p5ab_to_p5ad_root_noop_readiness.py`, `test_p5ab_to_p5ad_root_noop_readiness.py`, `probe_p5ad_root_ready_trace.py`, and `test_p5ad_root_ready_trace_probe.py` verify `root_runtime_ready_for_real_test=1`, no real hidden/KV commit, no real rejected-branch discard, no real publish, no visible state change, no KV mutation, and no draft tokens.
- `jetspec_p5ae_topk_tree_runtime_candidate.md` records the approved P5AE synthetic top-k tree ABI source slice; `validate_p5ae_topk_tree_runtime.py`, `test_p5ae_topk_tree_runtime.py`, `probe_p5ae_topk_tree_trace.py`, and `test_p5ae_topk_tree_trace_probe.py` verify `actual_tree_nodes=3`, synthetic non-root tree arrays, no draft logits, no verify mask, no KV mutation, and no draft tokens.
- `jetspec_p5af_topk_verify_mask_runtime_candidate.md` records the approved P5AF synthetic top-k verify-mask ABI source slice; `validate_p5af_topk_verify_mask_runtime.py`, `test_p5af_topk_verify_mask_runtime.py`, `probe_p5af_topk_verify_mask_trace.py`, and `test_p5af_topk_verify_mask_trace_probe.py` verify `actual_verify_mask_entries=5`, ancestor-only edges, no mask tensor, no accept, no KV mutation, and no draft tokens.
- `jetspec_p5ag_topk_accept_boundary_runtime_candidate.md` records the approved P5AG synthetic top-k accept-boundary ABI source slice; `validate_p5ag_topk_accept_boundary_runtime.py`, `test_p5ag_topk_accept_boundary_runtime.py`, `probe_p5ag_topk_accept_boundary_trace.py`, and `test_p5ag_topk_accept_boundary_trace_probe.py` verify `accept_boundary_candidate_nodes=2`, `actual_verified_logits_rows=0`, no target logits walk, no target accept walk, no commit, no KV mutation, and no draft tokens.
- `jetspec_p5ah_draft_head_topk_readiness.md` records the inert P5AH draft-head top-k readiness descriptor; `jetspec_draft_head_topk_readiness.py`, `fixtures/jetspec_draft_head_topk_readiness_smoke.json`, `validate_p5ah_draft_head_topk_readiness.py`, and `test_p5ah_draft_head_topk_readiness.py` verify `draft_head_topk_readiness_verified_not_executed`, `planned_draft_head_logits_rows=1`, `actual_verified_logits_rows=0`, `future_logits_source=draft_head_full_vocab_logits`, `ctx_dft=nullptr`, no graph/logits execution, no production path touches, and no draft tokens.
- `parse_gguf_preview.py` includes `validate_tensor_payload_against_plan` for no-runtime 91-tensor GGUF tensor-info table validation against the conversion plan.
- `probe_p5c_route_gate.py` and `test_p5c_route_gate_probe.py` verify the explicit JetSpec env/context/preflight gates and no silent `draft-simple` fallback without creating contexts, reporting `p5c_route_gate_verified_no_model`.
- `jetspec_descriptor_chain_contract.py`, `fixtures/jetspec_descriptor_chain_smoke.json`, and `test_jetspec_descriptor_chain_contract.py` validate the P5N-W descriptor-chain state machine and zero-actual counters without runtime execution, reporting `descriptor_chain_verified_not_executed`.
- `jetspec_runtime_contract_compile_smoke.cpp` and `compile_runtime_contract_smoke.py` provide a standalone C++17 `-fsyntax-only` smoke for inert runtime headers outside CMake, reporting `runtime_contract_compile_smoke_passed`.

### P6: performance and promotion gate

Required before calling anything mature, production, fast, default, or promotion-ready:

- Correctness parity against greedy target output on deterministic fixtures.
- Correctness parity over a small prompt matrix with JetSpec disabled vs enabled as appropriate.
- Benchmark against the current proven speculative baseline.
- Matches or beats the baseline for the agreed minimal benchmark.
- Failed candidates are removed or demoted, not left as half-routed flags.

## Path touch policy

| Path | Current status | Earliest phase |
| --- | --- | --- |
| `experiments/jetspec/` | allowed | P0 |
| top-level CMake / production `.cmake` | forbidden | P5 |
| P5A production hook set | validation-only candidate approved | P5A |
| P5B private target-tap hook set | side-channel candidate approved | P5B |
| P5C speculative type hook set | fail-closed parser candidate approved | P5C |
| P5D target-tap ingestion hook set | non-drafting runtime slice approved | P5D |
| P5E runtime-state hook set | state-only runtime slice approved | P5E |
| P5F binding-preflight hook set | preflight-only runtime slice approved | P5F |
| P5N transaction-plan scaffold hook set | non-drafting transaction scaffold approved | P5N |
| P5O pre-round snapshot hook set | non-drafting snapshot descriptor approved | P5O |
| P5P transient-reservation descriptor hook set | descriptor-only reservation slice approved | P5P |
| P5Q tree-build descriptor hook set | descriptor-only tree-build slice approved | P5Q |
| P5R verify-mask descriptor hook set | descriptor-only verify-mask slice approved | P5R |
| P5S accept-path descriptor hook set | descriptor-only accept slice approved | P5S |
| P5T token-commit descriptor hook set | descriptor-only token-commit slice approved | P5T |
| P5U hidden/KV survivor commit descriptor hook set | descriptor-only hidden/KV commit slice approved | P5U |
| P5V rejected-branch discard descriptor hook set | descriptor-only discard slice approved | P5V |
| P5W publish-gate descriptor hook set | descriptor-only publish gate approved | P5W |
| P5X root-only runtime tree hook set | default-off one-node tree ABI approved | P5X |
| P5Y root-only verify-mask runtime hook set | default-off one-entry root mask ABI approved | P5Y |
| P5Z root-anchor accept-path runtime hook set | default-off root-anchor accept ABI approved | P5Z |
| P5AA root-token-commit no-op runtime hook set | default-off root token-commit no-op ABI approved | P5AA |
| P5AB-P5AD root no-op tail readiness hook set | default-off root no-op tail readiness approved | P5AB-P5AD |
| P5AE synthetic top-k tree ABI hook set | default-off synthetic top-k tree ABI approved | P5AE |
| P5AF synthetic top-k verify-mask ABI hook set | default-off synthetic top-k verify-mask ABI approved | P5AF |
| P5AG synthetic top-k accept-boundary ABI hook set | default-off synthetic top-k accept-boundary ABI approved | P5AG |
| P5AH draft-head top-k readiness descriptor | inert experiments-only readiness, no production hook approval | P5AH |
| non-P5A/P5B `src/models/*.cpp` | forbidden | tree-runtime approval |
| non-P5C/P5D/P5E/P5F/P5N/P5O/P5P/P5Q/P5R/P5S/P5T/P5U/P5V/P5W/P5X/P5Y/P5Z/P5AA/P5AB-P5AD/P5AE/P5AF/P5AG `common/` files | forbidden | next runtime approval |
| `tools/server/server-context.cpp` JetSpec model-only binding | allowed only with `ctx_dft=nullptr`, no draft context, and no graph execution | P5F model-only binding |
| other `tools/server/` behavior/routes | forbidden | runtime approval |
| repository `tests/` | forbidden | P5 |
| `examples/` | forbidden | P5 |
| `pocs/` | forbidden | P5 |
| `ggml/src/` | forbidden | P6 unless a kernel-specific P5 design is approved |

## Evidence gates

Every promotion proposal must cite concrete artifacts:

- aggregate verifier output;
- generated GGUF conversion plan;
- metadata-only GGUF parser result;
- runtime header validator result;
- target hidden tap parity result;
- tree verify mask result;
- hidden-cache rollback result;
- end-to-end round fixture result;
- CMake isolation scan;
- if performance-sensitive, benchmark command, artifact path, and baseline comparison.

## Fail-closed blockers

Any one of these blocks promotion:

- `runtime_supported=false` is still required for the artifact being loaded.
- Preview GGUF can run without an explicit experimental flag.
- Top-level CMake references `experiments/jetspec` before P5.
- Target embeddings or target `lm_head` are unavailable.
- Target hidden taps are missing, wrong width, wrong order, or change greedy target output (`target hidden taps` contract failure).
- Verify mask allows sibling, descendant, or rejected branch attention leaks.
- Hidden/KV rollback lets rejected branch rows influence the next proposal.
- Correction hidden is appended in the same round.
- JetSpec disabled path changes output, speed, memory, or routing.
- A candidate loses the agreed baseline but remains wired as if promoted.

## Rollback and demotion policy

- Candidate features are named `experimental` or `candidate`, never `mature`, `fast`,
  `final`, or `production`, until P6 passes.
- If a candidate is correct but slower than the baseline, keep it private or remove
  route plumbing.
- If a candidate changes output unexpectedly, disable it by default and preserve
  the failing artifact for investigation.
- Do not silently change GGUF metadata, tensor shapes, hidden-cache width, or KV
  ownership rules to make a candidate pass.

## Next allowed work

P5N consumed the first explicit tree-runtime approval as a bounded transaction-plan scaffold only, P5O consumed the next approval as a bounded pre-round snapshot descriptor only, P5P consumed the next approval as a descriptor-only transient reservation intent with `actual_pages_reserved=0`, and P5Q consumed the next approval as a descriptor-only tree-build intent; P5R-P5W consumed follow-on approval as descriptor-only gates for verify, accept, commit, discard, and publish intent with all actual verify/accept/commit/discard/publish counts held at zero. P5X consumed the next approval as a default-off root-only runtime tree ABI object with `actual_tree_nodes=1`; P5Y consumed the next approval as a default-off root-only verify-mask ABI object with `actual_verify_mask_entries=1`; P5Z consumed the next approval as a default-off root-anchor accept-path ABI object with `root_verified_anchor=1`, `actual_accepted_nodes=0`, and no commit/publish work. P5AA consumed the next approval as a default-off root-token-commit no-op ABI object with `actual_committed_tokens=0` and no visible publish/KV mutation. P5AB-P5AD consumed the next approval as default-off root no-op tail readiness objects with `root_runtime_ready_for_real_test=1`, no KV mutation, and no visible publish. P5AE/P5AF/P5AG consumed the next approval as default-off synthetic top-k tree, ancestor-mask, and accept-boundary ABI objects with `actual_tree_nodes=3`, `actual_verify_mask_entries=5`, `accept_boundary_candidate_nodes=2`, `actual_verified_logits_rows=0`, no draft logits, no mask tensor, no target logits walk, no target accept walk, no KV mutation, and no visible publish. P5AH consumed no production approval: it is inert experiments-only readiness for future `draft_head_full_vocab_logits`, keeps `planned_draft_head_logits_rows=1` and `actual_verified_logits_rows=0`, preserves `runtime_supported=false` and `ctx_dft=nullptr`, and confirms no graph/logits execution. The next safe implementation step is a separate explicit approval decision to wire real draft-head logits/top-k production construction. Without that approval, keep further work under `experiments/jetspec/` or the existing approved default-off hook sets. With approval, implement only the next bounded tree-runtime slice, keep it default-off, run the runtime verification matrix, and preserve preview-GGUF rejection. Until approval, continue to avoid server flags, executable server behavior beyond the existing model-only binding, repository `tests/`, `examples/`, `pocs/`, `ggml/src/`, production CMake wiring, public API, and non-P5A/P5B model registration changes.
