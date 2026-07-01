# JetSpec P5F binding preflight candidate

Status: approved bounded runtime slice. This includes model-only draft-head
binding, but not draft-head execution, tree verify, KV/hidden rollback, or
executable server drafting behavior.

## Production hook set

Primary preflight files in scope for this candidate:

- `common/speculative.cpp`
- `docs/speculative.md`

The follow-on model-only binding slice also uses the existing JetSpec model
loader hooks and `tools/server/server-context.cpp` to carry a draft model pointer
with `ctx_dft = nullptr`. The model-only linker fails closed unless the paired
target model exposes `token_embd.weight`, `output.weight`, and
`output_norm.weight` with the staged Qwen3.6 target shapes. No public
`include/llama.h`, repository `tests`, `examples`, `pocs`, `ggml/src`, production
CMake wiring, or executable graph path is approved by P5F.

## Runtime behavior

P5F adds a private draft-head/target binding preflight before the explicit
`draft-jetspec` implementation is instantiated. The preflight uses existing
public model/vocab metadata accessors only; it adds no public API.

Required preflight inputs:

- explicit `--spec-type draft-jetspec` or `--spec-type jetspec`;
- `LLAMA_JETSPEC_EXPERIMENTAL=1`;
- non-null target context;
- either a draft context or model-only loaded draft-head model;
- non-null target model/vocab and draft model metadata;
- draft metadata strings:
  - `general.architecture=jetspec_qwen3_draft_head`;
  - `jetspec.architecture=qwen3_draft_head`;
  - `jetspec.source_architecture=DFlashDraftModel`;
  - `jetspec.tensor_data_dtype=bfloat16`;
- target shape: hidden size 2048, layer count 40, vocab size 248320;
- draft shape: context train size 16, hidden size 2048, layer count 8, heads 32,
  KV heads 4, vocab size 248320;
- target tap shape: count 5, width 10240, and width equals `tap_count *
  target_hidden_size`;
- model-only target tensor presence/shape: `token_embd.weight` `[2048,248320]`,
  `output.weight` `[2048,248320]`, and `output_norm.weight` `[2048]`.

On any mismatch, P5F disables the private target-tap side channel, drops the
JetSpec implementation before runtime execution, logs the preflight reason, and
continues without silently falling back to `draft-simple`.

The route remains fail-closed and non-drafting:

- It emits no draft tokens.
- It does not execute the JetSpec draft head.
- It does not build a tree, verify mask, or rollback/commit KV/hidden state.
- The draft-head loader still preserves `runtime_supported=false` and graph
  construction still throws `unsupported_runtime`.

## Offline artifact-binding evidence

The real JetSpec draft-head manifest/tensor map/conversion plan is now checked
against the P5F source constants without creating a `llama_context` or executing
the draft head. `validate_p5f_artifact_binding.py` verifies:

- source repo `JetSpec/jetspec-Qwen3.6-35B-A3B` at commit
  `ffb38cf9917e0f426ab1b21d745e859f7788e467`;
- `general.architecture=jetspec_qwen3_draft_head`,
  `jetspec.architecture=qwen3_draft_head`,
  `jetspec.source_architecture=DFlashDraftModel`, and
  `jetspec.tensor_data_dtype=bfloat16` in the dry-run GGUF plan;
- 91 BF16 tensors, `byte_mismatch_count=0`, 473,995,264 params, and
  947,990,528 tensor payload bytes;
- target tap layers `[1, 10, 19, 28, 37]`, hidden size 2048, computed tap width
  10240, and `fc.weight` shape `[2048, 10240]`;
- draft shape constants: block size 16, 8 layers, 32 attention heads, 4 KV heads,
  and vocab size 248320;
- `requires_target_embeddings=true`, `requires_target_lm_head=true`, and
  `runtime_supported=false` remain explicit.

This upgrades P5F from source/build-only verification to
`artifact_verified_not_runtime_executed`. It still does not prove draft-head graph
execution, tree verification, or token drafting.

## Live metadata-only loader gate

`probe_p5f_loader_gate.py` writes the approved zero-tensor metadata-only preview
GGUF and invokes the built `llama-cli` loader. This is a loader-failure probe,
not a JetSpec runtime execution probe.

Expected outcomes:

- default load fails nonzero with `preview_not_allowed`;
- `LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1` plus `LLAMA_JETSPEC_EXPERIMENTAL=1` load
  still fails nonzero with `unsupported_runtime`;
- source guard proves `runtime_supported=true` is rejected before optional load
  gates;
- no draft-head graph, tree verify, or rollback path executes.

A passing probe reports `loader_gate_verified_preflight_still_blocked`. This is
stronger than source-only validation, but it confirms the current safety boundary:
The metadata-only loader-gate probe remains blocked before model-only binding;
model-only binding of the 91-tensor artifact is still non-drafting.

`probe_p5f_target_tensor_binding.py` is a separate header-only target probe. It
reads GGUF tensor-info headers for the paired target split files and verifies the
model-only linker requirements for `token_embd.weight`, `output.weight`, and
`output_norm.weight` without loading weights, creating contexts, executing
preflight, or running the draft-head graph.

## Source guard

`validate_p5f_binding_preflight.py` checks:

- P5F preflight state lives in `common/speculative.cpp` and `docs/speculative.md`, with the model-only target tensor linker guard in `src/models/jetspec_qwen3_draft_head.cpp`;
- the private preflight validates target context, draft context or model-only
  draft model, target vocab, draft metadata strings, shape constants, and target
  tap width/count;
- target tap width is cross-checked against target hidden size;
- the model-only linker requires target tensor presence/shape for
  `token_embd.weight`, `output.weight`, and `output_norm.weight`;
- preflight failure disables target taps and drops JetSpec before runtime;
- no draft tokens are emitted;
- no draft-head graph execution, tree verifier, public API/CMake/kernel/test,
  example, or poc route is added.

`validate_p5f_artifact_binding.py` checks:

- stored real draft-head artifacts match the P5F metadata and shape constants;
- `fc.weight` proves the draft head consumes the concatenated 5×2048 target tap
  input width;
- the artifact still requires target embeddings/lm-head sharing and keeps
  `runtime_supported=false`;
- the result is artifact-verified, still not runtime-executed.

`probe_p5f_loader_gate.py` checks:

- the metadata-only GGUF is rejected by the actual built loader as
  `preview_not_allowed` by default;
- even the explicit preview-load plus experimental escape hatch rejects as
  `unsupported_runtime`;
- `runtime_supported=true` is rejected before optional load gates;
- no graph, tree, or rollback execution occurs.

`probe_p5f_target_tensor_binding.py` checks:

- split target GGUF headers contain `token_embd.weight`, `output.weight`, and
  `output_norm.weight`;
- header shapes match the P5F model-only linker contract;
- the result reports `target_tensor_headers_verified_not_loaded` and keeps
  `model_loaded=false`, `context_created=false`, and `runtime_executed=false`.

## Verification evidence

Representative commands:

```bash
cmake --build build-rocm-qwen35-dev -j2 --target llama-server
python3 experiments/jetspec/validate_p5f_binding_preflight.py
python3 experiments/jetspec/validate_p5f_artifact_binding.py
python3 experiments/jetspec/probe_p5f_loader_gate.py
python3 experiments/jetspec/probe_p5f_target_tensor_binding.py --json
python3 experiments/jetspec/run_all_jetspec_contracts.py
```

Expected outcome: build passes, P5F source guard passes, P5F artifact-binding
check reports `artifact_verified_not_runtime_executed`, loader-gate probe reports
`loader_gate_verified_preflight_still_blocked`, target tensor probe reports
`target_tensor_headers_verified_not_loaded`, aggregate contracts pass, and JetSpec
remains explicit-opt-in/non-drafting.

## Next blocked work

Draft-head graph execution, causal-parallel tree construction, tree verify masks,
KV/hidden commit/rollback, server request behavior, and performance promotion
remain blocked until separate approval.
