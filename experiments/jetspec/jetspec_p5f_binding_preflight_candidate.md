# JetSpec P5F binding preflight candidate

Status: approved bounded runtime slice. This is not draft-head execution, tree
verify, KV/hidden rollback, or server behavior.

## Production hook set

Only these production files are in scope for this candidate:

- `common/speculative.cpp`
- `docs/speculative.md`

No public `include/llama.h`, `tools/server/`, repository `tests/`, `examples/`,
`pocs/`, `ggml/src/`, production CMake wiring, or non-P5A/P5B model registration
edits are approved by P5F.

## Runtime behavior

P5F adds a private draft-head/target binding preflight before the explicit
`draft-jetspec` implementation is instantiated. The preflight uses existing
public model/vocab metadata accessors only; it adds no public API.

Required preflight inputs:

- explicit `--spec-type draft-jetspec` or `--spec-type jetspec`;
- `LLAMA_JETSPEC_EXPERIMENTAL=1`;
- non-null target and draft contexts;
- non-null target and draft models/vocabs;
- draft metadata strings:
  - `general.architecture=jetspec_qwen3_draft_head`;
  - `jetspec.architecture=qwen3_draft_head`;
  - `jetspec.source_architecture=DFlashDraftModel`;
  - `jetspec.tensor_data_dtype=bfloat16`;
- target shape: hidden size 2048, layer count 40, vocab size 248320;
- draft shape: context train size 16, hidden size 2048, layer count 8, heads 32,
  KV heads 4, vocab size 248320;
- target tap shape: count 5, width 10240, and width equals `tap_count *
  target_hidden_size`.

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
`artifact_verified_not_runtime_executed`. It still does not prove live target
GGUF loading, live draft `llama_context` creation, or runtime preflight execution.

## Live metadata-only loader gate

`probe_p5f_loader_gate.py` writes the approved zero-tensor metadata-only preview
GGUF and invokes the built `llama-cli` loader. This is a loader-failure probe,
not a JetSpec runtime execution probe.

Expected outcomes:

- default load fails nonzero with `preview_not_allowed`;
- `LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1` load still fails nonzero with
  `unsupported_runtime`;
- no P5F preflight, draft-head graph, tree verify, or rollback path executes.

A passing probe reports `loader_gate_verified_preflight_still_blocked`. This is
stronger than source-only validation, but it confirms the current safety boundary:
P5F live preflight cannot run until a separately approved runnable draft context
exists.

## Source guard

`validate_p5f_binding_preflight.py` checks:

- P5F state lives only in `common/speculative.cpp` and `docs/speculative.md`;
- the private preflight validates target/draft context, model, vocab, draft
  metadata strings, shape constants, and target tap width/count;
- target tap width is cross-checked against target hidden size;
- preflight failure disables target taps and drops JetSpec before runtime;
- no draft tokens are emitted;
- no draft-head graph execution, tree verifier, public/server/CMake/kernel/test,
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
- even the explicit preview-load escape hatch rejects as `unsupported_runtime`;
- live P5F preflight remains blocked before any draft context, graph, tree, or
  rollback execution.

## Verification evidence

Representative commands:

```bash
cmake --build build-rocm-qwen35-dev -j2 --target llama-server
python3 experiments/jetspec/validate_p5f_binding_preflight.py
python3 experiments/jetspec/validate_p5f_artifact_binding.py
python3 experiments/jetspec/probe_p5f_loader_gate.py
python3 experiments/jetspec/run_all_jetspec_contracts.py
```

Expected outcome: build passes, P5F source guard passes, P5F artifact-binding
check reports `artifact_verified_not_runtime_executed`, loader-gate probe reports
`loader_gate_verified_preflight_still_blocked`, aggregate contracts pass, and
JetSpec remains explicit-opt-in/non-drafting.

## Next blocked work

Draft-head graph execution, causal-parallel tree construction, tree verify masks,
KV/hidden commit/rollback, server request behavior, and performance promotion
remain blocked until separate approval.
