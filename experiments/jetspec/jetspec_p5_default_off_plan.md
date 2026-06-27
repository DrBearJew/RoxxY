# JetSpec P5 default-off production candidate plan

Status: proposed inert plan only; not approved for production edits. This file is
not included by CMake and no llama.cpp runtime reads it.

Decision: after P1-P4 contract evidence, the first production-path candidate
should be a minimal default-off implementation sequence that fails closed before
runtime execution. In the context of loader risk, target-state ownership risk,
and existing speculative decoding paths, choose a staged P5A/P5B/P5C candidate
instead of wiring a full JetSpec runtime at once. Accept slower progress in
exchange for preserving default `llama-server` behavior and keeping rollback
trivial.

## Context

P1-P4 evidence exists under `experiments/jetspec/` only:

- P1 maps metadata-only GGUF previews to `draft_head_metadata` and blocks runtime
  because `runtime_supported=false`.
- P2 validates all 91 BF16 tensor metadata entries and raw BF16 no-transform
  payload rules.
- P3 validates target hidden taps against HF `hidden_states[layer_id + 1]` and
  proves tap capture is side-channel only in fixtures.
- P4 composes tree verify, greedy accept, hidden commit, rollback, and baseline
  token parity in one deterministic fixture.

The next step is planning only. Do not edit production paths in this plan.

## Evidence checked

- `src/CMakeLists.txt`: `file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")` means a
  new `src/models/*.cpp` file is production-compiled automatically.
- `src/llama-arch.h` and `src/llama-arch.cpp`: model architectures are registered
  as `llm_arch` enum values and architecture-name strings.
- `src/llama-model.cpp`: `llama_model_mapping()` maps an `llm_arch` to a concrete
  `llama_model_*` class.
- `src/models/models.h`: concrete model classes are declared for model-specific
  graph/loader code.
- `src/models/qwen35_mtp.cpp`: existing draft-head-like model shares target
  embedding tensors via `link_shared_tensors()` and builds a draft graph from
  target-provided hidden input.
- `common/common.h`: speculative decoding types live in `common_speculative_type`
  and params are grouped in `common_params_speculative`.
- `common/speculative.cpp`: string parsing, implementation priority, and
  `common_speculative_init()` instantiate speculative implementations.
- `common/speculative.h`: the public common speculative surface exposes begin,
  process, draft, accept, hidden-embedding needs, and branch-candidate helpers.
- `common/arg.cpp`: `--spec-type` and draft-model options populate
  `common_params_speculative`.
- `tools/server/server-context.cpp`: server speculative/MTP integration already
  carries high blast radius and target-state logic, so server behavior must remain
  unchanged until explicit opt-in.
- `include/llama.h`: current experimental MTP APIs expose hidden tensors and MTP
  context linking; JetSpec should not silently reuse them with different semantics.
- `docs/speculative.md`: public docs list current speculative types and options.

## Non-goals

- No full JetSpec runtime graph in the first P5 candidate.
- No default enablement, auto-detection, or silent fallback to JetSpec.
- No quantization, packing, or tensor-format conversion beyond raw BF16 metadata
  validation.
- No `ggml/src/` kernel work in P5; kernel work is P6 or a separate approved
  kernel-specific P5 design.
- No behavior change when `--spec-type draft-jetspec` is absent.
- Required proof phrase: no default behavior change.

## Default-off controls

A future P5 implementation must require both an explicit build/config choice and
an explicit runtime choice:

1. Build/config gate: `LLAMA_JETSPEC_EXPERIMENTAL=1` or an equivalent explicitly
   named experimental build option.
2. Runtime opt-in: `--spec-type draft-jetspec` plus an explicit draft-head path.
3. Preview-load development escape hatch: `LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1`,
   for loader inspection only, never for normal runtime.

Default disabled means default disabled:

- `--spec-type` without `draft-jetspec` behaves exactly as before.
- Metadata-only GGUF previews still fail closed as `preview_not_allowed` unless
  the preview-load development escape hatch is present.
- Even with preview inspection, `runtime_supported=false` still fails closed as
  `unsupported_runtime`.
- `llama-server` has no new behavior unless all explicit gates are present.

## Candidate sequence

### P5A: loader-registration candidate, default-off and fail-closed

Likely production files, only after explicit approval:

- `src/llama-arch.h`: add `LLM_ARCH_JETSPEC_QWEN3_DRAFT_HEAD`.
- `src/llama-arch.cpp`: map it to `jetspec_qwen3_draft_head`.
- `src/models/models.h`: declare `llama_model_jetspec_qwen3_draft_head`.
- `src/models/jetspec_qwen3_draft_head.cpp`: validate JetSpec metadata and 91
  tensor infos, then reject runtime unless the experimental gates are present.
- `src/llama-model.cpp`: map `LLM_ARCH_JETSPEC_QWEN3_DRAFT_HEAD` to the new model
  class.

Acceptance:

- Loading a metadata-only preview without `LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1`
  fails as `preview_not_allowed`.
- Loading a preview with the escape hatch still fails before execution as
  `unsupported_runtime` because `runtime_supported=false`.
- Existing target models load unchanged.
- Existing `draft-mtp`, `draft-simple`, and n-gram speculative paths still parse
  and initialize as before.

Rollback: revert only the five P5A production files above. rollback: revert the candidate subphase cleanly. No persisted data or
public API should require migration.

### P5B: target hidden tap capture candidate, default-off

Likely production files, only after P5A passes and explicit approval:

- `include/llama.h` or a private `src/llama-ext.h` staging API: add explicit
  JetSpec target-tap capture accessors; do not overload existing MTP hidden APIs.
- `src/llama-context.*` and graph/input code: capture target post-layer outputs
  for `[1, 10, 19, 28, 37]` only when JetSpec capture is enabled.
- No server route or default context change.

Acceptance:

- A/B decode with capture disabled is byte-identical to baseline logits and greedy
  output.
- Capture enabled on a deterministic fixture matches `hidden_states[layer_id + 1]`
  ordering and width `10240`.
- Missing tap, wrong width, or target mismatch disables JetSpec before drafting.

Rollback: remove the tap-capture accessors and private capture storage; no model
file or cache format changes remain.

### P5C: speculative type integration candidate, explicit opt-in only

Likely production files, only after P5A/P5B pass and explicit approval:

- `common/common.h`: add `COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC`.
- `common/speculative.cpp`: parse `draft-jetspec`, instantiate a JetSpec
  implementation only when target bindings and experimental gates pass, and leave
  existing implementation priority unchanged when absent.
- `common/speculative.h`: expose only the minimal internal hooks required by the
  implementation.
- `common/arg.cpp`: accept `--spec-type draft-jetspec` and reuse the existing
  draft model path options.
- `tools/server/server-context.cpp`: no server behavior change unless the request
  explicitly chooses `--spec-type draft-jetspec`.
- `docs/speculative.md`: document the feature as experimental and default-off.

Acceptance:

- Default `llama-server` launch, no-spec launch, existing MTP launch, and existing
  n-gram launch produce unchanged routing and output.
- With `draft-jetspec`, any failed loader, target binding, tap, verify, or rollback
  check disables JetSpec before partial execution.
- Runtime trace must show `draft-jetspec` only when explicitly requested.

Rollback: revert the P5C enum/parser/implementation/docs/server edits; existing
speculative types remain unchanged.

## Preview-GGUF rejection contract

The production candidate must preserve the staging contract:

- `general.architecture=jetspec_qwen3_draft_head` is recognized only by the
  approved P5A loader.
- `jetspec.experimental.preview=true` rejects by default.
- `jetspec.experimental.metadata_only=true` never executes.
- `runtime_supported=false` never becomes runnable.
- Tensor payload mode requires exactly 91 BF16 tensors matching the P2 plan before
  any graph can be built.

## No-default-behavior-change proof

Before any P5 implementation is considered complete, run and save evidence for:

1. `python3 experiments/jetspec/run_all_jetspec_contracts.py` passes.
2. Default CMake/build without JetSpec gates succeeds.
3. Default `llama-server` command without speculative decoding has identical
   routing/output to the pre-P5 baseline on a deterministic prompt.
4. Existing `--spec-type draft-mtp` command still initializes the MTP path and
   keeps its prior verification result.
5. Existing n-gram speculative commands still parse and route unchanged; ngram compatibility remains covered.
6. Metadata-only JetSpec GGUF preview fails closed by default.
7. Any P5 benchmark is labeled candidate/experimental and compared against the
   current proven speculative baseline; losing candidates are demoted or removed.

## Verification matrix

| Gate | Command/evidence | Pass condition |
| --- | --- | --- |
| Staging contracts | `python3 experiments/jetspec/run_all_jetspec_contracts.py` | all checks pass |
| Loader preview rejection | metadata-only preview load attempt | `preview_not_allowed` by default |
| Runtime unsupported | preview-load escape hatch | `unsupported_runtime`, no graph execution |
| Default server path | deterministic no-spec server run | output/routing unchanged |
| Existing MTP path | deterministic `--spec-type draft-mtp` run | output/routing not regressed |
| Existing n-gram path | deterministic n-gram parse/run | output/routing not regressed |
| JetSpec opt-in path | `LLAMA_JETSPEC_EXPERIMENTAL=1 --spec-type draft-jetspec` | either passes all contract gates or disables before partial execution |
| Performance | agreed prompt matrix vs baseline | only promoted if matches or beats baseline |

## Rejected alternatives

- Full runtime first: rejected because loader, tap capture, rollback, and tree
  verify failures would be hard to isolate.
- Automatic enablement when a JetSpec GGUF is present: rejected because it changes default
  server behavior and can run preview files accidentally.
- Reusing `draft-mtp` semantics for JetSpec: rejected because JetSpec requires
  target hidden taps, tree-causal verify, and rejected-branch rollback semantics
  that differ from linear MTP.
- Adding `ggml/src/` kernels in P5: rejected because correctness and default-off
  plumbing must land before performance kernels.

## Approval gate

This plan is not an approval to edit production paths. A future P5 implementation
turn must explicitly state that P5 production edits are approved, cite this plan,
name the selected subphase (P5A, P5B, or P5C), and run the verification matrix
for that subphase. If approval is absent, keep all work under `experiments/jetspec/`.
