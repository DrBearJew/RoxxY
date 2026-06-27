# JetSpec P5A loader-registration candidate

Status: approved P5A validation-only production candidate. This is not P5B target
hidden capture and not P5C speculative/server integration.

## Production hook set

Only these production files are in scope:

- `src/llama-arch.h`
- `src/llama-arch.cpp`
- `src/llama-model.cpp`
- `src/models/models.h`
- `src/models/jetspec_qwen3_draft_head.cpp`

No `common/`, `tools/server/`, `include/llama.h`, repository `tests/`,
`examples/`, `pocs/`, `ggml/src/`, or production CMake wiring is approved by
P5A.

## Runtime behavior

P5A registers `general.architecture=jetspec_qwen3_draft_head` so the llama.cpp
loader can recognize JetSpec draft-head GGUF files, validate their metadata, and
then fail closed before runtime execution.

Required fail-closed outcomes:

- Metadata-only preview without `LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1` fails as
  `preview_not_allowed`.
- Metadata-only preview with `LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1` still fails as
  `unsupported_runtime` because `jetspec.experimental.runtime_supported=false`.
- 91-tensor payload mode must contain exactly 91 BF16 tensor infos matching the
  P2 pattern before any future runtime could proceed.
- `build_arch_graph()` is unreachable and throws `unsupported_runtime` if called.
- `LLAMA_JETSPEC_EXPERIMENTAL=1` is reserved for future runtime development and
  does not make P5A executable.

## Source guard

`validate_p5a_loader_candidate.py` checks:

- the five P5A files contain the expected arch/model registration tokens;
- the loader source contains `preview_not_allowed`, `unsupported_runtime`,
  `LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD`, `LLAMA_JETSPEC_EXPERIMENTAL`, BF16 checks,
  and the 0-or-91 tensor inventory rule;
- no P5C `draft-jetspec` speculative type is wired in `common/` or
  `tools/server/`;
- no explicit CMake references are added for the new model source.

## Verification evidence

Representative commands:

```bash
cmake -S . -B build-rocm-qwen35-dev
cmake --build build-rocm-qwen35-dev -j2 --target llama-server
python3 experiments/jetspec/run_all_jetspec_contracts.py
```

Preview-load proof:

```bash
python3 experiments/jetspec/convert_jetspec_head_to_gguf.py \
  --write-metadata-only --output /tmp/jetspec-qwen3-draft-head-preview-p5a.gguf --force --json
build-rocm-qwen35-dev/bin/llama-server -m /tmp/jetspec-qwen3-draft-head-preview-p5a.gguf -c 16 -ngl 0
LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1 \
  build-rocm-qwen35-dev/bin/llama-server -m /tmp/jetspec-qwen3-draft-head-preview-p5a.gguf -c 16 -ngl 0
```

Observed P5A outcomes:

- no-env load exits nonzero with `preview_not_allowed`;
- preview-allow load exits nonzero with `unsupported_runtime`;
- no graph execution occurs.

## Next blocked work

P5B target hidden capture and P5C speculative type integration require a new
explicit approval decision and their own verification matrices.
