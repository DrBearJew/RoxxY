# JetSpec P5C speculative type integration candidate

Status: approved P5C fail-closed speculative type candidate. This is not a full
JetSpec tree runtime and does not generate draft tokens.

## Production hook set

Only these production files are in scope for this candidate:

- `common/common.h`
- `common/speculative.cpp`
- `docs/speculative.md`

No `include/llama.h`, repository `tests/`, `examples/`, `pocs/`, `ggml/src/`,
production CMake wiring, or non-P5A/P5B model registration edits are approved by
P5C. A later approved runtime-loader slice may reference this type in
`tools/server/server-context.cpp` only for model-only draft-head binding.

## Runtime behavior

P5C adds the explicit speculative type name `draft-jetspec` and local alias
`jetspec` to the existing speculative parser. The type is default-off and is only
selected when the user explicitly passes `--spec-type draft-jetspec` or
`--spec-type jetspec`.

The implementation is intentionally fail-closed:

- Missing `LLAMA_JETSPEC_EXPERIMENTAL=1` disables JetSpec before runtime execution.
- Missing target context or loaded draft-head model disables JetSpec before runtime execution.
- A target tap binding mismatch disables JetSpec before runtime execution.
- Even when the route is accepted, `runtime_supported=false` is preserved and no
  draft tokens are emitted before tree verify/rollback runtime exists.
- A draft model path supplied with `--spec-type draft-jetspec` must not silently
  fall back to `draft-simple`.

The route touches the private P5B target-tap staging API only inside
`common/speculative.cpp`. P5D may keep that side channel enabled for explicit
JetSpec target-tap ingestion, but it still does not expose a public API.

## Source guard

`validate_p5c_speculative_type.py` checks:

- `COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC` exists in `common/common.h`;
- `draft-jetspec` and `jetspec` parse to the JetSpec type;
- `common_speculative_type_to_str()` returns `draft-jetspec`;
- `COMMON_SPECULATIVE_TYPE_COUNT` static assert is updated to `10`;
- `LLAMA_JETSPEC_EXPERIMENTAL` gates the route;
- the draft-model auto-fallback excludes `has_draft_jetspec`;
- the placeholder keeps `runtime_supported=false` and no-draft behavior;
- no public `draft-jetspec` route appears in `include/llama.h`, CMake,
  `ggml/src/`, `examples/`, or `pocs/`;
- any `tools/server/server-context.cpp` reference remains model-only binding with
  `ctx_dft = nullptr` and graph execution disabled.

## Verification evidence

Representative commands:

```bash
cmake --build build-rocm-qwen35-dev -j2 --target llama-server
python3 experiments/jetspec/validate_p5c_speculative_type.py
python3 experiments/jetspec/run_all_jetspec_contracts.py
```

Expected outcome: build passes, P5C source guard passes, aggregate contracts pass,
and JetSpec remains explicit-opt-in/fail-closed.

## Next blocked work

A real JetSpec tree runtime remains blocked until a separate approval covers tree
state, verify masks, hidden/KV rollback, draft-head graph execution, executable
server request behavior, and performance gates.
