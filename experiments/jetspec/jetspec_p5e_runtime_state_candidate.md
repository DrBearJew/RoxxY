# JetSpec P5E runtime-state bookkeeping candidate

Status: approved bounded runtime slice. This is not draft-head execution, tree
verify, KV/hidden rollback, or server behavior.

## Production hook set

Only these production files are in scope for this candidate:

- `common/speculative.cpp`
- `docs/speculative.md`

No public `include/llama.h`, `tools/server/`, repository `tests/`, `examples/`,
`pocs/`, `ggml/src/`, production CMake wiring, or non-P5A/P5B model registration
edits are approved by P5E.

## Runtime behavior

P5E extends the explicit `draft-jetspec` route with private runtime-state
bookkeeping around P5D target-tap ingestion. It records:

- runtime phase: `waiting_for_target_taps`, `target_taps_captured`, or `disabled`;
- failure reason: `none`, `missing_target_context`, or `invalid_target_taps`;
- cached target tap row count, tap count, tap width, last FNV-1a hash, total rows,
  process count, state reset count, and draft-call count;
- row metadata for each masked target output row: batch index, position, and
  primary sequence id.

The route remains fail-closed and non-drafting:

- It emits no draft tokens.
- It does not execute the JetSpec draft head.
- It does not build a tree, verify mask, or rollback/commit KV/hidden state.
- It disables tap ingestion and records `invalid_target_taps` if the side channel
  is missing, has tap count other than 5, has width other than 10240, returns a
  null row pointer, or row-state extraction disagrees with the masked row count.
- The implementation destructor disables the private target-tap side channel.

Optional trace:

```bash
LLAMA_JETSPEC_TRACE=1 LLAMA_JETSPEC_EXPERIMENTAL=1 llama-server [...] \
  --spec-type draft-jetspec --spec-draft-model <draft-head.gguf>
```

`LLAMA_JETSPEC_TAP_TRACE=1` and `LLAMA_JETSPEC_STATE_TRACE=1` are narrower trace
aliases. Trace prints phase, failure reason, cached row count, row-state count,
tap count, width, hash, total rows, process count, reset count, and draft-call
count.

## Source guard

`validate_p5e_runtime_state.py` checks:

- P5E state lives only in `common/speculative.cpp` and `docs/speculative.md`;
- the private JetSpec implementation contains runtime phase/failure enums;
- row-state metadata records batch index, position, and sequence id;
- failure handling disables the P5B tap side channel and records a reason;
- no draft tokens are emitted;
- no draft-head graph execution, tree verifier, public/server/CMake/kernel/test,
  example, or poc route is added.

## Verification evidence

Representative commands:

```bash
cmake --build build-rocm-qwen35-dev -j2 --target llama-server
python3 experiments/jetspec/validate_p5e_runtime_state.py
python3 experiments/jetspec/run_all_jetspec_contracts.py
```

Expected outcome: build passes, P5E source guard passes, aggregate contracts pass,
and JetSpec remains explicit-opt-in/non-drafting.

## Next blocked work

Draft-head graph execution, causal-parallel tree construction, tree verify masks,
KV/hidden commit/rollback, server request behavior, and performance promotion
remain blocked until separate approval.
