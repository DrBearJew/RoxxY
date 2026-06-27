# JetSpec P5D target-tap ingestion candidate

Status: approved bounded runtime slice. This is not draft-head execution, tree
verify, KV/hidden rollback, or server behavior.

## Production hook set

Only these production files are in scope for this candidate:

- `common/speculative.cpp`
- `docs/speculative.md`

No public `include/llama.h`, `tools/server/`, repository `tests/`, `examples/`,
`pocs/`, `ggml/src/`, production CMake wiring, or non-P5A/P5B model registration
edits are approved by P5D.

## Runtime behavior

When the user explicitly selects `--spec-type draft-jetspec` or `--spec-type
jetspec` and sets `LLAMA_JETSPEC_EXPERIMENTAL=1`, P5D keeps the private P5B
target-tap side channel enabled on the target context. After target decode,
`common_speculative_impl_draft_jetspec::process()` copies the masked target tap
rows into an internal buffer and records a FNV-1a hash.

The route remains fail-closed and non-drafting:

- It emits no draft tokens.
- It does not execute the JetSpec draft head.
- It does not build a tree, verify mask, or rollback/commit KV/hidden state.
- It disables tap ingestion if the target side channel is missing, has tap count
  other than 5, has width other than 10240, or returns a null row pointer.
- The implementation destructor disables the private target-tap side channel.

Optional trace:

```bash
LLAMA_JETSPEC_TRACE=1 LLAMA_JETSPEC_EXPERIMENTAL=1 llama-server [...] \
  --spec-type draft-jetspec --spec-draft-model <draft-head.gguf>
```

Trace prints the captured tap row count, width, hash, total rows, and process
count. `LLAMA_JETSPEC_TAP_TRACE=1` is accepted as a narrower alias.

## Source guard

`validate_p5d_target_tap_ingestion.py` checks:

- P5D keeps target taps enabled for the explicit JetSpec route;
- target rows are read with `llama_get_jetspec_target_hidden_taps()` after target
  decode;
- masked rows are counted from `batch.logits`;
- captured rows are copied into an internal buffer and hashed with FNV-1a;
- missing/wrong-width/null tap data disables ingestion instead of drafting;
- no draft tokens are emitted;
- no public/server/CMake/kernel/test/example/poc route is added.

## Verification evidence

Representative commands:

```bash
cmake --build build-rocm-qwen35-dev -j2 --target llama-server
python3 experiments/jetspec/validate_p5d_target_tap_ingestion.py
python3 experiments/jetspec/run_all_jetspec_contracts.py
```

Expected outcome: build passes, P5D source guard passes, aggregate contracts pass,
and JetSpec remains explicit-opt-in/non-drafting.

## Next blocked work

Draft-head graph execution, causal-parallel tree construction, tree verify masks,
KV/hidden commit/rollback, server request behavior, and performance promotion
remain blocked until separate approval.
