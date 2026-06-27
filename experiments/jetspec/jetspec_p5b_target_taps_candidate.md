# JetSpec P5B target-hidden tap capture candidate

Status: approved P5B private side-channel production candidate. This is not P5C
speculative/server integration and does not execute a JetSpec draft runtime.

## Production hook set

Only these production files are in scope:

- `src/llama-cparams.h`
- `src/llama-graph.h`
- `src/llama-graph.cpp`
- `src/llama-context.h`
- `src/llama-context.cpp`
- `src/llama-ext.h`
- `src/models/qwen35.cpp`
- `src/models/qwen35moe.cpp`

No public `include/llama.h`, `tools/server/`, repository `tests/`, `examples/`,
`pocs/`, `ggml/src/`, or production CMake wiring is approved by P5B. Later P5C
may call the private P5B staging API from its own approved connector, but P5B
itself remains a side channel.

## Runtime behavior

P5B adds a private staging API in `src/llama-ext.h`:

- `llama_set_jetspec_target_hidden_taps(ctx, enabled, masked)`
- `llama_get_jetspec_target_hidden_taps(ctx)`
- `llama_get_jetspec_target_hidden_tap_count(ctx)`
- `llama_get_jetspec_target_hidden_tap_width(ctx)`

Default is disabled. When enabled on supported Qwen35/Qwen35MoE targets with
hidden width 2048, normal target graph construction emits a side-channel tensor
for post-layer outputs after `build_cvec()` at layers `[1, 10, 19, 28, 37]`.
The host layout is contiguous `float[n_rows * 10240]`:

```text
[row0 layer1 | row0 layer10 | row0 layer19 | row0 layer28 | row0 layer37]
[row1 layer1 | row1 layer10 | row1 layer19 | row1 layer28 | row1 layer37]
...
```

The tensor is not consumed by logits, sampling, KV, recurrent state, MTP, or any
server route. It is a side channel only.

## Fail-closed rules

- Setter disables capture for non-Qwen35/Qwen35MoE targets.
- Setter disables capture if target hidden width is not 2048.
- Graph construction throws if the five fixed taps cannot produce concat width
  `10240`.
- P5B itself did not add `--spec-type draft-jetspec` and did not change existing
  speculative routing.
- Graph reuse compares the P5B capture booleans so toggling capture cannot reuse
  a stale no-tap graph.

## Source guard

`validate_p5b_target_taps.py` checks:

- only the P5B private hook files contain P5B source tokens;
- the fixed layer list `[1, 10, 19, 28, 37]` is present for Qwen35 and Qwen35MoE;
- tap concatenation is along hidden dimension and guarded by width `10240`;
- the API remains private in `src/llama-ext.h`, not `include/llama.h`;
- no P5B tap route appears in public `include/llama.h` or `tools/server/`;
- no explicit CMake wiring is added.

## Verification evidence

Representative commands:

```bash
cmake --build build-rocm-qwen35-dev -j2 --target llama-server
python3 experiments/jetspec/run_all_jetspec_contracts.py
python3 experiments/jetspec/target_hidden_taps.py \
  --fixture experiments/jetspec/fixtures/target_hidden_tap_parity_smoke.json --json
```

Observed P5B source/build outcomes:

- `llama-server` build passes.
- P5B source guard passes.
- Staged P3 hidden-tap parity fixture still passes.
- P5C/server/common wiring remains absent.

## Next blocked work

P5C may only consume this private side-channel through its own explicit approval
and verification matrix. A real JetSpec tree runtime remains separately blocked.
