# JetSpec runtime state contract draft

Status: inert design fixture. This file and `jetspec_runtime_contract.hpp` are not
included by CMake and no llama.cpp runtime reads them.

Purpose: sketch the future C++ data boundaries for JetSpec loader metadata,
target-model bindings, tree verification, hidden-cache commit, and per-round
state. This is a planning artifact only, not an ABI.

## Header scope

`jetspec_runtime_contract.hpp` defines declaration-only structs under the inert
namespace:

```cpp
namespace llama_jetspec_experiment { ... }
```

It may include only other inert `experiments/jetspec` headers and standard C++
headers. It must not include or reference production headers such as `llama.h`,
`ggml.h`, `common/`, `tools/server/`, or `src/models/` until runtime integration
is intentionally promoted.

## Required constant contract

The header freezes the Qwen3.6 JetSpec draft-head constants already validated by
Python fixtures:

| Constant | Value |
| --- | ---: |
| `jetspec_qwen36_block_size` | `16` |
| `jetspec_qwen36_draft_depth` | `15` |
| `jetspec_qwen36_mask_token_id` | `248070` |
| `jetspec_qwen36_target_layers` | `40` |
| `jetspec_qwen36_target_tap_count` | `5` |
| `jetspec_qwen36_hidden_size` | `2048` |
| `jetspec_qwen36_concat_width` | `10240` |
| `jetspec_qwen36_draft_layers` | `8` |
| `jetspec_qwen36_attention_heads` | `32` |
| `jetspec_qwen36_attention_heads_kv` | `4` |
| `jetspec_qwen36_head_dim` | `128` |
| `jetspec_qwen36_ffn_size` | `6144` |
| `jetspec_qwen36_vocab_size` | `248320` |
| `jetspec_qwen36_tensor_count` | `91` |

Static invariants:

```text
draft_depth == block_size - 1
concat_width == target_tap_count * hidden_size
```

## Struct boundaries

### `draft_head_metadata`

Future loader-visible metadata, copied from the GGUF contract:

- staged arch: `jetspec_qwen3_draft_head`
- head arch: `qwen3_draft_head`
- source arch: `DFlashDraftModel`
- target taps: `[1, 10, 19, 28, 37]`
- `requires_target_embeddings=true`
- `requires_target_lm_head=true`
- `runtime_supported=false` until compiled parity exists

### `draft_head_tensor_info` and `draft_head_loader_plan`

Future tensor-info and loader plan boundary. It records GGUF tensor names, HF names,
shapes, BF16 type, byte counts, and offsets. Preview files remain preview-only;
a future production loader must not run preview metadata unless an explicit
experimental mode is enabled.

### `target_model_bindings`

Future target-model compatibility boundary. It records whether the target exposes:

- token embeddings;
- `lm_head`;
- hidden taps;
- matching hidden width, vocab size, and layer count.

The inert header intentionally uses abstract numeric handles rather than llama.cpp
or ggml pointer types.

### `target_hidden_cache_state`

Future committed hidden-cache boundary. It carries the invariant:

```text
hidden_row_count == committed_token_count - 1
```

before and after every tree round.

### `tree_verify_plan`

Future verify-mask boundary:

- `past_len`
- real node count `N`
- bucket node count `B`
- node-only ancestor block
- optional bucket visibility block

This mirrors `jetspec_tree_verify_mask.md` without naming backend-specific mask
or kernel objects.

### `round_commit_plan` and `round_state`

Future per-round ownership boundary:

- accepted root-inclusive path;
- accepted draft tokens;
- correction token;
- appended node indices `[root | accepted]`;
- discarded node indices;
- correction-hidden no-append rule.

`round_state` ties loader, target binding, committed tokens, draft tree, verify plan,
and commit plan into one explicit state object.

## Fail-closed policy

A future implementation must enter `runtime_phase::failed` and set a non-`none`
`runtime_failure` if any contract fails:

- metadata mismatch;
- preview file used without explicit experimental mode;
- missing or wrong-shaped tensor;
- missing target embeddings or target lm_head;
- missing target hidden taps;
- hidden width mismatch;
- verify mask violation;
- rejected branch hidden/KV leak;
- runtime path is not yet supported.

## Inert validation tools

- `validate_runtime_contract.py`: text validator for the header constants, required
  structs/enums, local-only includes, and forbidden production references.
- `test_runtime_contract.py`: unittest wrapper for the validator plus key invariant checks.

These tools do not compile llama.cpp and do not make the original server see these
structs.
