# JetSpec draft-head loader contract draft

Status: inert design fixture. This file is not included by CMake and no llama.cpp
loader/server code reads it.

Purpose: define what a future `jetspec_qwen3_draft_head` GGUF loader must accept
or reject before compiled integration begins.

## Source scope

Target draft-head artifact:

- HF repo: `JetSpec/jetspec-Qwen3.6-35B-A3B`
- Observed HF architecture name: `DFlashDraftModel`
- Actual mode: causal JetSpec head, `dflash_config.causal_head=true`
- Staged GGUF architecture: `jetspec_qwen3_draft_head`
- Runtime support today: false

The staged GGUF is a draft-head-only artifact. It is not a standalone language
model and must be attached to a compatible Qwen3.6 target model at runtime.

## Required metadata

A future loader must reject the GGUF unless these metadata fields exist and match:

| Key | Required value |
| --- | --- |
| `general.architecture` | `jetspec_qwen3_draft_head` |
| `jetspec.architecture` | `qwen3_draft_head` |
| `jetspec.source_architecture` | `DFlashDraftModel` |
| `jetspec.block_size` | `16` |
| `jetspec.draft_depth` | `15` |
| `jetspec.causal_head` | `true` |
| `jetspec.mask_token_id` | `248070` |
| `jetspec.target_layer_ids` | `[1, 10, 19, 28, 37]` |
| `jetspec.num_target_layers` | `40` |
| `jetspec.requires_target_embeddings` | `true` |
| `jetspec.requires_target_lm_head` | `true` |
| `jetspec.embedding_length` | `2048` |
| `jetspec.feed_forward_length` | `6144` |
| `jetspec.block_count` | `8` |
| `jetspec.attention.head_count` | `32` |
| `jetspec.attention.head_count_kv` | `4` |
| `jetspec.attention.key_length` | `128` |
| `jetspec.attention.value_length` | `128` |
| `jetspec.rope.freq_base` | `10000000.0` |
| `jetspec.attention.layer_norm_rms_epsilon` | `1e-6` |
| `jetspec.vocab_size` | `248320` |
| `jetspec.tensor_data_dtype` | `bfloat16` |

Preview-only metadata:

- `jetspec.experimental.preview=true`
- `jetspec.experimental.runtime_supported=false`
- `jetspec.experimental.metadata_only=true` for zero-tensor preview files

A production loader must reject `jetspec.experimental.preview=true` unless it is
running in an explicit experimental/test mode.

## Tensor-info contract

Metadata-only preview files are valid only for contract validation:

- `tensor_count == 0`
- no tensor payload
- no runtime loading

A real draft-head GGUF must contain exactly 91 BF16 tensor infos matching
`conversion_plans/JetSpec_jetspec-Qwen3.6-35B-A3B_gguf_plan.json`. P2 parity
requires raw BF16 copy semantics only: no quantization, no packing, and no dtype
conversion.

- `draft.fc.weight`, shape `[2048, 10240]`
- `draft.hidden_norm.weight`, shape `[2048]`
- `draft.norm.weight`, shape `[2048]`
- for each layer `0..7`:
  - `draft.layers.{i}.input_layernorm.weight`, `[2048]`
  - `draft.layers.{i}.post_attention_layernorm.weight`, `[2048]`
  - `draft.layers.{i}.self_attn.q_proj.weight`, `[4096, 2048]`
  - `draft.layers.{i}.self_attn.k_proj.weight`, `[512, 2048]`
  - `draft.layers.{i}.self_attn.v_proj.weight`, `[512, 2048]`
  - `draft.layers.{i}.self_attn.o_proj.weight`, `[2048, 4096]`
  - `draft.layers.{i}.self_attn.q_norm.weight`, `[128]`
  - `draft.layers.{i}.self_attn.k_norm.weight`, `[128]`
  - `draft.layers.{i}.mlp.gate_proj.weight`, `[6144, 2048]`
  - `draft.layers.{i}.mlp.up_proj.weight`, `[6144, 2048]`
  - `draft.layers.{i}.mlp.down_proj.weight`, `[2048, 6144]`

All tensor infos are BF16 in the current draft. Quantized variants are out of
scope until a numerically matched BF16 loader exists.

## Target-model compatibility checks

The draft-head loader must be paired with a target model that can provide:

1. Token embeddings for the committed anchor token and mask tokens.
2. Hidden-state taps for target layers `[1, 10, 19, 28, 37]`.
3. A target `lm_head` projection with vocab size `248320`.
4. Target hidden width `2048` for each selected tap.
5. A concatenated hidden tap input width of `5 * 2048 = 10240` for `draft.fc.weight`.

Reject if the target cannot expose these taps without changing normal greedy
outputs when JetSpec is disabled.

## Runtime graph contract

For one committed context tail, the draft-head graph consumes:

- `anchor_token_id`: scalar token id from target context.
- `noise_token_ids`: `[anchor_token_id, mask_token_id x 15]`.
- `target_hidden_taps`: five hidden vectors at target layers `[1, 10, 19, 28, 37]`.

The graph performs:

1. Embed `noise_token_ids` with target embeddings.
2. Concatenate target hidden taps into `[10240]`.
3. Apply `draft.fc.weight` then `draft.hidden_norm.weight`.
4. Run 8 causal draft-head decoder blocks over block size 16.
5. Apply `draft.norm.weight`.
6. Project positions `1..15` with the target `lm_head`.
7. Return draft logits for 15 candidate positions.

Tree construction and acceptance remain separate contracts in
`jetspec_tree_contract.hpp` and `tree_semantics.py`.

## Rejection conditions

A future loader must fail closed if any condition is true:

- GGUF architecture is not `jetspec_qwen3_draft_head`.
- Preview metadata is present without an explicit experimental mode.
- Tensor count is neither `0` in metadata-only test mode nor `91` for payload mode.
- Any real tensor is not BF16 or has a mismatched shape/name.
- `block_size != draft_depth + 1`.
- Target layer IDs are missing, unsorted, duplicated, or incompatible with the target.
- Target embeddings or target lm_head sharing is unavailable.
- Runtime would mutate target KV/state for rejected branches without a rollback/commit contract.

## Inert validation tools

- `convert_jetspec_head_to_gguf.py --write-metadata-only` writes a zero-tensor GGUF preview.
- `parse_gguf_preview.py --validate-jetspec-loader` parses that preview and validates the metadata contract above.
- `draft_head_loader_prototype.py --self-test` is the P1 standalone loader/parser prototype. It maps validated metadata into the inert `draft_head_metadata` field shape, rejects malformed target-layer metadata, and applies the preview-runtime gate.
- `bf16_payload_parity.py --fixture fixtures/bf16_payload_parity_smoke.json` is the P2 synthetic/local payload parity fixture. It validates all 91 tensor names, BF16 dtypes, shapes, source `data_offsets`, GGUF tensor-info offsets, and the `raw_bf16_no_transform` policy.
- `bf16_payload_parity.py --self-test` also creates a sparse local safetensors shell and passes it through the converter's header validator without copying the 947,990,528 byte payload.

The P1 prototype may inspect metadata-only previews, but runtime preparation fails closed unless `--allow-preview-runtime` is passed explicitly. Even with that experimental flag, current previews remain unrunnable because `runtime_supported=false` and no BF16 tensor payload is loaded. P2 adds payload parity evidence only; it still does not make previews runnable.

These tools are fixtures only. They do not register a llama.cpp architecture and
do not make the original `llama-server` load JetSpec GGUF files.
