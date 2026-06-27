# JetSpec draft-head GGUF contract draft

Status: design draft only. Not wired into production converter, loader, CMake, or server. The only writer is the inert experimental `convert_jetspec_head_to_gguf.py` preview tool under this directory.

## Source artifact

HF repo: `JetSpec/jetspec-Qwen3.6-35B-A3B`

Observed files:
- `config.json`
- `dflash.py`
- `model.safetensors`
- `training_state.pt`

Observed config essentials:

```json
{
  "architectures": ["DFlashDraftModel"],
  "auto_map": {"AutoModel": "dflash.DFlashDraftModel"},
  "block_size": 16,
  "dflash_config": {
    "causal_head": true,
    "mask_token_id": 248070,
    "target_layer_ids": [1, 10, 19, 28, 37]
  },
  "dtype": "bfloat16",
  "hidden_size": 2048,
  "intermediate_size": 6144,
  "num_hidden_layers": 8,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "num_target_layers": 40,
  "vocab_size": 248320,
  "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
  "rms_norm_eps": 1e-6
}
```

## Runtime contract

The JetSpec head is not a standalone LM. Runtime must provide:
- Target token embedding for `[anchor, mask, ..., mask]` noise embeddings.
- Target hidden taps from layers `[1, 10, 19, 28, 37]`, concatenated along feature dim.
- Target `lm_head` projection for head hidden states to vocab logits.

Head forward:
1. Build `block_output_ids = [anchor_token, mask_token_id x 15]`.
2. Embed with target `embed_tokens` to produce `noise_embedding` `(1, 16, target_hidden_size)`.
3. Concatenate selected target hidden states for the committed context.
4. Apply draft-head `fc + hidden_norm` to tapped hidden.
5. Run 8 causal draft-head decoder layers.
6. Apply draft-head final norm.
7. Apply target `lm_head` to prediction positions.
8. Return draft logits `(1, 15, vocab)` for tree construction.

## Proposed GGUF metadata keys

Names are provisional. Only emit from inert preview tools until loader names are agreed.

- `jetspec.arch = "qwen3_draft_head"`
- `jetspec.block_size = 16`
- `jetspec.draft_depth = 15`
- `jetspec.causal_head = true`
- `jetspec.mask_token_id = 248070`
- `jetspec.target_layer_ids = [1, 10, 19, 28, 37]`
- `jetspec.num_target_layers = 40`
- `jetspec.hidden_size = 2048`
- `jetspec.intermediate_size = 6144`
- `jetspec.num_hidden_layers = 8`
- `jetspec.num_attention_heads = 32`
- `jetspec.num_key_value_heads = 4`
- `jetspec.head_dim = 128`
- `jetspec.rope_theta = 10000000`
- `jetspec.rms_norm_eps = 1e-6`
- `jetspec.requires_target_embeddings = true`
- `jetspec.requires_target_lm_head = true`

## Expected tensor families

Names are descriptive, not final GGUF names.

- `draft.fc.weight`
- `draft.hidden_norm.weight`
- `draft.norm.weight`
- Per-layer:
  - `draft.layers.{i}.input_layernorm.weight`
  - `draft.layers.{i}.self_attn.q_proj.weight`
  - `draft.layers.{i}.self_attn.k_proj.weight`
  - `draft.layers.{i}.self_attn.v_proj.weight`
  - `draft.layers.{i}.self_attn.o_proj.weight`
  - `draft.layers.{i}.self_attn.q_norm.weight`
  - `draft.layers.{i}.self_attn.k_norm.weight`
  - `draft.layers.{i}.post_attention_layernorm.weight`
  - `draft.layers.{i}.mlp.gate_proj.weight`
  - `draft.layers.{i}.mlp.up_proj.weight`
  - `draft.layers.{i}.mlp.down_proj.weight`

## Non-goals for first inert drop

- No CMake target.
- No server flag.
- No loader registration.
- No modification to existing MTP GGUF handling.
- No attempt to quantize or pack draft-head weights.

## Promotion gates

1. Converter can list and map all safetensors keys without missing/unexpected tensors.
2. Loader constructs a graph that numerically matches HF `DFlashDraftModel.forward()` on a tiny captured fixture.
3. Tree verifier is lossless with a deterministic greedy sampler.
4. Server build remains unchanged when JetSpec is not enabled.
