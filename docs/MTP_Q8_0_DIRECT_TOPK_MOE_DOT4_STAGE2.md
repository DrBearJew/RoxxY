# MTP stage2: Q8_0 LM-head top1 and Q8_0 MoE small-route dot4

This patch keeps the verifier exactness contract from the expert pack intact: do not promote causal-batched / active-state paths that already showed `state_match=0` before repair.  The additions here are opt-in where they can change routing and are wired through the existing active/shadow LM-head controls.

## What changed

1. `GGML_OP_LM_HEAD_TOP_K` now accepts `GGML_TYPE_Q8_0` output heads in the CUDA backend in addition to the existing `Q6_K` path.
2. `ggml_cuda_op_lm_head_top_k()` dispatches a new Q8_0 top1 implementation that computes `Q8_0 x Q8_1 -> top1` directly without materializing the full logits tensor.
3. The Qwen3.5 dense and Qwen3.5/3.6 MoE graph builders can use the direct LM-head top1 path when the output tensor is Q6_K or Q8_0 and LoRA is not active.
4. The Qwen3.5/3.6 MoE graph now has the same direct active and shadow wiring that the dense Qwen3.5 graph already had.
5. A Q8_0 MoE small-route dot4 selector has been added for route-direct `MUL_MAT_ID`/fused gate-up shapes.  It is deliberately opt-in with `LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4=1`.
6. Backend canaries were added for real-ish gate/up (`K=2048 -> rows=1024`) and down (`K=512 -> rows=2048`) Q8_0 route shapes with token counts 1, 3, and 4.

## Environment gates

Direct LM-head top1 active mode uses the existing lab gates:

```bash
LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE=1
LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE_RAW_UNSAFE=1
# unset/0 to use direct LM_HEAD_TOP_K; set to 1 for full-logits TOP_K comparison path
LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE_LOGITS=0
LLAMA_MTP_FUSED_LM_HEAD_TOPK_LOG=1
```

Shadow validation is a separate mode. Leave active mode off so the graph materializes
both full logits and a direct LM-head top1 tensor for comparison:

```bash
LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE=0
LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW=1
LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW_REQUIRE=1
LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW_LOG=1
LLAMA_MTP_FUSED_LM_HEAD_TOPK_LOG=1
```

Q8_0 MoE small-route dot4 is opt-in:

```bash
LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4=1
LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4_LOG=1
LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4_MAX_ROUTES=64
LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4_ROWS_PER_BLOCK=2   # allowed: 1,2,4,8; unset for shape default
```

The selected route log marker is:

```text
route=mmvq_moe_small_route_dot4_q8_0
```

The direct LM-head Q8_0 marker is:

```text
route=cuda_lm_head_top1_q8_0
```

## Validation checklist

Run the normal exactness gates first with unsafe causal-batched features disabled:

- status accepted/no rejects
- `token_match=1` vs serial oracle
- `state_match=1` vs serial oracle before repair
- stable generated/accepted counts
- route logs include the expected Q8_0 markers only when the env gates above are set
- speed beats the safe serial baseline without changing verifier state bytes

Backend canaries:

```bash
LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4=1 \
LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4_LOG=1 \
./build-rocm-fixed/bin/test-backend-ops test -o MUL_MAT_ID,MUL_MAT -p 'q8_0.*(1024|2048|512)'
```

LM-head shadow probe:

```bash
LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE=0 \
LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW=1 \
LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW_REQUIRE=1 \
LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW_LOG=1 \
LLAMA_MTP_FUSED_LM_HEAD_TOPK_LOG=1 \
./build-rocm-fixed/bin/llama-cli -m /path/to/qwen3.5-or-qwen3.6-q8-output.gguf -p 'hello' -n 16
```
