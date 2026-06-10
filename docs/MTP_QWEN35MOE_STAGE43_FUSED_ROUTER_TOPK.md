# Qwen35MoE MTP Stage4.3 fused router/top-k/weights candidate

Stage4.2 proved the important correctness boundary: replacing the row-serial router dense matmul with `router_mmvf_serial_columns` can keep `token_match=1` and `state_match=1` before repair, but the remaining top-k/weight island stayed row-serial and the speed result still lost to the safe/default serial verifier.

Stage4.3 therefore leaves projection routing alone.  It keeps the Stage4.1/4.2 exact state barriers and routed expert projection serial-column routes, then opens only the remaining router island:

```text
router logits
softmax
grouped expert masking
top-k
weight gather/sum/clamp/normalize/scale
```

The path is opt-in and diagnostic until target-machine logs prove both exactness and speed.

## Activation

```bash
LLAMA_MTP_SERIAL_EQUIV_PREFIX=1 \
LLAMA_MTP_PREFIX_ROWEQ_STAGE43_FUSED_ROUTER_TOPK=1 \
LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH=1 \
LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG=1 \
LLAMA_MTP_PREFIX_ROWEQ_STAGE41_NO_DEFAULT_SERIAL=1 \
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER_DENSE=0 \
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER=0 \
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS=0 \
LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS=1 \
LLAMA_MTP_ROWEQ_ROUTER_MMVF_ACTIVE=1 \
LLAMA_MTP_ROWEQ_ROUTER_TOPK_FUSED_ACTIVE=1 \
LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE=1 \
LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS=1 \
LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH=1 \
LLAMA_MTP_VERIFY_TRACE=1 \
LLAMA_MTP_VERIFY_COMPARE=1 \
LLAMA_MTP_PREFIX_HIDDEN_TRACE=1 \
./llama-server -v ... 2>&1 | tee stage43-fused-router-topk.log
```

The helper script exports those defaults:

```bash
scripts/run_qwen35moe_stage43_fused_router_topk_env.sh ./llama-server -v ... 2>&1 | tee stage43-fused-router-topk.log
```

## Expected markers

Backend reason:

```text
exact_roweq_stage43_fused_router_topk_prefix_requested
```

Graph marker fields:

```text
stage42_router_topk=1
stage43_router_topk_fused=1
serial_router=0
serial_topk_weights=0
batch_routed_proj=1
router_topk_mode=router_mmvf_roweq_fused_topk_weights_candidate
```

Backend route markers:

```text
route=router_mmvf_serial_columns
route=router_topk_weights_roweq_fused_candidate
route=mmvq_serial_columns
route=mmvq_serial_columns_single_launch
```

The fused route marker is emitted by the explicit `GGML_OP_ROUTER_TOPK_WEIGHTS` CUDA kernel.  It consumes the small-N router logits and produces a compact payload containing selected expert ids plus normalized/scaled weights.  If the route marker is absent, the graph did not select the Stage4.3 op and the run must not be promoted.

## Checker

Promotion-mode checker:

```bash
scripts/check_qwen35moe_stage43_fused_router_topk_log.py stage43-fused-router-topk.log --require-ncols 4
```

Diagnostic route-discovery mode, not promotion:

```bash
scripts/check_qwen35moe_stage43_fused_router_topk_log.py stage43-fused-router-topk.log \
  --allow-pre-repair-state-miss \
  --no-require-fused-route
```

## Promotion gate

Stage4.3 can only be promoted if the same prompt/settings show:

```text
token_match=1
state_match=1 before repair
stable acceptance
route=router_mmvf_serial_columns
route=router_topk_weights_roweq_fused_candidate
selected routed projection serial-column routes
speed > same-prompt safe/default serial verifier baseline
```

If `state_match=0` returns, this explicit fused router/top-k/weights op is not row-equivalent enough for Qwen35MoE's grouped router path; the next step is to keep top-k/weights serial or tighten the kernel arithmetic/order against the serial oracle.
