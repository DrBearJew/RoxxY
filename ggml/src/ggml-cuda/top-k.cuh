#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_lm_head_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_router_topk_weights(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_routed_lanes(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_routed_lanes_row_slot_map(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_routed_lanes_expert_bounds(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_routed_lanes_projection(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_routed_lanes_pack_slots(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_routed_lanes_unpack_slots(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_routed_lanes_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_routed_lanes_scatter_reduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
