#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
py=${TRITON_PYTHON:-/home/mrtrent/miniconda3/envs/LLM/bin/python}

cd "$repo"
"$py" scripts/hip/check-triton-feasibility.py
"$py" experiments/compressed_kv_triton/compat_gate.py
"$py" experiments/compressed_kv_triton/llama_cpp_tensor_layout_parity.py
"$py" experiments/compressed_kv_triton/llama_cpp_block_table_parity.py
"$py" experiments/compressed_kv_triton/dispatch_policy_contract.py
"$py" experiments/compressed_kv_triton/paged_row_mapping_contract.py
"$py" experiments/compressed_kv_triton/materializers.py
"$py" experiments/compressed_kv_triton/paged_materializers.py
"$py" experiments/compressed_kv_triton/qk_only.py
"$py" experiments/compressed_kv_triton/qk_2d_tiled.py
"$py" experiments/compressed_kv_triton/online_softmax.py
"$py" experiments/compressed_kv_triton/full_qkv.py
"$py" experiments/compressed_kv_triton/qkv_2d_tiled.py
"$py" experiments/compressed_kv_triton/varlen_qkv.py
"$py" experiments/compressed_kv_triton/mask_semantics.py
"$py" experiments/compressed_kv_triton/segmented_qkv.py
"$py" experiments/compressed_kv_triton/compare_2d_segmented.py
"$py" experiments/compressed_kv_triton/tbq4_domain_parity.py
"$py" experiments/compressed_kv_triton/planar_iso_domain_parity.py
"$py" experiments/compressed_kv_triton/autotune_metadata.py
