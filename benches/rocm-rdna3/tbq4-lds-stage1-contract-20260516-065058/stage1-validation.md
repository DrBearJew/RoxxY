# Stage 1 validation

Artifact: benches/rocm-rdna3/tbq4-lds-stage1-contract-20260516-065058

```text
py_compile: pass
git diff --check: pass
contract-json: pass
tbq4_lds_route_d_k_d128: total_lds=40704 env=GGML_CUDA_TBQ4_LDS_ROUTE=D_K fallback=tbq4_vec tau=0 status=stage1_contract_locked
tbq4_lds_route_b_k_d128: total_lds=25344 env=GGML_CUDA_TBQ4_LDS_ROUTE=B_K fallback=tbq4_vec tau=0 status=backup_contract_locked
tbq4_lds_route_b_k_d128_rows128: total_lds=33792 env=GGML_CUDA_TBQ4_LDS_ROUTE=B_K fallback=tbq4_vec tau=0 status=backup_contract_locked
```
