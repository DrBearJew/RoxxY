# q8_0 K + TBQ4 V VEC + Sparse V checkpoint

## Scope

- Enable `q8_0` K + `tbq4_0` V on ROCm/RDNA3 VEC FlashAttention.
- Add default-off sparse V dequant for decode-only TBQ4 V.
- No WMMA/default-wrapper/policy promotion changes.

## Code changes

- `ggml/src/ggml-cuda/fattn.cu`
  - allows `K=q8_0, V=tbq4_0` mixed pair;
  - adds `FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_TBQ4_0)` selector cases;
  - route labels include `q8k_tbq4v_vec` and `q8k_tbq4v_sparsev`.
- `ggml/src/ggml-cuda/fattn-vec.cuh`
  - adds explicit `Q8_0/TBQ4_0` VEC instantiations;
  - adds decode-only sparse V template variant for `type_V == GGML_TYPE_TBQ4_0` and `cols_per_block == 1`.
- `ggml/src/ggml-cuda/fattn-common.cuh`
  - adds default-off HIP env gate `GGML_CUDA_SPARSE_V_DEQUANT=1`;
  - adds tau selector `GGML_CUDA_SPARSE_V_TAU_LEVEL=0..5`.

## Verification

- Build passed: `cmake --build build-rocm --target llama-bench -j6`.
- Mixed route smoke passed after selector case fix:
  - `llama-bench -ctk q8_0 -ctv tbq4_0 -p 128 -n 1 -fa 1`, rc=0.
- Sparse V smoke passed:
  - `llama-bench -ctk q8_0 -ctv tbq4_0 -p 128 -n 16 -fa 1`, sparse off/on rc=0.
- KLD-reduction tau-level smoke passed:
  - `GGML_CUDA_SPARSE_V_DEQUANT=1 GGML_CUDA_SPARSE_V_TAU_LEVEL=2 llama-bench -ctk q8_0 -ctv tbq4_0 -p 128 -n 1 -fa 1`, rc=0.
- Aggressive tau-level smoke passed:
  - `GGML_CUDA_SPARSE_V_DEQUANT=1 GGML_CUDA_SPARSE_V_TAU_LEVEL=5 llama-bench -ctk q8_0 -ctv tbq4_0 -p 128 -n 1 -fa 1`, rc=0.

## Long-context decode ladder, 35B IQ4_XS, q8_0 K + tbq4_0 V

| ctx | prefill off | prefill sparse | delta | decode off | decode sparse | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 2141.54 | 2074.14 | -3.15% | 79.02 | 80.50 | +1.88% |
| 16384 | 1571.82 | 1574.91 | +0.20% | 78.99 | 80.46 | +1.86% |
| 32768 | 1031.37 | 1028.94 | -0.24% | 79.29 | 80.76 | +1.86% |

## Notes

- Sparse V threshold is ratio-based in the online-softmax numerator space: skip when `KQ_k < tau * KQ_sparse_sum`.
- Tau defaults to current behavior at `GGML_CUDA_SPARSE_V_TAU_LEVEL=0` (`tau=1e-6`); lower-KLD sweep levels are `1` (`3e-7`), `2` (`1e-7`), `3` (`3e-8`); aggressive speed-side levels are `4` (`1e-5`) and `5` (`1e-4`).
- Current first implementation adds per-tile denominator reduction before V accumulation; it is correct-oriented, not yet tuned.
- Initial PPL/NIAH canaries are clean; artifact `../q8k-tbq4v-sparsev-quality-sweep-20260516-052442/` adds ctx8192 PPL/KLD and 8K/16K/32K NIAH across tau0..5.

## Quality canaries

See `quality-canaries.md`.

- PPL quick canary (`llama-perplexity -c 512 --chunks 8 -b 1 -ub 1`): sparse off 6.3087 ±0.34412, sparse on 6.3316 ±0.34574 (+0.36%, inside CI).
- NIAH server canary: 8K ctx, depths 10/50/90, sparse off 3/3, sparse on 3/3.
- Route evidence: sparse-on server log contains `route=q8k_tbq4v_sparsev` for decode (`nq=1`).

Promotion remains blocked on higher-power PPL and longer NIAH; this checkpoint is canary-clean only.
