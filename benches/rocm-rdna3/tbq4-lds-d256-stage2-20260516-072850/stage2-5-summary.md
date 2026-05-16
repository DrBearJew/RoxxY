# TBQ4 LDS D256 Stage 2-5 summary

Model: `/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`.

## Stage 2 implementation

- Added default-off `GGML_CUDA_TBQ4_LDS_ROUTE=D_K`.
- D256-first route: `tbq4_lds_route_d_k_d256`.
- Contract: D=256, tile_rows=48, packed_stride=144 B, f16_stride=130 half2, total LDS=38,784 B, fallback=`tbq4_vec`.
- D128 remains secondary/dev-shape only.

## Stage 3 canaries

- Build targets passed: `llama-bench`, `llama-server`, `llama-perplexity`.
- Qwen D256 fallback route log: `route=tbq4_vec`.
- Qwen D256 D_K route log: `route=tbq4_lds_route_d_k_d256`.
- Qwen D256 D_K sparse-V tau0 route log: `route=tbq4_lds_route_d_k_d256_sparsev`.
- Sparse-V non-tau0 guard check: tau5 falls back to `route=tbq4_vec_sparsev`, not D_K.
- No fatal/error/OOM lines in recorded canary logs.
- Scope note: tests used `ngl=4` because the live full 35B server was already occupying VRAM; this exercises Qwen D256 GPU FA layers but is not a full production-context NIAH/coherence sweep.

## Stage 4 quick performance/profile

| case | route | avg tg tok/s | stddev | route logs |
|---|---|---:|---:|---:|
| fallback | `tbq4_vec` | 12.4873 | 0.8092 | 9 |
| d_k | `tbq4_lds_route_d_k_d256` | 12.4453 | 0.6258 | 9 |
| d_k_sparsev_tau0 | `tbq4_lds_route_d_k_d256_sparsev` | 12.6695 | 0.7717 | 9 |

- Quick ngl=4 microbench result: D_K is parity/slightly below fallback; D_K+sparseV tau0 is slightly above in this low-power test.
- rocprofv3 kernel trace summary: `benches/rocm-rdna3/tbq4-lds-d256-stage2-20260516-072850/rocprof-dk-summary.txt`.
- D_K kernel observed: `flash_attn_ext_vec<256, ..., tbq4_lds_d_k=true>` with 4 calls, ~1.11 ms total in the profile run.
- rocprofv3 DB kept local only (not intended for git).

## Stage 5 decision

- Keep D_K default-off/env-gated.
- Do not promote: no full-context NIAH/coherence sweep and no profiler-backed speedup yet.
- Next expansion should be D256 full-context canaries after freeing VRAM or stopping/restarting the live server deliberately; V integration remains later and must preserve sparse-V tau0 pre-dequant skip.
