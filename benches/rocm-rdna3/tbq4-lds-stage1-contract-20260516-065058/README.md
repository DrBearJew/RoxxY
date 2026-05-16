# TBQ4 FlashAttention calculator

Offline planning model; values are estimates from kernel constants, not benchmark results.

## Scope labels
| label | status | rule |
|---|---|---|
| `TBQ4_0` | current format | Production-compatible compressed KV cache format. Near-term optimizations must preserve this contract. |
| `tbq4_vec` | current default route | Baseline VEC FlashAttention route; do not replace without env-gated canaries + sweep evidence. |
| `tbq4_vec_norm_hoist` | proposed current-format optimization | Same TBQ4_0 data contract; hoist repeated norm loads before trying heavier LDS/WMMA paths. |
| `tbq4_lds_route_a/b/c/d` | proposed env-gated experiments | LDS materialization/staging experiments only; default-off until benchmark and coherence gates pass. |
| `future_format_v2_harness_only` | future-format research placeholder | Not TBQ4_0, not a runtime flag, and not compatible with current GGUF/cache rows. Exact future-format label stays in harness context. |

## Row storage
| D | blocks | TBQ4 raw row | TBQ4 padded16 | f16 row | q8_0 approx row | TBQ4 bpw | TBQ4/f16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 66 | 80 | 256 | 136 | 4.125 | 0.258 |
| 256 | 2 | 132 | 144 | 512 | 272 | 4.125 | 0.258 |

## Current VEC dequant load model
Current KQ TBQ4 VEC loads the block norm once per half2 pair; norm-hoist means one norm per 128-value TBQ4 block.
| D | K current B/row | K hoisted B/row | V current B/row | V hoisted B/row | K+V current | K+V hoisted | savings |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 192 | 66 | 128 | 66 | 320 | 132 | 58.8% |
| 256 | 384 | 132 | 256 | 132 | 640 | 264 | 58.8% |

## KQ norm-hoist ownership model
HIP/RDNA TBQ4 K uses Q in registers and currently maps `nthreads_KQ = 128 / cpy_nb`; defaults here are cpy_nb=16, nthreads_KQ=8.
Per-thread cache is safer but leaves one norm load per lane per TBQ4 block; subwarp broadcast targets one norm load per block for the active K row.
| D | nthreads_KQ | cpy_ne | blocks | current norm loads | per-thread norm loads | subwarp norm loads | current K B | per-thread K B | subwarp K B | ideal K B | per-thread savings | subwarp savings |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 4 | 1 | 64 | 8 | 1 | 192 | 80 | 66 | 66 | 58.3% | 65.6% |
| 256 | 8 | 4 | 2 | 128 | 16 | 2 | 384 | 160 | 132 | 132 | 58.3% | 65.6% |

## Rotation model
Future 4x32 transform estimates are format-incompatible with current TBQ4 GGUFs; exact future-format labels stay in harness context.
| D | FWHT128 add/sub | FWHT128 barriers | future32 add/sub | future32 barriers | add/sub savings |
|---:|---:|---:|---:|---:|---:|
| 128 | 896 | 3 | 640 | 0 | 28.6% |
| 256 | 1792 | 6 | 1280 | 0 | 28.6% |

## Materialized tile / WMMA expansion
Shows why direct TBQ4 WMMA must be carefully gated: compressed rows expand to f16 tiles in LDS.
| D | rows | raw staging | f16 tile | total LDS-ish | expansion vs raw | fits 64KiB |
|---:|---:|---:|---:|---:|---:|:---:|
| 128 | 16 | 1280 | 4096 | 5376 | 5.09x | yes |
| 128 | 32 | 2560 | 8192 | 10752 | 5.09x | yes |
| 128 | 64 | 5120 | 16384 | 21504 | 5.09x | yes |
| 128 | 128 | 10240 | 32768 | 43008 | 5.09x | yes |
| 128 | 256 | 20480 | 65536 | 86016 | 5.09x | NO |
| 256 | 16 | 2304 | 8192 | 10496 | 4.97x | yes |
| 256 | 32 | 4608 | 16384 | 20992 | 4.97x | yes |
| 256 | 64 | 9216 | 32768 | 41984 | 4.97x | yes |
| 256 | 128 | 18432 | 65536 | 83968 | 4.97x | NO |
| 256 | 256 | 36864 | 131072 | 167936 | 4.97x | NO |

## LDS route gates
Rows are max resident tile rows under practical 48KiB/56KiB budgets and the hard 64KiB limit. Scratch, padding, and VGPR pressure still need route-specific validation.
| route | D | bytes/row | rows @48KiB | rows @56KiB | rows @64KiB | preserves compressed KV | notes |
|---|---:|---:|---:|---:|---:|:---:|---|
| `tbq4_lds_route_a` | 128 | 160 | 307 | 358 | 409 | yes | packed K+V TBQ4 staging in LDS; dequant to registers, no f16 tile stored |
| `tbq4_lds_route_b` | 128 | 336 | 146 | 170 | 195 | yes | single f16 materialization tile reused K -> V; needs online-softmax state |
| `tbq4_lds_route_c` | 128 | 672 | 73 | 85 | 97 | yes | dual K+V f16 materialization; LDS-heavy hold/negative-control route |
| `tbq4_lds_route_d` | 128 | 416 | 118 | 137 | 157 | yes | ping-pong raw TBQ4 staging plus one f16 tile; barrier/scheduling risk |
| `f16_mma_fa` | 128 | 512 | 96 | 112 | 128 | no | f16 K+V comparison route; no compressed KV savings |
| `tbq4_lds_route_a` | 256 | 288 | 170 | 199 | 227 | yes | packed K+V TBQ4 staging in LDS; dequant to registers, no f16 tile stored |
| `tbq4_lds_route_b` | 256 | 656 | 74 | 87 | 99 | yes | single f16 materialization tile reused K -> V; needs online-softmax state |
| `tbq4_lds_route_c` | 256 | 1312 | 37 | 43 | 49 | yes | dual K+V f16 materialization; LDS-heavy hold/negative-control route |
| `tbq4_lds_route_d` | 256 | 800 | 61 | 71 | 81 | yes | ping-pong raw TBQ4 staging plus one f16 tile; barrier/scheduling risk |
| `f16_mma_fa` | 256 | 1024 | 48 | 56 | 64 | no | f16 K+V comparison route; no compressed KV savings |

## Stage 1 LDS experiment contracts
Concrete route contracts from `tbq4-lds-research-report.md`. These are planning rows only: no runtime behavior changes until Stage 2.
| route | role | env | D | tile rows | packed stages | packed stride | f16 stride half2 | LDS bytes/row | total LDS | fits 48/56/64KiB | sparse tau | fallback | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|---|---|
| `tbq4_lds_route_d_k_d128` | first_experiment | `GGML_CUDA_TBQ4_LDS_ROUTE=D_K` | 128 | 96 | 2 | 80 | 66 | 424 | 39.8 KiB | yes/yes/yes | tau0 `1e-6` | `tbq4_vec` | stage1_contract_locked |
| `tbq4_lds_route_b_k_d128` | backup_diagnostic | `GGML_CUDA_TBQ4_LDS_ROUTE=B_K` | 128 | 96 | 0 | 80 | 66 | 264 | 24.8 KiB | yes/yes/yes | tau0 `1e-6` | `tbq4_vec` | backup_contract_locked |
| `tbq4_lds_route_b_k_d128_rows128` | backup_diagnostic_larger_tile | `GGML_CUDA_TBQ4_LDS_ROUTE=B_K` | 128 | 128 | 0 | 80 | 66 | 264 | 33.0 KiB | yes/yes/yes | tau0 `1e-6` | `tbq4_vec` | backup_contract_locked |

Route log contracts:
- `route=tbq4_lds_route_d_k_d128 env=GGML_CUDA_TBQ4_LDS_ROUTE=D_K d=128 tile_rows=96 f16_stride=66 packed_stride=80 sparse_v_tau_level=0 fallback=tbq4_vec`
- `route=tbq4_lds_route_b_k_d128 env=GGML_CUDA_TBQ4_LDS_ROUTE=B_K d=128 tile_rows=96 f16_stride=66 packed_stride=direct_global sparse_v_tau_level=0 fallback=tbq4_vec`
- `route=tbq4_lds_route_b_k_d128_rows128 env=GGML_CUDA_TBQ4_LDS_ROUTE=B_K d=128 tile_rows=128 f16_stride=66 packed_stride=direct_global sparse_v_tau_level=0 fallback=tbq4_vec`

## FlashAttention route model
This includes the newer FA route families as calculator lanes, even when they stay env-gated.
| route | current TBQ4_0 compatible | cache savings | estimated target | calculator gate | status |
|---|:---:|:---:|---|---|---|
| VEC inline D128 | yes | yes | baseline; K+V logical load 320 B/pair | none | production default |
| VEC norm-hoist D128 | yes | yes | reduce repeated norm loads; model saves 58.8% of K+V TBQ4 dequant load bytes | no format change | best first optimization candidate |
| LDS D-lite K-only D128 | yes | yes | first Stage-1 contract: ping-pong packed TBQ4 K staging + padded f16 K tile; V unchanged | `GGML_CUDA_TBQ4_LDS_ROUTE=D_K`, tile_rows=96, f16_stride=66 | Stage 2 implementation target; default-off |
| LDS B-lite K-only D128 | yes | yes | backup diagnostic: direct TBQ4 K -> padded f16 K tile without ping-pong | `GGML_CUDA_TBQ4_LDS_ROUTE=B_K`, tile_rows=96/128, f16_stride=66 | backup only if D-lite needs diagnostic fallback |
| TBQ4 WMMA/materialized FA D128 | yes | yes | new FA path with f16 tile materialization | max materialized rows within 64KiB: 128 | env-gated; previous sweep slower |
| f16 MMA FA D128 | no | no | compare tensor-core FA ceiling with f16 KV | f16 row 256 B vs TBQ4 row 66 B | benchmark comparison only |
| future_format_v2_harness_only VEC D128 | no | yes | estimate future 4x32 transform upside, 28.6% fewer add/sub ops | new format + quality canaries | harness-only future format experiment |
| VEC inline D256 | yes | yes | baseline; K+V logical load 640 B/pair | none | production default |
| VEC norm-hoist D256 | yes | yes | reduce repeated norm loads; model saves 58.8% of K+V TBQ4 dequant load bytes | no format change | best first optimization candidate |
| TBQ4 WMMA/materialized FA D256 | yes | yes | new FA path with f16 tile materialization | max materialized rows within 64KiB: 64 | env-gated; previous sweep slower |
| f16 MMA FA D256 | no | no | compare tensor-core FA ceiling with f16 KV | f16 row 512 B vs TBQ4 row 132 B | benchmark comparison only |
| future_format_v2_harness_only VEC D256 | no | yes | estimate future 4x32 transform upside, 28.6% fewer add/sub ops | new format + quality canaries | harness-only future format experiment |

## Calculator decision gate
This table is the current routing gate. Lower priority means earlier work; `baseline` remains fallback.
| priority | route | D | decision | reason | next evidence |
|---:|---|---:|---|---|---|
| 0 | `tbq4_vec` | 128 | baseline | current production-compatible TBQ4_0 VEC route | keep as fallback for every experiment |
| 0 | `tbq4_vec` | 256 | baseline | current production-compatible TBQ4_0 VEC route | keep as fallback for every experiment |
| 1 | `tbq4_vec_norm_hoist` | 128 | next_candidate | same TBQ4_0 format; KQ-only model saves 58.3% with per-thread cache or 65.6% with subwarp broadcast before later V work (full K+V ideal 58.8%) | design/code review of norm load ownership, then correctness canaries before benchmarks |
| 1 | `tbq4_vec_norm_hoist` | 256 | next_candidate | same TBQ4_0 format; KQ-only model saves 58.3% with per-thread cache or 65.6% with subwarp broadcast before later V work (full K+V ideal 58.8%) | design/code review of norm load ownership, then correctness canaries before benchmarks |
| 2 | `tbq4_lds_route_d_k_d128` | 128 | stage1_first_experiment | research report recommends Route D-lite D128 K-only: ping-pong packed TBQ4 K staging plus one padded f16/half2 K tile, V unchanged | Stage 2 default-off implementation for GGML_CUDA_TBQ4_LDS_ROUTE=D_K with route log and fallback proof |
| 3 | `tbq4_lds_route_b_k_d128` | 128 | stage1_backup_diagnostic | backup B-lite D128 K-only isolates whether direct f16 K materialization helps without ping-pong complexity | implement only if D-lite fails from complexity or profiler needs simpler diagnostic comparison |
| 4 | `tbq4_lds_route_a` | 128 | model_before_code | packed TBQ4 LDS staging may preserve cache savings but register pressure is not modeled yet | add LDS/register gate rows for packed staging plus VGPR estimate |
| 4 | `tbq4_lds_route_a` | 256 | model_before_code | packed TBQ4 LDS staging may preserve cache savings but register pressure is not modeled yet | add LDS/register gate rows for packed staging plus VGPR estimate |
| 5 | `tbq4_lds_route_b` | 128 | model_before_code | single f16 materialization can fit up to 128 rows under 64KiB in this simple model, but scratch/padding/occupancy are not modeled yet | generic route only; Stage-1 concrete backup is tbq4_lds_route_b_k_d128 |
| 5 | `tbq4_lds_route_b` | 256 | model_before_code | single f16 materialization can fit up to 64 rows under 64KiB in this simple model, but scratch/padding/occupancy are not modeled yet | generic route only; Stage-1 concrete backup is tbq4_lds_route_b_k_d128 |
| 6 | `tbq4_lds_route_d` | 128 | model_before_code | raw ping-pong plus one f16 tile may overlap loads without doubling f16 LDS cost, but scheduling/barrier cost is unknown | generic route only; Stage-1 concrete first experiment is tbq4_lds_route_d_k_d128 |
| 6 | `tbq4_lds_route_d` | 256 | model_before_code | raw ping-pong plus one f16 tile may overlap loads without doubling f16 LDS cost, but scheduling/barrier cost is unknown | generic route only; Stage-1 concrete first experiment is tbq4_lds_route_d_k_d128 |
| 8 | `f16_mma_fa` | 128 | comparison_only | useful tensor-core FA ceiling comparison, but loses compressed KV savings | benchmark only as reference; do not promote as compressed-KV optimization |
| 8 | `f16_mma_fa` | 256 | comparison_only | useful tensor-core FA ceiling comparison, but loses compressed KV savings | benchmark only as reference; do not promote as compressed-KV optimization |
| 9 | `tbq4_lds_route_c` | 128 | hold_negative_control | dual K+V f16 materialization is LDS-heavy and previous TBQ4 WMMA/materialized sweep was slower | only revisit after Route A/B/D evidence or as a bounded negative-control probe |
| 9 | `tbq4_lds_route_c` | 256 | hold_negative_control | dual K+V f16 materialization is LDS-heavy and previous TBQ4 WMMA/materialized sweep was slower | only revisit after Route A/B/D evidence or as a bounded negative-control probe |
| 99 | `future_format_v2_harness_only` | 128 | future_format_only | future-format-only and incompatible with current GGUF/cache rows | separate format contract, quantizer/converter, and quality canaries |
| 99 | `future_format_v2_harness_only` | 256 | future_format_only | future-format-only and incompatible with current GGUF/cache rows | separate format contract, quantizer/converter, and quality canaries |

## Request-scale estimate
Using D=128, nq=512, kv=4096, q_heads=1, kv_heads=1, V_TBQ4=True.
| metric | value |
|---|---:|
| nq | 512 |
| kv | 4096 |
| q_heads | 1 |
| kv_heads | 1 |
| gqa_ratio | 1.000 |
| qk_pairs | 2097152 |
| cache_tbq4_bytes | 528.0 KiB |
| cache_f16_bytes | 2.00 MiB |
| vec_current_logical_kv_bytes | 640.00 MiB |
| vec_norm_hoist_logical_kv_bytes | 264.00 MiB |
| vec_norm_hoist_savings_bytes | 376.00 MiB |
| vec_norm_hoist_savings_percent | 58.750 |
| q_rotate_rows | 512 |
| out_rotate_rows | 512 |
| fwht128_barriers_total_if_kv_tbq4 | 3072 |
| future32_barriers_total_if_kv_tbq4 | 0 |

## Stage execution hypotheses
1. **Stage 1 contract gate**: calculator rows lock `tbq4_lds_route_d_k_d128` as first experiment and `tbq4_lds_route_b_k_d128` as backup; no runtime behavior changes.
2. **Stage 2 D-lite implementation**: add `GGML_CUDA_TBQ4_LDS_ROUTE=D_K` for D128 K-only with fallback to `tbq4_vec` for env-absent/unsupported shapes.
3. **Stage 3 canaries before benchmarks**: route log, tiny decode, OOB/tile-tail, GQA/mask/KV_max, sparse-V tau0 unchanged, NIAH/coherence.
4. **Stage 4 profiler-driven iteration**: compare against `tbq4_vec`, `q8k_tbq4v_sparsev` tau0, and opt-in norm-hoist; sweep tile rows and f16 stride.
5. **Stage 5 expansion only after D128 success**: D256 then optional V integration with sparse-V tau0 pre-dequant skip; Route C remains negative-control only.
