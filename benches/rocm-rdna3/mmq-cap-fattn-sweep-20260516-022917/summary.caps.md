# RDNA3 MMQ cap / FlashAttention sweep

Artifact: `/home/mrtrent/llama.cpp-mtp-tbq4-rdna3/benches/rocm-rdna3/mmq-cap-fattn-sweep-20260516-022917`

Base command shape: `llama-bench -p 128,256,512,1024,2048,4096 -n 0 -fa 1 -b ${BATCH:-1024} -ub ${UBATCH:-512} -r ${REPS:-3} -o jsonl`

| variant | route | cache K/V | env | pp128 tok/s | pp256 tok/s | pp512 tok/s | pp1024 tok/s | pp2048 tok/s | pp4096 tok/s | rc |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | `tbq4_vec` | `tbq4_0/tbq4_0` | `none` | 1270.2 ± 56.7 | 1939.7 ± 74.5 | 2640.8 ± 53.4 | 2557.0 ± 13.8 | 2383.1 ± 4.3 | 2138.7 ± 13.3 | 0 |
| `rdna2_opt` | `tbq4_vec` | `tbq4_0/tbq4_0` | `RDNA2_MATMUL_OPT_V1=1` | 1262.3 ± 56.9 | 1924.0 ± 74.6 | 2627.4 ± 18.3 | 2533.7 ± 20.4 | 2376.0 ± 12.1 | 2131.1 ± 11.8 | 0 |
| `maxx32` | `tbq4_vec` | `tbq4_0/tbq4_0` | `GGML_CUDA_MMQ_MAX_X=32 RDNA2_MATMUL_OPT_V1=1` | 1650.7 ± 43.6 | 2242.0 ± 50.2 | 2683.1 ± 27.6 | 2589.3 ± 19.0 | 2423.6 ± 4.8 | 2176.2 ± 4.2 | 0 |
| `maxx48` | `tbq4_vec` | `tbq4_0/tbq4_0` | `GGML_CUDA_MMQ_MAX_X=48 RDNA2_MATMUL_OPT_V1=1` | 1767.0 ± 86.1 | 2484.8 ± 62.0 | 3126.1 ± 47.5 | 3023.9 ± 2.7 | 2802.6 ± 7.5 | 2467.1 ± 8.4 | 0 |
| `maxx64` | `tbq4_vec` | `tbq4_0/tbq4_0` | `GGML_CUDA_MMQ_MAX_X=64 RDNA2_MATMUL_OPT_V1=1` | 1666.4 ± 95.6 | 2415.7 ± 72.9 | 3100.4 ± 45.5 | 2965.7 ± 8.6 | 2749.5 ± 4.6 | 2421.1 ± 17.9 | 0 |
| `maxx128` | `tbq4_vec` | `tbq4_0/tbq4_0` | `GGML_CUDA_MMQ_MAX_X=128 RDNA2_MATMUL_OPT_V1=1` | 1183.5 ± 122.4 | 1911.7 ± 48.4 | 2607.1 ± 33.6 | 2512.9 ± 20.9 | 2352.1 ± 18.9 | 2116.3 ± 17.0 | 0 |
| `maxx48_f16_mma` | `f16_mma` | `f16/f16` | `GGML_CUDA_MMQ_MAX_X=48 RDNA2_MATMUL_OPT_V1=1` | 1739.1 ± 60.3 | 2410.5 ± 70.8 | 2989.6 ± 3.5 | 2802.1 ± 19.1 | 2499.5 ± 10.5 | 2035.7 ± 5.0 | 0 |
| `maxx64_f16_mma` | `f16_mma` | `f16/f16` | `GGML_CUDA_MMQ_MAX_X=64 RDNA2_MATMUL_OPT_V1=1` | 1638.4 ± 60.4 | 2318.2 ± 37.5 | 2932.7 ± 44.0 | 2760.3 ± 19.2 | 2457.3 ± 10.7 | 2003.5 ± 6.7 | 0 |
| `maxx48_tbq4_wmma` | `tbq4_wmma` | `tbq4_0/tbq4_0` | `GGML_CUDA_MMQ_MAX_X=48 RDNA2_MATMUL_OPT_V1=1 TBQ4_WMMA_FATTN=1` | 1383.8 ± 52.7 | 1963.6 ± 14.2 | 1879.2 ± 2.5 | 1514.9 ± 4.8 | 1083.8 ± 2.6 | 689.5 ± 1.3 | 0 |
| `maxx64_tbq4_wmma` | `tbq4_wmma` | `tbq4_0/tbq4_0` | `GGML_CUDA_MMQ_MAX_X=64 RDNA2_MATMUL_OPT_V1=1 TBQ4_WMMA_FATTN=1` | 1326.0 ± 42.0 | 1921.0 ± 29.6 | 1852.6 ± 9.9 | 1504.6 ± 2.5 | 1076.3 ± 1.0 | 691.2 ± 4.5 | 0 |
