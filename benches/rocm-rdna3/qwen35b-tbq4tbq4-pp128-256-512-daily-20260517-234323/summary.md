# Daily Qwen3.6 35B MoE tbq4/tbq4 pp128/256/512 check

Command shape:

```bash
RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=48 \
/home/mrtrent/.local/bin/llama-bench-daily \
  -m /mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  -ngl 999 -p 128,256,512 -n 0 -fa 1 \
  -ctk tbq4_0 -ctv tbq4_0 -b 1024 -ub 512 -r 5
```

Build: `cf7ccff23 (9135)` daily, ROCm, RX 7900 XTX.

| test | daily result | prior maxx48 artifact |
|---|---:|---:|
| pp128 | 1781.82 ± 42.97 tok/s | 1780.7 ± 44.5 tok/s |
| pp256 | 2490.73 ± 34.66 tok/s | 2479.6 ± 25.0 tok/s |
| pp512 | 3157.42 ± 20.45 tok/s | 3150.0 ± 40.5 tok/s |

Conclusion: daily matches/slightly exceeds the prior `maxx48` working result from `qwen35b-pp128-256-512-20260516-005350/summary.variants.clean.md`.
