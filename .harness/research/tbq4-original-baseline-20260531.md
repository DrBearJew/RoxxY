# TBQ4 original baseline numbers — 2026-05-31

User-provided original performance table for qwen3.6-27b with TBQ4 model.

| model | test | t/s | peak t/s | ttfr (ms) | est_ppt (ms) | e2e_ttft (ms) |
|:--|--:|--:|--:|--:|--:|--:|
| qwen3.6-27b | pp2048 | 732.33 ± 15.96 | | 2676.06 ± 46.22 | 2573.61 ± 46.22 | 2676.06 ± 46.22 |
| qwen3.6-27b | tg32 | 60.26 ± 2.94 | 62.07 ± 3.03 | | | |
| qwen3.6-27b | pp2048 @ d4096 | 726.03 ± 1.27 | | 7739.74 ± 148.82 | 7637.29 ± 148.82 | 7739.74 ± 148.82 |
| qwen3.6-27b | tg32 @ d4096 | 59.29 ± 2.27 | 61.07 ± 2.34 | | | |
| qwen3.6-27b | pp2048 @ d8192 | 700.15 ± 4.25 | | 13472.83 ± 68.84 | 13370.38 ± 68.84 | 13472.83 ± 68.84 |
| qwen3.6-27b | tg32 @ d8192 | 58.97 ± 2.54 | 60.78 ± 2.57 | | | |
| qwen3.6-27b | pp2048 @ d16384 | 661.88 ± 0.78 | | 25194.42 ± 248.86 | 25091.97 ± 248.86 | 25194.42 ± 248.86 |
| qwen3.6-27b | tg32 @ d16384 | 51.63 ± 4.06 | 53.11 ± 4.19 | | | |
| qwen3.6-27b | pp2048 @ d32768 | 589.57 ± 1.27 | | 53772.35 ± 291.04 | 53669.90 ± 291.04 | 53772.35 ± 291.04 |
| qwen3.6-27b | tg32 @ d32768 | 51.52 ± 2.45 | 53.02 ± 2.53 | | | |

## Derived reference bands

- Decode target at short context: ~60 tok/s tg32, peak ~62 tok/s.
- Decode target through d8192: ~59 tok/s, peak ~61 tok/s.
- Decode target at d16384/d32768: ~51.5 tok/s, peak ~53 tok/s.
- Prefill pp2048 target: ~732 tok/s at short context, declining to ~590 tok/s by d32768.

Use this as the baseline for post-v3 comparisons; do not compare production v3 against the older b10d538b3 local no-spec ~32 tok/s traces except as historical evidence from a different checkout/path.
