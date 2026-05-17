# PR #23198 MTP prefill optimization check

Test worktree: `/home/mrtrent/llama.cpp-mtp-pr23198-test`

Patch tested:
- Upstream PR #23198 / commit `3e12fbdea`: `llama: avoid copying logits during prompt decode in MTP`
- Cherry-picked onto the GitHub working branch after `cf7ccff23`; local landed commit is `899097b81` (`9136`).
- Built with ROCm/HIP (`GGML_HIP=ON`, `AMDGPU_TARGETS=gfx1100`, `GGML_HIP_ROCWMMA_FATTN=ON`).

Request shape for both models:
- `--ctx-size 8192`, 7281-token prompt, 128 generated tokens
- `--flash-attn on`, `K=q8_0`, `V=tbq4_0`, `--cache-ram 0`, `--no-warmup`
- 35B env: `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=48`
- 27B env: `RDNA2_MATMUL_OPT_V1=1`

## 35B MoE

Baseline daily prior artifact: `qwen35b-moe-mtp-8k-speed-20260517-233955`.

| config | prior prompt tok/s | PR#23198 prompt tok/s | prior gen tok/s | PR#23198 gen tok/s | PR#23198 accept |
|---|---:|---:|---:|---:|---:|
| no MTP | 2259.71 | 2240.20 | 76.43 | 75.11 | - |
| n3 | 1376.82 | 1927.00 | 102.90 | 101.67 | 81/135, 60.0% |
| n4 | 1358.95 | 1927.09 | 93.99 | 91.22 | 85/160, 53.1% |
| n5 | 1356.56 | 1929.44 | 87.62 | 82.99 | 86/195, 44.1% |
| n6 | 1353.36 | 1940.35 | 73.49 | 78.22 | 88/222, 39.6% |
| n7 | 1350.56 | 1928.12 | 76.10 | 73.58 | 89/258, 34.5% |

35B conclusion: PR #23198 fixes most of the MTP prefill penalty. n3 prompt speed improves from 1376.82 to 1927.00 tok/s (+40.0%), now ~14% below no-MTP instead of ~39% below. Generation is essentially unchanged/slightly lower; n3 remains best.

## 27B

Baseline daily prior artifact: `qwen27b-mtp-8k-speed-20260517-234727`.

| config | prior prompt tok/s | PR#23198 prompt tok/s | prior gen tok/s | PR#23198 gen tok/s | PR#23198 accept |
|---|---:|---:|---:|---:|---:|
| no MTP | 657.31 | 682.84 | 26.08 | 26.10 | - |
| n3 | 506.34 | 632.04 | 51.23 | 47.26 | 90/110, 81.8% |
| n4 | 513.29 | 630.33 | 47.01 | 47.04 | 96/121, 79.3% |
| n5 | 503.02 | 631.38 | 44.37 | 43.50 | 97/139, 69.8% |
| n6 | 508.38 | 628.83 | 47.93 | 44.05 | 101/149, 67.8% |
| n7 | 483.65 | 632.33 | 39.93 | 41.73 | 102/164, 62.2% |

27B conclusion: PR #23198 also fixes most of the MTP prefill penalty. n3 prompt speed improves from 506.34 to 632.04 tok/s (+24.8%), now ~7% below no-MTP instead of ~23% below. Generation did not improve in this one-pass prompt; n3/n4 are effectively tied here.

## Note

An earlier attempt accidentally configured the PR test worktree as CPU-only (`GGML_HIP=OFF`), producing CPU logs and no useful speed data. The results above are from the rebuilt ROCm/HIP test worktree.
