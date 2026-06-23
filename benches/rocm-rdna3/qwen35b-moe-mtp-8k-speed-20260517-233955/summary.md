# Qwen3.6 35B MoE daily MTP 8K speed sweep

> Superseded speed note: PR #23198 (`899097b81`, avoid copying logits during MTP prompt decode) updates the MTP prefill result for this same 8K shape. The old n3 prompt speed below was `1376.82 tok/s`; the updated n3 prompt speed is `1927.00 tok/s` with generation `101.67 tok/s`. See `../pr23198-mtp-prefill-check-20260518-001214/summary.md`.

- Build: daily `cf7ccff23` / 9135
- Model: `/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`
- Env: `RDNA2_MATMUL_OPT_V1=1 GGML_CUDA_MMQ_MAX_X=48`
- Server ctx: `--ctx-size 8192`, `--flash-attn on`, `K=q8_0`, `V=tbq4_0`, `--cache-ram 0`, `--no-warmup`
- Request: 7281-token prompt, 128 generated tokens, `temperature=0`, no prompt cache

| config | prompt tok/s | generation tok/s | gen ms | draft accepted | accept rate | vs no-MTP |
|---|---:|---:|---:|---:|---:|---:|
| n0 | 2259.71 | 76.43 | 1674.7 | 0/0 | 0.0% | +0.0% |
| n3 | 1376.82 | 102.90 | 1244.0 | 81/133 | 60.9% | +34.6% |
| n4 | 1358.95 | 93.99 | 1361.8 | 86/160 | 53.8% | +23.0% |
| n5 | 1356.56 | 87.62 | 1460.8 | 88/190 | 46.3% | +14.6% |
| n6 | 1353.36 | 73.49 | 1741.8 | 84/246 | 34.1% | -3.8% |
| n7 | 1350.56 | 76.10 | 1682.1 | 89/253 | 35.2% | -0.4% |

Best generation result in this one-pass sweep: `n3` at 102.90 tok/s, +34.6% vs no-MTP baseline.
