# Qwen3.6 27B MTP daily 8K speed sweep

> Superseded speed note: PR #23198 (`899097b81`, avoid copying logits during MTP prompt decode) updates the MTP prefill result for this same 8K shape. The old n3 prompt speed below was `506.34 tok/s`; the updated n3 prompt speed is `632.04 tok/s` with generation `47.26 tok/s`. See `../pr23198-mtp-prefill-check-20260518-001214/summary.md`.

- Build: daily `cf7ccff23` / 9135
- Model: `/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf`
- Env: `RDNA2_MATMUL_OPT_V1=1`
- Server ctx: `--ctx-size 8192`, `--flash-attn on`, `K=q8_0`, `V=tbq4_0`, `--cache-ram 0`, `--no-warmup`
- Request: 7281-token prompt, 128 generated tokens, `temperature=0`, no prompt cache

| config | prompt tok/s | generation tok/s | gen ms | draft accepted | accept rate | vs no-MTP |
|---|---:|---:|---:|---:|---:|---:|
| n0 | 657.31 | 26.08 | 4907.9 | 0/0 | 0.0% | +0.0% |
| n3 | 506.34 | 51.23 | 2498.7 | 92/102 | 90.2% | +96.4% |
| n4 | 513.29 | 47.01 | 2722.9 | 95/122 | 77.9% | +80.3% |
| n5 | 503.02 | 44.37 | 2885.1 | 97/142 | 68.3% | +70.1% |
| n6 | 508.38 | 47.93 | 2670.4 | 103/137 | 75.2% | +83.8% |
| n7 | 483.65 | 39.93 | 3205.4 | 100/172 | 58.1% | +53.1% |

Best generation result in this one-pass sweep: `n3` at 51.23 tok/s, +96.4% vs no-MTP baseline. Prefill drops from 657.31 tok/s to about 483-513 tok/s with MTP enabled.
