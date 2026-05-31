# PACKED16 PWMMA x4 / BM64 Feasibility Analysis

## Context

- **Part 5 result**: DOT4-MMQ is the prefill performance leader, beating PWMMA by 9.6–34% across pp512–pp4096.
- **Part 3 result**: BM32 was correct but only +0.6% at pp512, +1.7% at pp1024 — not enough to auto-route.
- **Part 4 result**: GQA2 correct but slightly behind at pp512 due to LDS pressure.
- **GPU**: RDNA3 gfx1100, 64 KiB LDS per CU, 96 CUs, 256 VGPRs per thread (waves of 32).
- **Target model**: Qwen3.6-35B-A3B MoE, heads_q=16, heads_kv=2, gqa_ratio=8, D=256.

## Resource Math

Baseline dimensions: D=256, BN=16, BM=64 for x4 designs.

### Design A — Naive BM64_X4

```
Q tile:       half[64][256]    = 32 KiB
V tile:       half[16][256]    = 8 KiB
logits:       float[64][16]    = 4 KiB
probs:        float[64][16]    = 4 KiB
out_smem:     float[64][256]   = 64 KiB
row state:    float[64] × 3    = < 1 KiB
Total:        ≈ 113 KiB
```

**Verdict: REJECT.** Exceeds 64 KiB LDS limit. Even with no-Q-LDS (-32 KiB), out_smem alone is 64 KiB.
No room for V tile, logits, or row state.

### Design B — BM64_X4 no-Q-LDS

```
Q tile:       0 (reload from global, +bandwidth cost)
V tile:       8 KiB
logits/probs: 8 KiB
out_smem:     64 KiB
row state:    < 1 KiB
Total:        ≈ 81 KiB
```

**Verdict: REJECT.** Still exceeds 64 KiB. out_smem[64][256] = 64 KiB alone uses the entire LDS budget.

### Design C — BM64_X4 half-out + no-Q-LDS

```
Q tile:       0
V tile:       8 KiB
logits/probs: 8 KiB
out_smem:     half[64][256] = 32 KiB
row state:    < 1 KiB
Total:        ≈ 49 KiB
```

**Verdict: CONDITIONAL.** Fits at ~49 KiB. Requires:
1. BM16 half-out proven correct first (not yet tested)
2. no-Q-LDS (Q reloaded from global for every K tile — 2× bandwidth cost at pp2048)
3. FP32→FP16 accumulation precision loss — unknown impact on softmax stability

### Design D — D-split (D_CHUNK=64) + no-Q-LDS

```
Q tile:       0
V tile:       8 KiB
logits/probs: 8 KiB
out_smem:     float[64][64] = 16 KiB
row state:    < 1 KiB
Total:        ≈ 33 KiB
```

**Verdict: REJECT.** Fits at ~33 KiB but QK recomputed 4× (D=256 split into 4 chunks).
At pp2048 with causal skip, this multiplies QK cost by 4× for the fill region,
likely making it slower than BM16. Does not justify the complexity.

### Design E — global scratch

```
Q tile:       32 KiB
V tile:       8 KiB
logits/probs: 8 KiB
out_smem:     0 (global scratch)
Total:        ≈ 49 KiB
```

**Verdict: REJECT for performance.** Each K tile reads/writes out state from global memory.
At pp2048 with 128 K tiles × 64 rows × 256 D × 4 bytes = 8 MiB extra traffic per FA op.
Would be slower than BM16. Correctness prototype only — not competitive.

### Design F — GQA4 naive

```
q_tile:       half[4][16][256] = 32 KiB
V tile:        8 KiB
logits/probs: float[4][16][16] = 4 + 4 = 8 KiB
out_smem:     float[4][16][256] = 64 KiB
row state:    float[4][16] × 3  = < 1 KiB
Total:        ≈ 113 KiB
```

**Verdict: REJECT.** Same LDS problem as BM64. out_smem dominates.

### Design G — GQA4 half-out no-Q-LDS

```
q_tile:       0
V tile:       8 KiB
logits/probs: 8 KiB
out_smem:     half[4][16][256] = 32 KiB
row state:    < 1 KiB
Total:        ≈ 49 KiB
```

**Verdict: CONDITIONAL, same as Design C.** GQA4 is more structurally useful than BM64
because the model has gqa_ratio=8 and GQA2 already proved K/V reuse works.
But requires half-out proven first.

## Practical Comparison

DOT4-MMQ is already the winner. The gap is large and grows with context:

| pp | DOT4-MMQ | PWMMA BM16 | Δ |
|---|---|---|---|
| 512 | 2774 | 2530 | +9.6% |
| 1024 | 2656 | 2346 | +13.2% |
| 2048 | 2479 | 2052 | +20.8% |
| 4096 | 2195 | 1638 | +34.0% |

Even if BM64_X4 were possible and achieved a 2.5% speedup over BM32 (optimistic,
given LDS pressure and Q-reload costs), it would still trail DOT4-MMQ by ~31% at pp4096.

BM64 and GQA4 address PWMMA's internal efficiency, but PWMMA is already the
losing backend. Optimizing it further while DOT4-MMQ runs +20-34% faster is
a diminishing-returns investment.

## Recommendation

**Option B: Skip x4 PWMMA, do DOT4-MMQ GQA reuse instead.**

Rationale:
1. DOT4-MMQ is the performance leader at every measured context length.
2. BM64/GQA4 resource math shows no design fits cleanly without precision tradeoffs (half-out) or bandwidth regressions (no-Q-LDS, D-split, global scratch).
3. GQA2 already proved that K/V reuse works structurally. Porting GQA reuse to DOT4-MMQ is the higher-return investment.
4. Even if BM64 were implemented perfectly, it would not catch DOT4-MMQ.

**Part 7 direction**: DOT4-MMQ GQA2/GQA4 K/V reuse, or auto-routing policy that defaults to DOT4-MMQ when packed16 K cache is active.
