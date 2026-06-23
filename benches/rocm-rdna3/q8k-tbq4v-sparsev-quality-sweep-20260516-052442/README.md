# q8_0 K + TBQ4 V Sparse V tau quality sweep

## Scope

- Add aggressive sparse-V tau levels while preserving existing levels/defaults:
  - `0 = 1e-6` default/current behavior
  - `1 = 3e-7`
  - `2 = 1e-7`
  - `3 = 3e-8`
  - `4 = 1e-5`
  - `5 = 1e-4`
- Compare sparse-off baseline against sparse-on tau0..5.
- Keep sparse V default-off behind `GGML_CUDA_SPARSE_V_DEQUANT=1`.

## Verification

- Build passed: `cmake --build build-rocm --target llama-bench llama-perplexity llama-server -j6`.
- `git diff --check` passed for the tau code edits.
- Tau5 smoke passed: `GGML_CUDA_SPARSE_V_DEQUANT=1 GGML_CUDA_SPARSE_V_TAU_LEVEL=5 llama-bench -ctk q8_0 -ctv tbq4_0 -p 128 -n 1 -fa 1`, rc=0.

## PPL/KLD sweep

See `ppl-kld-summary.md` and `ppl-kld-summary.json`.

- Model: Qwen3.6-35B-A3B IQ4_XS.
- Cache: `q8_0` K + `tbq4_0` V, FA on.
- Dataset: `wikitext-2-raw/wiki.test.raw`.
- Command shape: `llama-perplexity -c 8192 --chunks 2 -b 1 -ub 1 --save-all-logits/--kl-divergence`.
- `-b 1 -ub 1` intentionally forces decode-style VEC FA so sparse V is exercised.

| variant | tau | PPL(Q) | PPL(base) | PPL ratio | KLD | max KLD | RMS Δp % | same top % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse off | off | 4.7661 ± 0.11684 | n/a | n/a | n/a | n/a | n/a | n/a |
| tau0 | 1e-6 | 4.76817 | 4.76609 | 1.00044 | 0.004364 | 0.366522 | 1.990 | 97.155 |
| tau1 | 3e-7 | 4.77136 | 4.76609 | 1.00111 | 0.004370 | 0.250977 | 2.044 | 97.546 |
| tau2 | 1e-7 | 4.77090 | 4.76609 | 1.00101 | 0.004505 | 0.817499 | 2.001 | 97.241 |
| tau3 | 3e-8 | 4.77480 | 4.76609 | 1.00183 | 0.004279 | 0.363441 | 2.003 | 97.326 |
| tau4 | 1e-5 | 4.77550 | 4.76609 | 1.00198 | 0.004370 | 0.536875 | 1.939 | 97.131 |
| tau5 | 1e-4 | 4.77119 | 4.76609 | 1.00107 | 0.004417 | 0.518501 | 1.988 | 97.228 |

## NIAH sweep

See `niah/summary.md` and `niah/summary.json`.

- Server ctx: 32768.
- Context targets: 8192, 16384, 32768.
- Depths: 10, 50, 90.
- Variants: sparse off + tau0..5.

Result: all variants passed `9/9`; aggregate `63/63`.

Route evidence from server logs:

- sparse off: `q8k_tbq4v_vec` only.
- sparse on: `q8k_tbq4v_sparsev` present for decode and `q8k_tbq4v_vec` present for prefill.
- tau logs include the selected levels for tau1..5.

## Decision

- New tau levels compile and pass direct quality canaries.
- `1e-5` and `1e-4` did not show a quality cliff in this sweep.
- KLD/PPL differences are small and not clearly monotonic across tau levels; repeat/perf sweeps are still needed before choosing a promotion policy.
- Keep sparse V default-off.
