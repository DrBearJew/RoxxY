# Sparse V quality canaries

Config: 35B IQ4_XS, `-ctk q8_0 -ctv tbq4_0 -fa on`, ROCm/RDNA3 VEC.

## PPL canary

Command shape: `llama-perplexity -c 512 --chunks 8 -b 1 -ub 1 --no-warmup` on `wikitext-2-raw/wiki.test.raw`.

| sparse V | PPL | CI | delta vs off |
|---|---:|---:|---:|
| off | 6.3087 | ±0.34412 | baseline |
| on | 6.3316 | ±0.34574 | +0.36% |

Result: pass as a quick canary; sparse-on delta is inside the reported uncertainty band. This is not a full statistical PPL campaign.

## NIAH canary

Server canary: 8K ctx, generated ~6.2K-token prompt, depths 10/50/90, exact-code retrieval.

| variant | pass | depths | route evidence |
|---|---:|---|---|
| `q8k_tbq4v_sparse_off` | 3/3 | 10,50,90 | `q8k_tbq4v_vec` in server log |
| `q8k_tbq4v_sparse_on` | 3/3 | 10,50,90 | `q8k_tbq4v_sparsev` in server log |

Result: pass; sparse off and sparse on both retrieved all needles.

## Promotion note

Quality smoke is clean enough to continue evaluation, but not enough for default promotion. Before promotion, repeat with longer NIAH contexts/depths and a higher-power PPL run.
