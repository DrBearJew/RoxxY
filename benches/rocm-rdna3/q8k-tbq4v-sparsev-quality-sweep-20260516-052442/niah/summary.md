# Sparse V NIAH sweep

- server ctx: `32768`
- contexts: `[8192, 16384, 32768]`
- depths: `[10, 50, 90]`

| variant | tau | pass | routes | tau logs |
|---|---:|---:|---|---|
| sparse_off | off | 9/9 | q8k_tbq4v_vec:2280 | [0] |
| sparse_tau0_1e-6 | 1e-6 | 9/9 | q8k_tbq4v_sparsev:180, q8k_tbq4v_vec:2100 | [0] |
| sparse_tau1_3e-7 | 3e-7 | 9/9 | q8k_tbq4v_sparsev:180, q8k_tbq4v_vec:2100 | [0, 1] |
| sparse_tau2_1e-7 | 1e-7 | 9/9 | q8k_tbq4v_sparsev:180, q8k_tbq4v_vec:2100 | [0, 2] |
| sparse_tau3_3e-8 | 3e-8 | 9/9 | q8k_tbq4v_sparsev:180, q8k_tbq4v_vec:2100 | [0, 3] |
| sparse_tau4_1e-5 | 1e-5 | 9/9 | q8k_tbq4v_sparsev:180, q8k_tbq4v_vec:2100 | [0, 4] |
| sparse_tau5_1e-4 | 1e-4 | 9/9 | q8k_tbq4v_sparsev:180, q8k_tbq4v_vec:2100 | [0, 5] |
