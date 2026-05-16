# Sparse V tau sweep vs baseline

Artifact: `benches/rocm-rdna3/q8k-tbq4v-vec-sparsev-20260516-041159/tau-sweep-20260516-050459`

Model: Qwen3.6-35B-A3B IQ4_XS, `-ctk q8_0 -ctv tbq4_0 -fa 1`.
Baseline: sparse V off. Tau levels: 0=1e-6, 1=3e-7, 2=1e-7, 3=3e-8.

| variant | test | tok/s | delta vs baseline |
|---|---:|---:|---:|
| baseline-off | pp8192 | 2143.70 | +0.00% |
| baseline-off | pp16384 | 1585.52 | +0.00% |
| baseline-off | pp32768 | 1024.51 | +0.00% |
| baseline-off | tg64 | 81.14 | +0.00% |
| tau0 | pp8192 | 2119.63 | -1.12% |
| tau0 | pp16384 | 1583.93 | -0.10% |
| tau0 | pp32768 | 1030.37 | +0.57% |
| tau0 | tg64 | 81.01 | -0.16% |
| tau1 | pp8192 | 2108.74 | -1.63% |
| tau1 | pp16384 | 1560.83 | -1.56% |
| tau1 | pp32768 | 1043.92 | +1.89% |
| tau1 | tg64 | 80.94 | -0.24% |
| tau2 | pp8192 | 2154.49 | +0.50% |
| tau2 | pp16384 | 1598.85 | +0.84% |
| tau2 | pp32768 | 1029.87 | +0.52% |
| tau2 | tg64 | 80.55 | -0.72% |
| tau3 | pp8192 | 2102.14 | -1.94% |
| tau3 | pp16384 | 1560.45 | -1.58% |
| tau3 | pp32768 | 1031.25 | +0.66% |
| tau3 | tg64 | 82.83 | +2.09% |
