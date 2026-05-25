# 35B MoE ROCm temp0.6 production matrix

Artifact: `/home/mrtrent/llama.cpp-mtp-tbq4-rdna3-daily/benches/rocm-rdna3/mmq-35b-moe-prod-temp06-matrix-20260525-083402`

| case | prompt toks | pred toks | prompt tok/s | gen tok/s | draft accept | peak VRAM GiB | route logs | warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mtp_n2` | 16093 | 4096 | 2777.68 | 75.68 | 2595/3061 | 21.24 | mmq 8, f16 9069 | 0 |
| `mtp_n3` | 16093 | 4096 | 2677.28 | 98.80 | 3192/4044 | 21.23 | mmq 8, f16 10502 | 0 |
| `no_mtp` | 16093 | 4096 | 3157.29 | 71.47 |  | 19.95 | mmq 3, f16 170 | 0 |
