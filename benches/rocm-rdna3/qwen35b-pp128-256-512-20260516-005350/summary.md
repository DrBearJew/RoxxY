# Qwen3.6 35B-A3B IQ4_XS pp128/pp256/pp512 bench

Artifact: `/home/mrtrent/llama.cpp-mtp-tbq4-rdna3/benches/rocm-rdna3/qwen35b-pp128-256-512-20260516-005350`

Command shape: `llama-bench -pg 128,0 -pg 256,0 -pg 512,0 -fa 1 -ctk tbq4_0 -ctv tbq4_0 -b 1024 -ub 512 -r 5 -o jsonl`
Model: `/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`
Mode: non-MTP, ROCm/HIP, RX 7900 XTX.

| pp | baseline tok/s | RDNA2_MATMUL_OPT_V1=1 tok/s | delta |
|---:|---:|---:|---:|
| 128 | 1186.1 ± 39.9 | 1239.7 ± 24.2 | +4.5% |
| 256 | 1765.9 ± 37.2 | 1901.9 ± 43.0 | +7.7% |
| 512 | 2307.0 ± 32.9 | 2649.7 ± 18.6 | +14.9% |

Raw logs:
- `baseline.jsonl.log`
- `rdna2_opt.jsonl.log`
