# Qwen3.6 35B-A3B IQ4_XS clean pp128/pp256/pp512 bench

Artifact: `/home/mrtrent/llama.cpp-mtp-tbq4-rdna3/benches/rocm-rdna3/qwen35b-pp128-256-512-20260516-005350`

Command shape: `llama-bench -p 128,256,512 -n 0 -fa 1 -ctk tbq4_0 -ctv tbq4_0 -b 1024 -ub 512 -r 5 -o jsonl`
Model: `/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`
Mode: non-MTP, ROCm/HIP, RX 7900 XTX.

| pp | baseline tok/s | RDNA2_MATMUL_OPT_V1=1 tok/s | delta |
|---:|---:|---:|---:|
| 128 | 1241.3 ± 40.1 | 1245.5 ± 51.2 | +0.3% |
| 256 | 1910.9 ± 39.8 | 1901.8 ± 33.3 | -0.5% |
| 512 | 2621.7 ± 14.0 | 2622.3 ± 25.3 | +0.0% |

Raw logs:
- `baseline.clean.jsonl.log`
- `rdna2_opt.clean.jsonl.log`
