# RDNA3 MMQ cap sweep

Artifact: `/home/mrtrent/llama.cpp-mtp-tbq4-rdna3/benches/rocm-rdna3/mmq-cap-sweep-20260516-015451-quick`

Command shape: `llama-bench -p 128,256,512 -n 0 -fa 1 -ctk tbq4_0 -ctv tbq4_0 -b 1024 -ub 512 -r 2 -o jsonl`

| variant | pp128 tok/s | pp256 tok/s | pp512 tok/s | rc |
|---|---:|---:|---:|---:|
| `baseline` | 1268.5 ± 39.3 | 1936.8 ± 22.1 | 2572.1 ± 178.8 | 0 |
| `rdna2_opt` | 1267.2 ± 42.5 | 1944.0 ± 91.1 | 2677.8 ± 23.4 | 0 |
| `maxx32` | 1657.6 ± 54.2 | 2236.5 ± 104.1 | 2711.3 ± 27.9 | 0 |
| `maxx48` | 1799.8 ± 82.1 | 2515.9 ± 96.1 | 3184.3 ± 31.5 | 0 |
| `maxx64` | 1690.0 ± 85.1 | 2457.5 ± 81.5 | 3155.8 ± 27.0 | 0 |
| `maxx128` | 1282.0 ± 73.3 | 1924.9 ± 42.4 | 2676.9 ± 20.6 | 0 |
