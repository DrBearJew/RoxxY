# llama.cpp — TBQ4 + MTP + RotorQuant for ROCm/RDNA3 (AMD RX 7900 XTX)

> **Fork of [Indras-Mirror/llama.cpp-mtp](https://github.com/Indras-Mirror/llama.cpp-mtp)** with fused TurboQuant (TBQ4) VEC flash attention for AMD ROCm/RDNA3 GPUs. Coherence-verified against q8_0 baseline. Also supports NVIDIA CUDA (unchanged from upstream).

**ROCm/RX 7900 XTX: 38-54 tok/s generation at 64k context with TBQ4 KV (4.25 bpv), MTP 83% acceptance. Full precision gate: 100% top-1 match vs q8_0, 0.945 Jaccard, 0.0017 JSD.**

---

## What's New (ROCm/AMD)

| Feature | Description | Status |
|---------|-------------|--------|
| **TBQ4 VEC Flash Attention** | Quantized-KV dequant inside FA via vectorized inline lookup — no separate dequant pass | ✅ Working, coherence-verified |
| **MTP Speculative Decoding** | Multi-Token Prediction for Qwen3.6 with `--spec-draft-n-max 3` | ✅ 83% acceptance vs q8_0 77% |
| **Coherence/Precision Gate** | Automated Python harness comparing TBQ4 vs q8_0 next-token distributions | ✅ All layers pass |
| **RotorQuant** | PlanarQuant/IsoQuant KV types (from Indras upstream) | ⚠️ VEC dispatch, untested |
| **Arch Support Matrix** | RDNA3 tested, RDNA3.5/RDNA4 enabled | See matrix below |

### Key Finding: MTP Acceptance

The default `--spec-draft-n-max` in llama.cpp is **16**, which tanks aggregate MTP acceptance to ~36%. Setting it to **3** (as recommended by PR #22673) restores expected acceptance. **TBQ4 does NOT degrade MTP acceptance vs q8_0.**

| KV Cache | n_max | draft_n | accepted | accept % | gen tok/s |
|---|---:|---:|---:|---:|---:|
| q8_0 | 3 | 57 | 44 | 77.2% | 49.8 |
| tbq4_0 | 3 | 54 | 45 | **83.3%** | **54.0** |
| tbq4_0 | 16 (default) | 144 | 53 | 36.8% | 38.1 |

---

## ROCm Build Instructions (AMD RX 7900 XTX / gfx1100)

### Prerequisites

- **ROCm 7.2.3+** (tested: `/opt/rocm-7.2.3`)
- **HIP compiler**: `/opt/rocm-7.2.3/bin/amdclang++`
- **GPU**: RDNA3 (`gfx1100`/`gfx1101`/`gfx1102`/`gfx1103`)
- **Model**: Qwen3.6 MTP GGUF (see [Getting an MTP GGUF](#getting-an-mtp-capable-gguf))

### Build

```bash
git clone https://github.com/DrBearJew/llama.cpp.git
cd llama.cpp
git checkout tbq4-rdna3-experiment

cmake -B build-rocm -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.3/bin/amdclang++ \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-rocm --target llama-server -j8
```

### Run

```bash
# TBQ4 KV (max VRAM savings, 4.25 bpv) with MTP
./build-rocm/bin/llama-server \
  -m /path/to/Qwen3.6-27B-Q4_K_M-mtp.gguf \
  --cache-type-k tbq4_0 --cache-type-v tbq4_0 \
  --spec-type mtp --spec-draft-n-max 3 \
  --jinja --chat-template-file /path/to/qwen36-merged-template.jinja \
  -c 65536 --port 8080 --no-webui --no-warmup --parallel 1

# q8_0 KV (max speed, more VRAM)
./build-rocm/bin/llama-server \
  -m /path/to/Qwen3.6-27B-Q4_K_M-mtp.gguf \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -c 16384 --port 8080 --no-webui --no-warmup
```

### Chat Template (Qwen 3.6)

Use the merged template from [allanchan339/vLLM-Qwen3-Chat-Template-Fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix) (supports `<|think_off|>`, developer role, tool calls):

```bash
--jinja --chat-template-file /path/to/qwen36-merged-template.jinja
```

Template available in this repo: `docs/rocm-tbq4-paths/qwen36-merged-template.jinja`

---

## Coherence & Precision Gate (ROCm)

A Python harness (`docs/rocm-tbq4-paths/harness.py`) compares TBQ4 against q8_0 across 6 test layers.

### Quick Run

```bash
cd docs/rocm-tbq4-paths
python3 harness.py --quick
```

### Results (May 14, 2026 — quick mode, 4k ctx, RX 7900 XTX)

| Layer | Result | Details |
|---|---|---|
| **Smoke** | ✅ PASS | France→Paris, 2+2→4, deterministic 3× repeat |
| **Precision** | ✅ PASS | 100% top1_match, 0.945 Jaccard, 0.092 MAE, 0.0017 JSD vs q8_0 |
| **Canaries** | ✅ PASS | JSON, ChatML leak guard, tool-call schema, code syntax |
| **MTP** | ✅ PASS | 83.3% acceptance (54 draft / 45 accepted) |
| **Cache** | ✅ PASS | Identical output with/without `cache_prompt` |

**Full summary**: `docs/rocm-tbq4-paths/gate-summary.json`

### Full Gate (64k context, needles, long prompts)

```bash
python3 harness.py --tbq4-ctx 65536 --q8-ctx 16384
```

---

## Architecture Support Matrix

| Family | Targets | Status | Notes |
|---|---|---|---|
| **RDNA3** | gfx1100/1101/1102/1103 | ✅ Tested (gfx1100) | Primary target; VEC FA + MTP verified |
| **RDNA3.5** | gfx1150/1151/1152 | ✅ Enabled, untested | Same VEC path as RDNA3 |
| **RDNA4** | gfx1200/1201+ | ✅ Enabled, untested | Dispatch gate uses `amd_wmma_available` |
| **RDNA1/RDNA2** | gfx10xx | ❌ Not enabled | Could investigate as VEC fallback |
| **CDNA/MI** | gfx9x/gfx94x | ❌ Not enabled | MFMA path needs separate validation |
| **NVIDIA** | sm_80/sm_89/sm_90 | ✅ Working (upstream) | Fused MMA TBQ4 path from Indras |

---

## Benchmarks (RX 7900 XTX, 24GB, Qwen3.6-27B Q4_K_M)

### Single-User Generation (tok/s)

| KV Cache | Context | MTP | tok/s | VRAM |
|---|---|---|---|---|
| tbq4_0 | 64k | ✅ n_max=3 | 38-54 | ~20 GB |
| tbq4_0 | 16k | ✅ n_max=3 | 36-38 | ~17 GB |
| q8_0 | 32k | ❌ | ~31 | ~23 GB |
| q8_0 | 16k | ✅ n_max=3 | ~50 | ~22 GB |

### Prefill Speed

| KV Cache | Prefill Size | Context | tok/s |
|---|---|---|---|
| tbq4_0 (after fix) | 28k tokens | 64k | 360.8 |
| tbq4_0 (before fix) | 28k tokens | 64k | 100.8 |
| q8_0 | 14k tokens | 16k | 394.2 |
| tbq4_0 (after fix) | 14k tokens | 16k | 537.7 |

### MTP Acceptance (n_max=3)

| KV Cache | draft_n | accepted | accept % | tok/s |
|---|---:|---:|---:|---:|
| tbq4_0 | 54 | 45 | 83.3% | 54.0 |
| q8_0 | 57 | 44 | 77.2% | 49.8 |

---

## TBQ4 VEC Attention (AMD) — Technical Notes

Unlike NVIDIA's rocWMMA path, the AMD path uses **vectorized inline dequant** in the flash attention inner loop — the same approach TheTom used in turboquant_plus.

### Why VEC Instead of WMMA?

- TheTom's working AMD turboquant_plus uses VEC inline dequant, not WMMA
- Our rocWMMA TBQ4 prototype was stable but produced incorrect text
- VEC path is simpler, more portable, and correctness-verified

### Key Implementation Details

| File | Purpose |
|---|---|
| `ggml/src/ggml-cuda/fattn-common.cuh` | `vec_dot_fattn_vec_KQ_tbq4_0`, `dequantize_V_tbq4_0` |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | `DECL_FATTN_VEC_CASE` for TBQ4 D=64/128/256 |
| `ggml/src/ggml-cuda/fattn.cu` | Routes AMD WMMA-capable targets to TBQ4 VEC |
| `src/llama-kv-cache.cpp` | Disables generic `attn_rot_*` for TBQ4 |
| `ggml/src/ggml-cuda/fattn-wmma-tbq4.cu` | Experimental rocWMMA path (research only) |

### Bugs Fixed During Development

1. **VEC correctness**: `dequantize_V_tbq4_0` used wrong index; TBQ4 was double-rotated via generic Hadamard
2. **VEC prefill speed**: TBQ4 KQ dot used f16-style `Q_reg` but RDNA quantized lane mapping → fixed with `KQ_uses_Q_reg` + `nthreads_KQ=8`
3. **Rotation model**: TBQ4 stores K/V in signed-FWHT domain; Q pre-rotated, output inverse-rotated

---

## Key Flags

| Flag | Purpose |
|---|---|
| `--cache-type-k tbq4_0 --cache-type-v tbq4_0` | TBQ4 KV cache (lossless, 4.25 bpv) |
| `--cache-type-k q8_0 --cache-type-v q8_0` | q8_0 KV cache (higher speed, more VRAM) |
| `--spec-type mtp --spec-draft-n-max 3` | MTP with optimal draft depth |
| `--jinja --chat-template-file <path>` | Qwen merged chat template |
| `--parallel 1` | Required for MTP |
| `--no-warmup` | Skip startup warmup |
| `-c 65536` | Context length (64k) |

---

## Getting an MTP-Capable GGUF

**Option A: Pre-built (Recommended)**

```bash
# llmfan46's pre-built GGUF with 15 native MTP heads (~17 GB, Q4_K_M)
wget https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF/resolve/main/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_M.gguf
```

**Option B: Graft MTP heads onto any Qwen3.6 GGUF**

```bash
wget https://huggingface.co/havenoammo/Qwen3.6-27B-MTP-UD-GGUF/resolve/main/MTP-Q8_0.gguf
uv pip install gguf
python convert.py base-model.gguf MTP-Q8_0.gguf output-mtp.gguf
```

---

## Documentation

- **Coherence/Precision Gate**: `docs/rocm-tbq4-paths/07-coherence-precision-test.md`
- **ROCm Integration Steps**: `docs/rocm-tbq4-paths/04-rocwmma-integration-steps.md`
- **RDNA3 Layout Notes**: `docs/rocm-tbq4-paths/06-rdna3-layout-notes.md`
- **Harness**: `docs/rocm-tbq4-paths/harness.py`
- **Blog post (NVIDIA upstream)**: https://indrasmirror.au/blog-mtp-shared-tensors-200k.html

---

## Credits

### ROCm/AMD TBQ4 VEC Path
- **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)** — Working AMD VEC inline dequant implementation that proved the VEC approach correct
- **[adelj88/rocm_wmma_gemm](https://github.com/adelj88/rocm_wmma_gemm)** — rocWMMA reference implementation used in experimental WMMA prototype
- **[Kaden-Schutt/hipfire](https://github.com/Kaden-Schutt/hipfire)** — MMQ screening concept that inspired the coherence/precision gate design

### MTP & TurboQuant Foundation
- **[Indras-Mirror/llama.cpp-mtp](https://github.com/Indras-Mirror/llama.cpp-mtp)** — Base fork with fused TBQ4 MMA FA, MTP, RotorQuant, tensor sharing
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — PR #22673 (MTP support by ngxson, am17an), PR #21089 (CPU TBQ)
- **[spiritbuun](https://github.com/spiritbuun)** — dflash fork with CUDA TurboQuant kernels (FWHT kernels adapted from this)
- **[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** — MTP improvements PR #1736

### Models & Tooling
- **[llmfan46](https://huggingface.co/llmfan46)** — Qwen3.6-27B-Heretic-v2 Native-MTP-Preserved GGUF (15 native MTP heads)
- **[HauhauCS](https://huggingface.co/HauhauCS)** — Original Qwen3.6-Heretic-v2 uncensored base model
- **[havenoammo](https://huggingface.co/havenoammo)** — MTP graft tooling, first Qwen3.6-MTP GGUF release
- **[Radamanthys11](https://huggingface.co/Radamanthys11)** — MTP-Q8_0 GGUF extraction

### Chat Templates
- **[allanchan339/vLLM-Qwen3-Chat-Template-Fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix)** — Long strict tool system prompt, developer role
- **[froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)** — `<|think_on|>` / `<|think_off|>` toggles, non-ASCII escaping, `</thinking>` recognition

---

<details>
<summary><strong>Original NVIDIA Benchmarks & Documentation (from Indras-Mirror)</strong></summary>

## NVIDIA Results (RTX 4090 24GB, Qwen3.6-27B-Heretic-v2-MTP Q4_K_M)

| Config | Context | KV Cache | tok/s | Draft Accept | VRAM |
|--------|---------|----------|-------|-------------|------|
| **MTP + Fused TBQ4 FA (May 11)** | **262K** | **TBQ4_0 (4.25 bpv)** | **179.4** | **81.4%** | **~20 GB** |
| **MTP + Fused TBQ4 FA** | **262K** | **TBQ4_0 (4.25 bpv)** | **80-87** | **73-93%** | **~20 GB** |
| MTP + Fused TBQ4 FA | 200K | TBQ4_0 (4.25 bpv) | 82-87 | 73% | ~20 GB |
| MTP + Q4_0 KV | 200K | Q4_0 (4.5 bpv) | 92-97 | 93.6% | 23.96 GB |
| Baseline (no MTP, Q4_0 KV) | 200K | Q4_0 | ~40 | - | 23.96 GB |

### NVIDIA Build

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --config Release

./build/bin/llama-server \
  -m your-qwen3.6-mtp.gguf \
  --spec-type mtp --spec-draft-n-max 3 \
  -ctk tbq4_0 -ctv tbq4_0 -c 262144 -ngl 99 \
  --flash-attn on --mlock -t 8 -ub 32 --parallel 1 --no-warmup
```

### RotorQuant Benchmarks (RTX 4090)

| Type | 4K ctx | 32K ctx | 262K ctx | Notes |
|------|--------|---------|----------|-------|
| `tbq4_0` | 55.3 t/s | 51.5 t/s | 77 t/s | Baseline — fused MMA kernel |
| `planar3_0` | 53.9 t/s | 50.6 t/s | ~47 t/s | Best speed/compression tradeoff |
| `iso3_0` | 53.5 t/s | 50.5 t/s | — | Same compression as planar3 |

### NVIDIA TBQ4 MMA Architecture

Fused TBQ4 Flash Attention pipeline:
1. `k_tbq4_rotate_input` → Pre-rotate Q via FWHT
2. Fused FA kernel → Read raw TBQ4 blocks, centroid×norm dequant inline
3. `k_tbq4_rotate_output` → Post-rotate VKQ back to original domain

TBQ4_0 block: 66 bytes per 128 elements (4.25 bpv)
- `ggml_half d` — corrected L2 norm (2 bytes)
- `uint8_t qs[64]` — packed 4-bit centroid indices (64 bytes)
- 16 Lloyd-Max centroids in `__constant__` memory

### Tensor Sharing — `link_shared_tensors()`

MTP loads `token_embd.weight` as a separate 682 MiB GPU allocation. The API prevents duplication:

```cpp
LLAMA_API void llama_model_link_shared_tensors(
    struct llama_model * model,
    const struct llama_model * trunk);
```

### Known Issues (from upstream)

- Vision + MTP crashes. Use `--spec-type none` for vision tasks
- MTP requires `--parallel 1`
- 7B models crash with TBQ4 (16-byte alignment)
- MoE models may fail with `vector::_M_range_check` if GGUF metadata is incomplete

</details>

---

## License

MIT. See upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) for full license text.
