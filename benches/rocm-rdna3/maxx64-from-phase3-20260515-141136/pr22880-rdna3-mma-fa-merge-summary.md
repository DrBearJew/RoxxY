# PR #22880 RDNA3 MMA FA local merge summary

Base branch: `tbq4-rdna3-experiment` at `846992e5f`.
PR head: `refs/remotes/upstream-ggml/pr/22880` = `356de7839`.

Applied locally:
- `ggml/src/ggml-cuda/mma.cuh`: PR AMD/RDNA3 transpose + tile layout helper changes applied cleanly.
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`: PR RDNA3 MMA FA changes merged while preserving local TBQ4 staging argument and staging calls.
- `ggml/src/ggml-cuda/fattn.cu`: PR `gqa_ratio_eff` and AMD MFMA/WMMA selector tuning merged; local compressed-KV/TBQ4 routing preserved.

Manual conflict resolutions:
- Kept `tbq4_staging` parameter in `flash_attn_ext_f16_iter(...)` and widened PR guard to `defined(AMD_WMMA_AVAILABLE)` for RDNA3 MMA FA.
- Removed duplicated local Volta-only `gqa_ratio_eff` calculation because PR computes it once before Volta/AMD selectors.
- Kept local CDNA MFMA fallback for other supported head sizes, then added PR AMD WMMA selector for RDNA3/RDNA4.

Verification:
- `git diff --check -- ggml/src/ggml-cuda/fattn-mma-f16.cuh ggml/src/ggml-cuda/fattn.cu ggml/src/ggml-cuda/mma.cuh` passed.
- `cmake --build build-rocm --target llama-bench llama-server -j 8` passed after `mma.cuh` only.
- `cmake --build build-rocm --target llama-bench llama-server -j 8` passed after full three-file merge.

Blocked/notes:
- `scripts/hip/check-compressed-kv-fa-invariants.sh` currently fails before source checks because `/home/mrtrent/.local/bin/llama-server-wrapper` contains experimental compressed-KV FA env flags. No wrapper changes were made in this merge step.

Additional smoke results:
- F16 KV FA canary: HTTP 200, `SAFE_OUTPUT`, server rc 0; log shows `kernel=wmma_f16` for prefill/decode chunks and `kernel=vec` for nq=1 tail/decode.
  - JSON: `pr22880-f16-fa-long-canary-reasonoff.json`
  - log: `pr22880-f16-fa-long-canary-reasonoff-server.log`
- TBQ4 KV FA canary: HTTP 200, `SAFE_OUTPUT`, server rc 0; log shows `kernel=vec` for `K=tbq4_0 V=tbq4_0`, preserving compressed-KV savings.
  - JSON: `pr22880-tbq4-vec-long-canary-reasonoff.json`
  - log: `pr22880-tbq4-vec-long-canary-reasonoff-server.log`
- Planar/Iso compressed-KV canaries: `planar3_0` and `iso3_0` both HTTP 200, `SAFE_OUTPUT`, server rc 0; logs show `kernel=vec` for compressed KV.
  - JSON: `pr22880-planar3_0-vec-canary-reasonoff.json`, `pr22880-iso3_0-vec-canary-reasonoff.json`
  - logs: `pr22880-planar3_0-vec-canary-reasonoff-server.log`, `pr22880-iso3_0-vec-canary-reasonoff-server.log`
