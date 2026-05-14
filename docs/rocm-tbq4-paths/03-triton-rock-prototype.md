# Path 3 — Triton/ROCK prototype lab

## Goal
Use conda ROCK/Triton only to explore layouts and dequant strategy. Final target remains ROCm 7.2.3.

## Implement
- Prototype TBQ4 decode + QK/VKQ tile math in Triton 3.7.
- Use `D=128` first.
- Measure dequant cost, LDS/shared-memory shape, and tile ordering.
- Export useful indexing/layout formulas back to HIP/rocWMMA implementation.

## Do not
- Do not make ROCK the production dependency.
- Do not mix ROCK runtime libs into ROCm 7.2.3 build accidentally.
- Do not optimize Triton beyond what informs HIP implementation.

## Verify
- Run in isolated conda LLM env.
- Compare Triton prototype output to CPU/non-fused TBQ4 reference.
- Record any layout formula that reduces HIP trial-and-error.
