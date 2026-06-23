# DESLOPPIFY Backlog

Read-only scan date: 2026-06-23

Scope scanned:

- `/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github`
- Baseline copy checked: `/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github (Copy)`
- Note: the copy is byte-identical to this tree outside excluded git/build paths, so it is not a clean upstream reference.
- No code changes were made for this review. This file is the cleanup backlog requested by the user.

Verification/evidence used:

- `diff -qr` against the copy, excluding `.git` and build dirs: no differences found.
- `rg` scans over MTP/QBlock/packed16/V4/CUDA/server paths.
- Background read-only scans:
  - `desloppify-repo-hygiene-scout`
  - `desloppify-cuda-mtp-scout`
- `bash -n scripts/hip/*.sh .harness/patches/*.sh`
- `python3 -m py_compile scripts/hip/*.py`

## Archive policy

Archive means **out of the source tree**. Do not keep dead implementations in `src/`, `ggml/`, `tools/`, tests, or repo-local `.harness` just because they might be useful later.

For deleted experimental branches, preserve only:

1. an external patch under `/home/mrtrent/.harness/artifacts/llama.cpp-tree-tbq4-rdna3-github/archived-code/<date>-<topic>/`,
2. a small manifest with original files, reason for removal, last known hash/speed evidence, and restore command,
3. a one-line pointer in the canonical project wiki if the experiment is historically important.

The live repo should keep at most a tiny compatibility parser/error message when removing a public env/API would otherwise confuse users. It must not keep the old kernel/body/route branch as an in-repo graveyard.

## Critical issues

### C1. Broken git/worktree metadata blocks safe cleanup

- Where: `.git`
- Evidence: `.git` points to `/home/mrtrent/llama.cpp-mtp-tbq4-rdna3/.git/worktrees/llama.cpp-tree-tbq4-rdna3-github`; that parent worktree metadata path is missing, and `git status` fails with `fatal: not a git repository`.
- Why it matters: cannot reliably classify tracked vs untracked files, inspect diffs, clean generated files, or protect active work before deleting large artifacts.
- Recommendation: restore/prune the parent worktree metadata, reattach this checkout, or reclone before any destructive cleanup. After git works, rerun `git status --short --untracked-files=all` and snapshot any local changes.
- Safe to fix now: yes, but do this before deleting files or refactoring code.

### C2. Public QBlock direct API appears orphaned/uncompiled

- Where:
  - `src/llama-ext.h:139` declares `llama_decode_qblock_verify(...)`.
  - `src/llama-qblock-direct.inc:3484`, `src/llama-qblock-direct.inc:10767` define `decode_qblock_verify(...)` / `llama_decode_qblock_verify(...)`.
  - No include/build reference found for `src/llama-qblock-direct.inc` in `src/`, `CMakeLists.txt`, or `compile_commands.json`.
- Why it matters: a public experimental API is declared but its implementation appears not to be compiled. That creates false confidence, breaks external callers if they link to it, and makes 10k+ lines of QBlock direct logic ambiguous: live code or archived debug scaffold.
- Recommendation: decide one path:
  1. If live: add explicit build integration and compile gate, then fix any compile errors.
  2. If dead/debug-only: move to an archived research file or test utility and remove public declarations.
- Safe to fix now: wait until git is restored; then safe as a focused architecture cleanup, but it needs build verification.

### C3. QBlock direct file references an undefined speed-candidate flag

- Where:
  - `src/llama-ext.h:122-128` defines only `LLAMA_QBLOCK_VERIFY_FLAG_WHOLE_TARGET_CANDIDATE`.
  - `src/llama-qblock-direct.inc:3507` references `LLAMA_QBLOCK_VERIFY_FLAG_WHOLE_TARGET_SPEED_CANDIDATE`.
  - `docs/MTP_QBLOCK_ENV_KNOBS.md:98` documents the missing flag.
- Why it matters: if the QBlock direct implementation is reintroduced into the build, this is an immediate compile failure or stale-doc mismatch. It also shows the flag/API/docs are drifting.
- Recommendation: either add the missing enum value intentionally with tests, or delete the speed-candidate path/docs if it is no longer live.
- Safe to fix now: wait until C2 is resolved; then safe as part of the same QBlock API cleanup.

### C4. Process-global env mutation is used for request/graph routing

- Where:
  - `tools/server/server-context.cpp:895-965`, `tools/server/server-context.cpp:5504-5562`
  - `src/llama-context.cpp:5898-5923`
  - `ggml/src/ggml-cuda/fattn.cu:3979-4030`
  - `src/models/qwen35.cpp:1448-1476`
- Why it matters: `setenv`/`unsetenv` are process-global. In a server with multiple slots/contexts, request-local routing can leak into other graph builds or kernel selection. This makes route proofs order-dependent and hard to reproduce.
- Recommendation: stop adding new env-mutation scopes. Move these settings into graph/op params or a scoped launcher/config object passed through the call chain. Short term, document which scopes are server-global and add logging when they override user env.
- Safe to fix now: wait for a small design refactor; do not attempt a broad rewrite during active performance work.

### C5. Unused experimental branches are now a build-time and review-time tax

- Where:
  - `ggml/src/ggml-cuda/fattn-packed16-dot4-mmq-impl.cuh`, ~5,829 lines with many compile/runtime route cells.
  - `ggml/src/ggml-cuda/mmvq.cu`, ~7,561 lines.
  - `src/llama-qblock-direct.inc`, ~10,775 lines.
  - `ggml/src/ggml-hip/CMakeLists.txt:60-70`, many compile gates for demoted/optional PDMQ variants.
  - `ggml/src/ggml-hip/CMakeLists.txt:209-211`, fast-compile exists because this translation unit is expensive.
  - `scripts/hip/build-qwen35-dev.sh:4-24`, explicitly says the dev build narrows the matrix to avoid compiling broader experimental packed16 routes.
- Why it matters: even when code is default-off, large branch matrices slow HIP rebuilds, make compile failures harder to localize, and consume reviewer/LLM context. The current source has compile gates, but demoted paths still stay physically near the hot path and keep growing the cognitive surface.
- Recommendation: create an explicit `dead/demoted route deletion` pass. For each default-off branch, classify as `live`, `diagnostic`, `archive-out-of-tree`, or `delete`. For `archive-out-of-tree`, first write an external patch+manifest under harness artifacts, then delete the implementation from source. Keep only a tiny centralized compatibility error if a public env/API name remains; do not keep old kernel bodies or route branches in the repo.
- Safe to fix now: yes after C1, but do it in small deletion batches with one build/hash gate per batch. Do not mix with new performance work.

## Medium cleanup items

### M1. Packed16 sidecar registry has no unregister/invalidation path

- Where:
  - `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu:27-28` static `s_packed16_registry`.
  - `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu:119-128` register path by `k_view_data`.
  - No `erase`, `clear`, or unregister path found.
- Why it matters: stale metadata can survive KV cache/model teardown. If tensor/data pointers are reused, a new tensor can inherit old payload/scales/layout metadata.
- Recommendation: add unregister/invalidate hooks on KV cache/free/reset paths, include owner/generation validation, and reject null keys at registration.
- Safe to fix now: yes, after git restoration. Add a focused unit/variant test for stale pointer rejection.

### M2. Large build artifacts are inside the source tree

- Where: root `build*` directories, 26 dirs observed.
- Evidence examples: `build-rocm-rdna2-fa` ~846M, `build-rocm-fixed` ~686M, `build-rocm-ninja` ~623M; repo working directory around 14G.
- Why it matters: searches are noisy and slow, generated files appear in scans, and cleanup/diff review becomes risky.
- Recommendation: keep one active build dir if needed, move others outside the source tree or delete after archiving any unique logs.
- Safe to fix now: only after C1, because git must work before destructive deletion.

### M3. In-source CMake generated files pollute real source dirs

- Where: `CMakeFiles/`, `CMakeCache.txt`, `cmake_install.cmake`, `Makefile` under root, `src/`, `ggml/`, `tools/`, `tests/`, `examples/`, `common/`, etc.
- Why it matters: generated files mask source changes, make grep results unreliable, and can contain hardcoded local paths.
- Recommendation: delete generated CMake files after git is restored; enforce out-of-source builds and ignore coverage.
- Safe to fix now: after C1.

### M4. Benchmark/artifact logs are stored in the repo

- Where: `benches/` and repo-local `.harness/research`, `.harness/tmp`, `.harness/patches`.
- Evidence: `benches` contains thousands of files and ~5.8G of data; many dated reports duplicate canonical wiki state.
- Why it matters: the active source tree doubles as artifact storage, which makes provenance and current truth confusing.
- Recommendation: move raw artifacts to `/home/mrtrent/.harness/artifacts/...`; keep curated summaries in `/home/mrtrent/.harness/project/llama.cpp-tree-tbq4-rdna3-github/`.
- Safe to fix now: wait until C1 and after selecting which artifacts are still authoritative.

### M5. Nested reference repo and generated dependencies are inside the source tree

- Where: `reference/`, including nested `.git`, `node_modules`, `.svelte-kit`.
- Evidence: `reference` ~913M and ~33k files.
- Why it matters: duplicates symbols and files, slows grep/build scans, and confuses what belongs to this repo versus external reference material.
- Recommendation: move references under harness read-only research storage, or make them explicit submodules without generated dependency trees.
- Safe to fix now: wait; classify whether any scripts depend on these paths first.

### M6. Qwen35 MTP output-head logic is duplicated and drifting

- Where:
  - `src/models/qwen35.cpp:2208-2219`
  - `src/models/qwen35_mtp.cpp:208-233`
- Why it matters: main Qwen35 and Qwen35_MTP head paths can diverge. The main path guards fused top-k on `head_s == nullptr`; the MTP path does not mirror the same scale handling, and fallback calls may ignore `head_s`.
- Recommendation: consolidate MTP head construction into one helper or mirror `head_s`/scale/fused-topk handling exactly.
- Safe to fix now: yes with a narrow parity test; otherwise wait until MTP speed work is not in flight.

### M7. Packed16 decode variant test leaks route env between variants

- Where: `tests/test-packed16-decode-variants.cpp:330-450`.
- Why it matters: `packed16_fa2_vec` sets `GGML_CUDA_ROCM_PACKED16_FA2_VEC=1`, and later variants may inherit it. Test outcomes can depend on variant order.
- Recommendation: add an RAII env guard or reset all route envs at the start/end of every `run_variant` call.
- Safe to fix now: yes. Low-risk test harness cleanup.

### M8. Invalid route-forcing env values fail silently

- Where: `src/llama-graph.cpp:2439-2455`, `LLAMA_MTP_FA_INST` force parsing.
- Why it matters: a mistyped route force silently falls back to default behavior, wasting debugging time and weakening routeproof evidence.
- Recommendation: log a warning or fail closed when the env is non-empty and not one of the accepted route names.
- Safe to fix now: yes. Prefer warning first to avoid breaking old scripts.

### M9. CUDA/storage env implicitly changes MTP verifier policy

- Where: `tools/server/server-context.cpp:374-380`, `mtp_target_batch_verify_unsafe_requested()` returns true when `GGML_CUDA_ROCM_V4_K16D16_144_PV4` is set.
- Why it matters: backend storage/kernel selection is coupled to speculative verification policy. This makes correctness behavior harder to reason about and can surprise users enabling a V4/PV4 storage route.
- Recommendation: require explicit `LLAMA_MTP_TARGET_BATCH_VERIFY_UNSAFE=1`, or convert to typed config with a clear startup warning/error.
- Safe to fix now: wait; current PV4 workflows may depend on this shortcut.

### M10. Server CORS/proxy defaults are permissive when exposed beyond localhost

- Where:
  - `tools/server/server-http.cpp:229-235`
  - `tools/server/server.cpp:207-214`
  - `tools/server/server-cors-proxy.h:11-50`
- Why it matters: reflected `Origin` plus `Access-Control-Allow-Credentials: true` and unauthenticated `OPTIONS` can be risky if the server is bound publicly or used with browser/API-key clients. MCP proxy allows arbitrary HTTP/HTTPS target when enabled.
- Recommendation: add explicit CORS allowlist support; default credentials off unless configured; keep MCP proxy localhost-only or allowlisted.
- Safe to fix now: wait for product compatibility decision.

### M11. Broken benchmark helper script

- Where: `scripts/hip/bench-compare.sh`.
- Evidence: `bash -n` reports `line 32: syntax error near unexpected token '}'`.
- Why it matters: benchmark helper is unusable and can mislead future perf checks.
- Recommendation: fix the unmatched subshell/function block and quote `$MODEL`, `$TESTFILE`, `$BUILD`.
- Safe to fix now: yes after C1.

### M12. Patch apply helper references missing patch files

- Where: `.harness/patches/apply-packed8-v4-stack.sh`.
- Why it matters: helper implies an available WIP stack but referenced patch files are missing; it also depends on working git.
- Recommendation: restore the patch files, or demote this script to archived docs/delete it.
- Safe to fix now: wait until C1 and artifact provenance are clarified.

### M13. Effective LLM context is polluted by generated, duplicate, and monolithic files

- Where:
  - Generated/build files in source dirs: `CMakeFiles/`, `CMakeCache.txt`, `cmake_install.cmake`, `Makefile` across root/subdirs.
  - Duplicate/reference material: `reference/`, `packed16-pdmq-dll-i8-gpt55pro/`, repo-local `.harness/research`.
  - Monolithic hot/debug files: `src/llama-qblock-direct.inc`, `tools/server/server-context.cpp`, `ggml/src/ggml-cuda/fattn-packed16-dot4-mmq-impl.cuh`, `ggml/src/ggml-cuda/mmvq.cu`.
- Why it matters: agent/code-search context becomes noisy. The LLM sees generated linker files, duplicate old code, and dead branches unless every search is carefully excluded. This raises the chance of editing stale code or missing the real hot path.
- Recommendation: add an explicit repo `LLM_CONTEXT.md` or `.agentignore`/search wrapper policy: exclude build dirs, reference dirs, generated CMake, node_modules, archived experiments, and dated artifacts by default. Split monoliths into `live` and `diagnostics`; move true archive material out of the repo entirely so retrieval finds current code first.
- Safe to fix now: yes for ignore/search-policy docs after C1; splitting/deleting monoliths should wait for build/hash gates.

### M14. UI/UX rough-edge scan is shallow and server/web UI risks are under-triaged

- Where:
  - `tools/server/server-http.cpp:229-235`, `tools/server/server.cpp:207-214`, `tools/server/server-cors-proxy.h:11-50` for browser-facing CORS/proxy behavior.
  - `tools/server/webui/` and generated/dependency material under the server UI tree need a focused pass before user-facing changes.
- Why it matters: the current scan found security/config rough edges in the browser-facing server surface, but did not deeply review user experience: confusing startup warnings, WebUI configuration affordances, proxy enablement messaging, route/debug knob discoverability, and error text shown to users/operators. That can make a technically correct server hard or unsafe to operate.
- Recommendation: run a focused WebUI/server UX pass after repo hygiene: check CORS/proxy defaults, warnings, docs, browser error messages, config names, and whether dangerous knobs are visibly marked local-only/debug-only.
- Safe to fix now: wait until C1 and generated UI/build artifacts are cleaned, unless touching server CORS/proxy policy first.

## Nice-to-have polish

### N1. QBlock direct executor is too large and env-driven

- Where: `src/llama-qblock-direct.inc`, ~10,775 lines, especially dense `LLAMA_MTP_QBLOCK_DIRECT_*` flag cluster around `src/llama-qblock-direct.inc:3770-3823`.
- Why it matters: many debug/probe toggles make route state hard to reproduce and review. Unsafe escape hatches remain present, e.g. `LLAMA_MTP_QBLOCK_DIRECT_UNSAFE_ALLOW_STATE_DRIFT` around `src/llama-qblock-direct.inc:8270-8278`.
- Recommendation: after C2, split live executor, diagnostics, and archived probes. Put all direct debug routes behind one explicit `debug executor` gate.
- Safe to fix now: wait; requires owner decision and routeproof coverage.

### N2. Env knob registry exists but is not authoritative

- Where:
  - `docs/MTP_QBLOCK_ENV_KNOBS.md`
  - `.harness/research/v4-mtp-q4-env-knob-registry-20260620.md`
  - scattered `getenv`/`llama_env_i32` in `src/`, `tools/server/`, `ggml/src/ggml-cuda/`.
- Why it matters: documentation and code drift, as shown by the missing speed-candidate flag.
- Recommendation: generate an env-knob inventory from source and diff it against docs in CI or a local script.
- Safe to fix now: yes, non-runtime tooling only.

### N3. `tile16` / `d16_planar` naming remains confusing

- Where:
  - `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cuh:56-82`
  - `tests/test-packed16-decode-variants.cpp:34-45`
  - `src/llama-kv-cache.h:252`
- Why it matters: docs now define D16-planar as address grammar/metadata, but some code still accepts aliases like `tile16`, `native`, `v2`; this can make route intent unclear.
- Recommendation: keep backward aliases but log canonical `d16_planar`; update comments to say `tile16` is legacy alias.
- Safe to fix now: yes, but low priority.

### N4. Python bytecode and generated caches are present

- Where: `scripts/hip/__pycache__`.
- Why it matters: small but noisy generated artifacts in script directories.
- Recommendation: delete and add ignore coverage.
- Safe to fix now: after C1.

### N5. Hardcoded local paths in scripts/docs reduce portability

- Where: benchmark scripts/docs under `benchmarks/`, `scripts/hip/`, `docs/`, generated CMake files.
- Why it matters: some are expected for local benchmarking, but unmarked local paths make scripts brittle for future reuse.
- Recommendation: move local model/build paths into env defaults at top of scripts and document required vars.
- Safe to fix now: wait unless touching a script for another reason.

## Suggested task order

1. C1, repair git/worktree metadata.
2. M3 + N4, remove in-source generated build/cache artifacts once git can prove what is safe.
3. M13, add LLM/search ignore policy so future scans stop ingesting generated/reference noise.
4. M11, fix `scripts/hip/bench-compare.sh`.
5. C2 + C3, decide and clean QBlock direct API/build status.
6. C5, delete/archive unused experimental branches in small verified batches.
7. M1, add packed16 sidecar unregister/invalidation.
8. M7 + M8, fix low-risk test/env validation issues.
9. M2 + M4 + M5, move/archive bulky repo artifacts and references.
10. C4 + M9, design typed route/config path to reduce env mutation/coupling.
11. M6, consolidate Qwen35/Qwen35_MTP head handling.
12. M10 + M14, tighten CORS/proxy defaults and run focused WebUI/server UX pass after compatibility decision.

## Current selection menu

Pick one backlog ID to work next:

- `C1` repair git/worktree metadata
- `M3` clean in-source CMake files
- `M11` fix broken benchmark script
- `C2/C3` resolve QBlock direct API/build drift
- `C5` delete/archive unused experimental branches to reduce build/review tax
- `M13` add LLM/search ignore policy and context hygiene
- `M1` add packed16 sidecar lifecycle cleanup
- `M7` fix test env leakage
- `M8` warn/fail on invalid route force
- `M2/M4/M5` archive bulky repo artifacts
- `C4/M9` replace env mutation/coupling with typed config
- `M6` consolidate Qwen35 MTP head path
- `M10/M14` review/tighten server CORS/proxy and WebUI/server UX rough edges
