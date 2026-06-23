# LLM / Search Context Hygiene

Default agent and search scans should focus on live source, tests, scripts, and curated docs. Do not ingest generated build output, external reference dumps, or archived experiments unless a task explicitly asks for them.

## Exclude by default

- `build*/`
- `CMakeFiles/`, `CMakeCache.txt`, `cmake_install.cmake`, generated `Makefile`, `CTestTestfile.cmake`
- `__pycache__/`, `*.pyc`
- `node_modules/`, `.svelte-kit/`, generated WebUI `dist/`
- `benches/` raw benchmark output
- `reference/` nested source/reference trees
- `packed16-pdmq-dll-i8-gpt55pro/` and similar prompt/source dump bundles
- repo-local `.harness/research/`, `.harness/tmp/`, `.harness/artifacts/`
- any in-repo `archive/`, `archives/`, `dead_code/`, or `graveyard/` directory if one appears

## Canonical project memory

Use the external project wiki instead of repo-local research dumps:

- `/home/mrtrent/.harness/project/llama.cpp-tree-tbq4-rdna3-github/`

Use external archived-code artifacts instead of in-repo graveyards:

- `/home/mrtrent/.harness/artifacts/llama.cpp-tree-tbq4-rdna3-github/archived-code/`

## Live high-signal paths

- `src/`
- `ggml/src/ggml-cuda/`
- `ggml/src/ggml-hip/`
- `ggml/include/`
- `common/`
- `tools/server/`
- `tests/`
- `scripts/hip/`
- `docs/` only for maintained docs, not generated logs

Archive policy: if dead experimental code must be preserved, write an external patch+manifest under the harness artifacts path, then delete it from the source tree.
