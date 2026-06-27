# JetSpec P5H KV/hidden commit ownership readiness

Status: inert readiness contract only. This is not production tree-runtime
approval, not a llama.cpp KV mutation, and not compiled by CMake.

## Scope

P5H stays entirely under `experiments/jetspec/`. It turns upstream JetSpec
`reserve_tree_slots()`/`gather()`/commit behavior into a llama.cpp-facing
ownership contract before any production runtime approval. It does not edit
`common/speculative.cpp`, `src/`, `include/`, `tools/server/`, repository
`tests/`, `examples/`, `pocs/`, `ggml/src/`, top-level `CMakeLists.txt`, or
production `.cmake` files.

P5H keeps the no-runtime boundary: no production runtime execution, no `llama_context` draft runtime instantiation, no draft-head graph execution, no draft tokens emitted, no real KV cache mutation, no server route, and `runtime_supported=false` remains the loader/runtime boundary. This is not production tree-runtime approval.

## Ownership contract

The smoke fixture models abstract slots, not real llama.cpp cache state:

- model hidden/KV rows trail committed tokens by one correction anchor;
- committed tokens are `[accepted draft tokens | correction]`;
- hidden/KV survivors are exactly `[root | accepted]`;
- correction hidden is not appended in the same round;
- rejected tree nodes are unreachable after commit;
- future gather positions are `past_len + accepted_path`, matching the upstream
  `prefix + accepted_path`/`max_len + accepted_path` gather shape;
- duplicate accepted path nodes are rejected;
- out-of-range accepted path nodes are rejected;
- cross-sequence slots remain isolated and unchanged;
- refcounts for committed survivor slots are one in the abstract model;
- future production mapping must name an exact `llama_kv_cache_*` primitive or
  explicitly state `missing primitive`, never silently map upstream behavior to
  `seq_rm`/`seq_cp`/`seq_import_physical`.

## Files

- `jetspec_kv_commit_readiness.py`: stdlib evaluator for the P5H abstract slot,
  gather, hidden/KV commit, cross-sequence isolation, ownership mapping, and
  no-runtime boundary contract.
- `fixtures/jetspec_kv_commit_readiness_smoke.json`: deterministic P5H fixture.
- `fixtures/jetspec_kv_commit_readiness_smoke.out.json`: expected P5H output.
- `validate_p5h_kv_commit_readiness.py`: source/governance validator that checks
  the P5H files, fixture output, forbidden production path boundary, and CMake
  isolation.
- `test_p5h_kv_commit_readiness.py`: unittest coverage for the smoke fixture,
  duplicate/out-of-range path rejection, gather mismatch, cross-sequence
  isolation, silent ownership mapping rejection, runtime-boundary failures, and
  validator success.

## Verification evidence

Representative commands:

```bash
python3 experiments/jetspec/jetspec_kv_commit_readiness.py \
  --fixture experiments/jetspec/fixtures/jetspec_kv_commit_readiness_smoke.json
python3 experiments/jetspec/validate_p5h_kv_commit_readiness.py
python3 experiments/jetspec/test_p5h_kv_commit_readiness.py
python3 experiments/jetspec/run_all_jetspec_contracts.py
```

Expected outcome: the fixture reports
`kv_commit_readiness_verified_not_executed`, the validator passes, aggregate
contracts pass, and JetSpec remains explicit-opt-in/non-drafting.

## Blocked next work

Production tree runtime remains blocked until explicit tree-runtime approval. The
blocked surface includes draft-head graph execution, runtime tree verify, real
KV/hidden commit/rollback mutation, public API, server request behavior,
repository tests/examples/pocs, kernels, and performance promotion.
