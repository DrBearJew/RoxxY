# JetSpec P5K KV ownership primitive design packet

Status: inert design packet only. This is not production tree-runtime approval,
not runtime execution, not a performance claim, and not compiled by CMake.

## Scope

P5K stays entirely under `experiments/jetspec/`. It converts P5J's three
`exact_missing_primitive` gaps into explicit source-backed primitive contracts,
still without implementation. It does not edit `common/speculative.cpp`,
`common/common.h`, `src/`, `include/`, `tools/server/`, repository `tests/`,
`examples/`, `pocs/`, `ggml/src/`, top-level `CMakeLists.txt`, or production
`.cmake` files.

P5K keeps the no-runtime boundary: no `llama_context` creation, no draft-head graph execution, no draft tokens emitted, no real KV mutation, no server route, no performance claim, no promotion claim, and `runtime_supported=false` remains required.

## Design contract

`kv_ownership_primitive_design.py` reads `src/llama-kv-cache.h` and
`src/llama-kv-cache.cpp` as source text only. Existing helpers may be cited only
as audited non-exact helpers:

- `seq_rm`
- `seq_cp`
- `seq_import_physical`
- `seq_keep`
- `find_slot`
- `apply_ubatch`

The design packet names three future candidate primitives, but keeps them
`design_only_missing_implementation`:

- `llama_kv_cache_jetspec_commit_survivor_path_candidate` for hidden/KV survivor
  commit;
- `llama_kv_cache_jetspec_discard_rejected_tree_candidate` for rejected branch
  discard;
- `llama_kv_cache_jetspec_assert_cross_sequence_isolation_candidate` for
  cross-sequence isolation.

Required invariants:

- hidden/KV survivor commit: accepted_path physical gather/compact,
  `[root | accepted] only`, correction hidden deferred, committed tail compact;
- rejected branch discard: rejected transient tree slots unreachable, accepted
  path preserved, not range removal only, rollback restores pre-round state;
- cross-sequence isolation: other-sequence slots unchanged, `seq_to_stream`
  isolation, no shared-slot corruption, rollback preserves other sequences.

P5K fails if a fixture tries to treat `seq_cp`, `seq_rm`, or
`seq_import_physical` as an implicit exact mapping for any JetSpec tree ownership
primitive.

## Files

- `kv_ownership_primitive_design.py`: stdlib source-backed design validator.
- `fixtures/kv_ownership_primitive_design_smoke.json`: deterministic P5K design
  fixture.
- `fixtures/kv_ownership_primitive_design_smoke.out.json`: expected P5K output.
- `validate_p5k_kv_ownership_primitive_design.py`: source/governance validator
  that checks P5K files, fixture output, forbidden production path boundary, and
  CMake isolation.
- `test_p5k_kv_ownership_primitive_design.py`: unittest coverage for the smoke
  fixture, audited helper coverage, missing design rejection, implicit mapping
  rejection, implementation-approval rejection, runtime-boundary failures, and
  validator success.

## Verification evidence

Representative commands:

```bash
python3 experiments/jetspec/kv_ownership_primitive_design.py \
  --fixture experiments/jetspec/fixtures/kv_ownership_primitive_design_smoke.json
python3 experiments/jetspec/validate_p5k_kv_ownership_primitive_design.py
python3 experiments/jetspec/test_p5k_kv_ownership_primitive_design.py
python3 experiments/jetspec/run_all_jetspec_contracts.py
```

Expected outcome: the fixture reports
`kv_ownership_primitive_design_verified_not_executed`, the validator passes,
aggregate contracts pass, and JetSpec remains explicit-opt-in/non-drafting.

## Blocked next work

Production tree runtime remains blocked until explicit tree-runtime approval and
until these design-only primitives are implemented and verified in a bounded
compiled-path slice. The blocked surface includes draft-head graph execution,
runtime tree verify, real KV/hidden commit/rollback mutation, public API, server
request behavior, repository tests/examples/pocs, kernels, and performance
promotion.
