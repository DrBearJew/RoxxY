# JetSpec P5J KV/runtime primitive audit

Status: inert primitive audit only. This is not production tree-runtime approval,
not runtime execution, not a performance claim, and not compiled by CMake.

## Scope

P5J stays entirely under `experiments/jetspec/`. It reads source text to classify
the P5I actions that remain `missing_primitive`: hidden/KV survivor commit,
rejected branch discard, and cross-sequence isolation. It does not edit
`common/speculative.cpp`, `common/common.h`, `src/`, `include/`, `tools/server/`,
repository `tests/`, `examples/`, `pocs/`, `ggml/src/`, top-level
`CMakeLists.txt`, or production `.cmake` files.

P5J keeps the no-runtime boundary: no `llama_context` creation, no draft-head graph execution, no draft tokens emitted, no real KV mutation, no server route, no performance claim, no promotion claim, and `runtime_supported=false` remains required.

## Audit contract

`kv_primitive_audit.py` scans `src/llama-kv-cache.h` and
`src/llama-kv-cache.cpp` as read-only source text and records locations for
existing primitives such as `seq_rm`, `seq_cp`, `seq_import_physical`,
`seq_keep`, `find_slot`, and `apply_ubatch`.

The audit then classifies each P5I missing action as one of:

- `exact_existing_primitive_candidate` with source location;
- `exact_missing_primitive`;
- `blocked_pending_explicit_approval`.

The smoke fixture intentionally keeps all three P5I actions as
`exact_missing_primitive`:

- hidden/KV survivor commit needs accepted-path physical gather/compact from
  transient tree slots to the committed tail;
- rejected branch discard needs tree-node ownership discard that makes rejected
  transient slots unreachable;
- cross-sequence isolation needs explicit proof that other-sequence slots remain
  unchanged after gather/discard.

P5J fails if a future fixture silently maps these actions to `seq_cp`, `seq_rm`,
or `seq_import_physical`. Those helpers exist, but they are not an exact
JetSpec accepted-path tree gather/compact/discard ownership primitive; not an exact JetSpec accepted-path tree gather/compact/discard ownership primitive.

## Files

- `kv_primitive_audit.py`: stdlib read-only source scanner and primitive
  classifier.
- `fixtures/kv_primitive_audit_smoke.json`: deterministic P5J audit fixture.
- `fixtures/kv_primitive_audit_smoke.out.json`: expected P5J audit output.
- `validate_p5j_kv_primitive_audit.py`: source/governance validator that checks
  the P5J files, fixture output, forbidden production path boundary, and CMake
  isolation.
- `test_p5j_kv_primitive_audit.py`: unittest coverage for the smoke fixture,
  missing action rejection, implicit primitive mapping rejection, missing source
  symbol rejection, runtime-boundary failures, production touch rejection, and
  validator success.

## Verification evidence

Representative commands:

```bash
python3 experiments/jetspec/kv_primitive_audit.py \
  --fixture experiments/jetspec/fixtures/kv_primitive_audit_smoke.json
python3 experiments/jetspec/validate_p5j_kv_primitive_audit.py
python3 experiments/jetspec/test_p5j_kv_primitive_audit.py
python3 experiments/jetspec/run_all_jetspec_contracts.py
```

Expected outcome: the fixture reports `kv_primitive_audit_verified_not_executed`,
the validator passes, aggregate contracts pass, and JetSpec remains
explicit-opt-in/non-drafting.

## Blocked next work

Production tree runtime remains blocked until explicit tree-runtime approval and
until missing KV/hidden ownership primitives are designed or exact existing
primitives are proven by source-backed evidence. The blocked surface includes
draft-head graph execution, runtime tree verify, real KV/hidden commit/rollback
mutation, public API, server request behavior, repository tests/examples/pocs,
kernels, and performance promotion.
