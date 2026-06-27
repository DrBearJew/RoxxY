# JetSpec P5I tree-runtime approval packet

Status: inert approval packet only. This is not production tree-runtime approval,
not runtime execution, not a performance claim, and not compiled by CMake.

## Scope

P5I stays entirely under `experiments/jetspec/`. It converts P5G/P5H readiness
contracts into a machine-checked approval matrix for the first future production
tree-runtime slice. It does not edit `common/speculative.cpp`, `common/common.h`,
`src/`, `include/`, `tools/server/`, repository `tests/`, `examples/`, `pocs/`,
`ggml/src/`, top-level `CMakeLists.txt`, or production `.cmake` files.

P5I keeps the no-runtime boundary: no `llama_context` creation, no draft-head graph execution, no draft tokens emitted, no real KV mutation, no server route, no performance claim, no promotion claim, and `runtime_supported=false` remains required.

## Approval matrix contract

`tree_runtime_approval_matrix.py` verifies that every future runtime action is
mapped to one of these statuses:

- `validated_by_p5g`
- `validated_by_p5h`
- `missing_primitive`
- `blocked_pending_explicit_approval`

The required future runtime actions are explicit, not implied:

- tree build;
- verify mask;
- accept path;
- token commit;
- hidden/KV survivor commit;
- rejected branch discard;
- cross-sequence isolation;
- rollback/fail-closed disable.

P5I fails if any action touches production now, executes runtime now, claims
performance, claims promotion, omits explicit approval, uses an implicit
primitive such as `seq_cp`/`seq_rm`, or maps a missing production primitive
without stating `missing primitive`.

## Approval gates

The approval packet requires future evidence for:

- aggregate contracts;
- default build;
- disabled no-spec path;
- existing draft-MTP path;
- explicit JetSpec opt-in fail-closed behavior;
- correctness matrix;
- baseline benchmark comparison before any promotion discussion.

## Files

- `tree_runtime_approval_matrix.py`: stdlib evaluator for the P5I approval
  matrix.
- `fixtures/tree_runtime_approval_matrix_smoke.json`: deterministic approval
  matrix fixture.
- `fixtures/tree_runtime_approval_matrix_smoke.out.json`: expected P5I output.
- `validate_p5i_tree_runtime_approval_packet.py`: source/governance validator
  that checks the P5I files, fixture output, forbidden production path boundary,
  and CMake isolation.
- `test_p5i_tree_runtime_approval_packet.py`: unittest coverage for the smoke
  fixture, missing required actions, production path touch rejection, runtime
  execution claim rejection, implicit primitive rejection, promotion claim
  rejection, and validator success.

## Verification evidence

Representative commands:

```bash
python3 experiments/jetspec/tree_runtime_approval_matrix.py \
  --fixture experiments/jetspec/fixtures/tree_runtime_approval_matrix_smoke.json
python3 experiments/jetspec/validate_p5i_tree_runtime_approval_packet.py
python3 experiments/jetspec/test_p5i_tree_runtime_approval_packet.py
python3 experiments/jetspec/run_all_jetspec_contracts.py
```

Expected outcome: the fixture reports
`tree_runtime_approval_packet_verified_not_executed`, the validator passes,
aggregate contracts pass, and JetSpec remains explicit-opt-in/non-drafting.

## Blocked next work

Production tree runtime remains blocked until explicit tree-runtime approval. The
blocked surface includes draft-head graph execution, runtime tree verify, real
KV/hidden commit/rollback mutation, public API, server request behavior,
repository tests/examples/pocs, kernels, and performance promotion.
