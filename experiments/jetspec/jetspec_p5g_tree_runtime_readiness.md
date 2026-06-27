# JetSpec P5G tree-runtime readiness contract

Status: inert readiness contract only. This is not production tree-runtime
approval and it is not compiled by CMake.

## Scope

P5G stays entirely under `experiments/jetspec/`. It translates the upstream
JetSpec tree/verify/gather semantics into llama.cpp-facing contracts so a later
runtime slice has a precise boundary. It does not edit `common/speculative.cpp`,
`src/`, `include/`, `tools/server/`, repository `tests/`, `examples/`, `pocs/`,
`ggml/src/`, top-level `CMakeLists.txt`, or production `.cmake` files.

P5G is the safe next slice after P5F because P5F is still
`artifact_verified_not_runtime_executed` and
`loader_gate_verified_preflight_still_blocked`. P5G therefore keeps the same
safety boundary: no target/draft `llama_context` runtime instantiation, no draft-head graph execution, no draft tokens, no KV cache mutation, and no server route. This is not production tree-runtime approval.

## Upstream ABI translated

P5G records the following upstream-facing contracts:

- DraftTree ABI: `token_ids`, `parent_indices`, `depth`, and `num_nodes` are flat
  parent-before-child arrays; root parent is `-1`; root depth is `0`; every child
  parent index precedes the child; every child depth is parent depth plus one;
  `num_nodes <= budget`.
- Top-k build ABI: input logprobs come from full-vocab softmax, not top-k-only renormalization. The smoke fixture uses `topk_logprob_source=full_vocab_softmax`
  and requires each top-k probability row to sum below one.
- Accum-logp ABI: the deterministic fixture validates that upstream-style
  cumulative-logprob tree construction matches the expected `token_ids`,
  `parent_indices`, `depth`, `rank`, and `cum_logprob` arrays.
- Ancestor/mask ABI: prefix keys are visible to every tree query; tree-key
  visibility is ancestor-only with self included; siblings, descendants, rejected branches, padding columns, and other-sequence tree columns are hidden; additive
  mask entries are only `0` and `-inf`.
- Accept ABI: accepted_path is root-inclusive; acceptance_length excludes the root; `correction_token` is the target greedy token at the last accepted node;
  duplicate child tokens follow upstream CPU map semantics where the later child overwrites the earlier one deterministically.
- Commit/gather ABI: committed tokens are `[accepted draft tokens | correction]`;
  hidden/KV commit is [root | accepted] only; correction hidden is not appended
  in the same round; rejected tree nodes cannot be reached after commit; gather
  positions are exactly `max_len + accepted_path`.

## Files

- `jetspec_tree_runtime_readiness.py`: stdlib fixture evaluator for P5G.
- `fixtures/jetspec_tree_runtime_readiness_smoke.json`: deterministic tree,
  verify-mask, accept, commit, and gather fixture.
- `fixtures/jetspec_tree_runtime_readiness_smoke.out.json`: expected P5G output.
- `validate_p5g_tree_runtime_readiness.py`: source/governance validator that
  checks the P5G files, smoke fixture, CMake isolation, and forbidden production
  path boundary.
- `test_p5g_tree_runtime_readiness.py`: unittest coverage for the smoke fixture,
  top-k renormalization rejection, gather-position contract, duplicate-child
  deterministic overwrite, readiness-boundary failures, and the validator.

## Verification evidence

Representative commands:

```bash
python3 experiments/jetspec/jetspec_tree_runtime_readiness.py \
  --fixture experiments/jetspec/fixtures/jetspec_tree_runtime_readiness_smoke.json
python3 experiments/jetspec/validate_p5g_tree_runtime_readiness.py
python3 experiments/jetspec/test_p5g_tree_runtime_readiness.py
python3 experiments/jetspec/run_all_jetspec_contracts.py
```

Expected outcome: the fixture reports
`tree_runtime_readiness_verified_not_executed`, the validator passes, aggregate
contracts pass, and JetSpec remains explicit-opt-in/non-drafting.

## Blocked next work

Production tree runtime remains blocked until explicit tree-runtime approval. The
blocked surface includes draft-head graph execution, runtime tree construction,
tree verify masks in llama.cpp, KV/hidden commit/rollback mutation, server
request behavior, repository tests/examples/pocs, kernels, and performance
promotion.
