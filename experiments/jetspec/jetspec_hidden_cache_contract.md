# JetSpec committed hidden-cache / rollback contract draft

Status: inert design fixture. This file is not included by CMake and no llama.cpp
runtime reads it.

Purpose: define how target-hidden rows are committed after tree verification, so
accepted branches feed the draft head and rejected branches cannot leak into the
next proposal context.

## Upstream reference invariant

Pinned upstream checkout:
`/home/mrtrent/.harness/tmp/jetspec-upstream-master`

Relevant source points:

- `jetspec/inference_engine/engine.py:870` documents the round invariant:
  `cache_len == committed_len - 1 == target_hidden_len` when hidden taps are enabled.
- `jetspec/inference_engine/engine.py:1129` gathers/keeps prefix plus accepted path
  KV and drops rejected branches.
- `jetspec/inference_engine/engine.py:1138` appends `new_hidden[:, path_t, :]`.
- `jetspec/core/llm.py:590` mirrors the same `selected_hidden = verify_hidden.index_select(1, accepted_indices)` contract.

The important invariant is:

```text
before verify:  target_hidden_len == committed_len - 1
                the missing token is the current anchor / tree root

after verify:   append hidden rows for [root | accepted draft nodes]
                append committed tokens [accepted draft tokens | correction]
                target_hidden_len == committed_len - 1 again
```

The correction token is appended to committed tokens, but no correction hidden row
exists yet. It becomes the next round's anchor and enters the hidden/KV cache when
that next tree is verified.

## Commit rule

Given:

- `pre_hidden_cache`: persistent target-hidden rows before the round.
- `tree_token_ids`: flat tree node tokens with node `0` as root/anchor.
- `node_hidden_rows`: verify-produced hidden row for every real tree node.
- `accepted_path`: root-inclusive path from `tree_accept`, e.g. `[0, 1, 3]`.
- `correction_token`: target greedy/sample token at the final accepted node.

The commit step must:

1. Validate `accepted_path[0] == 0`.
2. Validate the tree root token equals the current committed anchor token.
3. Append hidden rows for exactly `accepted_path`, including root.
4. Append committed tokens for `tree_token_ids[accepted_path[1:]]` plus `correction_token`.
5. Discard hidden rows for all tree nodes not in `accepted_path`.
6. Do not append hidden for `correction_token` in the same round.
7. Preserve `target_hidden_len == committed_len - 1`.

## Why root hidden is appended

The draft tree root is the current anchor token, already present in `committed`,
but hidden/KV intentionally trail `committed` by one token before verify. Tree
verify runs the target over the root and candidate nodes, so root hidden becomes
available during this round and must be appended before accepted child-node rows.

This is why the hidden append count is `len(accepted_path)` while committed token
append count is `len(accepted_path)`: accepted child tokens are `path[1:]`, and the
last committed token is the correction.

## Rejected branch rollback

Rejected tree nodes may have target hidden rows and KV rows from verify. They are
transient only.

Required behavior:

- Hidden rows for rejected nodes are not copied into the committed hidden cache.
- KV rows for rejected nodes are dropped/ignored by the equivalent of gather/select.
- Logical-slot/window implementations may keep leased physical storage, but the
  committed window/map must not reference rejected nodes after commit.
- Rejected rows must not influence the next draft-head proposal.

A simple sentinel test should be possible: put a unique value in every rejected
hidden row, commit a different accepted path, then assert the sentinel never
appears in the persistent hidden cache.

## Fail-closed checks

A future integration must reject or disable the JetSpec path if any condition is
true:

- Pre-round hidden/KV length does not equal `committed_len - 1`.
- Tree root token does not equal the current committed anchor token.
- Accepted path is empty, not root-inclusive, duplicated, or out of tree range.
- Any accepted node is missing a target-hidden row.
- Any appended hidden row width differs from the target-hidden concat width.
- Correction token hidden is appended in the same round.
- Rejected node hidden rows remain referenced by the committed hidden cache.
- Rejected node KV slots remain reachable through the committed KV map/window.

## Inert validation tools

- `committed_hidden_cache.py`: stdlib model of hidden-cache commit/rollback.
- `test_committed_hidden_cache.py`: unit tests for accepted-row append, zero-accept
  rounds, `tree_accept` integration, fail-closed shape/path/invariant checks, and
  immutable copies.
- `fixtures/committed_hidden_cache_smoke.json`: accepted path plus rejected sentinel rows.
- `fixtures/committed_hidden_cache_smoke.out.json`: expected committed append tokens,
  hidden rows, discarded node indices, and invariants.

These fixtures are contracts only. They do not modify llama.cpp KV handling and do
not make `llama-server` run JetSpec.
