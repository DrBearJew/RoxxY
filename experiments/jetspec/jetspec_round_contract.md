# JetSpec end-to-end round contract draft

Status: inert design fixture. This file is not included by CMake and no llama.cpp
runtime reads it.

Purpose: compose the prior staged contracts into one deterministic speculative
round:

1. build an accum-logp draft tree;
2. build the tree-causal verify mask;
3. perform greedy target acceptance;
4. commit accepted hidden rows and discard rejected rows;
5. prove the committed `[accepted draft tokens | correction]` matches a deterministic baseline greedy target path.

This is still not a runtime implementation. It is a fixture-level acceptance
contract for future compiled integration.

## Round inputs

The smoke fixture uses:

- root/anchor token: `7`
- committed tokens before the round: `[5, 6, 7]`
- hidden cache before the round: rows for `[5, 6]`, so hidden trails committed by one anchor
- accum-logp top-k draft tree with 7 nodes
- target argmax rows that accept path `[0, 1, 4]`
- correction token `999`
- rejected hidden rows containing unique sentinels

The generated tree is:

```text
node 0 token 7   parent -1
node 1 token 11  parent 0   accepted
node 2 token 12  parent 0   rejected sibling
node 3 token 21  parent 1   rejected sibling under accepted parent
node 4 token 22  parent 1   accepted
node 5 token 21  parent 2   rejected branch child
node 6 token 22  parent 2   rejected branch child
```

## Expected round output

Greedy acceptance:

```text
accepted_path        = [0, 1, 4]
accepted_draft_tokens = [11, 22]
correction_token      = 999
```

Committed token update:

```text
pre_committed  = [5, 6, 7]
append         = [11, 22, 999]
post_committed = [5, 6, 7, 11, 22, 999]
```

Hidden cache update:

```text
pre_hidden rows    = hidden([5, 6])
append hidden rows = hidden([root node 0, accepted node 1, accepted node 4])
post_hidden rows   = hidden([5, 6, 7, 11, 22])
```

No hidden row is appended for correction token `999` in the same round. It becomes
the next round's anchor.

## Verify mask expectations

For each tree query row:

- committed prefix columns are visible;
- tree columns use ancestor-only visibility;
- rejected sibling branches cannot influence accepted-branch logits;
- accepted branch nodes cannot attend rejected sibling branch nodes;
- parent queries cannot attend descendant keys.

The fixture checks node `4` can attend prefix, root, node `1`, and itself, but not
node `2`, node `3`, node `5`, or node `6`. The P4 parity fixture also checks every
accepted-path query is isolated from rejected tree nodes.

## Baseline greedy parity expectation

The P4 parity fixture supplies a deterministic baseline greedy target path:

```text
baseline_greedy_token_ids = [11, 22, 999]
```

The composed JetSpec round must append exactly the same token sequence:

```text
committed_append_tokens = [accepted draft tokens | correction] = [11, 22, 999]
```

A mismatch fails closed. This is a fixture-level proxy for the future runtime
requirement that JetSpec acceptance/rollback preserves target greedy output on
deterministic prompts.

## Rejected-row rollback expectation

Rejected node hidden rows include sentinel values:

```text
node 2 -> 12000
node 3 -> 21000
node 5 -> 31000
node 6 -> 32000
```

The round contract fails if any sentinel appears in the post-commit hidden cache.
This models the future requirement that rejected hidden/KV rows cannot be reached
by the next draft-head proposal.

## Fail-closed checks

The composed round must fail closed if any condition is true:

- built tree differs from the expected top-k/budget result;
- verify mask violates prefix or ancestor-only visibility;
- greedy accept result differs from the expected path/correction;
- pre-round hidden cache does not trail committed tokens by one anchor;
- accepted hidden rows are missing or wrong width;
- correction hidden is appended in the same round;
- rejected sentinel rows leak into post-commit hidden cache;
- accepted-path queries can attend rejected tree nodes;
- `[accepted draft tokens | correction]` differs from the deterministic baseline greedy target path.

## Inert validation tools

- `jetspec_round_contract.py`: composes tree build, verify mask, accept, hidden-cache commit, rollback checks, and baseline greedy-output parity.
- `test_jetspec_round_contract.py`: unit tests for the deterministic round, invariant wiring, sentinel rejection, accepted-path isolation, baseline mismatch, and fail-closed mismatch handling.
- `fixtures/jetspec_round_smoke.json`: round input fixture.
- `fixtures/jetspec_round_smoke.out.json`: complete expected output.
- `fixtures/jetspec_round_parity_smoke.json`: P4 parity fixture with baseline greedy tokens.
- `fixtures/jetspec_round_parity_smoke.out.json`: expected P4 parity output.

These fixtures are contracts only. They do not modify llama.cpp decoding or
`llama-server` behavior.
