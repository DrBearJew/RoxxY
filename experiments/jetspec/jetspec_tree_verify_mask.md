# JetSpec tree-causal verify mask contract draft

Status: inert design fixture. This file is not included by CMake and no llama.cpp
runtime reads it.

Purpose: define the verify-time attention mask for causal-parallel tree nodes so
one target forward can score every tree node without siblings or rejected branches
seeing each other.

## Upstream reference semantics

Pinned upstream checkout:
`/home/mrtrent/.harness/tmp/jetspec-upstream-master`

Relevant source points:

- `jetspec/tree/_core/ancestor.py:34` `build_ancestor_matrix(tree)`
- `jetspec/inference_engine/engine.py:1072` builds the dense 4D fallback mask:
  prefix keys visible, tree keys filtered by ancestor matrix.
- `jetspec/inference_engine/engine.py:930` folds the same ancestor matrix into
  `qq_bias` for the paged tree-attention kernel path.
- `jetspec/inference_engine/engine.py:465` pads `qq_bias` to buckets with real rows
  isolated from pad rows and pad rows self-only.
- `jetspec/inference_engine/paged_tree_attn.py:236` applies `qq_bias` over query-key
  tree columns while prefix keys remain visible.

## Dense verify mask

For a tree with `N` nodes and committed prefix length `past_len`, target verify
uses query rows for the `N` tree nodes and key columns for:

```text
[prefix keys, tree node keys]
```

The bool visibility mask has shape:

```text
[N, past_len + N]
```

Rules:

1. Every query node can attend every committed prefix key.
2. Tree-key visibility is the ancestor matrix, including self.
3. A query node cannot attend siblings, sibling descendants, or its own descendants.
4. Root can attend root plus prefix, not child nodes.
5. A child can attend root, its parent chain, itself, and prefix.

Equivalent construction:

```text
allowed[:, :past_len] = true
allowed[:, past_len:] = build_ancestor_matrix(tree)
```

## `qq_bias` contract

Kernel paths may use an `(N, N)` additive bias over tree-node columns instead of a
full dense mask:

```text
qq_bias[row, col] = 0    if col is an ancestor of row or col == row
qq_bias[row, col] = -inf otherwise
```

Prefix keys are always visible and are not represented in `qq_bias`.

## Bucket padding contract

Compiled/graph verify may pad a real tree of `N` nodes to bucket size `B >= N`.
The padded `qq_bias` must preserve the real `(N,N)` block exactly and isolate pad
rows/columns:

- real row -> pad col: `-inf`
- pad row -> real col: `-inf`
- pad row -> other pad col: `-inf`
- pad row -> same pad col: `0`

The pad self-edge avoids all-masked softmax rows, but pad rows are never accepted
or committed.

## Rejected branch isolation

Rejected nodes are not known until after target logits are scored, so the verify
mask must be correct for all tree nodes before acceptance. Isolation is structural:

- sibling node `A` cannot attend sibling node `B`;
- a child on the accepted branch cannot attend a rejected sibling branch;
- a rejected sibling branch cannot attend the accepted branch except shared ancestors;
- a parent query cannot attend child/descendant keys.

This ensures rejected branch content cannot influence accepted-branch logits
through attention, and later hidden/KV rollback can safely discard rejected rows.

## Fail-closed checks

A future integration must reject or disable tree verify if any condition is true:

- The ancestor matrix differs from parent-before-child ancestry.
- Prefix keys are not visible to every real tree query.
- A sibling or sibling-descendant attention edge is present.
- A query can attend its descendant key.
- `qq_bias` uses anything other than `0` for allowed and `-inf` for forbidden.
- Bucket padding lets real rows attend pad columns or pad rows attend real rows.
- Pad rows have no self-edge and can trigger all-masked softmax behavior.

## Inert validation tools

- `tree_verify_mask.py`: stdlib helper for dense verify masks, `qq_bias`, bucket padding, and sibling-isolation checks.
- `test_tree_verify_mask.py`: unit tests for prefix visibility, ancestor-only tree visibility, sibling/rejected-branch isolation, additive bias, and bucket pad semantics.
- `fixtures/tree_verify_mask_smoke.json`: small two-branch tree with prefix length and bucket size.
- `fixtures/tree_verify_mask_smoke.out.json`: expected ancestor matrix, dense verify mask, `qq_bias`, and bucketed `qq_bias`.

These fixtures are contracts only. They do not modify llama.cpp attention masks or
JetSpec runtime routing.
