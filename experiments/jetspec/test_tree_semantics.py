#!/usr/bin/env python3
"""Self-tests for inert JetSpec tree semantics prototype."""

from __future__ import annotations

import unittest

from tree_semantics import DraftCandidate, DraftTree, build_accum_logp_tree, build_ancestor_matrix, tree_accept


class TreeSemanticsTests(unittest.TestCase):
    def test_ancestor_matrix_chain_and_sibling(self) -> None:
        tree = DraftTree(
            token_ids=[10, 20, 30, 40],
            parent_indices=[-1, 0, 1, 0],
            depth=[0, 1, 2, 1],
            cum_logprob=[0.0, -0.1, -0.2, -0.3],
        )
        self.assertEqual(
            build_ancestor_matrix(tree),
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, False, False, True],
            ],
        )

    def test_tree_accept_longest_matching_path(self) -> None:
        tree = DraftTree(
            token_ids=[101, 201, 202, 301, 302, 303],
            parent_indices=[-1, 0, 0, 1, 1, 2],
            depth=[0, 1, 1, 2, 2, 2],
            cum_logprob=[0.0, -0.1, -0.2, -0.3, -0.4, -0.5],
        )
        # root predicts child token 201 -> node 1; node 1 predicts child token
        # 302 -> node 4; node 4 has no matching child, so correction is 999.
        path, acc, correction = tree_accept(tree, [201, 302, 303, 0, 999, 0])
        self.assertEqual(path, [0, 1, 4])
        self.assertEqual(acc, 2)
        self.assertEqual(correction, 999)

    def test_tree_accept_duplicate_child_matches_upstream_cpu_overwrite(self) -> None:
        tree = DraftTree(
            token_ids=[10, 20, 20],
            parent_indices=[-1, 0, 0],
            depth=[0, 1, 1],
            cum_logprob=[0.0, -0.1, -0.2],
        )
        # Upstream CPU child-map construction assigns child_maps[parent][token]
        # in increasing node order, so the later duplicate node wins.
        path, acc, correction = tree_accept(tree, [20, 111, 222])
        self.assertEqual(path, [0, 2])
        self.assertEqual(acc, 1)
        self.assertEqual(correction, 222)

    def test_accum_logp_tree_budget_and_order(self) -> None:
        topk = [
            [DraftCandidate(11, -0.1, 0), DraftCandidate(12, -0.3, 1)],
            [DraftCandidate(21, -0.2, 0), DraftCandidate(22, -0.4, 1)],
            [DraftCandidate(31, -0.5, 0), DraftCandidate(32, -0.6, 1)],
        ]
        tree = build_accum_logp_tree(root_token=7, topk_by_depth=topk, budget=7)
        # Root expands first: 11, 12. Then node 11 has better cum logp than 12,
        # so its children appear next. Then node 12 and node 21 contend at -0.5;
        # insertion order keeps node 12 first, matching upstream heap tuple.
        self.assertEqual(tree.token_ids, [7, 11, 12, 21, 22, 21, 22])
        self.assertEqual(tree.parent_indices, [-1, 0, 0, 1, 1, 2, 2])
        self.assertEqual(tree.depth, [0, 1, 1, 2, 2, 2, 2])
        self.assertEqual(tree.rank, [-1, 0, 1, 0, 1, 0, 1])
        self.assertEqual(tree.max_depth, 2)
        self.assertEqual(tree.num_nodes, 7)

    def test_accum_logp_budget_one_is_root_only(self) -> None:
        tree = build_accum_logp_tree(
            root_token=5,
            topk_by_depth=[[DraftCandidate(6, -0.1)]],
            budget=1,
        )
        self.assertEqual(tree.token_ids, [5])
        self.assertEqual(tree.parent_indices, [-1])
        self.assertEqual(tree.depth, [0])
        self.assertEqual(tree.ancestor, [[True]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
