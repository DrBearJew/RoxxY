#!/usr/bin/env python3
"""Tests for inert JetSpec committed target-hidden cache contract."""

from __future__ import annotations

import json
import pathlib
import unittest

import committed_hidden_cache as cache
from tree_semantics import DraftTree, tree_accept


HERE = pathlib.Path(__file__).resolve().parent
FIXTURE_PATH = HERE / "fixtures/committed_hidden_cache_smoke.json"
EXPECTED_PATH = HERE / "fixtures/committed_hidden_cache_smoke.out.json"


class CommittedHiddenCacheTests(unittest.TestCase):
    def test_fixture_appends_only_root_and_accepted_nodes(self) -> None:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
        actual = cache.evaluate_fixture(data)

        for key, value in expected.items():
            self.assertEqual(actual[key], value, key)
        flattened = [x for row in actual["post_hidden_cache"] for x in row]
        self.assertNotIn(9999.0, flattened)
        self.assertNotIn(8888.0, flattened)
        self.assertNotIn(7777.0, flattened)
        self.assertTrue(actual["invariants"]["pre_hidden_trails_committed_by_one"])
        self.assertTrue(actual["invariants"]["post_hidden_trails_committed_by_one"])
        self.assertTrue(actual["invariants"]["correction_has_no_hidden_until_next_round"])

    def test_zero_accept_appends_root_hidden_and_correction_token_only(self) -> None:
        result = cache.commit_tree_hidden_round(
            pre_committed_token_ids=[10, 11],
            pre_hidden_cache=[[10.0, 10.0]],
            tree_token_ids=[11, 21, 22],
            node_hidden_rows={0: [11.0, 11.5], 1: [999.0, 999.0], 2: [888.0, 888.0]},
            accepted_path=[0],
            correction_token=77,
            width=2,
        )

        self.assertEqual(result.accepted_draft_tokens, ())
        self.assertEqual(result.committed_append_tokens, (77,))
        self.assertEqual(result.appended_node_indices, (0,))
        self.assertEqual(result.discarded_node_indices, (1, 2))
        self.assertEqual(result.post_hidden_cache, ((10.0, 10.0), (11.0, 11.5)))
        self.assertEqual(result.post_hidden_len, result.post_committed_len - 1)

    def test_tree_accept_output_feeds_cache_commit_contract(self) -> None:
        tree = DraftTree(
            token_ids=[103, 201, 202, 301],
            parent_indices=[-1, 0, 0, 1],
            depth=[0, 1, 1, 2],
            cum_logprob=[0.0, -0.1, -0.2, -0.3],
        )
        accepted_path, acc, correction = tree_accept(tree, [201, 301, 999, 555])
        result = cache.commit_tree_hidden_round(
            pre_committed_token_ids=[101, 102, 103],
            pre_hidden_cache=[[1.0], [2.0]],
            tree_token_ids=tree.token_ids,
            node_hidden_rows={0: [103.0], 1: [201.0], 2: [202.0], 3: [301.0]},
            accepted_path=accepted_path,
            correction_token=correction,
            width=1,
        )

        self.assertEqual(acc, 2)
        self.assertEqual(accepted_path, [0, 1, 3])
        self.assertEqual(correction, 555)
        self.assertEqual(result.accepted_draft_tokens, (201, 301))
        self.assertEqual(result.committed_append_tokens, (201, 301, 555))
        self.assertEqual(result.appended_node_indices, (0, 1, 3))
        self.assertEqual(result.discarded_node_indices, (2,))

    def test_missing_accepted_hidden_row_fails_closed(self) -> None:
        with self.assertRaisesRegex(cache.HiddenCacheContractError, "missing hidden rows"):
            cache.commit_tree_hidden_round(
                pre_committed_token_ids=[1, 2],
                pre_hidden_cache=[[1.0]],
                tree_token_ids=[2, 3],
                node_hidden_rows={0: [2.0]},
                accepted_path=[0, 1],
                correction_token=4,
                width=1,
            )

    def test_wrong_hidden_width_fails_closed(self) -> None:
        with self.assertRaisesRegex(cache.HiddenCacheContractError, "width mismatch"):
            cache.commit_tree_hidden_round(
                pre_committed_token_ids=[1, 2],
                pre_hidden_cache=[[1.0]],
                tree_token_ids=[2],
                node_hidden_rows={0: [2.0, 2.0]},
                accepted_path=[0],
                correction_token=4,
                width=1,
            )

    def test_path_must_be_root_inclusive(self) -> None:
        with self.assertRaisesRegex(cache.HiddenCacheContractError, "root-inclusive"):
            cache.commit_tree_hidden_round(
                pre_committed_token_ids=[1, 2],
                pre_hidden_cache=[[1.0]],
                tree_token_ids=[2, 3],
                node_hidden_rows={0: [2.0], 1: [3.0]},
                accepted_path=[1],
                correction_token=4,
                width=1,
            )

    def test_pre_round_hidden_must_trail_committed_by_one(self) -> None:
        with self.assertRaisesRegex(cache.HiddenCacheContractError, "trail committed"):
            cache.commit_tree_hidden_round(
                pre_committed_token_ids=[1, 2, 3],
                pre_hidden_cache=[[1.0]],
                tree_token_ids=[3],
                node_hidden_rows={0: [3.0]},
                accepted_path=[0],
                correction_token=4,
                width=1,
            )

    def test_root_token_must_match_committed_anchor(self) -> None:
        with self.assertRaisesRegex(cache.HiddenCacheContractError, "committed anchor"):
            cache.commit_tree_hidden_round(
                pre_committed_token_ids=[1, 2],
                pre_hidden_cache=[[1.0]],
                tree_token_ids=[99],
                node_hidden_rows={0: [99.0]},
                accepted_path=[0],
                correction_token=4,
                width=1,
            )

    def test_post_cache_is_immutable_copy(self) -> None:
        pre = [[1.0]]
        rows = {0: [2.0]}
        result = cache.commit_tree_hidden_round(
            pre_committed_token_ids=[1, 2],
            pre_hidden_cache=pre,
            tree_token_ids=[2],
            node_hidden_rows=rows,
            accepted_path=[0],
            correction_token=3,
            width=1,
        )
        pre[0][0] = 999.0
        rows[0][0] = 888.0

        self.assertEqual(result.post_hidden_cache, ((1.0,), (2.0,)))
        with self.assertRaises(TypeError):
            result.post_hidden_cache[0][0] = 5.0  # type: ignore[index]


if __name__ == "__main__":
    unittest.main(verbosity=2)
