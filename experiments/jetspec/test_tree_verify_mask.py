#!/usr/bin/env python3
"""Tests for inert JetSpec tree-causal verify mask helpers."""

from __future__ import annotations

import json
import pathlib
import unittest

import tree_verify_mask as mask
from tree_semantics import DraftTree


HERE = pathlib.Path(__file__).resolve().parent
FIXTURE_PATH = HERE / "fixtures/tree_verify_mask_smoke.json"
EXPECTED_PATH = HERE / "fixtures/tree_verify_mask_smoke.out.json"


class TreeVerifyMaskTests(unittest.TestCase):
    def test_fixture_matches_expected_ancestor_and_verify_mask(self) -> None:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
        actual = mask.evaluate_fixture(data)

        self.assertTrue(actual["ok"], actual["errors"])
        for key, value in expected.items():
            self.assertEqual(actual[key], value, key)

    def test_prefix_columns_are_visible_to_every_tree_query(self) -> None:
        tree = DraftTree(
            token_ids=[100, 201, 202],
            parent_indices=[-1, 0, 0],
            depth=[0, 1, 1],
            cum_logprob=[0.0, -0.1, -0.2],
        )
        allowed = mask.build_verify_allowed_mask(tree, past_len=3)

        self.assertEqual([row[:3] for row in allowed], [[True, True, True]] * 3)

    def test_siblings_and_rejected_branch_nodes_cannot_attend_each_other(self) -> None:
        tree = mask.tree_from_obj(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))

        self.assertFalse(mask.can_attend_tree_node(tree, 1, 2))  # sibling -> sibling
        self.assertFalse(mask.can_attend_tree_node(tree, 2, 1))  # sibling -> sibling
        self.assertFalse(mask.can_attend_tree_node(tree, 3, 2))  # accepted branch child -> rejected sibling branch
        self.assertFalse(mask.can_attend_tree_node(tree, 4, 1))  # rejected branch child -> accepted sibling branch
        self.assertFalse(mask.can_attend_tree_node(tree, 1, 3))  # parent query cannot attend descendant key
        self.assertTrue(mask.can_attend_tree_node(tree, 3, 1))   # child query can attend accepted parent key
        mask.assert_sibling_isolation(tree)

    def test_qq_bias_uses_zero_for_allowed_and_neg_inf_for_forbidden(self) -> None:
        tree = mask.tree_from_obj(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))
        qq = mask.build_qq_bias(tree)

        self.assertEqual(qq[3][0], "0")
        self.assertEqual(qq[3][1], "0")
        self.assertEqual(qq[3][3], "0")
        self.assertEqual(qq[3][2], "-inf")
        self.assertEqual(qq[3][4], "-inf")

    def test_bucket_padding_keeps_real_rows_isolated_from_pad_rows(self) -> None:
        tree = mask.tree_from_obj(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))
        qq = mask.build_qq_bias(tree)
        padded = mask.pad_qq_bias_to_bucket(qq, 7)

        self.assertEqual([row[:5] for row in padded[:5]], qq)
        for row in range(5):
            self.assertEqual(padded[row][5:], ["-inf", "-inf"])
        self.assertEqual(padded[5], ["-inf", "-inf", "-inf", "-inf", "-inf", "0", "-inf"])
        self.assertEqual(padded[6], ["-inf", "-inf", "-inf", "-inf", "-inf", "-inf", "0"])

    def test_invalid_bucket_size_fails_closed(self) -> None:
        tree = mask.tree_from_obj(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))
        with self.assertRaisesRegex(mask.TreeVerifyMaskError, "must be >="):
            mask.pad_qq_bias_to_bucket(mask.build_qq_bias(tree), 4)

    def test_negative_past_len_fails_closed(self) -> None:
        tree = mask.tree_from_obj(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))
        with self.assertRaisesRegex(mask.TreeVerifyMaskError, "past_len"):
            mask.build_verify_allowed_mask(tree, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
