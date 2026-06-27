#!/usr/bin/env python3
"""Tests for inert end-to-end JetSpec round contract fixture."""

from __future__ import annotations

import copy
import json
import pathlib
import unittest

import jetspec_round_contract as round_contract


HERE = pathlib.Path(__file__).resolve().parent
FIXTURE_PATH = HERE / "fixtures/jetspec_round_smoke.json"
EXPECTED_PATH = HERE / "fixtures/jetspec_round_smoke.out.json"
PARITY_FIXTURE_PATH = HERE / "fixtures/jetspec_round_parity_smoke.json"
PARITY_EXPECTED_PATH = HERE / "fixtures/jetspec_round_parity_smoke.out.json"


class JetSpecRoundContractTests(unittest.TestCase):
    def test_round_fixture_matches_expected_output(self) -> None:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
        actual = round_contract.evaluate_fixture(data)

        self.assertEqual(actual, expected)

    def test_round_connects_tree_mask_accept_and_commit(self) -> None:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        actual = round_contract.evaluate_fixture(data)

        self.assertEqual(actual["tree"]["token_ids"], [7, 11, 12, 21, 22, 21, 22])
        self.assertEqual(actual["verify"]["verify_allowed_mask"][4], [True, True, True, True, False, False, True, False, False])
        self.assertEqual(actual["accept"]["accepted_path"], [0, 1, 4])
        self.assertEqual(actual["accept"]["accepted_tokens"], [11, 22])
        self.assertEqual(actual["commit"]["committed_append_tokens"], [11, 22, 999])
        self.assertEqual(actual["commit"]["post_committed_token_ids"], [5, 6, 7, 11, 22, 999])
        self.assertEqual(actual["commit"]["post_hidden_len"], len(actual["commit"]["post_committed_token_ids"]) - 1)
        self.assertFalse(actual["round_contract"]["correction_hidden_appended"])
        self.assertTrue(actual["round_contract"]["accepted_path_isolated_from_rejected_nodes"])

    def test_rejected_sentinel_rows_are_absent_from_committed_hidden_cache(self) -> None:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        actual = round_contract.evaluate_fixture(data)
        flat = [x for row in actual["commit"]["post_hidden_cache"] for x in row]

        for sentinel in data["rejected_sentinels"]:
            self.assertNotIn(float(sentinel), flat)
        self.assertEqual(actual["commit"]["discarded_node_indices"], [2, 3, 5, 6])

    def test_round_parity_fixture_matches_expected_output(self) -> None:
        data = json.loads(PARITY_FIXTURE_PATH.read_text(encoding="utf-8"))
        expected = json.loads(PARITY_EXPECTED_PATH.read_text(encoding="utf-8"))
        actual = round_contract.evaluate_fixture(data)

        self.assertEqual(actual, expected)
        self.assertEqual(actual["parity"]["baseline_greedy_token_ids"], [11, 22, 999])
        self.assertEqual(actual["parity"]["jetspec_committed_append_tokens"], [11, 22, 999])
        self.assertTrue(actual["parity"]["greedy_output_matches_baseline"])
        self.assertTrue(actual["round_contract"]["accepted_path_isolated_from_rejected_nodes"])
        self.assertTrue(actual["round_contract"]["rejected_sentinels_absent"])

    def test_expected_accept_mismatch_fails_closed(self) -> None:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        data["expected_accept"] = {"accepted_path": [0], "acceptance_length": 0, "correction_token": 11}

        with self.assertRaisesRegex(round_contract.RoundContractError, "accepted_path mismatch"):
            round_contract.evaluate_fixture(data)

    def test_rejected_sentinel_leak_fails_closed(self) -> None:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        # Force acceptance down the branch whose hidden row carries rejected sentinel 12000.
        data["target_argmax_by_node"] = [12, 22, 21, 777, 999, 555, 666]
        data.pop("expected_accept")

        with self.assertRaisesRegex(round_contract.RoundContractError, "sentinel leaked"):
            round_contract.evaluate_fixture(data)

    def test_baseline_greedy_mismatch_fails_closed(self) -> None:
        data = json.loads(PARITY_FIXTURE_PATH.read_text(encoding="utf-8"))
        data["baseline_greedy_token_ids"] = [11, 22, 123]

        with self.assertRaisesRegex(round_contract.RoundContractError, "baseline greedy output mismatch"):
            round_contract.evaluate_fixture(data)

    def test_pre_round_hidden_commit_invariant_is_enforced(self) -> None:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        data["pre_hidden_cache"] = [[5, 5, 5]]

        with self.assertRaisesRegex(Exception, "trail committed"):
            round_contract.evaluate_fixture(data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
