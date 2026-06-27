#!/usr/bin/env python3
"""Tests for the inert P5G JetSpec tree-runtime readiness contract."""

from __future__ import annotations

import copy
import json
import math
import pathlib
import unittest

import jetspec_tree_runtime_readiness as readiness
import validate_p5g_tree_runtime_readiness as validator


HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "jetspec_tree_runtime_readiness_smoke.json"
EXPECTED = HERE / "fixtures" / "jetspec_tree_runtime_readiness_smoke.out.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class P5GTreeRuntimeReadinessTests(unittest.TestCase):
    def test_smoke_fixture_matches_expected_output(self) -> None:
        result = readiness.evaluate_fixture(load_fixture())
        expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

        self.assertEqual(result, expected)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "tree_runtime_readiness_verified_not_executed")
        self.assertEqual(result["accept"]["accepted_path"], [0, 1, 4])
        self.assertEqual(result["gather"]["positions"], [100, 101, 104])
        self.assertFalse(result["readiness_boundary"]["runtime_supported"])
        self.assertTrue(result["readiness_boundary"]["no_draft_runtime_execution"])

    def test_rejects_topk_only_renormalization(self) -> None:
        data = load_fixture()
        data["topk_by_depth"][0][0]["p"] = 0.5
        data["topk_by_depth"][0][0]["logprob"] = math.log(0.5)
        data["topk_by_depth"][0][1]["p"] = 0.5
        data["topk_by_depth"][0][1]["logprob"] = math.log(0.5)

        with self.assertRaisesRegex(readiness.TreeRuntimeReadinessError, "renormalized over top-k"):
            readiness.evaluate_fixture(data)

    def test_rejects_gather_position_mismatch(self) -> None:
        data = load_fixture()
        data["expected_gather_positions"] = [100, 101, 105]

        with self.assertRaisesRegex(readiness.TreeRuntimeReadinessError, "gather_positions mismatch"):
            readiness.evaluate_fixture(data)

    def test_duplicate_child_token_overwrite_is_deterministic(self) -> None:
        data = {
            "root_token": 10,
            "budget": 3,
            "topk_logprob_source": "full_vocab_softmax",
            "renormalize_topk": False,
            "topk_by_depth": [[
                {"token": 11, "rank": 0, "p": 0.2, "logprob": math.log(0.2)},
                {"token": 11, "rank": 1, "p": 0.15, "logprob": math.log(0.15)},
            ]],
            "expected_tree": {
                "token_ids": [10, 11, 11],
                "parent_indices": [-1, 0, 0],
                "depth": [0, 1, 1],
                "rank": [-1, 0, 1],
                "cum_logprob": [0.0, math.log(0.2), math.log(0.15)],
            },
            "past_len": 1,
            "bucket_size": 4,
            "other_sequence_tree_cols": 1,
            "target_argmax_by_node": [11, 99, 42],
            "expected_accept": {"accepted_path": [0, 2], "acceptance_length": 1, "correction_token": 42},
            "pre_committed_token_ids": [10],
            "hidden_width": 2,
            "pre_hidden_cache": [],
            "node_hidden_rows": {
                "0": [10.0, 10.1],
                "1": [111.0, 111.1],
                "2": [11.0, 11.1],
            },
            "rejected_sentinels": [111.0],
            "kv_max_len": 20,
            "expected_gather_positions": [20, 22],
            "readiness_boundary": {
                "runtime_supported": False,
                "draft_head_graph_executed": False,
                "llama_context_runtime_instantiated": False,
                "draft_tokens_emitted": False,
                "kv_cache_mutated": False,
                "server_route_added": False,
            },
        }

        result = readiness.evaluate_fixture(data)
        policy = result["accept"]["duplicate_child_policy"]

        self.assertTrue(policy["deterministic"])
        self.assertEqual(policy["duplicates"][0]["selected_child"], 2)
        self.assertEqual(result["accept"]["accepted_path"], [0, 2])
        self.assertEqual(result["gather"]["positions"], [20, 22])

    def test_rejects_runtime_execution_boundary_crossing(self) -> None:
        data = load_fixture()
        data["readiness_boundary"] = copy.deepcopy(data["readiness_boundary"])
        data["readiness_boundary"]["draft_head_graph_executed"] = True

        with self.assertRaisesRegex(readiness.TreeRuntimeReadinessError, "draft_head_graph_executed=false"):
            readiness.evaluate_fixture(data)

    def test_rejects_runtime_supported_true(self) -> None:
        data = load_fixture()
        data["readiness_boundary"] = copy.deepcopy(data["readiness_boundary"])
        data["readiness_boundary"]["runtime_supported"] = True

        with self.assertRaisesRegex(readiness.TreeRuntimeReadinessError, "runtime_supported=false"):
            readiness.evaluate_fixture(data)

    def test_validator_passes(self) -> None:
        result = validator.validate_p5g_tree_runtime_readiness()

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "p5g_tree_runtime_readiness_validated")
        self.assertEqual(result["forbidden_root_hits"], [])
        self.assertEqual(result["cmake_hits"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
