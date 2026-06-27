#!/usr/bin/env python3
"""Tests for the inert P5I JetSpec tree-runtime approval packet."""

from __future__ import annotations

import copy
import json
import pathlib
import unittest

import tree_runtime_approval_matrix as matrix
import validate_p5i_tree_runtime_approval_packet as validator


HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "tree_runtime_approval_matrix_smoke.json"
EXPECTED = HERE / "fixtures" / "tree_runtime_approval_matrix_smoke.out.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class P5ITreeRuntimeApprovalPacketTests(unittest.TestCase):
    def test_smoke_fixture_matches_expected_output(self) -> None:
        result = matrix.evaluate_fixture(load_fixture())
        expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

        self.assertEqual(result, expected)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "tree_runtime_approval_packet_verified_not_executed")
        self.assertTrue(result["coverage"]["required_actions_present"])
        self.assertEqual(result["coverage"]["mapping_counts"]["missing_primitive"], 3)
        self.assertTrue(result["runtime_claims"]["no_runtime_execution"])
        self.assertEqual(result["production_paths_touched"], [])

    def test_rejects_missing_required_action(self) -> None:
        data = load_fixture()
        data["future_runtime_actions"] = [
            action for action in data["future_runtime_actions"] if action["id"] != "rollback_fail_closed_disable"
        ]

        with self.assertRaisesRegex(matrix.ApprovalMatrixError, "missing required future runtime actions"):
            matrix.evaluate_fixture(data)

    def test_rejects_production_path_touch(self) -> None:
        data = load_fixture()
        data["production_paths_touched"] = ["common/speculative.cpp"]

        with self.assertRaisesRegex(matrix.ApprovalMatrixError, "must not list touched production paths"):
            matrix.evaluate_fixture(data)

    def test_rejects_runtime_execution_claim(self) -> None:
        data = load_fixture()
        data["runtime_claims"] = copy.deepcopy(data["runtime_claims"])
        data["runtime_claims"]["draft_head_graph_executed"] = True

        with self.assertRaisesRegex(matrix.ApprovalMatrixError, "draft_head_graph_executed=true"):
            matrix.evaluate_fixture(data)

    def test_rejects_implicit_primitive(self) -> None:
        data = load_fixture()
        data["future_runtime_actions"] = copy.deepcopy(data["future_runtime_actions"])
        data["future_runtime_actions"][4]["future_primitive"] = "seq_cp"
        data["future_runtime_actions"][4]["mapping_status"] = "validated_by_p5h"

        with self.assertRaisesRegex(matrix.ApprovalMatrixError, "implicit or unsafe"):
            matrix.evaluate_fixture(data)

    def test_rejects_promotion_claim(self) -> None:
        data = load_fixture()
        data["future_runtime_actions"] = copy.deepcopy(data["future_runtime_actions"])
        data["future_runtime_actions"][0]["claims_promotion"] = True

        with self.assertRaisesRegex(matrix.ApprovalMatrixError, "claims_promotion must be false"):
            matrix.evaluate_fixture(data)

    def test_rejects_blocked_action_without_explicit_approval_reason(self) -> None:
        data = load_fixture()
        data["future_runtime_actions"] = copy.deepcopy(data["future_runtime_actions"])
        data["future_runtime_actions"][-1]["blocked_reason"] = "later"

        with self.assertRaisesRegex(matrix.ApprovalMatrixError, "explicit approval"):
            matrix.evaluate_fixture(data)

    def test_validator_passes(self) -> None:
        result = validator.validate_p5i_tree_runtime_approval_packet()

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "p5i_tree_runtime_approval_packet_validated")
        self.assertEqual(result["forbidden_root_hits"], [])
        self.assertEqual(result["cmake_hits"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
