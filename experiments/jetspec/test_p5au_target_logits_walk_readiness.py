#!/usr/bin/env python3
"""Tests for the inert P5AU target-logits walk readiness packet."""

from __future__ import annotations

import json
import pathlib
import unittest

import jetspec_p5au_target_logits_walk_readiness as readiness
import validate_p5au_target_logits_walk_readiness as validator

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "jetspec_p5au_target_logits_walk_readiness_smoke.json"
EXPECTED_OUT = HERE / "fixtures" / "jetspec_p5au_target_logits_walk_readiness_smoke.out.json"


class P5AUTargetLogitsWalkReadinessTests(unittest.TestCase):
    def load_fixture(self) -> dict:
        return json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_smoke_fixture_matches_expected_output(self) -> None:
        actual = readiness.evaluate_fixture(self.load_fixture())
        expected = json.loads(EXPECTED_OUT.read_text(encoding="utf-8"))
        self.assertEqual(actual, expected)
        self.assertTrue(actual["ok"])
        self.assertEqual(actual["status"], readiness.STATUS)
        self.assertEqual(actual["target_logits_plan"]["planned_target_logits_rows"], 1)
        self.assertEqual(actual["target_logits_plan"]["actual_target_logits_rows_walked"], 0)
        self.assertFalse(actual["target_logits_walk_executed"])
        self.assertFalse(actual["draft_tokens_emitted"])

    def test_rejects_missing_p5at_predecessor(self) -> None:
        data = self.load_fixture()
        data["completed_chain"] = [item for item in data["completed_chain"] if item != "P5AT"]
        out = readiness.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn("missing required predecessor slices: ['P5AT']", out["errors"])

    def test_rejects_candidate_tree_mismatch(self) -> None:
        data = self.load_fixture()
        data["source_tree"]["real_tree_token_ids"] = [13, 111, 222]
        out = readiness.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn("source_tree.real_tree_token_ids[1:] must equal candidate_ids", out["errors"])

    def test_rejects_wrong_planned_target_rows(self) -> None:
        data = self.load_fixture()
        data["target_logits_plan"]["planned_target_logits_rows"] = 2
        data["target_logits_plan"]["planned_parent_nodes"] = [0, 1]
        out = readiness.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn("target_logits_plan.planned_target_logits_rows must be 1", out["errors"])
        self.assertIn("target_logits_plan.planned_parent_nodes must be [0]", out["errors"])

    def test_rejects_runtime_hook_and_approval(self) -> None:
        data = self.load_fixture()
        data["approvals"]["target_logits_walk_approved"] = True
        data["runtime_hooks"]["p5au_target_logits_walk_runtime_hook"] = True
        out = readiness.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn(
            "target_logits_walk_approved must remain false until explicit architecture approval",
            out["errors"],
        )
        self.assertIn("p5au_target_logits_walk_runtime_hook must remain false in P5AU", out["errors"])

    def test_rejects_actual_target_walk_or_accept(self) -> None:
        data = self.load_fixture()
        data["target_logits_plan"]["actual_target_logits_rows_walked"] = 1
        data["side_effect_counters"]["actual_target_logits_rows_walked"] = 1
        data["runtime_boundary"]["target_logits_walk_executed"] = True
        data["runtime_boundary"]["target_accept_walk_executed"] = True
        out = readiness.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn("target_logits_plan.actual_target_logits_rows_walked must be 0", out["errors"])
        self.assertIn("actual_target_logits_rows_walked must be 0", out["errors"])
        self.assertIn("target_logits_walk_executed must be false", out["errors"])
        self.assertIn("target_accept_walk_executed must be false", out["errors"])

    def test_rejects_commit_publish_or_draft_tokens(self) -> None:
        data = self.load_fixture()
        data["side_effect_counters"]["actual_committed_tokens"] = 1
        data["side_effect_counters"]["actual_publish_visible_state"] = 1
        data["runtime_boundary"]["draft_tokens_emitted"] = True
        out = readiness.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn("actual_committed_tokens must be 0", out["errors"])
        self.assertIn("actual_publish_visible_state must be 0", out["errors"])
        self.assertIn("draft_tokens_emitted must be false", out["errors"])

    def test_validator_passes(self) -> None:
        out = validator.validate_p5au_target_logits_walk_readiness()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5au_target_logits_walk_readiness_validated")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["target_logits_walk_executed"])
        self.assertFalse(out["production_source_hook_present"])
        self.assertEqual(out["forbidden_root_hits"], [])
        self.assertEqual(out["cmake_hits"], [])


if __name__ == "__main__":
    unittest.main()
