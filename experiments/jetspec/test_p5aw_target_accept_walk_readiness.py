#!/usr/bin/env python3
"""Tests for the inert P5AW target-accept walk readiness packet."""

from __future__ import annotations

import json
import pathlib
import unittest

import jetspec_p5aw_target_accept_walk_readiness as readiness
import validate_p5aw_target_accept_walk_readiness as validator

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "jetspec_p5aw_target_accept_walk_readiness_smoke.json"
EXPECTED_OUT = HERE / "fixtures" / "jetspec_p5aw_target_accept_walk_readiness_smoke.out.json"


class P5AWTargetAcceptWalkReadinessTests(unittest.TestCase):
    def load_fixture(self) -> dict:
        return json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_smoke_fixture_matches_expected_output(self) -> None:
        actual = readiness.evaluate_fixture(self.load_fixture())
        expected = json.loads(EXPECTED_OUT.read_text(encoding="utf-8"))
        self.assertEqual(actual, expected)
        self.assertTrue(actual["ok"])
        self.assertEqual(actual["status"], readiness.STATUS)
        self.assertEqual(actual["p5av_target_logits_walk_evidence"]["actual_target_logits_rows_walked"], 1)
        self.assertEqual(actual["target_accept_plan"]["planned_target_accept_steps"], 1)
        self.assertEqual(actual["target_accept_plan"]["actual_target_accept_steps"], 0)
        self.assertFalse(actual["target_accept_walk_executed"])
        self.assertFalse(actual["draft_tokens_emitted"])

    def test_rejects_missing_p5av_predecessor(self) -> None:
        data = self.load_fixture()
        data["completed_chain"] = [item for item in data["completed_chain"] if item != "P5AV"]
        out = readiness.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn("missing required predecessor slices: ['P5AV']", out["errors"])

    def test_rejects_bad_p5av_evidence(self) -> None:
        data = self.load_fixture()
        data["p5av_target_logits_walk_evidence"]["actual_target_logits_rows_walked"] = 0
        data["p5av_target_logits_walk_evidence"]["trace_status"] = "wrong"
        data["p5av_target_logits_walk_evidence"]["target_candidate_logits"] = [1.0]
        out = readiness.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn(
            "p5av_target_logits_walk_evidence.actual_target_logits_rows_walked must be 1",
            out["errors"],
        )
        self.assertIn(
            "p5av_target_logits_walk_evidence.trace_status must be p5av_target_logits_walk_canary_trace_contract_verified",
            out["errors"],
        )
        self.assertIn(
            "p5av_target_logits_walk_evidence.target_candidate_logits must contain two scores",
            out["errors"],
        )

    def test_rejects_candidate_tree_mismatch(self) -> None:
        data = self.load_fixture()
        data["source_tree"]["real_tree_token_ids"] = [13, 111, 222]
        data["p5av_target_logits_walk_evidence"]["candidate_ids"] = [111, 222]
        data["target_accept_plan"]["planned_accept_candidate_ids"] = [111, 222]
        out = readiness.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn("source_tree.real_tree_token_ids[1:] must equal candidate_ids", out["errors"])
        self.assertIn("p5av_target_logits_walk_evidence.candidate_ids must equal source_tree.candidate_ids", out["errors"])
        self.assertIn("target_accept_plan.planned_accept_candidate_ids must equal source_tree.candidate_ids", out["errors"])

    def test_rejects_wrong_planned_accept_steps(self) -> None:
        data = self.load_fixture()
        data["target_accept_plan"]["planned_target_accept_steps"] = 2
        data["target_accept_plan"]["planned_accept_parent_nodes"] = [0, 1]
        data["target_accept_plan"]["planned_accept_rule"] = "sampled_accept"
        out = readiness.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn("target_accept_plan.planned_target_accept_steps must be 1", out["errors"])
        self.assertIn("target_accept_plan.planned_accept_parent_nodes must be [0]", out["errors"])
        self.assertIn("target_accept_plan.planned_accept_rule must be greedy_target_argmax_child_match_or_correction", out["errors"])

    def test_rejects_runtime_hook_and_approval(self) -> None:
        data = self.load_fixture()
        data["approvals"]["target_accept_walk_approved"] = True
        data["runtime_hooks"]["p5aw_target_accept_walk_runtime_hook"] = True
        out = readiness.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn(
            "target_accept_walk_approved must remain false until explicit architecture approval",
            out["errors"],
        )
        self.assertIn("p5aw_target_accept_walk_runtime_hook must remain false in P5AW", out["errors"])

    def test_rejects_actual_accept_commit_publish_or_draft_tokens(self) -> None:
        data = self.load_fixture()
        data["target_accept_plan"]["actual_target_accept_steps"] = 1
        data["target_accept_plan"]["actual_accepted_nodes"] = 1
        data["side_effect_counters"]["actual_target_accept_steps"] = 1
        data["side_effect_counters"]["actual_committed_tokens"] = 1
        data["side_effect_counters"]["actual_publish_visible_state"] = 1
        data["runtime_boundary"]["target_accept_walk_executed"] = True
        data["runtime_boundary"]["draft_tokens_emitted"] = True
        out = readiness.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn("target_accept_plan.actual_target_accept_steps must be 0", out["errors"])
        self.assertIn("target_accept_plan.actual_accepted_nodes must be 0", out["errors"])
        self.assertIn("actual_target_accept_steps must be 0", out["errors"])
        self.assertIn("actual_committed_tokens must be 0", out["errors"])
        self.assertIn("actual_publish_visible_state must be 0", out["errors"])
        self.assertIn("target_accept_walk_executed must be false", out["errors"])
        self.assertIn("draft_tokens_emitted must be false", out["errors"])

    def test_validator_passes(self) -> None:
        out = validator.validate_p5aw_target_accept_walk_readiness()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5aw_target_accept_walk_readiness_validated")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["target_accept_walk_executed"])
        self.assertFalse(out["production_source_hook_present"])
        self.assertEqual(out["forbidden_root_hits"], [])
        self.assertEqual(out["cmake_hits"], [])


if __name__ == "__main__":
    unittest.main()
