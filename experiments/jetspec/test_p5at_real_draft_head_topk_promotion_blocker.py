#!/usr/bin/env python3
"""Tests for the inert P5AT real draft-head top-k promotion blocker audit."""

from __future__ import annotations

import copy
import json
import pathlib
import unittest

import jetspec_p5at_real_draft_head_topk_promotion_blocker as blocker
import validate_p5at_real_draft_head_topk_promotion_blocker as validator

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "jetspec_p5at_real_draft_head_topk_promotion_blocker_smoke.json"
EXPECTED_OUT = HERE / "fixtures" / "jetspec_p5at_real_draft_head_topk_promotion_blocker_smoke.out.json"


class P5ATRealDraftHeadTopKPromotionBlockerTests(unittest.TestCase):
    def load_fixture(self) -> dict:
        return json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_smoke_fixture_matches_expected_output(self) -> None:
        actual = blocker.evaluate_fixture(self.load_fixture())
        expected = json.loads(EXPECTED_OUT.read_text(encoding="utf-8"))
        self.assertEqual(actual, expected)
        self.assertTrue(actual["ok"])
        self.assertEqual(actual["status"], blocker.STATUS)
        self.assertEqual(actual["promotion_result"], "blocked_pending_explicit_approval")
        self.assertFalse(actual["runtime_executed"])
        self.assertFalse(actual["draft_tokens_emitted"])

    def test_rejects_missing_p5as_predecessor(self) -> None:
        data = self.load_fixture()
        data["completed_chain"] = [item for item in data["completed_chain"] if item != "P5AS"]
        out = blocker.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn("missing required predecessor slices: ['P5AS']", out["errors"])

    def test_rejects_product_runtime_hook_approval(self) -> None:
        data = self.load_fixture()
        data["approvals"]["product_runtime_hooks_approved"] = True
        out = blocker.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn(
            "product_runtime_hooks_approved must remain false until explicit architecture approval",
            out["errors"],
        )

    def test_rejects_p5t_p5w_p5ad_runtime_hooks(self) -> None:
        for key in (
            "p5t_token_commit_runtime_product_hook",
            "p5w_publish_gate_runtime_product_hook",
            "p5ad_root_publish_noop_product_hook_reuse",
        ):
            with self.subTest(key=key):
                data = self.load_fixture()
                data["runtime_hooks"][key] = True
                out = blocker.evaluate_fixture(data)
                self.assertFalse(out["ok"])
                self.assertIn(f"{key} must remain false in P5AT", out["errors"])

    def test_rejects_side_effects_and_draft_tokens(self) -> None:
        data = self.load_fixture()
        data["side_effect_counters"]["actual_committed_tokens"] = 1
        data["side_effect_counters"]["actual_publish_visible_state"] = 1
        data["runtime_boundary"]["draft_tokens_emitted"] = True
        out = blocker.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn("actual_committed_tokens must be 0", out["errors"])
        self.assertIn("actual_publish_visible_state must be 0", out["errors"])
        self.assertIn("draft_tokens_emitted must be false", out["errors"])

    def test_rejects_target_walk_execution(self) -> None:
        data = self.load_fixture()
        data["side_effect_counters"]["actual_target_logits_rows_walked"] = 1
        data["runtime_boundary"]["target_logits_walk_executed"] = True
        data["runtime_boundary"]["target_accept_walk_executed"] = True
        out = blocker.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn("actual_target_logits_rows_walked must be 0", out["errors"])
        self.assertIn("target_logits_walk_executed must be false", out["errors"])
        self.assertIn("target_accept_walk_executed must be false", out["errors"])

    def test_rejects_wrong_allowed_path(self) -> None:
        data = self.load_fixture()
        data["allowed_paths"] = ["common"]
        out = blocker.evaluate_fixture(data)
        self.assertFalse(out["ok"])
        self.assertIn("allowed_paths must be exactly ['experiments/jetspec']", out["errors"])

    def test_validator_passes(self) -> None:
        out = validator.validate_p5at_real_draft_head_topk_promotion_blocker()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5at_real_draft_head_topk_promotion_blocker_validated")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["production_source_hook_present"])
        self.assertEqual(out["forbidden_root_hits"], [])
        self.assertEqual(out["cmake_hits"], [])


if __name__ == "__main__":
    unittest.main()
