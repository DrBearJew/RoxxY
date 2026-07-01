#!/usr/bin/env python3
"""Tests for P5AV target-logits walk canary trace probe and validator."""

from __future__ import annotations

import tempfile
import pathlib
import unittest

import probe_p5av_real_draft_head_topk_target_logits_walk_canary_trace as probe
import validate_p5av_real_draft_head_topk_target_logits_walk_canary_runtime as validator


class P5AVRealDraftHeadTopKTargetLogitsWalkCanaryTraceProbeTests(unittest.TestCase):
    def test_probe_passes_without_live_log(self) -> None:
        out = probe.probe_p5av_real_draft_head_topk_target_logits_walk_canary_trace()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5av_target_logits_walk_canary_trace_contract_verified")
        self.assertFalse(out["accept_executed"])
        self.assertFalse(out["commit_executed"])
        self.assertFalse(out["publish_executed"])
        self.assertFalse(out["kv_mutated"])
        self.assertFalse(out["draft_tokens_emitted"])

    def test_trace_line_requires_target_logits_walk_and_zero_side_effects(self) -> None:
        errors = probe.validate_trace_line(probe.SELF_TEST_TRACE)
        self.assertEqual(errors, [])
        self.assertIn("actual_target_logits_rows_walked=1", probe.SELF_TEST_TRACE)
        self.assertIn("actual_target_accept_steps=0", probe.SELF_TEST_TRACE)
        self.assertIn("actual_committed_tokens=0", probe.SELF_TEST_TRACE)
        self.assertIn("actual_publish_visible_state=0", probe.SELF_TEST_TRACE)

    def test_trace_line_rejects_no_target_logits_walk_marker(self) -> None:
        bad = probe.SELF_TEST_TRACE + " no_target_logits_walk=1"
        errors = probe.validate_trace_line(bad)
        self.assertIn("forbidden P5AV trace token present: no_target_logits_walk=1", errors)

    def test_trace_line_rejects_missing_or_zero_target_walk(self) -> None:
        bad = probe.SELF_TEST_TRACE.replace("actual_target_logits_rows_walked=1", "actual_target_logits_rows_walked=0")
        errors = probe.validate_trace_line(bad)
        self.assertIn("forbidden P5AV trace token present: actual_target_logits_rows_walked=0", errors)
        self.assertIn("P5AV trace actual_target_logits_rows_walked must be 1", errors)
        missing = probe.SELF_TEST_TRACE.replace("planned_target_logits_rows=1", "planned_target_logits_rows=2")
        self.assertIn("P5AV trace planned_target_logits_rows must be 1", probe.validate_trace_line(missing))

    def test_trace_line_rejects_accept_commit_publish_and_generation(self) -> None:
        bad = (
            probe.SELF_TEST_TRACE
            .replace("actual_target_accept_steps=0", "actual_target_accept_steps=1")
            .replace("actual_accepted_nodes=0", "actual_accepted_nodes=1")
            .replace("actual_committed_tokens=0", "actual_committed_tokens=1")
            .replace("actual_publish_visible_state=0", "actual_publish_visible_state=1")
            + " #gen drafts = 1 #gen tokens = 1"
        )
        errors = probe.validate_trace_line(bad)
        joined = "\n".join(errors)
        self.assertIn("actual_target_accept_steps=1", joined)
        self.assertIn("actual_accepted_nodes=1", joined)
        self.assertIn("actual_committed_tokens=1", joined)
        self.assertIn("actual_publish_visible_state=1", joined)
        self.assertIn("#gen drafts = 1", joined)
        self.assertIn("#gen tokens = 1", joined)

    def test_trace_line_rejects_bad_candidate_lists(self) -> None:
        bad_ids = probe.SELF_TEST_TRACE.replace("candidate_ids=[92637,2054]", "candidate_ids=[42,42]")
        self.assertIn(
            "P5AV trace candidate_ids must contain two distinct non-negative ids",
            probe.validate_trace_line(bad_ids),
        )
        bad_scores = probe.SELF_TEST_TRACE.replace("target_candidate_logits=[12.5,11.25]", "target_candidate_logits=[bad,11.25]")
        self.assertIn(
            "P5AV trace target_candidate_logits must contain two scores",
            probe.validate_trace_line(bad_scores),
        )
        bad_nodes = probe.SELF_TEST_TRACE.replace("planned_candidate_nodes=[1,2]", "planned_candidate_nodes=[2,3]")
        self.assertIn("P5AV trace planned_candidate_nodes must be [1,2]", probe.validate_trace_line(bad_nodes))

    def test_trace_log_file_mode_requires_p5av_line(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "server.log"
            path.write_text("no p5av trace here\n", encoding="utf-8")
            out = probe.probe_p5av_real_draft_head_topk_target_logits_walk_canary_trace(path)
        self.assertFalse(out["ok"])
        self.assertIn(
            "trace log does not contain draft-jetspec p5av_real_draft_head_topk_target_logits_walk_canary_runtime",
            out["errors"],
        )

    def test_validator_passes(self) -> None:
        out = validator.validate_p5av_real_draft_head_topk_target_logits_walk_canary_runtime()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5av_real_draft_head_topk_target_logits_walk_canary_contract_wiring_validated")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["published_visible_state"])
        self.assertFalse(out["kv_mutated"])
        self.assertFalse(out["draft_tokens_emitted"])
        self.assertEqual(out["cmake_hits"], [])
        self.assertEqual(out["forbidden_wiring_hits"], [])


if __name__ == "__main__":
    unittest.main()
