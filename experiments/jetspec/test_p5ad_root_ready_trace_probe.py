#!/usr/bin/env python3
"""Tests for P5AD root-ready trace probe."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import probe_p5ad_root_ready_trace as probe


class P5ADRootReadyTraceProbeTests(unittest.TestCase):
    def test_probe_default_passes_without_runtime(self) -> None:
        out = probe.probe_p5ad_root_ready_trace()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5ad_root_ready_trace_contract_verified")
        self.assertTrue(out["ready_to_start_real_root_only_test"])
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["draft_tokens_emitted"])

    def test_self_test_trace_scalars(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE)
        self.assertTrue(parsed["ok"], parsed["errors"])
        self.assertEqual(parsed["root_runtime_ready_for_real_test"], 1)
        self.assertEqual(parsed["actual_committed_tokens"], 0)
        self.assertEqual(parsed["actual_survivor_pages_committed"], 0)
        self.assertEqual(parsed["actual_pages_discarded"], 0)
        self.assertEqual(parsed["rejected_branch_pages_reachable_after_discard"], 0)
        self.assertEqual(parsed["actual_publish_visible_state"], 0)

    def test_trace_rejects_visible_publish(self) -> None:
        bad = probe.SELF_TEST_TRACE.replace("actual_publish_visible_state=0", "actual_publish_visible_state=1")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("P5AD trace actual_publish_visible_state must be 0", parsed["errors"])

    def test_probe_accepts_live_trace_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.log"
            path.write_text(probe.SELF_TEST_TRACE, encoding="utf-8")
            out = probe.probe_p5ad_root_ready_trace(path)
        self.assertTrue(out["ok"], out["errors"])
        self.assertTrue(out["runtime_executed"])
        self.assertIsNotNone(out["live_trace"])


if __name__ == "__main__":
    unittest.main()
