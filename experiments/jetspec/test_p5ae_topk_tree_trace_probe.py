#!/usr/bin/env python3
"""Tests for P5AE top-k tree trace probe."""

from __future__ import annotations

import unittest

import probe_p5ae_topk_tree_trace as probe


class P5AETopkTreeTraceProbeTests(unittest.TestCase):
    def test_self_test_trace_is_valid(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE)
        self.assertTrue(parsed["ok"], parsed["errors"])
        self.assertEqual(parsed["actual_tree_nodes"], 3)
        self.assertEqual(parsed["topk_width"], 2)
        self.assertEqual(parsed["topk_depth"], 1)
        self.assertEqual(parsed["non_root_nodes"], 2)

    def test_rejects_missing_no_draft_tokens(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE.replace(" no_draft_tokens=1", ""))
        self.assertFalse(parsed["ok"])
        self.assertIn("missing P5AE trace token: no_draft_tokens=1", parsed["errors"])

    def test_rejects_wrong_actual_tree_nodes(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE.replace("actual_tree_nodes=3", "actual_tree_nodes=2"))
        self.assertFalse(parsed["ok"])
        self.assertIn("P5AE trace actual_tree_nodes must be 3", parsed["errors"])

    def test_probe_passes_without_live_log(self) -> None:
        out = probe.probe_p5ae_topk_tree_trace()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5ae_topk_tree_trace_contract_verified")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["draft_tokens_emitted"])


if __name__ == "__main__":
    unittest.main()
