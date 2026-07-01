#!/usr/bin/env python3
"""Tests for P5AF top-k verify-mask trace probe."""

from __future__ import annotations

import unittest

import probe_p5af_topk_verify_mask_trace as probe


class P5AFTopkVerifyMaskTraceProbeTests(unittest.TestCase):
    def test_self_test_trace_is_valid(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE)
        self.assertTrue(parsed["ok"], parsed["errors"])
        self.assertEqual(parsed["actual_tree_nodes"], 3)
        self.assertEqual(parsed["actual_verify_mask_entries"], 5)
        self.assertEqual(parsed["verify_mask_rows"], 3)
        self.assertEqual(parsed["verify_mask_cols"], 3)

    def test_rejects_missing_no_mask_tensor(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE.replace(" no_mask_tensor=1", ""))
        self.assertFalse(parsed["ok"])
        self.assertIn("missing P5AF trace token: no_mask_tensor=1", parsed["errors"])

    def test_rejects_wrong_mask_entry_count(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE.replace("actual_verify_mask_entries=5", "actual_verify_mask_entries=4"))
        self.assertFalse(parsed["ok"])
        self.assertIn("P5AF trace actual_verify_mask_entries must be 5", parsed["errors"])

    def test_probe_passes_without_live_log(self) -> None:
        out = probe.probe_p5af_topk_verify_mask_trace()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5af_topk_verify_mask_trace_contract_verified")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["draft_tokens_emitted"])


if __name__ == "__main__":
    unittest.main()
