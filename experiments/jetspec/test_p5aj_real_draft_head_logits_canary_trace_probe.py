#!/usr/bin/env python3
"""Tests for the P5AJ real draft-head logits/top-k canary trace probe."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import probe_p5aj_real_draft_head_logits_canary_trace as probe


class P5AJRealDraftHeadLogitsCanaryTraceProbeTests(unittest.TestCase):
    def test_probe_passes_without_live_log(self) -> None:
        out = probe.probe_p5aj_real_draft_head_logits_canary_trace()
        self.assertTrue(out["ok"], out["errors"])
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["draft_tokens_emitted"])
        self.assertTrue(out["source_contract"]["ok"])

    def test_trace_line_requires_logits_topk_and_no_tokens(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE)
        self.assertTrue(parsed["ok"], parsed["errors"])
        self.assertEqual(parsed["ctx_dft_present"], 1)
        self.assertEqual(parsed["decode_rc"], 0)
        self.assertEqual(parsed["input_width"], 10240)
        self.assertEqual(parsed["logits_width"], 248320)
        self.assertEqual(parsed["logits_rows"], parsed["input_rows"])
        self.assertEqual(parsed["topk_rows"], parsed["input_rows"])
        self.assertEqual(parsed["topk_k"], 2)
        self.assertGreaterEqual(parsed["top1_id"], 0)
        self.assertGreaterEqual(parsed["top2_id"], 0)

    def test_probe_accepts_live_trace_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "server.log"
            path.write_text("prefix\n" + probe.SELF_TEST_TRACE + "\n", encoding="utf-8")
            out = probe.probe_p5aj_real_draft_head_logits_canary_trace(path)
        self.assertTrue(out["ok"], out["errors"])
        self.assertTrue(out["runtime_executed"])
        self.assertIsNotNone(out["live_trace"])
        self.assertEqual(out["live_trace"]["logits_rows"], out["live_trace"]["input_rows"])

    def test_probe_rejects_missing_live_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "server.log"
            path.write_text("no p5aj trace here\n", encoding="utf-8")
            out = probe.probe_p5aj_real_draft_head_logits_canary_trace(path)
        self.assertFalse(out["ok"])
        self.assertIn("trace log does not contain draft-jetspec p5aj_real_draft_head_logits_canary", out["errors"])


if __name__ == "__main__":
    unittest.main()
