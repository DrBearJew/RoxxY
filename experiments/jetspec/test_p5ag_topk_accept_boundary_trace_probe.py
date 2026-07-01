#!/usr/bin/env python3
"""Tests for P5AG top-k accept-boundary trace probe."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe_p5ag_topk_accept_boundary_trace as probe


class P5AGTopKAcceptBoundaryTraceProbeTests(unittest.TestCase):
    def test_default_source_and_self_test_contract(self) -> None:
        out = probe.probe_p5ag_topk_accept_boundary_trace()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5ag_topk_accept_boundary_trace_contract_verified")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["draft_tokens_emitted"])
        self.assertEqual(out["self_test_trace"]["accept_boundary_candidate_nodes"], 2)
        self.assertEqual(out["self_test_trace"]["accept_boundary_verified_edges"], 5)
        self.assertEqual(out["self_test_trace"]["actual_verified_logits_rows"], 0)
        self.assertEqual(out["self_test_trace"]["actual_accepted_nodes"], 0)

    def test_trace_line_rejects_missing_boundary(self) -> None:
        parsed = probe.validate_trace_line("draft-jetspec p5ag_topk_accept_boundary_runtime phase=accept_path_runtime_ready")
        self.assertFalse(parsed["ok"])
        self.assertIn("accept_boundary_candidate_nodes=2", "\n".join(parsed["errors"]))

    def test_trace_log_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "server.log"
            path.write_text("noise\n" + probe.SELF_TEST_TRACE + "\n", encoding="utf-8")
            out = probe.probe_p5ag_topk_accept_boundary_trace(path)
        self.assertTrue(out["ok"], out["errors"])
        self.assertTrue(out["runtime_executed"])
        self.assertEqual(out["live_trace"]["accept_boundary_candidate_nodes"], 2)
        self.assertEqual(out["live_trace"]["actual_accepted_nodes"], 0)


if __name__ == "__main__":
    unittest.main()
