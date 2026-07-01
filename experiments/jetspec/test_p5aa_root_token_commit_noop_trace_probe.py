#!/usr/bin/env python3
"""Tests for the P5AA root-token-commit no-op trace contract probe."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import probe_p5aa_root_token_commit_noop_trace as probe


class P5AARootTokenCommitNoopTraceProbeTests(unittest.TestCase):
    def test_probe_default_passes_without_runtime(self) -> None:
        out = probe.probe_p5aa_root_token_commit_noop_trace()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5aa_root_token_commit_noop_trace_contract_verified")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["model_loaded"])
        self.assertFalse(out["context_created"])
        self.assertFalse(out["draft_tokens_emitted"])

    def test_self_test_trace_parses_expected_scalars(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE)
        self.assertTrue(parsed["ok"], parsed["errors"])
        self.assertEqual(parsed["root_verified_anchor"], 1)
        self.assertEqual(parsed["accept_path_len"], 0)
        self.assertEqual(parsed["actual_tree_nodes"], 1)
        self.assertEqual(parsed["actual_verify_mask_entries"], 1)
        self.assertEqual(parsed["actual_accepted_nodes"], 0)
        self.assertEqual(parsed["correction_token_present"], 0)
        self.assertEqual(parsed["actual_committed_tokens"], 0)

    def test_trace_rejects_nonzero_commit(self) -> None:
        bad = probe.SELF_TEST_TRACE.replace("actual_committed_tokens=0", "actual_committed_tokens=1")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("P5AA trace actual_committed_tokens must stay 0", parsed["errors"])
        self.assertIn("actual_committed_tokens=1", parsed["forbidden_hits"])

    def test_trace_rejects_missing_no_visible_publish(self) -> None:
        bad = probe.SELF_TEST_TRACE.replace(" no_visible_token_publish=1", "")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("missing P5AA trace token: no_visible_token_publish=1", parsed["errors"])

    def test_probe_accepts_live_trace_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.log"
            path.write_text("prefix\n" + probe.SELF_TEST_TRACE + "\nsuffix\n", encoding="utf-8")
            out = probe.probe_p5aa_root_token_commit_noop_trace(path)
        self.assertTrue(out["ok"], out["errors"])
        self.assertTrue(out["runtime_executed"])
        self.assertIsNotNone(out["live_trace"])

    def test_probe_rejects_missing_live_trace_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.log"
            path.write_text("no jetspec trace here\n", encoding="utf-8")
            out = probe.probe_p5aa_root_token_commit_noop_trace(path)
        self.assertFalse(out["ok"])
        self.assertIn("trace log does not contain draft-jetspec p5aa_root_token_commit_noop_runtime", out["errors"])


if __name__ == "__main__":
    unittest.main()
