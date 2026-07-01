#!/usr/bin/env python3
"""Tests for the P5Z root-anchor accept-path trace contract probe."""

from __future__ import annotations

import tempfile
import unittest

import probe_p5z_root_anchor_accept_path_trace as probe


class P5ZRootAnchorAcceptPathTraceProbeTests(unittest.TestCase):
    def test_probe_passes_without_model(self) -> None:
        result = probe.probe_p5z_root_anchor_accept_path_trace()
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "p5z_root_anchor_accept_path_trace_contract_verified")
        self.assertFalse(result["runtime_executed"])
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["context_created"])
        self.assertFalse(result["draft_tokens_emitted"])
        self.assertTrue(result["source_contract"]["ok"])
        self.assertTrue(result["self_test_trace"]["ok"])

    def test_self_test_trace_has_root_anchor_boundary(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE)
        self.assertTrue(parsed["ok"], parsed["errors"])
        self.assertEqual(parsed["root_verified_anchor"], 1)
        self.assertEqual(parsed["accept_path_len"], 0)
        self.assertEqual(parsed["actual_tree_nodes"], 1)
        self.assertEqual(parsed["actual_verify_mask_entries"], 1)
        self.assertEqual(parsed["actual_accepted_nodes"], 0)
        self.assertEqual(parsed["correction_token_present"], 0)
        self.assertEqual(parsed["forbidden_hits"], [])

    def test_trace_rejects_missing_boundary_tokens(self) -> None:
        parsed = probe.validate_trace_line(
            "draft-jetspec p5z_root_anchor_accept_path_runtime phase=accept_path_runtime_ready "
            "root_anchor_accept_path_runtime_ready=1 root_verified_anchor=1"
        )
        self.assertFalse(parsed["ok"])
        self.assertTrue(any("root_verify_mask_runtime_ready=1" in err for err in parsed["errors"]))
        self.assertTrue(any("no_token_commit=1" in err for err in parsed["errors"]))

    def test_trace_rejects_accepted_node(self) -> None:
        bad = probe.SELF_TEST_TRACE.replace("actual_accepted_nodes=0", "actual_accepted_nodes=1")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("P5Z trace actual_accepted_nodes must stay 0", parsed["errors"])

    def test_trace_rejects_correction_token(self) -> None:
        bad = probe.SELF_TEST_TRACE.replace("correction_token_present=0", "correction_token_present=1")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("P5Z trace correction_token_present must stay 0", parsed["errors"])

    def test_trace_rejects_forbidden_downstream_ready_state(self) -> None:
        bad = probe.SELF_TEST_TRACE + " token_commit_descriptor_ready=1"
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("token_commit_descriptor_ready=1", parsed["forbidden_hits"])

    def test_optional_log_validation_accepts_captured_line(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as tmp:
            tmp.write("prefix\n")
            tmp.write(probe.SELF_TEST_TRACE + "\n")
            tmp.flush()
            result = probe.probe_p5z_root_anchor_accept_path_trace(probe.pathlib.Path(tmp.name))
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(result["runtime_executed"])
        self.assertTrue(result["model_loaded"])
        self.assertTrue(result["context_created"])
        self.assertFalse(result["draft_tokens_emitted"])
        self.assertIsNotNone(result["live_trace"])

    def test_optional_log_validation_requires_p5z_line(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as tmp:
            tmp.write("no p5z here\n")
            tmp.flush()
            result = probe.probe_p5z_root_anchor_accept_path_trace(probe.pathlib.Path(tmp.name))
        self.assertFalse(result["ok"])
        self.assertIn("trace log does not contain draft-jetspec p5z_root_anchor_accept_path_runtime", result["errors"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
