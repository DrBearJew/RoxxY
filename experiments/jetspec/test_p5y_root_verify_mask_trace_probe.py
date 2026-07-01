#!/usr/bin/env python3
"""Tests for the P5Y root verify-mask trace contract probe."""

from __future__ import annotations

import tempfile
import unittest

import probe_p5y_root_verify_mask_trace as probe


class P5YRootVerifyMaskTraceProbeTests(unittest.TestCase):
    def test_probe_passes_without_model(self) -> None:
        result = probe.probe_p5y_root_verify_mask_trace()
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "p5y_root_verify_mask_trace_contract_verified")
        self.assertFalse(result["runtime_executed"])
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["context_created"])
        self.assertFalse(result["draft_tokens_emitted"])
        self.assertTrue(result["source_contract"]["ok"])
        self.assertTrue(result["self_test_trace"]["ok"])

    def test_self_test_trace_has_root_only_mask_boundary(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE)
        self.assertTrue(parsed["ok"], parsed["errors"])
        self.assertEqual(parsed["actual_tree_nodes"], 1)
        self.assertEqual(parsed["actual_verify_mask_entries"], 1)
        self.assertEqual(parsed["verify_mask_rows"], 1)
        self.assertEqual(parsed["verify_mask_cols"], 1)
        self.assertEqual(parsed["root_attends_self"], 1)
        self.assertEqual(parsed["root_mask_row"], 0)
        self.assertEqual(parsed["root_mask_col"], 0)
        self.assertEqual(parsed["forbidden_hits"], [])

    def test_trace_rejects_missing_boundary_tokens(self) -> None:
        parsed = probe.validate_trace_line(
            "draft-jetspec p5y_root_verify_mask_runtime phase=verify_mask_runtime_ready "
            "root_verify_mask_runtime_ready=1 actual_verify_mask_entries=1"
        )
        self.assertFalse(parsed["ok"])
        self.assertTrue(any("root_tree_runtime_ready=1" in err for err in parsed["errors"]))
        self.assertTrue(any("no_accept=1" in err for err in parsed["errors"]))

    def test_trace_rejects_non_self_root_mask(self) -> None:
        bad = probe.SELF_TEST_TRACE.replace("root_attends_self=1", "root_attends_self=0")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("P5Y trace root must attend to itself", parsed["errors"])

    def test_trace_rejects_non_root_coordinate(self) -> None:
        bad = probe.SELF_TEST_TRACE.replace("root_mask_col=0", "root_mask_col=1")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("P5Y trace root mask coordinate must be (0,0)", parsed["errors"])

    def test_trace_rejects_forbidden_downstream_ready_state(self) -> None:
        bad = probe.SELF_TEST_TRACE + " accept_path_descriptor_ready=1"
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("accept_path_descriptor_ready=1", parsed["forbidden_hits"])

    def test_optional_log_validation_accepts_captured_line(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as tmp:
            tmp.write("prefix\n")
            tmp.write(probe.SELF_TEST_TRACE + "\n")
            tmp.flush()
            result = probe.probe_p5y_root_verify_mask_trace(probe.pathlib.Path(tmp.name))
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(result["runtime_executed"])
        self.assertTrue(result["model_loaded"])
        self.assertTrue(result["context_created"])
        self.assertFalse(result["draft_tokens_emitted"])
        self.assertIsNotNone(result["live_trace"])

    def test_optional_log_validation_requires_p5y_line(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as tmp:
            tmp.write("no p5y here\n")
            tmp.flush()
            result = probe.probe_p5y_root_verify_mask_trace(probe.pathlib.Path(tmp.name))
        self.assertFalse(result["ok"])
        self.assertIn("trace log does not contain draft-jetspec p5y_root_verify_mask_runtime", result["errors"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
