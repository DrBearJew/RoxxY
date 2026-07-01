#!/usr/bin/env python3
"""Tests for the P5X root-tree trace contract probe."""

from __future__ import annotations

import tempfile
import unittest

import probe_p5x_root_tree_trace as probe


class P5XRootTreeTraceProbeTests(unittest.TestCase):
    def test_probe_passes_without_model(self) -> None:
        result = probe.probe_p5x_root_tree_trace()
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "p5x_root_tree_trace_contract_verified")
        self.assertFalse(result["runtime_executed"])
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["context_created"])
        self.assertFalse(result["draft_tokens_emitted"])
        self.assertTrue(result["source_contract"]["ok"])
        self.assertTrue(result["self_test_trace"]["ok"])

    def test_self_test_trace_has_root_only_boundary(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE)
        self.assertTrue(parsed["ok"], parsed["errors"])
        self.assertEqual(parsed["actual_tree_nodes"], 1)
        self.assertEqual(parsed["root_token"], 42)
        self.assertEqual(parsed["root_parent"], -1)
        self.assertEqual(parsed["root_depth"], 0)
        self.assertEqual(parsed["root_rank"], -1)
        self.assertGreaterEqual(parsed["tree_build_node_budget"], 1)
        self.assertEqual(parsed["forbidden_hits"], [])

    def test_trace_rejects_missing_boundary_tokens(self) -> None:
        parsed = probe.validate_trace_line(
            "draft-jetspec p5x_root_tree_runtime phase=tree_build_runtime_ready "
            "root_tree_runtime_ready=1 actual_tree_nodes=1 tree_token_ids=[7]"
        )
        self.assertFalse(parsed["ok"])
        self.assertTrue(any("no_verify_mask=1" in err for err in parsed["errors"]))
        self.assertTrue(any("tree_parent_indices" in err for err in parsed["errors"]))

    def test_trace_rejects_non_root_parent(self) -> None:
        bad = probe.SELF_TEST_TRACE.replace("tree_parent_indices=[-1]", "tree_parent_indices=[0]")
        bad = bad.replace("root_parent=-1", "root_parent=0")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("P5X trace root parent must be -1", parsed["errors"])

    def test_trace_rejects_forbidden_downstream_ready_state(self) -> None:
        bad = probe.SELF_TEST_TRACE + " verify_mask_descriptor_ready=1"
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("verify_mask_descriptor_ready=1", parsed["forbidden_hits"])

    def test_optional_log_validation_accepts_captured_line(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as tmp:
            tmp.write("prefix\n")
            tmp.write(probe.SELF_TEST_TRACE + "\n")
            tmp.flush()
            result = probe.probe_p5x_root_tree_trace(probe.pathlib.Path(tmp.name))
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(result["runtime_executed"])
        self.assertTrue(result["model_loaded"])
        self.assertTrue(result["context_created"])
        self.assertFalse(result["draft_tokens_emitted"])
        self.assertIsNotNone(result["live_trace"])

    def test_optional_log_validation_requires_p5x_line(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as tmp:
            tmp.write("no p5x here\n")
            tmp.flush()
            result = probe.probe_p5x_root_tree_trace(probe.pathlib.Path(tmp.name))
        self.assertFalse(result["ok"])
        self.assertIn("trace log does not contain draft-jetspec p5x_root_tree_runtime", result["errors"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
