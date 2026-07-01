#!/usr/bin/env python3
"""Tests for the P5F metadata-only loader-gate probe."""

from __future__ import annotations

import unittest

from probe_p5f_loader_gate import probe_loader_gate


class P5FLoaderGateProbeTests(unittest.TestCase):
    def test_contract_mode_generates_preview_without_claiming_runtime(self) -> None:
        result = probe_loader_gate(run_binary=False)

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "loader_gate_contract_ready_binary_not_run")
        self.assertFalse(result["runtime_executed"])
        self.assertFalse(result["draft_context_created"])
        self.assertFalse(result["p5f_preflight_executed"])
        self.assertEqual(result["preview_metadata"]["tensor_count"], 0)
        self.assertTrue(result["preview_metadata"]["metadata_only"])
        self.assertTrue(result["runtime_supported_true_negative"]["ok"], result["runtime_supported_true_negative"])
        self.assertTrue(result["runtime_supported_true_negative"]["reject_before_optional_load_gate"])

    def test_contract_documents_remaining_live_preflight_blocker(self) -> None:
        result = probe_loader_gate(run_binary=False)

        self.assertIn("does not instantiate target/draft llama_context pair", result["limitations"])
        self.assertIn("does not execute common_speculative_jetspec_preflight", result["limitations"])
        self.assertIn("does not execute draft-head graph/tree/rollback runtime", result["limitations"])
        self.assertEqual(result["runtime_supported_true_negative"]["name"], "runtime_supported_true_source_rejected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
