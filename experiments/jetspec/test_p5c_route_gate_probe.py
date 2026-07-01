#!/usr/bin/env python3
"""Tests for the no-model P5C route-gate source probe."""

from __future__ import annotations

import unittest

import probe_p5c_route_gate as probe


class P5CRouteGateProbeTests(unittest.TestCase):
    def test_route_gate_probe_passes_without_runtime(self) -> None:
        result = probe.probe_p5c_route_gate()

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "p5c_route_gate_verified_no_model")
        self.assertFalse(result["runtime_executed"])
        self.assertFalse(result["draft_context_created"])
        self.assertGreater(result["line_map"]["env_gate"], 0)
        self.assertGreater(result["line_map"]["jetspec_config_push"], result["line_map"]["preflight_gate"])

    def test_route_gate_source_contains_no_silent_draft_simple_fallback(self) -> None:
        result = probe.probe_p5c_route_gate()

        self.assertTrue(result["ok"], result["errors"])
        source = probe.SOURCE.read_text(encoding="utf-8", errors="replace")
        self.assertIn("has_draft_model_path && !has_mtp && !has_draft_eagle3 && !has_draft_jetspec", source)
        self.assertIn("draft-jetspec requires LLAMA_JETSPEC_EXPERIMENTAL=1", source)
        self.assertIn("draft-jetspec preflight failed", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
