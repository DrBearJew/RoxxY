#!/usr/bin/env python3
"""Tests for the P5B target hidden tap source validator."""

from __future__ import annotations

import pathlib
import unittest

from validate_p5b_target_taps import P5B_ALLOWED_FILES, REPO_ROOT, validate_p5b_target_taps


class P5BTargetTapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validate_p5b_target_taps()

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["cmake_hits"], [])

    def test_allowlist_is_private_p5b_only(self) -> None:
        expected = {
            "src/llama-cparams.h",
            "src/llama-graph.h",
            "src/llama-graph.cpp",
            "src/llama-context.h",
            "src/llama-context.cpp",
            "src/llama-ext.h",
            "src/models/qwen35.cpp",
            "src/models/qwen35moe.cpp",
        }
        self.assertEqual({str(path) for path in P5B_ALLOWED_FILES}, expected)

    def test_fixed_layer_contract_and_width(self) -> None:
        for rel, helper in [
            ("src/models/qwen35.cpp", "qwen35_jetspec_target_hidden_tap_layer"),
            ("src/models/qwen35moe.cpp", "qwen35moe_jetspec_target_hidden_tap_layer"),
        ]:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            self.assertIn(helper, text)
            self.assertIn("il == 1 || il == 10 || il == 19 || il == 28 || il == 37", text)
            self.assertIn("ggml_concat(ctx0, jetspec_target_hidden_taps, tap, 0)", text)
            self.assertIn("concat width 10240", text)

    def test_private_api_only_no_public_or_server_route(self) -> None:
        self.assertIn("llama_set_jetspec_target_hidden_taps", (REPO_ROOT / "src/llama-ext.h").read_text(encoding="utf-8"))
        for rel in [
            "include/llama.h",
            "common/common.h",
            "common/speculative.h",
            "common/arg.cpp",
            "tools/server/server-context.cpp",
        ]:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
            self.assertNotIn("llama_set_jetspec_target_hidden_taps", text, rel)
            self.assertNotIn("jetspec_target_hidden_taps", text, rel)
        text = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8", errors="replace")
        self.assertIn("llama_set_jetspec_target_hidden_taps", text)
        self.assertIn("draft-jetspec", text)

    def test_graph_reuse_checks_capture_flags(self) -> None:
        text = (REPO_ROOT / "src/llama-graph.h").read_text(encoding="utf-8")
        self.assertIn("cparams.jetspec_target_hidden_taps        == other.cparams.jetspec_target_hidden_taps", text)
        self.assertIn("cparams.jetspec_target_hidden_taps_masked == other.cparams.jetspec_target_hidden_taps_masked", text)
        self.assertIn("cparams.embeddings_pre_norm               == other.cparams.embeddings_pre_norm", text)


if __name__ == "__main__":
    unittest.main()
