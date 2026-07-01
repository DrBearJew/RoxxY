#!/usr/bin/env python3
"""Tests for the P5E JetSpec runtime-state source validator."""

from __future__ import annotations

import pathlib
import unittest

from validate_p5e_runtime_state import P5E_ALLOWED_FILES, REPO_ROOT, validate_p5e_runtime_state


class P5ERuntimeStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validate_p5e_runtime_state()

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["cmake_hits"], [])
        self.assertEqual(self.result["forbidden_root_hits"], [])

    def test_allowlist_is_bounded(self) -> None:
        expected = {
            "common/speculative.cpp",
            "docs/speculative.md",
        }
        self.assertEqual({str(path) for path in P5E_ALLOWED_FILES}, expected)

    def test_impl_tracks_runtime_phase_failure_and_rows(self) -> None:
        text = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        self.assertIn("enum class jetspec_runtime_phase", text)
        self.assertIn("waiting_for_target_taps", text)
        self.assertIn("target_taps_captured", text)
        self.assertIn("enum class jetspec_runtime_failure", text)
        self.assertIn("missing_target_context", text)
        self.assertIn("invalid_target_taps", text)
        self.assertIn("struct jetspec_target_tap_row_state", text)
        self.assertIn("target_tap_row_state", text)
        self.assertIn("n_target_tap_rows_cached", text)
        self.assertIn("LLAMA_JETSPEC_STATE_TRACE", text)

    def test_fail_closed_no_draft_or_graph_execution(self) -> None:
        text = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        start = text.index("struct common_speculative_impl_draft_jetspec")
        end = text.index("struct common_speculative_impl_draft_mtp")
        impl = text[start:end]
        self.assertIn("disable_runtime_state(jetspec_runtime_failure::invalid_target_taps, tap_count, tap_width, taps)", impl)
        self.assertIn("target_taps_active = false", impl)
        self.assertIn("no_draft=1", impl)
        self.assertIn("do not emit draft tokens", impl)
        self.assertNotIn("result->push_back", impl)
        self.assertNotIn("llama_decode", impl)
        self.assertNotIn("llama_graph", impl)
        self.assertNotIn("tree_accept", impl)

    def test_private_trace_and_no_surface_area(self) -> None:
        for rel in [
            pathlib.Path("include/llama.h"),
            pathlib.Path("tools/server/server-context.cpp"),
            pathlib.Path("common/speculative.h"),
            pathlib.Path("common/arg.cpp"),
        ]:
            other = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
            self.assertNotIn("LLAMA_JETSPEC_STATE_TRACE", other, rel)
            self.assertNotIn("jetspec_runtime_phase", other, rel)
            self.assertNotIn("jetspec_runtime_failure", other, rel)

    def test_loader_remains_fail_closed(self) -> None:
        text = (REPO_ROOT / "src/models/jetspec_qwen3_draft_head.cpp").read_text(encoding="utf-8")
        self.assertIn("preview_not_allowed", text)
        self.assertIn("unsupported_runtime", text)
        self.assertIn("runtime_supported=false", text)
        self.assertIn("throw std::runtime_error(\"unsupported_runtime: JetSpec draft-head graph execution is not implemented\")", text)

    def test_docs_mark_state_only_no_draft_tokens(self) -> None:
        text = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        self.assertIn("runtime-state", text)
        self.assertIn("LLAMA_JETSPEC_STATE_TRACE=1", text)
        self.assertIn("failure state", text)
        self.assertIn("no draft tokens", text)


if __name__ == "__main__":
    unittest.main()
