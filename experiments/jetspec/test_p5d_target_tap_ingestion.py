#!/usr/bin/env python3
"""Tests for the P5D JetSpec target-tap ingestion source validator."""

from __future__ import annotations

import pathlib
import unittest

from validate_p5d_target_tap_ingestion import P5D_ALLOWED_FILES, REPO_ROOT, validate_p5d_target_tap_ingestion


class P5DTargetTapIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validate_p5d_target_tap_ingestion()

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["cmake_hits"], [])
        self.assertEqual(self.result["forbidden_root_hits"], [])

    def test_allowlist_is_bounded(self) -> None:
        expected = {
            "common/speculative.cpp",
            "docs/speculative.md",
        }
        self.assertEqual({str(path) for path in P5D_ALLOWED_FILES}, expected)

    def test_impl_captures_and_hashes_target_taps(self) -> None:
        text = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        self.assertIn("target_tap_rows", text)
        self.assertIn("n_target_tap_rows_total", text)
        self.assertIn("target_tap_hash_last", text)
        self.assertIn("batch.logits", text)
        self.assertIn("llama_get_jetspec_target_hidden_taps(params.ctx_tgt)", text)
        self.assertIn("target_tap_rows.assign(taps, taps + n_values)", text)
        self.assertIn("common_speculative_fnv1a64(target_tap_rows.data(), n_values * sizeof(float))", text)

    def test_fail_closed_no_draft_or_graph_execution(self) -> None:
        text = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        start = text.index("struct common_speculative_impl_draft_jetspec")
        end = text.index("struct common_speculative_impl_draft_mtp")
        impl = text[start:end]
        self.assertIn("tap_count != 5 || tap_width != 10240 || taps == nullptr", impl)
        self.assertIn("target_taps_active = false", impl)
        self.assertIn("do not emit draft tokens", impl)
        self.assertNotIn("result->push_back", impl)
        self.assertNotIn("llama_decode", impl)
        self.assertNotIn("llama_graph", impl)

    def test_trace_and_cleanup_are_private(self) -> None:
        text = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        self.assertIn("LLAMA_JETSPEC_TRACE", text)
        self.assertIn("LLAMA_JETSPEC_TAP_TRACE", text)
        self.assertIn("llama_set_jetspec_target_hidden_taps(params.ctx_tgt, false, true)", text)
        for rel in [
            pathlib.Path("include/llama.h"),
            pathlib.Path("tools/server/server-context.cpp"),
            pathlib.Path("common/speculative.h"),
            pathlib.Path("common/arg.cpp"),
        ]:
            other = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
            self.assertNotIn("LLAMA_JETSPEC_TAP_TRACE", other, rel)
            self.assertNotIn("target_tap_rows", other, rel)

    def test_docs_mark_no_draft_tokens(self) -> None:
        text = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        self.assertIn("target tap", text)
        self.assertIn("LLAMA_JETSPEC_TRACE=1", text)
        self.assertIn("LLAMA_JETSPEC_TAP_TRACE=1", text)
        self.assertIn("no draft tokens", text)


if __name__ == "__main__":
    unittest.main()
