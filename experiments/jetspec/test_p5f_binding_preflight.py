#!/usr/bin/env python3
"""Tests for the P5F JetSpec binding-preflight source validator."""

from __future__ import annotations

import pathlib
import unittest

from validate_p5f_binding_preflight import P5F_ALLOWED_FILES, REPO_ROOT, validate_p5f_binding_preflight


class P5FBindingPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validate_p5f_binding_preflight()

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["cmake_hits"], [])
        self.assertEqual(self.result["forbidden_root_hits"], [])

    def test_allowlist_is_bounded(self) -> None:
        expected = {
            "common/speculative.cpp",
            "docs/speculative.md",
        }
        self.assertEqual({str(path) for path in P5F_ALLOWED_FILES}, expected)

    def test_preflight_checks_target_and_draft_shapes(self) -> None:
        text = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        self.assertIn("common_speculative_jetspec_preflight", text)
        self.assertIn("JETSPEC_QWEN36_DRAFT_BLOCK_SIZE  = 16", text)
        self.assertIn("JETSPEC_QWEN36_TARGET_HIDDEN     = 2048", text)
        self.assertIn("JETSPEC_QWEN36_TARGET_LAYERS     = 40", text)
        self.assertIn("JETSPEC_QWEN36_TARGET_TAP_WIDTH  = JETSPEC_QWEN36_TARGET_TAP_COUNT * JETSPEC_QWEN36_TARGET_HIDDEN", text)
        self.assertIn("JETSPEC_QWEN36_DRAFT_LAYERS      = 8", text)
        self.assertIn("JETSPEC_QWEN36_DRAFT_HEADS       = 32", text)
        self.assertIn("JETSPEC_QWEN36_DRAFT_HEADS_KV    = 4", text)
        self.assertIn("JETSPEC_QWEN36_VOCAB_SIZE        = 248320", text)
        self.assertIn("target.n_embd", text)
        self.assertIn("target.n_layer", text)
        self.assertIn("draft.n_ctx_train", text)
        self.assertIn("draft.n_head_kv", text)
        self.assertIn("target_tap_width_vs_target", text)
        self.assertIn("tap_count * target_hidden", text)

    def test_preflight_checks_draft_metadata_strings(self) -> None:
        text = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        self.assertIn("common_speculative_jetspec_expect_meta_str", text)
        self.assertIn("general.architecture", text)
        self.assertIn("jetspec_qwen3_draft_head", text)
        self.assertIn("jetspec.architecture", text)
        self.assertIn("qwen3_draft_head", text)
        self.assertIn("jetspec.source_architecture", text)
        self.assertIn("DFlashDraftModel", text)
        self.assertIn("jetspec.tensor_data_dtype", text)
        self.assertIn("bfloat16", text)

    def test_fail_closed_no_draft_or_graph_execution(self) -> None:
        text = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        start = text.index("struct common_speculative_impl_draft_jetspec")
        end = text.index("struct common_speculative_impl_draft_mtp")
        impl = text[start:end]
        self.assertIn("invalid_binding", impl)
        self.assertIn("do not emit draft tokens", impl)
        self.assertNotIn("result->push_back", impl)
        self.assertNotIn("llama_decode", impl)
        self.assertNotIn("llama_graph", impl)
        preflight_start = text.index("static bool common_speculative_jetspec_preflight")
        preflight_end = text.index("static bool common_speculative_are_compatible")
        preflight = text[preflight_start:preflight_end]
        self.assertNotIn("llama_decode", preflight)
        self.assertNotIn("llama_graph", preflight)
        self.assertNotIn("tree_accept", preflight)

    def test_private_preflight_and_no_surface_area(self) -> None:
        for rel in [
            pathlib.Path("include/llama.h"),
            pathlib.Path("tools/server/server-context.cpp"),
            pathlib.Path("common/speculative.h"),
            pathlib.Path("common/arg.cpp"),
        ]:
            other = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
            self.assertNotIn("common_speculative_jetspec_preflight", other, rel)
            self.assertNotIn("JETSPEC_QWEN36_TARGET_TAP_WIDTH", other, rel)
            self.assertNotIn("draft-jetspec preflight failed", other, rel)

    def test_loader_remains_fail_closed(self) -> None:
        text = (REPO_ROOT / "src/models/jetspec_qwen3_draft_head.cpp").read_text(encoding="utf-8")
        self.assertIn("preview_not_allowed", text)
        self.assertIn("unsupported_runtime", text)
        self.assertIn("runtime_supported=false", text)
        self.assertIn("throw std::runtime_error(\"unsupported_runtime: JetSpec P5A has no graph execution path\")", text)

    def test_docs_mark_preflight_only_no_draft_tokens(self) -> None:
        text = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        self.assertIn("binding preflight", text)
        self.assertIn("metadata/shape", text)
        self.assertIn("target tap count/width", text)
        self.assertIn("emits no draft", text)
        self.assertIn("tokens and does not execute the draft head", text)


if __name__ == "__main__":
    unittest.main()
