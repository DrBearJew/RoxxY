#!/usr/bin/env python3
"""Tests for the P5C JetSpec speculative type source validator."""

from __future__ import annotations

import pathlib
import unittest

from validate_p5c_speculative_type import P5C_ALLOWED_FILES, REPO_ROOT, validate_p5c_speculative_type


class P5CSpeculativeTypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validate_p5c_speculative_type()

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["cmake_hits"], [])
        self.assertEqual(self.result["forbidden_root_hits"], [])

    def test_allowlist_is_bounded(self) -> None:
        expected = {
            "common/common.h",
            "common/speculative.cpp",
            "docs/speculative.md",
            "tools/server/server-context.cpp",
        }
        self.assertEqual({str(path) for path in P5C_ALLOWED_FILES}, expected)

    def test_parser_and_string_roundtrip_are_wired(self) -> None:
        common_h = (REPO_ROOT / "common/common.h").read_text(encoding="utf-8")
        spec_cpp = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")

        self.assertIn("COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC", common_h)
        self.assertIn('{"draft-jetspec", COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC}', spec_cpp)
        self.assertIn('{"jetspec",       COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC}', spec_cpp)
        self.assertIn('case COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC: return "draft-jetspec";', spec_cpp)
        self.assertIn("static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 10)", spec_cpp)

    def test_fail_closed_gates_are_present(self) -> None:
        spec_cpp = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        self.assertIn("LLAMA_JETSPEC_EXPERIMENTAL", spec_cpp)
        self.assertIn("disabling JetSpec before runtime execution", spec_cpp)
        self.assertIn("runtime_supported=false", spec_cpp)
        self.assertIn("no draft tokens will be generated", spec_cpp)
        self.assertIn("!has_draft_jetspec", spec_cpp)
        self.assertIn("llama_set_jetspec_target_hidden_taps(params.draft.ctx_tgt, true, true)", spec_cpp)
        self.assertIn("llama_set_jetspec_target_hidden_taps(params.draft.ctx_tgt, false, true)", spec_cpp)

    def test_no_public_api_or_cmake_route(self) -> None:
        for rel in [
            pathlib.Path("include/llama.h"),
            pathlib.Path("common/speculative.h"),
        ]:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
            self.assertNotIn("COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC", text, rel)
            self.assertNotIn("draft-jetspec", text, rel)

    def test_server_reference_is_model_only_binding(self) -> None:
        text = (REPO_ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8", errors="replace")
        self.assertIn("COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC", text)
        self.assertIn("spec_jetspec", text)
        self.assertIn("llama_model_link_shared_tensors(model_dft.get(), model_tgt)", text)
        self.assertIn("params_base.speculative.draft.ctx_dft = nullptr", text)
        self.assertIn("loaded JetSpec draft-head model-only binding; draft context and graph execution remain disabled", text)

    def test_docs_mark_route_experimental(self) -> None:
        text = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        self.assertIn("draft-jetspec", text)
        self.assertIn("experimental", text.lower())
        self.assertIn("fail-closed", text)
        self.assertIn("LLAMA_JETSPEC_EXPERIMENTAL=1", text)


if __name__ == "__main__":
    unittest.main()
