#!/usr/bin/env python3
"""Tests for the P5A production loader candidate validator."""

from __future__ import annotations

import pathlib
import unittest

from validate_p5a_loader_candidate import (
    P5A_ALLOWED_PRODUCTION_FILES,
    P5A_APPROVED_DOWNSTREAM_TOKEN_HITS,
    REPO_ROOT,
    validate_p5a_loader_candidate,
)


class P5ALoaderCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validate_p5a_loader_candidate()

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["cmake_hits"], [])

    def test_allowlist_is_only_p5a_files(self) -> None:
        expected = {
            "src/llama-arch.h",
            "src/llama-arch.cpp",
            "src/llama-model.cpp",
            "src/models/models.h",
            "src/models/jetspec_qwen3_draft_head.cpp",
        }
        self.assertEqual({str(path) for path in P5A_ALLOWED_PRODUCTION_FILES}, expected)

    def test_loader_source_preserves_fail_closed_gates(self) -> None:
        text = (REPO_ROOT / "src/models/jetspec_qwen3_draft_head.cpp").read_text(encoding="utf-8")
        for token in [
            "LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD",
            "LLAMA_JETSPEC_EXPERIMENTAL",
            "preview_not_allowed",
            "unsupported_runtime",
            "jetspec.experimental.runtime_supported",
            "jetspec.experimental.runtime_supported must remain false until JetSpec draft-head graph execution is implemented",
            "jetspec_expect(!meta.runtime_supported",
            "n_tensors == 0 || n_tensors == 91",
            "GGML_TYPE_BF16",
        ]:
            self.assertIn(token, text)
        self.assertNotIn("ggml_build_forward_expand", text)
        self.assertNotIn("return std::make_unique", text)

    def test_p5a_does_not_add_public_loader_api(self) -> None:
        text = (REPO_ROOT / "include/llama.h").read_text(encoding="utf-8", errors="replace")
        self.assertNotIn("jetspec_qwen3_draft_head", text)
        self.assertNotIn("LLM_ARCH_JETSPEC_QWEN3_DRAFT_HEAD", text)

    def test_production_token_hits_stay_in_allowlist_or_downstream_approval(self) -> None:
        allowed = {str(path) for path in P5A_ALLOWED_PRODUCTION_FILES}
        downstream = {(str(path), token) for path, token in P5A_APPROVED_DOWNSTREAM_TOKEN_HITS}
        for hit in self.result["token_hits"]:
            self.assertTrue(hit["path"] in allowed or (hit["path"], hit["token"]) in downstream, hit)


if __name__ == "__main__":
    unittest.main()
