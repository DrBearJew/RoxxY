#!/usr/bin/env python3
"""Tests for the inert JetSpec P5 default-off candidate plan."""

from __future__ import annotations

import pathlib
import unittest

import validate_p5_plan as validator


HERE = pathlib.Path(__file__).resolve().parent
PLAN_PATH = HERE / "jetspec_p5_default_off_plan.md"


class P5PlanTests(unittest.TestCase):
    def test_p5_plan_validator_passes(self) -> None:
        result = validator.validate_plan(PLAN_PATH)

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["phase_order"], ["P5A", "P5B", "P5C"])

    def test_plan_is_not_production_edit_approval(self) -> None:
        text = PLAN_PATH.read_text(encoding="utf-8")

        self.assertIn("Status: proposed inert plan only; not approved for production edits.", text)
        self.assertIn("This plan is not an approval to edit production paths.", text)
        self.assertIn("If approval is absent, keep all work under `experiments/jetspec/`.", text)
        self.assertNotIn("Status: accepted", text)
        self.assertNotIn("runtime_supported=true", text)

    def test_default_off_gates_are_explicit(self) -> None:
        text = PLAN_PATH.read_text(encoding="utf-8")

        self.assertIn("LLAMA_JETSPEC_EXPERIMENTAL=1", text)
        self.assertIn("LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1", text)
        self.assertIn("--spec-type draft-jetspec", text)
        self.assertIn("default disabled", text)
        self.assertIn("no default behavior change", text)

    def test_preview_files_remain_rejected_and_unrunnable(self) -> None:
        text = PLAN_PATH.read_text(encoding="utf-8")

        self.assertIn("preview_not_allowed", text)
        self.assertIn("unsupported_runtime", text)
        self.assertIn("metadata_only=true", text)
        self.assertIn("runtime_supported=false", text)
        self.assertIn("exactly 91 BF16 tensors", text)

    def test_minimal_candidate_files_are_named(self) -> None:
        text = PLAN_PATH.read_text(encoding="utf-8")

        for path_token in [
            "src/llama-arch.h",
            "src/llama-arch.cpp",
            "src/models/models.h",
            "src/models/jetspec_qwen3_draft_head.cpp",
            "src/llama-model.cpp",
            "common/common.h",
            "common/speculative.cpp",
            "common/speculative.h",
            "common/arg.cpp",
            "tools/server/server-context.cpp",
            "docs/speculative.md",
        ]:
            self.assertIn(path_token, text)

    def test_verification_matrix_protects_existing_paths(self) -> None:
        text = PLAN_PATH.read_text(encoding="utf-8")

        self.assertIn("Default server path", text)
        self.assertIn("Existing MTP path", text)
        self.assertIn("Existing n-gram path", text)
        self.assertIn("Metadata-only JetSpec GGUF preview fails closed by default", text)
        self.assertIn("matches or beats baseline", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
