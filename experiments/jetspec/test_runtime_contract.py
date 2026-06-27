#!/usr/bin/env python3
"""Tests for inert JetSpec runtime C++ contract header."""

from __future__ import annotations

import pathlib
import re
import unittest

import validate_runtime_contract as validator


HERE = pathlib.Path(__file__).resolve().parent
HEADER_PATH = HERE / "jetspec_runtime_contract.hpp"


class RuntimeContractTests(unittest.TestCase):
    def test_runtime_header_validator_passes(self) -> None:
        result = validator.validate_header(HEADER_PATH)

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["constants"]["jetspec_qwen36_block_size"], 16)
        self.assertEqual(result["constants"]["jetspec_qwen36_draft_depth"], 15)
        self.assertEqual(result["constants"]["jetspec_qwen36_concat_width"], 10240)
        self.assertEqual(result["constants"]["jetspec_qwen36_tensor_count"], 91)

    def test_constants_preserve_shape_invariants(self) -> None:
        constants = validator.validate_header(HEADER_PATH)["constants"]

        self.assertEqual(constants["jetspec_qwen36_draft_depth"], constants["jetspec_qwen36_block_size"] - 1)
        self.assertEqual(
            constants["jetspec_qwen36_concat_width"],
            constants["jetspec_qwen36_target_tap_count"] * constants["jetspec_qwen36_hidden_size"],
        )

    def test_header_contains_required_struct_boundaries(self) -> None:
        text = HEADER_PATH.read_text(encoding="utf-8")

        for struct_name in validator.REQUIRED_STRUCTS:
            self.assertRegex(text, rf"struct\s+{re.escape(struct_name)}\b")
        self.assertIn("draft_head_loader_plan loader", text)
        self.assertIn("target_model_bindings target", text)
        self.assertIn("target_hidden_cache_state hidden_cache", text)
        self.assertIn("tree_verify_plan verify", text)
        self.assertIn("round_commit_plan commit", text)

    def test_header_has_no_production_includes_or_types(self) -> None:
        result = validator.validate_header(HEADER_PATH)

        self.assertEqual(result["includes"], ["jetspec_tree_contract.hpp", "cstdint", "string", "vector"])
        self.assertFalse(any("llama.h" in error or "ggml" in error for error in result["errors"]))

    def test_header_keeps_runtime_unsupported_by_default(self) -> None:
        text = HEADER_PATH.read_text(encoding="utf-8")

        self.assertIn("bool runtime_supported = false", text)
        self.assertIn("bool preview_file = true", text)
        self.assertIn("bool allow_preview_runtime = false", text)
        self.assertIn("unsupported_runtime", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
