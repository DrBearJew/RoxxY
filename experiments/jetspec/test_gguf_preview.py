#!/usr/bin/env python3
"""Tests for inert JetSpec GGUF preview parsing and loader-contract validation."""

from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest

import convert_jetspec_head_to_gguf as convert
import parse_gguf_preview as preview


HERE = pathlib.Path(__file__).resolve().parent
PLAN_PATH = HERE / "conversion_plans/JetSpec_jetspec-Qwen3.6-35B-A3B_gguf_plan.json"


class GGUFPreviewTests(unittest.TestCase):
    def test_metadata_only_preview_satisfies_loader_contract(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="jetspec-gguf-preview-test-") as tmp_s:
            out = pathlib.Path(tmp_s) / "preview.gguf"
            result = convert.write_gguf(out, plan)
            parsed = preview.parse_gguf(out)
            validation = preview.validate_jetspec_loader_contract(parsed)

        self.assertEqual(result["tensor_count"], 0)
        self.assertEqual(result["metadata_count"], 30)
        self.assertEqual(parsed["version"], 3)
        self.assertEqual(parsed["tensor_count"], 0)
        self.assertEqual(parsed["kv_count"], 30)
        self.assertTrue(validation["ok"], validation["errors"])
        self.assertEqual(validation["loader_contract"]["target_layer_ids"], [1, 10, 19, 28, 37])

    def test_loader_contract_fails_closed_on_wrong_architecture(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="jetspec-gguf-preview-test-") as tmp_s:
            out = pathlib.Path(tmp_s) / "preview.gguf"
            convert.write_gguf(out, plan)
            parsed = preview.parse_gguf(out)

        tampered = copy.deepcopy(parsed)
        tampered["metadata"]["general.architecture"]["value"] = "qwen3"
        validation = preview.validate_jetspec_loader_contract(tampered)

        self.assertFalse(validation["ok"])
        self.assertTrue(any("general.architecture" in error for error in validation["errors"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
