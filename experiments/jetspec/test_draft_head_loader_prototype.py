#!/usr/bin/env python3
"""Tests for the P1 inert JetSpec draft-head loader prototype."""

from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest

import convert_jetspec_head_to_gguf as convert
import draft_head_loader_prototype as loader
import parse_gguf_preview as preview


HERE = pathlib.Path(__file__).resolve().parent
PLAN_PATH = HERE / "conversion_plans/JetSpec_jetspec-Qwen3.6-35B-A3B_gguf_plan.json"


class DraftHeadLoaderPrototypeTests(unittest.TestCase):
    def _write_preview(self, tmp_s: str) -> pathlib.Path:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        out = pathlib.Path(tmp_s) / "preview.gguf"
        convert.write_gguf(out, plan)
        return out

    def test_metadata_only_preview_maps_to_draft_head_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jetspec-loader-p1-test-") as tmp_s:
            gguf = self._write_preview(tmp_s)
            result = loader.build_loader_plan(gguf)

        self.assertTrue(result["ok"], result["errors"])
        plan = result["loader_plan"]
        metadata = plan["metadata"]
        self.assertEqual(plan["payload_mode"], "metadata_only")
        self.assertTrue(plan["preview_file"])
        self.assertTrue(plan["metadata_only"])
        self.assertFalse(plan["allow_preview_runtime"])
        self.assertEqual(metadata["gguf_arch"], "jetspec_qwen3_draft_head")
        self.assertEqual(metadata["head_arch"], "qwen3_draft_head")
        self.assertEqual(metadata["source_arch"], "DFlashDraftModel")
        self.assertEqual(metadata["block_size"], 16)
        self.assertEqual(metadata["draft_depth"], 15)
        self.assertEqual(metadata["target_layer_ids"], [1, 10, 19, 28, 37])
        self.assertEqual(metadata["hidden_size"], 2048)
        self.assertEqual(metadata["concat_width"], 10240)
        self.assertEqual(metadata["draft_layers"], 8)
        self.assertEqual(metadata["attention_heads"], 32)
        self.assertEqual(metadata["attention_heads_kv"], 4)
        self.assertEqual(metadata["head_dim"], 128)
        self.assertEqual(metadata["ffn_size"], 6144)
        self.assertEqual(metadata["vocab_size"], 248320)
        self.assertTrue(metadata["causal_head"])
        self.assertTrue(metadata["requires_target_embeddings"])
        self.assertTrue(metadata["requires_target_lm_head"])
        self.assertFalse(metadata["runtime_supported"])
        self.assertEqual(result["runtime_gate"]["failure"], "unsupported_runtime")
        self.assertFalse(result["runtime_gate"]["can_prepare_runtime"])

    def test_prepare_runtime_fails_closed_without_preview_flag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jetspec-loader-p1-test-") as tmp_s:
            gguf = self._write_preview(tmp_s)
            result = loader.build_loader_plan(gguf, prepare_runtime=True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["runtime_gate"]["failure"], "preview_not_allowed")
        self.assertTrue(any("preview_not_allowed" in error for error in result["errors"]))
        self.assertFalse(result["runtime_gate"]["allow_preview_runtime"])

    def test_prepare_runtime_with_preview_flag_still_preserves_runtime_unsupported(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jetspec-loader-p1-test-") as tmp_s:
            gguf = self._write_preview(tmp_s)
            result = loader.build_loader_plan(gguf, prepare_runtime=True, allow_preview_runtime=True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["runtime_gate"]["failure"], "unsupported_runtime")
        self.assertTrue(any("runtime_supported=false" in error for error in result["errors"]))
        self.assertTrue(result["runtime_gate"]["allow_preview_runtime"])
        self.assertFalse(result["loader_plan"]["metadata"]["runtime_supported"])

    def test_mapping_fails_closed_for_duplicate_target_layers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jetspec-loader-p1-test-") as tmp_s:
            gguf = self._write_preview(tmp_s)
            parsed = preview.parse_gguf(gguf)

        tampered = copy.deepcopy(parsed)
        tampered["metadata"]["jetspec.target_layer_ids"]["value"] = [1, 10, 10, 37, 28]
        result = loader.build_loader_plan_from_parsed(tampered)

        self.assertFalse(result["ok"])
        self.assertTrue(any("jetspec.target_layer_ids" in error or "target_layer_ids" in error for error in result["errors"]))

    def test_loader_fails_closed_for_required_metadata_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jetspec-loader-p1-test-") as tmp_s:
            gguf = self._write_preview(tmp_s)
            parsed = preview.parse_gguf(gguf)

        cases = [
            ("jetspec.requires_target_embeddings", False, "requires_target_embeddings"),
            ("jetspec.requires_target_lm_head", False, "requires_target_lm_head"),
            ("jetspec.causal_head", False, "causal_head"),
            ("jetspec.experimental.runtime_supported", True, "runtime_supported"),
        ]
        for key, value, needle in cases:
            with self.subTest(key=key):
                tampered = copy.deepcopy(parsed)
                tampered["metadata"][key]["value"] = value
                result = loader.build_loader_plan_from_parsed(tampered)
                self.assertFalse(result["ok"])
                self.assertTrue(any(needle in error for error in result["errors"]), result["errors"])

    def test_loader_fails_closed_for_head_dim_mismatch_and_missing_key(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jetspec-loader-p1-test-") as tmp_s:
            gguf = self._write_preview(tmp_s)
            parsed = preview.parse_gguf(gguf)

        mismatch = copy.deepcopy(parsed)
        mismatch["metadata"]["jetspec.attention.value_length"]["value"] = 64
        mismatch_result = loader.build_loader_plan_from_parsed(mismatch)
        self.assertFalse(mismatch_result["ok"])
        self.assertTrue(any("value_length" in error or "head_dim mismatch" in error for error in mismatch_result["errors"]))

        missing = copy.deepcopy(parsed)
        missing["metadata"].pop("jetspec.requires_target_lm_head")
        missing_result = loader.build_loader_plan_from_parsed(missing)
        self.assertFalse(missing_result["ok"])
        self.assertTrue(any("jetspec.requires_target_lm_head" in error for error in missing_result["errors"]))

    def test_self_test_passes(self) -> None:
        result = loader.self_test()

        self.assertTrue(result["ok"])
        self.assertEqual(result["payload_mode"], "metadata_only")
        self.assertEqual(result["runtime_gate_without_flag"]["failure"], "preview_not_allowed")
        self.assertEqual(result["runtime_gate_with_flag"]["failure"], "unsupported_runtime")


if __name__ == "__main__":
    unittest.main(verbosity=2)
