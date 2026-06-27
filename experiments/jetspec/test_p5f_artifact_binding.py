#!/usr/bin/env python3
"""Tests for the offline P5F JetSpec artifact-binding validator."""

from __future__ import annotations

import unittest

from validate_p5f_artifact_binding import EXPECTED_ARTIFACT, validate_p5f_artifact_binding


class P5FArtifactBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validate_p5f_artifact_binding()
        cls.binding = cls.result["artifact_binding"]

    def test_validator_passes_but_does_not_claim_runtime_execution(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["status"], "artifact_verified_not_runtime_executed")
        self.assertTrue(self.result["artifact_verified"])
        self.assertFalse(self.result["runtime_executed"])
        self.assertTrue(self.result["not_runtime_executed"])
        self.assertIn("does not instantiate llama_context", self.result["limitations"])

    def test_real_manifest_matches_p5f_binding_metadata(self) -> None:
        self.assertEqual(self.binding["repo"], EXPECTED_ARTIFACT["repo"])
        self.assertEqual(self.binding["model_sha"], EXPECTED_ARTIFACT["model_sha"])
        self.assertEqual(self.binding["general_architecture"], "jetspec_qwen3_draft_head")
        self.assertEqual(self.binding["jetspec_architecture"], "qwen3_draft_head")
        self.assertEqual(self.binding["source_architecture"], "DFlashDraftModel")
        self.assertEqual(self.binding["tensor_dtype"], "bfloat16")
        self.assertFalse(self.binding["runtime_supported"])

    def test_real_tensors_prove_concatenated_target_tap_width(self) -> None:
        self.assertEqual(self.binding["target_layer_ids"], [1, 10, 19, 28, 37])
        self.assertEqual(self.binding["target_hidden"], 2048)
        self.assertEqual(self.binding["target_tap_count"], 5)
        self.assertEqual(self.binding["target_tap_width"], 5 * 2048)
        self.assertEqual(self.binding["fc_weight_shape"], [2048, 10240])
        self.assertTrue(self.binding["requires_target_embeddings"])
        self.assertTrue(self.binding["requires_target_lm_head"])

    def test_real_tensors_match_expected_qwen36_draft_shape(self) -> None:
        self.assertEqual(self.binding["draft_block_size"], 16)
        self.assertEqual(self.binding["draft_layers"], 8)
        self.assertEqual(self.binding["draft_heads"], 32)
        self.assertEqual(self.binding["draft_heads_kv"], 4)
        self.assertEqual(self.binding["vocab_size"], 248320)
        self.assertEqual(self.binding["tensor_count"], 91)
        self.assertEqual(self.binding["bf16_tensor_count"], 91)
        self.assertEqual(self.binding["byte_mismatch_count"], 0)

    def test_source_constants_match_artifact_facts(self) -> None:
        constants = self.result["source_constants"]
        self.assertEqual(constants["JETSPEC_QWEN36_DRAFT_BLOCK_SIZE"], 16)
        self.assertEqual(constants["JETSPEC_QWEN36_TARGET_TAP_COUNT"], 5)
        self.assertEqual(constants["JETSPEC_QWEN36_TARGET_HIDDEN"], 2048)
        self.assertEqual(constants["JETSPEC_QWEN36_TARGET_LAYERS"], 40)
        self.assertEqual(constants["JETSPEC_QWEN36_TARGET_TAP_WIDTH"], 10240)
        self.assertEqual(constants["JETSPEC_QWEN36_DRAFT_LAYERS"], 8)
        self.assertEqual(constants["JETSPEC_QWEN36_DRAFT_HEADS"], 32)
        self.assertEqual(constants["JETSPEC_QWEN36_DRAFT_HEADS_KV"], 4)
        self.assertEqual(constants["JETSPEC_QWEN36_VOCAB_SIZE"], 248320)


if __name__ == "__main__":
    unittest.main(verbosity=2)
