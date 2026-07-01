#!/usr/bin/env python3
"""Tests for the P5F target tensor header-only binding probe."""

from __future__ import annotations

import copy
import pathlib
import unittest

from probe_p5f_target_tensor_binding import (
    EXPECTED_TARGET_TENSORS,
    _derive_split_paths,
    _self_test,
    validate_target_tensor_headers,
)


GOOD_HEADERS = [
    {
        "path": "target-00001-of-00002.gguf",
        "tensor_count": 0,
        "metadata": {"split.tensors.count": {"value": 3}},
        "tensors": [],
    },
    {
        "path": "target-00002-of-00002.gguf",
        "tensor_count": 3,
        "metadata": {},
        "tensors": [
            {"name": "output.weight", "shape": [248320, 2048], "ggml_type": 8, "offset": 0},
            {"name": "output_norm.weight", "shape": [2048], "ggml_type": 0, "offset": 1},
            {"name": "token_embd.weight", "shape": [248320, 2048], "ggml_type": 8, "offset": 2},
        ],
    },
]


class P5FTargetTensorBindingProbeTests(unittest.TestCase):
    def test_self_test_passes_without_runtime_execution(self) -> None:
        result = _self_test()
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "target_tensor_headers_verified_not_loaded")
        self.assertFalse(result["runtime_executed"])
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["context_created"])
        self.assertFalse(result["draft_tokens_emitted"])
        self.assertIn("does not execute draft-head graph/tree/rollback runtime", result["limitations"])

    def test_expected_tensor_shapes_match_linker_contract(self) -> None:
        self.assertEqual(EXPECTED_TARGET_TENSORS["token_embd.weight"]["gguf_shape"], [248320, 2048])
        self.assertEqual(EXPECTED_TARGET_TENSORS["token_embd.weight"]["linker_shape"], [2048, 248320])
        self.assertEqual(EXPECTED_TARGET_TENSORS["output.weight"]["gguf_shape"], [248320, 2048])
        self.assertEqual(EXPECTED_TARGET_TENSORS["output.weight"]["linker_shape"], [2048, 248320])
        self.assertEqual(EXPECTED_TARGET_TENSORS["output_norm.weight"]["gguf_shape"], [2048])
        self.assertEqual(EXPECTED_TARGET_TENSORS["output_norm.weight"]["linker_shape"], [2048])

    def test_validation_accepts_split_headers_with_required_target_tensors(self) -> None:
        result = validate_target_tensor_headers(copy.deepcopy(GOOD_HEADERS))
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["headers_checked"], 2)
        self.assertEqual(result["split_tensor_count"], 3)
        self.assertEqual(result["split_declared_total"], 3)
        self.assertEqual(set(result["checked_tensors"]), set(EXPECTED_TARGET_TENSORS))

    def test_validation_rejects_missing_lm_head(self) -> None:
        headers = copy.deepcopy(GOOD_HEADERS)
        headers[1]["tensors"] = [t for t in headers[1]["tensors"] if t["name"] != "output.weight"]
        result = validate_target_tensor_headers(headers)
        self.assertFalse(result["ok"])
        self.assertTrue(any("missing target tensor output.weight" in error for error in result["errors"]))

    def test_validation_rejects_shape_drift(self) -> None:
        headers = copy.deepcopy(GOOD_HEADERS)
        for tensor in headers[1]["tensors"]:
            if tensor["name"] == "token_embd.weight":
                tensor["shape"] = [2048, 248320]
        result = validate_target_tensor_headers(headers)
        self.assertFalse(result["ok"])
        self.assertTrue(any("token_embd.weight shape" in error for error in result["errors"]))

    def test_split_path_derivation_is_bounded(self) -> None:
        paths = _derive_split_paths(pathlib.Path("model-00001-of-00002.gguf"))
        self.assertEqual([path.name for path in paths], ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
