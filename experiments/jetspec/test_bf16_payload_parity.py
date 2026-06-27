#!/usr/bin/env python3
"""Tests for the P2 inert BF16 payload parity fixture."""

from __future__ import annotations

import json
import pathlib
import unittest

import bf16_payload_parity as parity


HERE = pathlib.Path(__file__).resolve().parent
PLAN_PATH = HERE / "conversion_plans/JetSpec_jetspec-Qwen3.6-35B-A3B_gguf_plan.json"
FIXTURE_PATH = HERE / "fixtures/bf16_payload_parity_smoke.json"
EXPECTED_PATH = HERE / "fixtures/bf16_payload_parity_smoke.out.json"


class BF16PayloadParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = parity.load_plan(PLAN_PATH)
        self.header = parity.synthetic_safetensors_header(self.plan)

    def test_synthetic_header_matches_all_91_plan_tensors(self) -> None:
        result = parity.build_payload_loader_plan(self.plan, self.header)

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["payload_contract"]["tensor_count"], 91)
        self.assertEqual(result["payload_contract"]["expected_tensor_count"], 91)
        self.assertEqual(result["payload_contract"]["payload_bytes"], 947990528)
        self.assertEqual(result["payload_contract"]["dtype"], "BF16")
        self.assertEqual(result["payload_contract"]["ggml_type"], 30)
        self.assertEqual(result["payload_contract"]["copy_policy"], "raw_bf16_no_transform")
        self.assertEqual(result["payload_contract"]["quantization"], "none")
        self.assertEqual(result["payload_contract"]["packing"], "none")
        self.assertEqual(result["payload_contract"]["dtype_conversion"], "none")
        self.assertTrue(result["payload_contract"]["offsets_match_safetensors_header"])
        self.assertTrue(result["payload_contract"]["gguf_offsets_monotonic"])
        self.assertEqual(len(result["loader_tensors"]), 91)

    def test_loader_tensor_info_matches_future_contract_shape(self) -> None:
        result = parity.build_payload_loader_plan(self.plan, self.header)
        first = result["loader_tensors"][0]
        last = result["loader_tensors"][-1]

        self.assertEqual(first["gguf_name"], "draft.fc.weight")
        self.assertEqual(first["hf_name"], "fc.weight")
        self.assertEqual(first["shape"], [2048, 10240])
        self.assertEqual(first["ggml_type"], 30)
        self.assertEqual(first["nbytes"], 41943040)
        self.assertEqual(first["offset"], 0)
        self.assertEqual(first["source_data_offsets"], [0, 41943040])
        self.assertEqual(first["copy_policy"], "raw_bf16_no_transform")
        self.assertEqual(last["gguf_name"], "draft.norm.weight")
        self.assertEqual(last["hf_name"], "norm.weight")
        self.assertEqual(last["shape"], [2048])
        self.assertEqual(last["source_data_offsets"], [947986432, 947990528])

    def test_fixture_output_is_stable(self) -> None:
        actual = parity.evaluate_fixture(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))
        expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))

        self.assertEqual(actual, expected)

    def test_missing_tensor_fails_closed(self) -> None:
        bad = dict(self.header)
        bad.pop("fc.weight")
        result = parity.build_payload_loader_plan(self.plan, bad)

        self.assertFalse(result["ok"])
        self.assertTrue(any("missing tensors" in error for error in result["errors"]))
        self.assertEqual(result["loader_tensors"], [])

    def test_wrong_shape_fails_closed(self) -> None:
        bad = json.loads(json.dumps(self.header))
        bad["fc.weight"]["shape"] = [2048, 10239]
        result = parity.build_payload_loader_plan(self.plan, bad)

        self.assertFalse(result["ok"])
        self.assertTrue(any("shape mismatch" in error for error in result["errors"]))

    def test_wrong_dtype_fails_closed(self) -> None:
        bad = json.loads(json.dumps(self.header))
        bad["fc.weight"]["dtype"] = "F16"
        result = parity.build_payload_loader_plan(self.plan, bad)

        self.assertFalse(result["ok"])
        self.assertTrue(any("expected BF16 dtype" in error for error in result["errors"]))

    def test_wrong_offsets_fail_closed(self) -> None:
        bad = json.loads(json.dumps(self.header))
        bad["fc.weight"]["data_offsets"] = [0, 4]
        result = parity.build_payload_loader_plan(self.plan, bad)

        self.assertFalse(result["ok"])
        self.assertTrue(any("data_offsets mismatch" in error for error in result["errors"]))

    def test_self_test_covers_converter_sparse_header_validation(self) -> None:
        result = parity.self_test()

        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(result["converter_sparse_header_validation"]["ok"])
        self.assertFalse(result["converter_sparse_header_validation"]["sparse_payload_written"])
        self.assertEqual(sorted(result["rejected_cases"]), ["missing", "wrong_dtype", "wrong_offsets", "wrong_shape"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
