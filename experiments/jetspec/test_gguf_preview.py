#!/usr/bin/env python3
"""Tests for inert JetSpec GGUF preview parsing and loader-contract validation."""

from __future__ import annotations

import copy
import json
import pathlib
import struct
import tempfile
import unittest

import convert_jetspec_head_to_gguf as convert
import parse_gguf_preview as preview


HERE = pathlib.Path(__file__).resolve().parent
PLAN_PATH = HERE / "conversion_plans/JetSpec_jetspec-Qwen3.6-35B-A3B_gguf_plan.json"


class GGUFPreviewTests(unittest.TestCase):
    def _write_sparse_safetensors_shell(self, path: pathlib.Path, plan: dict) -> None:
        header = {
            tensor["hf_name"]: {
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "data_offsets": tensor["source_data_offsets"],
            }
            for tensor in plan["tensors"]
        }
        raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
        payload_bytes = int(plan["proposed_output"]["tensor_payload_bytes"])
        with path.open("wb") as f:
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)
            f.truncate(8 + len(raw) + payload_bytes)

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

    def test_payload_tensor_info_table_matches_plan_without_model_runtime(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="jetspec-gguf-payload-test-") as tmp_s:
            tmp = pathlib.Path(tmp_s)
            safetensors = tmp / "model.safetensors"
            out = tmp / "payload.gguf"
            self._write_sparse_safetensors_shell(safetensors, plan)

            original_copy = convert._copy_exact
            try:
                def sparse_copy(_src, dst, nbytes: int) -> None:
                    if nbytes <= 0:
                        return
                    dst.seek(nbytes - 1, 1)
                    dst.write(b"\0")

                convert._copy_exact = sparse_copy
                result = convert.write_gguf(out, plan, safetensors_path=safetensors)
            finally:
                convert._copy_exact = original_copy

            parsed = preview.parse_gguf(out)
            loader_validation = preview.validate_jetspec_loader_contract(parsed, allow_tensor_payload=True)
            payload_validation = preview.validate_tensor_payload_against_plan(parsed, plan)

        self.assertEqual(result["tensor_count"], 91)
        self.assertFalse(result["metadata_only"])
        self.assertEqual(parsed["tensor_count"], 91)
        self.assertTrue(loader_validation["ok"], loader_validation["errors"])
        self.assertTrue(payload_validation["ok"], payload_validation["errors"])
        self.assertFalse(payload_validation["runtime_executed"])
        self.assertEqual(payload_validation["payload_bytes"], 947990528)
        self.assertEqual(payload_validation["checked_first"][0]["name"], "draft.fc.weight")
        self.assertEqual(payload_validation["checked_last"][-1]["name"], "draft.norm.weight")

    def test_payload_tensor_info_validation_rejects_shape_drift(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        parsed = {
            "tensor_count": 1,
            "metadata": {"jetspec.experimental.metadata_only": {"value": False}},
            "tensors": [{"name": "draft.fc.weight", "shape": [1], "ggml_type": preview.GGML_TYPE_BF16, "offset": 0}],
            "data_start": 0,
            "file_size": 4,
        }
        validation = preview.validate_tensor_payload_against_plan(parsed, plan)

        self.assertFalse(validation["ok"])
        self.assertTrue(any("tensor_count mismatch" in error for error in validation["errors"]))
        self.assertTrue(any("shape mismatch" in error for error in validation["errors"]))

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
