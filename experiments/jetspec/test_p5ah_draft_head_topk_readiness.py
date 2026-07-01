#!/usr/bin/env python3
"""Tests for the inert P5AH JetSpec draft-head top-k readiness descriptor."""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jetspec_draft_head_topk_readiness as readiness
import validate_p5ah_draft_head_topk_readiness as validator


FIXTURE = HERE / "fixtures" / "jetspec_draft_head_topk_readiness_smoke.json"
EXPECTED = HERE / "fixtures" / "jetspec_draft_head_topk_readiness_smoke.out.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class P5AHDraftHeadTopKReadinessTests(unittest.TestCase):
    def test_smoke_fixture_matches_expected_output(self) -> None:
        result = readiness.evaluate_fixture(load_fixture())
        expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

        self.assertEqual(result, expected)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "draft_head_topk_readiness_verified_not_executed")
        self.assertFalse(result["production_source_approval"])
        self.assertEqual(result["future_topk_contract"]["future_logits_source"], "draft_head_full_vocab_logits")
        self.assertEqual(result["future_topk_contract"]["planned_draft_head_logits_rows"], 1)
        self.assertEqual(result["future_topk_contract"]["actual_verified_logits_rows"], 0)
        self.assertEqual(result["future_topk_contract"]["parent_logits_rows"], [
            {"logits_row": 0, "parent_node": 0, "candidate_nodes": [1, 2]},
        ])
        self.assertTrue(result["runtime_boundary"]["ctx_dft_null"])
        self.assertTrue(result["runtime_boundary"]["no_runtime_execution"])
        self.assertFalse(result["runtime_boundary"]["draft_head_graph_executed"])

    def test_rejects_missing_p5ag_gate(self) -> None:
        data = load_fixture()
        data["chain"] = copy.deepcopy(data["chain"])
        data["chain"]["requires_gates"].remove("LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1")

        with self.assertRaisesRegex(readiness.DraftHeadTopKReadinessError, "missing required P5AG chain gates"):
            readiness.evaluate_fixture(data)

    def test_rejects_missing_p5ag_hash(self) -> None:
        data = load_fixture()
        data["chain"] = copy.deepcopy(data["chain"])
        data["chain"]["p5ag_accept_boundary_hash_required"] = False

        with self.assertRaisesRegex(readiness.DraftHeadTopKReadinessError, "p5ag_accept_boundary_hash_required"):
            readiness.evaluate_fixture(data)

    def test_rejects_actual_logits_rows_without_approval(self) -> None:
        data = load_fixture()
        data["future_topk_contract"] = copy.deepcopy(data["future_topk_contract"])
        data["future_topk_contract"]["actual_verified_logits_rows"] = 1

        with self.assertRaisesRegex(readiness.DraftHeadTopKReadinessError, "actual_verified_logits_rows must be 0"):
            readiness.evaluate_fixture(data)

    def test_rejects_target_logits_source(self) -> None:
        data = load_fixture()
        data["future_topk_contract"] = copy.deepcopy(data["future_topk_contract"])
        data["future_topk_contract"]["future_logits_source"] = "target_logits"

        with self.assertRaisesRegex(readiness.DraftHeadTopKReadinessError, "future_logits_source"):
            readiness.evaluate_fixture(data)

    def test_rejects_sampler_source_missing_forbidden(self) -> None:
        data = load_fixture()
        data["future_topk_contract"] = copy.deepcopy(data["future_topk_contract"])
        data["future_topk_contract"]["forbidden_sources"].remove("sampler")

        with self.assertRaisesRegex(readiness.DraftHeadTopKReadinessError, "missing forbidden logits sources"):
            readiness.evaluate_fixture(data)

    def test_rejects_draft_context_creation(self) -> None:
        data = load_fixture()
        data["runtime_boundary"] = copy.deepcopy(data["runtime_boundary"])
        data["runtime_boundary"]["ctx_dft_null"] = False
        data["runtime_boundary"]["draft_context_created"] = True

        with self.assertRaisesRegex(readiness.DraftHeadTopKReadinessError, "draft_context_created"):
            readiness.evaluate_fixture(data)

    def test_rejects_llama_decode_call(self) -> None:
        data = load_fixture()
        data["runtime_boundary"] = copy.deepcopy(data["runtime_boundary"])
        data["runtime_boundary"]["llama_decode_called"] = True

        with self.assertRaisesRegex(readiness.DraftHeadTopKReadinessError, "llama_decode_called must be false"):
            readiness.evaluate_fixture(data)

    def test_rejects_production_path_touch(self) -> None:
        data = load_fixture()
        data["production_paths_touched"] = ["common/speculative.cpp"]

        with self.assertRaisesRegex(readiness.DraftHeadTopKReadinessError, "must not touch production paths"):
            readiness.evaluate_fixture(data)

    def test_rejects_production_source_approval(self) -> None:
        data = load_fixture()
        data["production_source_approval"] = True

        with self.assertRaisesRegex(readiness.DraftHeadTopKReadinessError, "production_source_approval must be false"):
            readiness.evaluate_fixture(data)

    def test_validator_passes(self) -> None:
        result = validator.validate_p5ah_draft_head_topk_readiness()

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "p5ah_draft_head_topk_readiness_validated")
        self.assertEqual(result["forbidden_root_hits"], [])
        self.assertEqual(result["cmake_hits"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
