#!/usr/bin/env python3
"""Tests for the inert P5K JetSpec KV ownership primitive design packet."""

from __future__ import annotations

import copy
import json
import pathlib
import unittest

import kv_ownership_primitive_design as design
import validate_p5k_kv_ownership_primitive_design as validator


HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "kv_ownership_primitive_design_smoke.json"
EXPECTED = HERE / "fixtures" / "kv_ownership_primitive_design_smoke.out.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class P5KKVOwnershipPrimitiveDesignTests(unittest.TestCase):
    def test_smoke_fixture_matches_expected_output(self) -> None:
        result = design.evaluate_fixture(load_fixture())
        expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

        self.assertEqual(result, expected)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "kv_ownership_primitive_design_verified_not_executed")
        self.assertEqual(result["coverage"]["design_count"], 3)
        self.assertTrue(result["coverage"]["all_designs_missing_implementation"])
        self.assertTrue(result["runtime_boundary"]["design_only"])
        self.assertTrue(result["runtime_boundary"]["no_runtime_execution"])

    def test_rejects_missing_required_design(self) -> None:
        data = load_fixture()
        data["primitive_designs"] = [item for item in data["primitive_designs"] if item["id"] != "cross_sequence_isolation"]

        with self.assertRaisesRegex(design.KVOwnershipPrimitiveDesignError, "missing required primitive designs"):
            design.evaluate_fixture(data)

    def test_rejects_helper_marked_exact(self) -> None:
        data = load_fixture()
        data["audited_non_exact_helpers"] = copy.deepcopy(data["audited_non_exact_helpers"])
        data["audited_non_exact_helpers"][0]["exact_for_actions"] = ["rejected_branch_discard"]

        with self.assertRaisesRegex(design.KVOwnershipPrimitiveDesignError, "exact_for_actions must be empty"):
            design.evaluate_fixture(data)

    def test_rejects_missing_forbidden_implicit_mapping(self) -> None:
        data = load_fixture()
        data["primitive_designs"] = copy.deepcopy(data["primitive_designs"])
        data["primitive_designs"][0]["must_not_use_as_implicit_mapping"] = ["seq_cp", "seq_rm"]

        with self.assertRaisesRegex(design.KVOwnershipPrimitiveDesignError, "must forbid implicit helpers"):
            design.evaluate_fixture(data)

    def test_rejects_implementation_approval(self) -> None:
        data = load_fixture()
        data["primitive_designs"] = copy.deepcopy(data["primitive_designs"])
        data["primitive_designs"][0]["implementation_approved"] = True

        with self.assertRaisesRegex(design.KVOwnershipPrimitiveDesignError, "implementation_approved must be false"):
            design.evaluate_fixture(data)

    def test_rejects_bad_primitive_prefix(self) -> None:
        data = load_fixture()
        data["primitive_designs"] = copy.deepcopy(data["primitive_designs"])
        data["primitive_designs"][0]["proposed_primitive"] = "seq_cp"

        with self.assertRaisesRegex(design.KVOwnershipPrimitiveDesignError, "llama_kv_cache_jetspec_"):
            design.evaluate_fixture(data)

    def test_rejects_runtime_boundary_crossing(self) -> None:
        data = load_fixture()
        data["runtime_boundary"] = copy.deepcopy(data["runtime_boundary"])
        data["runtime_boundary"]["real_kv_mutated"] = True

        with self.assertRaisesRegex(design.KVOwnershipPrimitiveDesignError, "real_kv_mutated=false"):
            design.evaluate_fixture(data)

    def test_validator_passes(self) -> None:
        result = validator.validate_p5k_kv_ownership_primitive_design()

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "p5k_kv_ownership_primitive_design_validated")
        self.assertEqual(result["forbidden_root_hits"], [])
        self.assertEqual(result["cmake_hits"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
