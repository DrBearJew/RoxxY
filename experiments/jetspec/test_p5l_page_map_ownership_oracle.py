#!/usr/bin/env python3
"""Tests for the inert P5L JetSpec page-map ownership oracle."""

from __future__ import annotations

import copy
import json
import pathlib
import unittest

import page_map_ownership_oracle as oracle
import validate_p5l_page_map_ownership_oracle as validator


HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "page_map_ownership_oracle_smoke.json"
EXPECTED = HERE / "fixtures" / "page_map_ownership_oracle_smoke.out.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class P5LPageMapOwnershipOracleTests(unittest.TestCase):
    def test_smoke_fixture_matches_expected_output(self) -> None:
        result = oracle.evaluate_fixture(load_fixture())
        expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

        self.assertEqual(result, expected)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "page_map_ownership_oracle_verified_not_executed")
        self.assertTrue(result["runtime_boundary"]["page_map_oracle_only"])
        self.assertTrue(result["coverage"]["accepted_survivor_mapping_exact"])
        self.assertTrue(result["coverage"]["rejected_branches_unreachable"])
        self.assertTrue(result["coverage"]["cross_sequence_isolation_holds"])

    def test_rejects_missing_required_oracle(self) -> None:
        data = load_fixture()
        data["oracle_cases"] = [item for item in data["oracle_cases"] if item["id"] != "cross_sequence_page_isolation"]

        with self.assertRaisesRegex(oracle.PageMapOwnershipOracleError, "missing required oracle cases"):
            oracle.evaluate_fixture(data)

    def test_rejects_duplicate_page_owner(self) -> None:
        data = load_fixture()
        data["page_ownership"] = copy.deepcopy(data["page_ownership"])
        data["page_ownership"]["page_owners"][5]["physical_page"] = 12

        with self.assertRaisesRegex(oracle.PageMapOwnershipOracleError, "duplicate mutable physical page ownership"):
            oracle.evaluate_fixture(data)

    def test_rejects_rejected_page_reachable(self) -> None:
        data = load_fixture()
        data["tree_round"] = copy.deepcopy(data["tree_round"])
        data["tree_round"]["post_commit_survivor_pages"] = [10, 11, 13]

        with self.assertRaisesRegex(oracle.PageMapOwnershipOracleError, "post_commit_survivor_pages must exactly match accepted path pages"):
            oracle.evaluate_fixture(data)

    def test_rejects_accepted_path_without_root(self) -> None:
        data = load_fixture()
        data["tree_round"] = copy.deepcopy(data["tree_round"])
        data["tree_round"]["accepted_path"] = [1, 4]

        with self.assertRaisesRegex(oracle.PageMapOwnershipOracleError, "accepted_path must begin with the root node"):
            oracle.evaluate_fixture(data)

    def test_rejects_implicit_helper_mapping(self) -> None:
        data = load_fixture()
        data["helper_mappings"] = copy.deepcopy(data["helper_mappings"])
        data["helper_mappings"][0]["implicit_mapping_allowed"] = True

        with self.assertRaisesRegex(oracle.PageMapOwnershipOracleError, "implicit mapping must be forbidden"):
            oracle.evaluate_fixture(data)

    def test_rejects_cross_sequence_mutation(self) -> None:
        data = load_fixture()
        data["page_ownership"] = copy.deepcopy(data["page_ownership"])
        data["page_ownership"]["other_sequence_pages_after"] = [50, 52]

        with self.assertRaisesRegex(oracle.PageMapOwnershipOracleError, "other-sequence pages must remain unchanged"):
            oracle.evaluate_fixture(data)

    def test_rejects_runtime_boundary_crossing(self) -> None:
        data = load_fixture()
        data["runtime_boundary"] = copy.deepcopy(data["runtime_boundary"])
        data["runtime_boundary"]["real_kv_mutated"] = True

        with self.assertRaisesRegex(oracle.PageMapOwnershipOracleError, "real_kv_mutated=false"):
            oracle.evaluate_fixture(data)

    def test_rejects_production_path_touch(self) -> None:
        data = load_fixture()
        data["production_paths_touched"] = ["ggml/src/ggml-cuda/fattn.cu"]

        with self.assertRaisesRegex(oracle.PageMapOwnershipOracleError, "must not touch production paths"):
            oracle.evaluate_fixture(data)

    def test_validator_passes(self) -> None:
        result = validator.validate_p5l_page_map_ownership_oracle()

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "p5l_page_map_ownership_oracle_validated")


if __name__ == "__main__":
    unittest.main(verbosity=2)
