#!/usr/bin/env python3
"""Tests for the inert P5M JetSpec transaction/failpoint plan oracle."""

from __future__ import annotations

import copy
import json
import pathlib
import unittest

import transaction_plan_oracle as oracle
import validate_p5m_transaction_plan_oracle as validator


HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "transaction_plan_oracle_smoke.json"
EXPECTED = HERE / "fixtures" / "transaction_plan_oracle_smoke.out.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def phase(data: dict, phase_id: str) -> dict:
    for item in data["transaction_phases"]:
        if item["id"] == phase_id:
            return item
    raise AssertionError(f"missing phase {phase_id}")


class P5MTransactionPlanOracleTests(unittest.TestCase):
    def test_smoke_fixture_matches_expected_output(self) -> None:
        result = oracle.evaluate_fixture(load_fixture())
        expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

        self.assertEqual(result, expected)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "transaction_plan_oracle_verified_not_executed")
        self.assertTrue(result["runtime_boundary"]["transaction_plan_oracle_only"])
        self.assertTrue(result["coverage"]["required_phases_ordered"])
        self.assertTrue(result["coverage"]["required_rollback_points_present"])
        self.assertTrue(result["coverage"]["rollback_restores_pre_round_snapshot"])
        self.assertTrue(result["coverage"]["post_publish_tokens_are_accepted_plus_correction"])

    def test_rejects_missing_rollback_point(self) -> None:
        data = load_fixture()
        data["rollback_points"] = [item for item in data["rollback_points"] if item["id"] != "after_accept"]

        with self.assertRaisesRegex(oracle.TransactionPlanOracleError, "missing rollback points"):
            oracle.evaluate_fixture(data)

    def test_rejects_publish_before_all_commits_and_discards(self) -> None:
        data = load_fixture()
        phase(data, "publish_post_commit_state")["requires_after"] = ["commit_tokens", "commit_hidden_kv_survivors"]

        with self.assertRaisesRegex(oracle.TransactionPlanOracleError, "must require token commit"):
            oracle.evaluate_fixture(data)

    def test_rejects_token_commit_before_accept_path(self) -> None:
        data = load_fixture()
        phase(data, "commit_tokens")["requires_after"] = ["build_verify_mask"]

        with self.assertRaisesRegex(oracle.TransactionPlanOracleError, "commit_tokens must require accept_path"):
            oracle.evaluate_fixture(data)

    def test_rejects_hidden_kv_commit_before_survivor_pages_validated(self) -> None:
        data = load_fixture()
        phase(data, "commit_hidden_kv_survivors")["survivor_pages_validated_by"] = "implicit_seq_cp"

        with self.assertRaisesRegex(oracle.TransactionPlanOracleError, "must cite p5l_page_map_oracle"):
            oracle.evaluate_fixture(data)

    def test_rejects_rejected_branch_reachable_after_discard(self) -> None:
        data = load_fixture()
        phase(data, "discard_rejected_branches")["rejected_pages_reachable_after_discard"] = True

        with self.assertRaisesRegex(oracle.TransactionPlanOracleError, "must not be reachable"):
            oracle.evaluate_fixture(data)

    def test_rejects_rollback_that_mutates_other_sequence_pages(self) -> None:
        data = load_fixture()
        data["rollback_points"] = copy.deepcopy(data["rollback_points"])
        data["rollback_points"][4]["other_sequence_pages_after_rollback"] = [50, 52]

        with self.assertRaisesRegex(oracle.TransactionPlanOracleError, "preserve other-sequence pages"):
            oracle.evaluate_fixture(data)

    def test_rejects_duplicate_mutable_page_ownership(self) -> None:
        data = load_fixture()
        data["page_ownership"] = copy.deepcopy(data["page_ownership"])
        data["page_ownership"]["page_owners"][5]["physical_page"] = 12

        with self.assertRaisesRegex(oracle.TransactionPlanOracleError, "duplicate mutable physical page ownership"):
            oracle.evaluate_fixture(data)

    def test_rejects_runtime_boundary_crossing(self) -> None:
        data = load_fixture()
        data["runtime_boundary"] = copy.deepcopy(data["runtime_boundary"])
        data["runtime_boundary"]["draft_tokens_emitted"] = True

        with self.assertRaisesRegex(oracle.TransactionPlanOracleError, "draft_tokens_emitted=false"):
            oracle.evaluate_fixture(data)

    def test_rejects_production_path_touch(self) -> None:
        data = load_fixture()
        data["production_paths_touched"] = ["common/speculative.cpp"]

        with self.assertRaisesRegex(oracle.TransactionPlanOracleError, "must not touch production paths"):
            oracle.evaluate_fixture(data)

    def test_rejects_implementation_approval(self) -> None:
        data = load_fixture()
        data["candidate_primitives"] = copy.deepcopy(data["candidate_primitives"])
        data["candidate_primitives"][0]["implementation_approved"] = True

        with self.assertRaisesRegex(oracle.TransactionPlanOracleError, "implementation_approved must be false"):
            oracle.evaluate_fixture(data)

    def test_rejects_performance_or_promotion_claim(self) -> None:
        data = load_fixture()
        data["candidate_primitives"] = copy.deepcopy(data["candidate_primitives"])
        data["candidate_primitives"][0]["claims_performance"] = True

        with self.assertRaisesRegex(oracle.TransactionPlanOracleError, "claims_performance must be false"):
            oracle.evaluate_fixture(data)

        data = load_fixture()
        data["candidate_primitives"] = copy.deepcopy(data["candidate_primitives"])
        data["candidate_primitives"][0]["claims_promotion"] = True

        with self.assertRaisesRegex(oracle.TransactionPlanOracleError, "claims_promotion must be false"):
            oracle.evaluate_fixture(data)

    def test_rejects_implicit_helper_mapping(self) -> None:
        data = load_fixture()
        data["helper_mappings"] = copy.deepcopy(data["helper_mappings"])
        data["helper_mappings"][0]["implicit_mapping_allowed"] = True

        with self.assertRaisesRegex(oracle.TransactionPlanOracleError, "implicit mapping must be forbidden"):
            oracle.evaluate_fixture(data)

    def test_validator_passes(self) -> None:
        result = validator.validate_p5m_transaction_plan_oracle()

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "p5m_transaction_plan_oracle_validated")


if __name__ == "__main__":
    unittest.main(verbosity=2)
