#!/usr/bin/env python3
"""Tests for the inert P5H JetSpec KV/hidden commit readiness contract."""

from __future__ import annotations

import copy
import json
import pathlib
import unittest

import jetspec_kv_commit_readiness as readiness
import validate_p5h_kv_commit_readiness as validator


HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "jetspec_kv_commit_readiness_smoke.json"
EXPECTED = HERE / "fixtures" / "jetspec_kv_commit_readiness_smoke.out.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class P5HKVCommitReadinessTests(unittest.TestCase):
    def test_smoke_fixture_matches_expected_output(self) -> None:
        result = readiness.evaluate_fixture(load_fixture())
        expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

        self.assertEqual(result, expected)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "kv_commit_readiness_verified_not_executed")
        self.assertEqual(result["slot_ownership"]["gather_positions"], [2, 3, 6])
        self.assertEqual(result["commit"]["committed_append_tokens"], [11, 22, 99])
        self.assertEqual(result["slot_ownership"]["post_kv_tokens"], [7, 8, 10, 11, 22])
        self.assertTrue(result["slot_ownership"]["cross_sequence_isolated"])
        self.assertTrue(result["readiness_boundary"]["no_production_kv_mutation"])

    def test_rejects_duplicate_accepted_path(self) -> None:
        data = load_fixture()
        data["accepted_path"] = [0, 1, 1]

        with self.assertRaisesRegex(readiness.KVCommitReadinessError, "duplicate nodes"):
            readiness.evaluate_fixture(data)

    def test_rejects_out_of_range_accepted_path(self) -> None:
        data = load_fixture()
        data["accepted_path"] = [0, 99]

        with self.assertRaisesRegex(readiness.KVCommitReadinessError, "out-of-range"):
            readiness.evaluate_fixture(data)

    def test_rejects_gather_position_mismatch(self) -> None:
        data = load_fixture()
        data["expected_gather_positions"] = [2, 3, 5]

        with self.assertRaisesRegex(readiness.KVCommitReadinessError, "gather_positions mismatch"):
            readiness.evaluate_fixture(data)

    def test_rejects_cross_sequence_slot_touch(self) -> None:
        data = load_fixture()
        data["other_sequence_slots"][0]["logical_slot"] = 3

        with self.assertRaisesRegex(readiness.KVCommitReadinessError, "cross-sequence isolation violation"):
            readiness.evaluate_fixture(data)

    def test_rejects_silent_ownership_mapping(self) -> None:
        data = load_fixture()
        data["llama_cpp_ownership_mapping"] = copy.deepcopy(data["llama_cpp_ownership_mapping"])
        data["llama_cpp_ownership_mapping"]["gather_accepted_path"] = "seq_cp"

        with self.assertRaisesRegex(readiness.KVCommitReadinessError, r"exact llama_kv_cache_\* primitive"):
            readiness.evaluate_fixture(data)

    def test_accepts_exact_llama_kv_cache_primitive_names(self) -> None:
        data = load_fixture()
        data["llama_cpp_ownership_mapping"] = copy.deepcopy(data["llama_cpp_ownership_mapping"])
        data["llama_cpp_ownership_mapping"]["gather_accepted_path"] = "llama_kv_cache_future_gather_accepted_path"

        result = readiness.evaluate_fixture(data)

        self.assertEqual(
            result["llama_cpp_ownership_mapping"]["gather_accepted_path"],
            "llama_kv_cache_future_gather_accepted_path",
        )

    def test_rejects_runtime_boundary_crossing(self) -> None:
        data = load_fixture()
        data["readiness_boundary"] = copy.deepcopy(data["readiness_boundary"])
        data["readiness_boundary"]["real_kv_cache_mutated"] = True

        with self.assertRaisesRegex(readiness.KVCommitReadinessError, "real_kv_cache_mutated=false"):
            readiness.evaluate_fixture(data)

    def test_rejects_runtime_supported_true(self) -> None:
        data = load_fixture()
        data["readiness_boundary"] = copy.deepcopy(data["readiness_boundary"])
        data["readiness_boundary"]["runtime_supported"] = True

        with self.assertRaisesRegex(readiness.KVCommitReadinessError, "runtime_supported=false"):
            readiness.evaluate_fixture(data)

    def test_validator_passes(self) -> None:
        result = validator.validate_p5h_kv_commit_readiness()

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "p5h_kv_commit_readiness_validated")
        self.assertEqual(result["forbidden_root_hits"], [])
        self.assertEqual(result["cmake_hits"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
