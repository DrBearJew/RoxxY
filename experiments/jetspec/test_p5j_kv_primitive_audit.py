#!/usr/bin/env python3
"""Tests for the inert P5J JetSpec KV/runtime primitive audit."""

from __future__ import annotations

import copy
import json
import pathlib
import unittest

import kv_primitive_audit as audit
import validate_p5j_kv_primitive_audit as validator


HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "kv_primitive_audit_smoke.json"
EXPECTED = HERE / "fixtures" / "kv_primitive_audit_smoke.out.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class P5JKVPrimitiveAuditTests(unittest.TestCase):
    def test_smoke_fixture_matches_expected_output(self) -> None:
        result = audit.evaluate_fixture(load_fixture())
        expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

        self.assertEqual(result, expected)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "kv_primitive_audit_verified_not_executed")
        self.assertEqual(result["coverage"]["classification_counts"]["exact_missing_primitive"], 3)
        self.assertTrue(result["runtime_boundary"]["source_text_read_only"])
        self.assertTrue(result["runtime_boundary"]["no_runtime_execution"])
        self.assertGreaterEqual(result["existing_symbols_audited"][0]["location_count"], 1)

    def test_rejects_missing_required_action(self) -> None:
        data = load_fixture()
        data["actions"] = [action for action in data["actions"] if action["id"] != "cross_sequence_isolation"]

        with self.assertRaisesRegex(audit.KVPrimitiveAuditError, "missing required P5J actions"):
            audit.evaluate_fixture(data)

    def test_rejects_implicit_seq_cp_mapping(self) -> None:
        data = load_fixture()
        data["actions"] = copy.deepcopy(data["actions"])
        data["actions"][0]["classification"] = "exact_existing_primitive_candidate"
        data["actions"][0]["primitive"] = "seq_cp"

        with self.assertRaisesRegex(audit.KVPrimitiveAuditError, "implicit mapping to seq_cp is forbidden"):
            audit.evaluate_fixture(data)

    def test_rejects_missing_source_symbol(self) -> None:
        data = load_fixture()
        data["existing_symbols_to_audit"] = copy.deepcopy(data["existing_symbols_to_audit"])
        data["existing_symbols_to_audit"][0]["symbol"] = "llama_kv_cache_definitely_missing_symbol"

        with self.assertRaisesRegex(audit.KVPrimitiveAuditError, "expected existing symbol not found"):
            audit.evaluate_fixture(data)

    def test_rejects_runtime_boundary_crossing(self) -> None:
        data = load_fixture()
        data["runtime_boundary"] = copy.deepcopy(data["runtime_boundary"])
        data["runtime_boundary"]["real_kv_mutated"] = True

        with self.assertRaisesRegex(audit.KVPrimitiveAuditError, "real_kv_mutated=false"):
            audit.evaluate_fixture(data)

    def test_rejects_production_path_touch(self) -> None:
        data = load_fixture()
        data["production_paths_touched"] = ["src/llama-kv-cache.cpp"]

        with self.assertRaisesRegex(audit.KVPrimitiveAuditError, "must not touch production paths"):
            audit.evaluate_fixture(data)

    def test_accepts_non_forbidden_exact_existing_candidate_with_location(self) -> None:
        data = load_fixture()
        data["actions"] = copy.deepcopy(data["actions"])
        data["actions"][0]["classification"] = "exact_existing_primitive_candidate"
        data["actions"][0]["primitive"] = "seq_keep"
        data["actions"][0]["blockers"] = []

        result = audit.evaluate_fixture(data)

        self.assertEqual(result["actions"][0]["classification"], "exact_existing_primitive_candidate")
        self.assertGreaterEqual(len(result["actions"][0]["locations"]), 1)

    def test_validator_passes(self) -> None:
        result = validator.validate_p5j_kv_primitive_audit()

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "p5j_kv_primitive_audit_validated")
        self.assertEqual(result["forbidden_root_hits"], [])
        self.assertEqual(result["cmake_hits"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
