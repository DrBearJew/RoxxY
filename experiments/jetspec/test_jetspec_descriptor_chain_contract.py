#!/usr/bin/env python3
"""Tests for the no-runtime JetSpec P5N-W descriptor-chain contract."""

from __future__ import annotations

import copy
import json
import pathlib
import unittest

import jetspec_descriptor_chain_contract as chain


HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures/jetspec_descriptor_chain_smoke.json"
EXPECTED = HERE / "fixtures/jetspec_descriptor_chain_smoke.out.json"


class JetSpecDescriptorChainContractTests(unittest.TestCase):
    def test_smoke_fixture_matches_expected_output(self) -> None:
        actual = chain.evaluate_descriptor_chain(json.loads(FIXTURE.read_text(encoding="utf-8")))
        expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

        self.assertEqual(actual, expected)
        self.assertTrue(actual["ok"], actual["errors"])
        self.assertEqual(actual["status"], chain.STATUS)
        self.assertEqual(actual["descriptor_count"], 10)
        self.assertFalse(actual["runtime_executed"])

    def test_missing_prerequisite_fails_closed(self) -> None:
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        data["descriptors"] = [item for item in data["descriptors"] if item["id"] != "p5q_tree_build"]
        result = chain.evaluate_descriptor_chain(data)

        self.assertFalse(result["ok"])
        self.assertTrue(any("descriptor order mismatch" in error for error in result["errors"]))
        self.assertTrue(any("p5r_verify_mask: prerequisite p5q_tree_build" in error for error in result["errors"]))

    def test_nonzero_actual_counter_fails_closed(self) -> None:
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(data)
        for item in tampered["descriptors"]:
            if item["id"] == "p5w_publish_gate":
                item["actuals"]["actual_publish_visible_state"] = 1
        result = chain.evaluate_descriptor_chain(tampered)

        self.assertFalse(result["ok"])
        self.assertTrue(any("actual_publish_visible_state must be 0" in error for error in result["errors"]))

    def test_runtime_boundary_crossing_fails_closed(self) -> None:
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        data["runtime_boundary"]["real_kv_mutated"] = True
        result = chain.evaluate_descriptor_chain(data)

        self.assertFalse(result["ok"])
        self.assertTrue(any("real_kv_mutated=false" in error for error in result["errors"]))

    def test_publish_gate_must_not_invent_rollback_point(self) -> None:
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        data["descriptors"][-1]["rollback_point"] = "after_publish"
        result = chain.evaluate_descriptor_chain(data)

        self.assertFalse(result["ok"])
        self.assertTrue(any("must not invent a post-publish rollback point" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
