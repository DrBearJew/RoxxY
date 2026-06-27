#!/usr/bin/env python3
"""Tests for the P5P JetSpec transient-reservation descriptor validator."""

from __future__ import annotations

import unittest

import validate_p5p_transient_reservation_descriptor as validator


REPO_ROOT = validator.REPO_ROOT


class P5PTransientReservationDescriptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validator.validate_p5p_transient_reservation_descriptor()
        cls.source = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        start = cls.source.index("struct common_speculative_impl_draft_jetspec")
        end = cls.source.index("struct common_speculative_impl_draft_mtp")
        cls.impl = cls.source[start:end]

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["cmake_hits"], [])
        self.assertEqual(self.result["status"], "p5p_transient_reservation_descriptor_validated")

    def test_allowlist_is_bounded(self) -> None:
        self.assertEqual(set(self.result["allowed_files"]), {"common/speculative.cpp", "docs/speculative.md"})

    def test_descriptor_fields_are_present(self) -> None:
        for token in [
            "JETSPEC_TRANSIENT_RESERVATION_PHASE",
            "JETSPEC_TRANSIENT_RESERVATION_ROLLBACK_POINT",
            "JETSPEC_TRANSIENT_RESERVATION_DESCRIPTOR",
            "transient_reservation_descriptor_ready",
            "transient_reservation_ready",
            "transient_reservation_hash_last",
            "n_transient_reservation_descriptors",
            "transient_reservation_node_budget_last",
            "transient_reservation_actual_pages_last",
            "invalid_transient_reservation_descriptor",
        ]:
            self.assertIn(token, self.impl)

    def test_descriptor_requires_p5o_and_p5n_prerequisites(self) -> None:
        self.assertIn("!transaction_plan_ready || transaction_plan_hash_last == 0", self.impl)
        self.assertIn("!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0", self.impl)
        self.assertIn("n_target_tap_rows_cached == 0 || target_tap_row_state.size() != n_target_tap_rows_cached", self.impl)
        self.assertIn("transient_reservation_node_budget_last <= 0 || transient_reservation_node_budget_last > JETSPEC_QWEN36_DRAFT_BLOCK_SIZE", self.impl)

    def test_descriptor_hash_inputs_are_bounded(self) -> None:
        for token in [
            "reservation_words.push_back((int64_t) transaction_plan_hash_last)",
            "reservation_words.push_back((int64_t) pre_round_snapshot_hash_last)",
            "reservation_words.push_back((int64_t) pre_round_seq_id_last)",
            "reservation_words.push_back((int64_t) pre_round_prompt_tokens_last)",
            "reservation_words.push_back((int64_t) target_tap_hash_last)",
            "reservation_words.push_back(JETSPEC_QWEN36_DRAFT_BLOCK_SIZE)",
            "JETSPEC_TRANSIENT_RESERVATION_PHASE",
            "JETSPEC_TRANSIENT_RESERVATION_ROLLBACK_POINT",
        ]:
            self.assertIn(token, self.impl)
        self.assertIn("transient_reservation_actual_pages_last = 0", self.impl)

    def test_fail_closed_and_still_no_runtime_reservation(self) -> None:
        self.assertIn("disable_runtime_state(jetspec_runtime_failure::invalid_transient_reservation_descriptor", self.impl)
        self.assertIn("// fail closed: do not emit draft tokens", self.impl)
        for token in ["llama_decode", "llama_graph", "tree_accept", "llama_kv_cache", "result->push_back", "seq_cp", "seq_rm", "seq_import_physical"]:
            self.assertNotIn(token, self.impl)

    def test_trace_marks_descriptor_only_boundary(self) -> None:
        for token in [
            "transient_reservation_ready=%d",
            "transient_reservation_hash=%016",
            "transient_reservation_phase=reserve_transient_tree_pages",
            "rollback_point=after_reserve",
            "transient_tree_node_budget=%d",
            "actual_pages_reserved=0",
            "pre_publish_visible_state_unmodified=1",
            "no_real_reserve=1",
            "no_page_map_write=1",
            "no_tree_build=1",
            "no_verify_mask=1",
            "no_kv_mutation=1",
            "no_publish=1",
            "no_draft_tokens=1",
        ]:
            self.assertIn(token, self.impl)

    def test_docs_mark_transient_reservation_boundaries(self) -> None:
        docs = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        self.assertIn("P5P transient-reservation descriptor", docs)
        self.assertIn("reserve_transient_tree_pages", docs)
        self.assertIn("actual_pages_reserved=0", docs)
        self.assertIn("no real page reservation", docs)
        self.assertIn("no llama_kv_cache primitive", docs)
        self.assertIn("no tree build", docs)
        self.assertIn("no verify mask", docs)
        self.assertIn("no draft tokens", docs)
        self.assertIn("no CUDA", docs)
        self.assertIn("server", docs)
        self.assertIn("public API", docs)
        self.assertIn("CMake", docs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
