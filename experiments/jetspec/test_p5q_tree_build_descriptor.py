#!/usr/bin/env python3
"""Tests for the P5Q JetSpec tree-build descriptor validator."""

from __future__ import annotations

import unittest

import validate_p5q_tree_build_descriptor as validator


REPO_ROOT = validator.REPO_ROOT


class P5QTreeBuildDescriptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validator.validate_p5q_tree_build_descriptor()
        cls.source = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        start = cls.source.index("struct common_speculative_impl_draft_jetspec")
        end = cls.source.index("struct common_speculative_impl_draft_mtp")
        cls.impl = cls.source[start:end]

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["cmake_hits"], [])
        self.assertEqual(self.result["status"], "p5q_tree_build_descriptor_validated")

    def test_allowlist_is_bounded(self) -> None:
        self.assertEqual(set(self.result["allowed_files"]), {"common/speculative.cpp", "docs/speculative.md"})

    def test_descriptor_fields_are_present(self) -> None:
        for token in [
            "JETSPEC_TREE_BUILD_PHASE",
            "JETSPEC_TREE_BUILD_ROLLBACK_POINT",
            "JETSPEC_TREE_BUILD_DESCRIPTOR",
            "JETSPEC_TREE_ROOT_PARENT",
            "JETSPEC_TREE_ROOT_DEPTH",
            "tree_build_descriptor_ready",
            "tree_build_descriptor_hash_last",
            "n_tree_build_descriptors",
            "tree_build_node_budget_last",
            "tree_build_root_parent_last",
            "tree_build_root_depth_last",
            "tree_build_actual_nodes_last",
            "invalid_tree_build_descriptor",
        ]:
            self.assertIn(token, self.impl)

    def test_descriptor_requires_prior_descriptors(self) -> None:
        self.assertIn("!transient_reservation_ready || transient_reservation_hash_last == 0", self.impl)
        self.assertIn("!transaction_plan_ready || transaction_plan_hash_last == 0", self.impl)
        self.assertIn("!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0", self.impl)
        self.assertIn("transient_reservation_actual_pages_last != 0", self.impl)
        self.assertIn("transient_reservation_node_budget_last <= 0 || transient_reservation_node_budget_last > JETSPEC_QWEN36_DRAFT_BLOCK_SIZE", self.impl)
        self.assertIn("n_target_tap_rows_cached == 0 || target_tap_row_state.size() != n_target_tap_rows_cached", self.impl)

    def test_descriptor_hash_inputs_are_bounded(self) -> None:
        for token in [
            "tree_words.push_back((int64_t) transaction_plan_hash_last)",
            "tree_words.push_back((int64_t) pre_round_snapshot_hash_last)",
            "tree_words.push_back((int64_t) transient_reservation_hash_last)",
            "tree_words.push_back((int64_t) pre_round_prompt_tokens_last)",
            "tree_words.push_back((int64_t) pre_round_prompt_hash_last)",
            "tree_words.push_back((int64_t) target_tap_hash_last)",
            "tree_words.push_back((int64_t) tree_build_actual_nodes_last)",
            "JETSPEC_TREE_BUILD_PHASE",
            "JETSPEC_TREE_BUILD_ROLLBACK_POINT",
        ]:
            self.assertIn(token, self.impl)
        self.assertIn("tree_build_actual_nodes_last = 0", self.impl)

    def test_fail_closed_and_still_no_real_tree_runtime(self) -> None:
        self.assertIn("disable_runtime_state(jetspec_runtime_failure::invalid_tree_build_descriptor", self.impl)
        self.assertIn("// fail closed: do not emit draft tokens", self.impl)
        for token in ["llama_decode", "llama_graph", "tree_accept", "llama_kv_cache", "result->push_back"]:
            self.assertNotIn(token, self.impl)
        for token in ["std::vector<llama_token> token_ids", "std::vector<int32_t> parent_indices", "std::vector<float> cum_logprob", "token_ids.resize", "parent_indices.resize"]:
            self.assertNotIn(token, self.impl)

    def test_trace_marks_descriptor_only_boundary(self) -> None:
        for token in [
            "tree_build_descriptor_ready=%d",
            "tree_build_descriptor_hash=%016",
            "tree_build_phase=build_tree",
            "rollback_point=after_build_tree",
            "planned_tree_node_budget=%d",
            "actual_tree_nodes=0",
            "tree_build_descriptor_only=1",
            "root_parent=%d",
            "root_depth=%d",
            "pre_publish_visible_state_unmodified=1",
            "no_real_tree_build=1",
            "no_tree_arrays=1",
            "no_verify_mask=1",
            "no_accept=1",
            "no_kv_mutation=1",
            "no_publish=1",
            "no_draft_tokens=1",
        ]:
            self.assertIn(token, self.impl)

    def test_docs_mark_tree_build_boundaries(self) -> None:
        docs = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        self.assertIn("P5Q tree-build descriptor", docs)
        self.assertIn("build_tree", docs)
        self.assertIn("actual_tree_nodes=0", docs)
        self.assertIn("no real tree build", docs)
        self.assertIn("no tree arrays", docs)
        self.assertIn("no verify mask", docs)
        self.assertIn("no accept", docs)
        self.assertIn("no draft tokens", docs)
        self.assertIn("no CUDA", docs)
        self.assertIn("server", docs)
        self.assertIn("public API", docs)
        self.assertIn("CMake", docs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
