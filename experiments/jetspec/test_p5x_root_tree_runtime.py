#!/usr/bin/env python3
"""Tests for the P5X JetSpec root-only runtime tree validator."""

from __future__ import annotations

import unittest

import validate_p5x_root_tree_runtime as validator


REPO_ROOT = validator.REPO_ROOT


class P5XRootTreeRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validator.validate_p5x_root_tree_runtime()
        cls.source = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        start = cls.source.index("struct common_speculative_impl_draft_jetspec")
        end = cls.source.index("struct common_speculative_impl_draft_mtp")
        cls.impl = cls.source[start:end]
        cls.branch = validator._p5x_branch(cls.impl)

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["status"], "p5x_root_tree_runtime_validated")
        self.assertEqual(self.result["cmake_hits"], [])
        self.assertEqual(self.result["dirty_p5x_hits"], [])

    def test_allowlist_is_bounded(self) -> None:
        self.assertEqual(set(self.result["allowed_files"]), {"common/speculative.cpp", "docs/speculative.md"})

    def test_env_gate_and_state_are_present(self) -> None:
        for token in [
            "LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY",
            "JETSPEC_ROOT_TREE_RUNTIME_PHASE",
            "tree_build_runtime_ready",
            "invalid_root_tree_runtime",
            "root_tree_runtime_ready",
            "root_tree_runtime_hash_last",
            "n_root_tree_runtime_builds",
            "pre_round_root_token_last",
        ]:
            self.assertIn(token, self.impl)

    def test_root_tree_arrays_are_root_only(self) -> None:
        for token in [
            "tree_token_ids[0] = pre_round_root_token_last",
            "tree_parent_indices[0] = JETSPEC_TREE_ROOT_PARENT",
            "tree_depth[0] = JETSPEC_TREE_ROOT_DEPTH",
            "tree_rank[0] = -1",
            "tree_cum_logprob[0] = 0.0f",
            "tree_build_actual_nodes_last = 1",
            "tree_build_actual_nodes_last > tree_build_node_budget_last",
        ]:
            self.assertIn(token, self.impl)
        self.assertNotIn("tree_token_ids.resize", self.impl)
        self.assertNotIn("tree_parent_indices.resize", self.impl)
        self.assertNotIn("tree_cum_logprob.resize", self.impl)

    def test_prompt_tail_is_root_token_source(self) -> None:
        self.assertIn("pre_round_root_token_last = prompt.empty() ? -1 : prompt.back()", self.impl)
        self.assertIn("p5x_root_tree_enabled && prompt.empty()", self.impl)

    def test_negative_guards_fail_closed_before_root_materialization(self) -> None:
        build_start = self.impl.index("bool build_root_only_runtime_tree()")
        build_end = self.impl.index("bool build_verify_mask_descriptor()")
        build = self.impl[build_start:build_end]
        for token in [
            "if (!p5x_root_tree_enabled) {",
            "if (!tree_build_descriptor_ready || tree_build_descriptor_hash_last == 0) {",
            "if (!transient_reservation_ready || transient_reservation_hash_last == 0) {",
            "if (!transaction_plan_ready || transaction_plan_hash_last == 0) {",
            "if (!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0) {",
            "if (pre_round_root_token_last < 0) {",
            "return false;",
        ]:
            self.assertIn(token, build)

    def test_descriptor_path_rejects_actual_tree_nodes(self) -> None:
        verify_start = self.impl.index("bool build_verify_mask_descriptor()")
        verify_end = self.impl.index("bool build_accept_path_descriptor()")
        verify = self.impl[verify_start:verify_end]
        self.assertIn("if (tree_build_actual_nodes_last != 0) {", verify)
        self.assertIn("return false;", verify)

    def test_p5x_branch_returns_before_verify_accept_commit_publish(self) -> None:
        self.assertIn("if (p5x_root_tree_enabled)", self.branch)
        self.assertIn("build_root_only_runtime_tree()", self.branch)
        self.assertIn("return true;", self.branch)
        for token in [
            "build_verify_mask_descriptor()",
            "build_accept_path_descriptor()",
            "build_token_commit_descriptor()",
            "build_hidden_kv_survivor_commit_descriptor()",
            "build_rejected_branch_discard_descriptor()",
            "build_publish_gate_descriptor()",
        ]:
            self.assertNotIn(token, self.branch)

    def test_trace_marks_no_runtime_mutation_boundary(self) -> None:
        for token in [
            "actual_tree_nodes=1",
            "tree_token_ids=[%d]",
            "tree_parent_indices=[%d]",
            "tree_depth=[%d]",
            "tree_cum_logprob=[%.1f]",
            "parent_before_child=1",
            "num_nodes_lte_budget=1",
            "no_draft_head_graph=1",
            "no_verify_mask=1",
            "no_accept=1",
            "no_token_commit=1",
            "no_hidden_kv_commit=1",
            "no_rejected_branch_discard=1",
            "no_publish=1",
            "no_visible_state_change=1",
            "no_draft_tokens=1",
        ]:
            self.assertIn(token, self.impl)

    def test_forbidden_runtime_calls_are_absent(self) -> None:
        for token in [
            "llama_decode",
            "llama_graph",
            "tree_accept",
            "llama_kv_cache",
            "result->push_back",
            "common_sampler_sample",
        ]:
            self.assertNotIn(token, self.impl)

    def test_docs_mark_p5x_boundaries(self) -> None:
        docs = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        for token in [
            "P5X root-only runtime tree materialization",
            "LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1",
            "actual_tree_nodes=1",
            "tree_token_ids=[root_token]",
            "tree_parent_indices=[-1]",
            "tree_depth=[0]",
            "tree_cum_logprob=[0.0]",
            "no draft-head graph execution",
            "no top-k/non-root tree expansion",
            "no verify mask",
            "no accept",
            "no token commit",
            "no hidden/KV commit",
            "no rejected-branch discard",
            "no publish",
            "no visible state change",
            "no KV mutation",
            "no draft tokens",
        ]:
            self.assertIn(token, docs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
