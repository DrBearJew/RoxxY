#!/usr/bin/env python3
"""Tests for the P5Y JetSpec root-only verify-mask runtime validator."""

from __future__ import annotations

import unittest

import validate_p5y_root_verify_mask_runtime as validator


REPO_ROOT = validator.REPO_ROOT


class P5YRootVerifyMaskRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validator.validate_p5y_root_verify_mask_runtime()
        cls.source = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        start = cls.source.index("struct common_speculative_impl_draft_jetspec")
        end = cls.source.index("struct common_speculative_impl_draft_mtp")
        cls.impl = cls.source[start:end]
        cls.branch = validator._p5y_branch(cls.impl)

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["status"], "p5y_root_verify_mask_runtime_validated")
        self.assertEqual(self.result["cmake_hits"], [])
        self.assertEqual(self.result["dirty_p5y_hits"], [])

    def test_allowlist_is_bounded(self) -> None:
        self.assertEqual(set(self.result["allowed_files"]), {"common/speculative.cpp", "docs/speculative.md"})

    def test_env_gate_and_state_are_present(self) -> None:
        for token in [
            "LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY",
            "JETSPEC_ROOT_VERIFY_MASK_RUNTIME_PHASE",
            "verify_mask_runtime_ready",
            "invalid_root_verify_mask_runtime",
            "root_verify_mask_runtime_ready",
            "root_verify_mask_runtime_hash_last",
            "n_root_verify_mask_runtime_builds",
            "root_verify_mask_runtime_seq_id_last",
        ]:
            self.assertIn(token, self.impl)

    def test_requires_p5x_root_tree_runtime(self) -> None:
        build_start = self.impl.index("bool build_root_only_verify_mask_runtime()")
        build_end = self.impl.index("bool build_verify_mask_descriptor()")
        build = self.impl[build_start:build_end]
        for token in [
            "if (!p5y_root_verify_mask_enabled) {",
            "if (!p5x_root_tree_enabled || !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {",
            "if (tree_build_actual_nodes_last != 1) {",
            "if (tree_token_ids[0] < 0",
            "return false;",
        ]:
            self.assertIn(token, build)
        self.assertIn("p5y_root_verify_mask_enabled && !p5x_root_tree_enabled", self.impl)

    def test_root_verify_mask_is_single_self_edge(self) -> None:
        for token in [
            "actual_verify_mask_entries_last = 1",
            "root_verify_mask_rows[0] = 0",
            "root_verify_mask_cols[0] = 0",
            "root_verify_mask_values[0] = 1",
            "actual_verify_mask_entries_last > tree_build_actual_nodes_last * tree_build_actual_nodes_last",
        ]:
            self.assertIn(token, self.impl)
        self.assertNotIn("root_verify_mask_rows.resize", self.impl)
        self.assertNotIn("root_verify_mask_cols.resize", self.impl)
        self.assertNotIn("root_verify_mask_values.resize", self.impl)

    def test_p5y_branch_returns_before_accept_commit_publish(self) -> None:
        self.assertIn("if (p5y_root_verify_mask_enabled)", self.branch)
        self.assertIn("build_root_only_verify_mask_runtime()", self.branch)
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

    def test_trace_marks_no_downstream_mutation_boundary(self) -> None:
        for token in [
            "p5y_root_verify_mask_runtime",
            "actual_tree_nodes=%d",
            "actual_verify_mask_entries=1",
            "verify_mask_rows=1",
            "verify_mask_cols=1",
            "root_attends_self=%d",
            "root_mask_row=%d",
            "root_mask_col=%d",
            "prefix_visible=1",
            "ancestor_only=1",
            "sibling_visible=0",
            "descendant_visible=0",
            "no_draft_head_graph=1",
            "no_mask_tensor=1",
            "no_accept=1",
            "no_token_commit=1",
            "no_hidden_kv_commit=1",
            "no_rejected_branch_discard=1",
            "no_publish=1",
            "no_visible_state_change=1",
            "no_kv_mutation=1",
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

    def test_docs_mark_p5y_boundaries(self) -> None:
        docs = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        for token in [
            "P5Y root-only verify-mask materialization",
            "LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY=1",
            "requires `LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY=1`",
            "actual_verify_mask_entries=1",
            "root_attends_self=1",
            "prefix_visible=1",
            "ancestor_only=1",
            "sibling_visible=0",
            "descendant_visible=0",
            "no mask tensor",
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
