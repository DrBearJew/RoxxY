#!/usr/bin/env python3
"""Tests for the P5Z JetSpec root-anchor accept-path runtime validator."""

from __future__ import annotations

import unittest

import validate_p5z_root_anchor_accept_path_runtime as validator


REPO_ROOT = validator.REPO_ROOT


class P5ZRootAnchorAcceptPathRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validator.validate_p5z_root_anchor_accept_path_runtime()
        cls.source = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        start = cls.source.index("struct common_speculative_impl_draft_jetspec")
        end = cls.source.index("struct common_speculative_impl_draft_mtp")
        cls.impl = cls.source[start:end]
        cls.branch = validator._p5z_branch(cls.impl)

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["status"], "p5z_root_anchor_accept_path_runtime_validated")
        self.assertEqual(self.result["cmake_hits"], [])
        self.assertEqual(self.result["dirty_p5z_hits"], [])

    def test_allowlist_is_bounded(self) -> None:
        self.assertEqual(set(self.result["allowed_files"]), {"common/speculative.cpp", "docs/speculative.md"})

    def test_env_gate_and_state_are_present(self) -> None:
        for token in [
            "LLAMA_JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_ONLY",
            "JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_RUNTIME_PHASE",
            "accept_path_runtime_ready",
            "invalid_root_anchor_accept_path_runtime",
            "root_anchor_accept_path_runtime_ready",
            "root_anchor_accept_path_runtime_hash_last",
            "n_root_anchor_accept_path_runtime_builds",
            "root_anchor_accept_path_runtime_seq_id_last",
        ]:
            self.assertIn(token, self.impl)

    def test_requires_p5x_and_p5y_runtime_state(self) -> None:
        build_start = self.impl.index("bool build_root_anchor_accept_path_runtime()")
        build_end = self.impl.index("bool build_accept_path_descriptor()")
        build = self.impl[build_start:build_end]
        for token in [
            "if (!p5z_root_anchor_accept_path_enabled) {",
            "if (!p5y_root_verify_mask_enabled || !root_verify_mask_runtime_ready || root_verify_mask_runtime_hash_last == 0) {",
            "if (!p5x_root_tree_enabled || !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {",
            "if (tree_build_actual_nodes_last != 1 || actual_verify_mask_entries_last != 1) {",
            "return false;",
        ]:
            self.assertIn(token, build)
        self.assertIn("p5z_root_anchor_accept_path_enabled && (!p5x_root_tree_enabled || !p5y_root_verify_mask_enabled)", self.impl)

    def test_root_anchor_accept_path_is_non_accepting(self) -> None:
        for token in [
            "root_verified_anchor_last = 1",
            "accept_path_len_last = 0",
            "actual_accepted_nodes_last = 0",
            "correction_token_present_last = 0",
            "root_verified_anchor_last != 1 || accept_path_len_last != 0 || actual_accepted_nodes_last != 0 || correction_token_present_last != 0",
        ]:
            self.assertIn(token, self.impl)

    def test_p5z_branch_returns_before_commit_publish(self) -> None:
        self.assertIn("if (p5z_root_anchor_accept_path_enabled)", self.branch)
        self.assertIn("build_root_anchor_accept_path_runtime()", self.branch)
        self.assertIn("return true;", self.branch)
        for token in [
            "build_accept_path_descriptor()",
            "build_token_commit_descriptor()",
            "build_hidden_kv_survivor_commit_descriptor()",
            "build_rejected_branch_discard_descriptor()",
            "build_publish_gate_descriptor()",
        ]:
            self.assertNotIn(token, self.branch)

    def test_trace_marks_no_downstream_mutation_boundary(self) -> None:
        for token in [
            "p5z_root_anchor_accept_path_runtime",
            "root_verified_anchor=%d",
            "accept_path_len=%d",
            "actual_accepted_nodes=%d",
            "correction_token_present=%d",
            "no_target_logits_walk=1",
            "no_target_accept_walk=1",
            "no_token_commit=1",
            "no_hidden_kv_commit=1",
            "no_rejected_branch_discard=1",
            "no_publish=1",
            "no_visible_state_change=1",
            "no_kv_mutation=1",
            "no_draft_head_graph=1",
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

    def test_docs_mark_p5z_boundaries(self) -> None:
        docs = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        for token in [
            "P5Z root-anchor accept-path materialization",
            "LLAMA_JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_ONLY=1",
            "root_verified_anchor=1",
            "accept_path_len=0",
            "actual_accepted_nodes=0",
            "correction_token_present=0",
            "returns before P5T",
            "no target logits walk",
            "no target accept walk",
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
