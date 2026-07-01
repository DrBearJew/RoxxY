#!/usr/bin/env python3
"""Tests for the P5AA JetSpec root-token-commit no-op runtime validator."""

from __future__ import annotations

import pathlib
import unittest

import validate_p5aa_root_token_commit_noop_runtime as validator


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


class P5AARootTokenCommitNoopRuntimeTests(unittest.TestCase):
    def test_validator_passes(self) -> None:
        result = validator.validate_p5aa_root_token_commit_noop_runtime()
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["status"], "p5aa_root_token_commit_noop_runtime_validated")

    def test_source_contains_gate_and_builder(self) -> None:
        text = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        for token in [
            "LLAMA_JETSPEC_ROOT_TOKEN_COMMIT_NOOP_ONLY",
            "JETSPEC_ROOT_TOKEN_COMMIT_NOOP_RUNTIME_PHASE",
            "root_token_commit_noop_runtime",
            "token_commit_runtime_ready",
            "invalid_root_token_commit_noop_runtime",
            "n_root_token_commit_noop_runtime_builds",
            "root_token_commit_noop_runtime_hash_last",
            "root_token_commit_noop_runtime_seq_id_last",
            "root_token_commit_noop_runtime_ready",
            "build_root_token_commit_noop_runtime",
            "actual_committed_tokens_last = 0",
            "runtime_phase = jetspec_runtime_phase::token_commit_runtime_ready",
        ]:
            self.assertIn(token, text)

    def test_p5aa_requires_prior_root_runtime_gates(self) -> None:
        text = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        self.assertIn(
            "p5aa_root_token_commit_noop_enabled && (!p5x_root_tree_enabled || !p5y_root_verify_mask_enabled || !p5z_root_anchor_accept_path_enabled)",
            text,
        )
        self.assertIn("!p5z_root_anchor_accept_path_enabled || !root_anchor_accept_path_runtime_ready || root_anchor_accept_path_runtime_hash_last == 0", text)
        self.assertIn("root_verified_anchor_last != 1 || accept_path_len_last != 0 || actual_accepted_nodes_last != 0 || correction_token_present_last != 0", text)

    def test_p5aa_branch_does_not_call_descriptor_chain(self) -> None:
        source = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        impl = validator._impl_slice(source)
        branch = validator._p5aa_branch(impl)
        builder = validator._builder_slice(impl)
        for token in [
            "build_token_commit_descriptor()",
            "build_hidden_kv_survivor_commit_descriptor()",
            "build_rejected_branch_discard_descriptor()",
            "build_publish_gate_descriptor()",
            "token_commit_descriptor_ready = true",
            "hidden_kv_survivor_commit_descriptor_ready = true",
            "rejected_branch_discard_descriptor_ready = true",
            "publish_gate_descriptor_ready = true",
        ]:
            self.assertNotIn(token, branch)
            self.assertNotIn(token, builder)
        self.assertIn("return true;", branch)

    def test_docs_and_candidate_record_boundaries(self) -> None:
        docs = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        candidate = (REPO_ROOT / "experiments/jetspec/jetspec_p5aa_root_token_commit_noop_runtime_candidate.md").read_text(encoding="utf-8")
        for text in [docs, candidate]:
            for token in [
                "P5AA root-token-commit no-op",
                "LLAMA_JETSPEC_ROOT_TOKEN_COMMIT_NOOP_ONLY=1",
                "root_token_commit_noop_runtime_ready=1",
                "actual_committed_tokens=0",
                "root_verified_anchor=1",
                "accept_path_len=0",
                "actual_accepted_nodes=0",
                "correction_token_present=0",
                "no real token commit",
                "no visible token publish",
                "no KV mutation",
                "no draft tokens",
            ]:
                self.assertIn(token, text)
        self.assertIn("returns before P5U", docs)
        self.assertIn("does not call the P5T descriptor builder", candidate)


if __name__ == "__main__":
    unittest.main()
