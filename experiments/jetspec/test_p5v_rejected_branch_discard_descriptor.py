#!/usr/bin/env python3
"""Tests for the P5V JetSpec rejected-branch discard descriptor validator."""

from __future__ import annotations

import unittest

import validate_p5v_rejected_branch_discard_descriptor as validator


REPO_ROOT = validator.REPO_ROOT


class RejectedbranchdiscarddescriptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validator.validate_p5v_rejected_branch_discard_descriptor()
        cls.source = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        start = cls.source.index("struct common_speculative_impl_draft_jetspec")
        end = cls.source.index("struct common_speculative_impl_draft_mtp")
        cls.impl = cls.source[start:end]

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["cmake_hits"], [])
        self.assertEqual(self.result["dirty_p5_hits"], [])
        self.assertEqual(self.result["status"], "p5v_rejected_branch_discard_descriptor_validated")

    def test_allowlist_is_bounded(self) -> None:
        self.assertEqual(set(self.result["allowed_files"]), {"common/speculative.cpp", "docs/speculative.md"})

    def test_descriptor_fields_are_present(self) -> None:
        for token in ['JETSPEC_REJECTED_BRANCH_DISCARD_PHASE', 'JETSPEC_REJECTED_BRANCH_DISCARD_ROLLBACK_POINT', 'JETSPEC_REJECTED_BRANCH_DISCARD_DESCRIPTOR', 'rejected_branch_discard_descriptor_ready', 'rejected_branch_discard_descriptor_hash_last', 'n_rejected_branch_discard_descriptors', 'invalid_rejected_branch_discard_descriptor']:
            self.assertIn(token, self.impl)

    def test_descriptor_requires_prior_descriptors(self) -> None:
        self.assertIn('!hidden_kv_survivor_commit_descriptor_ready || hidden_kv_survivor_commit_descriptor_hash_last == 0', self.impl)

    def test_descriptor_hash_and_zero_actuals_are_bounded(self) -> None:
        for token in ['- `actual_pages_discarded=0`', '- `rejected_branch_pages_reachable_after_discard=0`']:
            self.assertIn(token.strip('- `'), self.impl)
        self.assertIn("JETSPEC_REJECTED_BRANCH_DISCARD_PHASE", self.impl)

    def test_fail_closed_and_no_real_runtime(self) -> None:
        self.assertIn("disable_runtime_state(jetspec_runtime_failure::invalid_rejected_branch_discard_descriptor", self.impl)
        self.assertIn("// fail closed: do not emit draft tokens", self.impl)
        for token in ["llama_decode", "llama_graph", "tree_accept", "llama_kv_cache", "result->push_back", "seq_cp", "seq_rm", "seq_import_physical"]:
            self.assertNotIn(token, self.impl)
        for token in ["std::vector<llama_token> token_ids", "std::vector<int32_t> parent_indices", "std::vector<float> cum_logprob", "token_ids.resize", "parent_indices.resize"]:
            self.assertNotIn(token, self.impl)

    def test_trace_marks_descriptor_only_boundary(self) -> None:
        for token in ['rejected_branch_discard_descriptor_ready=%d', 'rejected_branch_discard_descriptor_hash=%016', 'actual_pages_discarded=0', 'rejected_branch_pages_reachable_after_discard=0', 'no_real_rejected_branch_discard=1', 'no_kv_mutation=1', 'no_publish=1', 'no_draft_tokens=1', 'rollback_point=after_rejected_discard']:
            self.assertIn(token, self.impl)

    def test_docs_mark_boundaries(self) -> None:
        docs = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        for token in ['P5V rejected-branch discard descriptor', 'discard_rejected_branches', 'rejected-branch discard descriptor', 'actual_pages_discarded=0', 'rejected_branch_pages_reachable_after_discard=0', 'no real rejected-branch discard', 'rejected_branch_pages_reachable_after_discard=0', 'no KV mutation', 'no publish', 'no draft tokens', 'no draft tokens']:
            self.assertIn(token, docs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
