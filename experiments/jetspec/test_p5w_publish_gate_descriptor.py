#!/usr/bin/env python3
"""Tests for the P5W JetSpec publish-gate descriptor validator."""

from __future__ import annotations

import unittest

import validate_p5w_publish_gate_descriptor as validator


REPO_ROOT = validator.REPO_ROOT


class PublishgatedescriptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validator.validate_p5w_publish_gate_descriptor()
        cls.source = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        start = cls.source.index("struct common_speculative_impl_draft_jetspec")
        end = cls.source.index("struct common_speculative_impl_draft_mtp")
        cls.impl = cls.source[start:end]

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["cmake_hits"], [])
        self.assertEqual(self.result["dirty_p5_hits"], [])
        self.assertEqual(self.result["status"], "p5w_publish_gate_descriptor_validated")

    def test_allowlist_is_bounded(self) -> None:
        self.assertEqual(set(self.result["allowed_files"]), {"common/speculative.cpp", "docs/speculative.md"})

    def test_descriptor_fields_are_present(self) -> None:
        for token in ['JETSPEC_PUBLISH_GATE_PHASE', 'JETSPEC_PUBLISH_GATE_DESCRIPTOR', 'publish_gate_descriptor_ready', 'publish_gate_descriptor_hash_last', 'n_publish_gate_descriptors', 'invalid_publish_gate_descriptor']:
            self.assertIn(token, self.impl)

    def test_descriptor_requires_prior_descriptors(self) -> None:
        self.assertIn('!rejected_branch_discard_descriptor_ready || rejected_branch_discard_descriptor_hash_last == 0', self.impl)

    def test_descriptor_hash_and_zero_actuals_are_bounded(self) -> None:
        for token in ['- `actual_publish_visible_state=0`', '- `publish_after_commit_and_discard_only=1`']:
            self.assertIn(token.strip('- `'), self.impl)
        self.assertIn("JETSPEC_PUBLISH_GATE_PHASE", self.impl)

    def test_fail_closed_and_no_real_runtime(self) -> None:
        self.assertIn("disable_runtime_state(jetspec_runtime_failure::invalid_publish_gate_descriptor", self.impl)
        self.assertIn("// fail closed: do not emit draft tokens", self.impl)
        for token in ["llama_decode", "llama_graph", "tree_accept", "llama_kv_cache", "result->push_back", "seq_cp", "seq_rm", "seq_import_physical"]:
            self.assertNotIn(token, self.impl)
        for token in ["std::vector<llama_token> token_ids", "std::vector<int32_t> parent_indices", "std::vector<float> cum_logprob", "token_ids.resize", "parent_indices.resize"]:
            self.assertNotIn(token, self.impl)

    def test_trace_marks_descriptor_only_boundary(self) -> None:
        for token in ['publish_gate_descriptor_ready=%d', 'publish_gate_descriptor_hash=%016', 'actual_publish_visible_state=0', 'publish_after_commit_and_discard_only=1', 'no_real_publish=1', 'no_visible_state_change=1', 'no_draft_tokens=1']:
            self.assertIn(token, self.impl)

    def test_docs_mark_boundaries(self) -> None:
        docs = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        for token in ['P5W publish-gate descriptor', 'publish_post_commit_state', 'publish-gate descriptor', 'actual_publish_visible_state=0', 'publish_after_commit_and_discard_only=1', 'no real publish', 'no visible state change', 'no draft tokens', 'no draft tokens']:
            self.assertIn(token, docs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
