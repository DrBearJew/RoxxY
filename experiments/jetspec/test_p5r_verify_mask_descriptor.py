#!/usr/bin/env python3
"""Tests for the P5R JetSpec verify-mask descriptor validator."""

from __future__ import annotations

import unittest

import validate_p5r_verify_mask_descriptor as validator


REPO_ROOT = validator.REPO_ROOT


class VerifymaskdescriptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validator.validate_p5r_verify_mask_descriptor()
        cls.source = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        start = cls.source.index("struct common_speculative_impl_draft_jetspec")
        end = cls.source.index("struct common_speculative_impl_draft_mtp")
        cls.impl = cls.source[start:end]

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["cmake_hits"], [])
        self.assertEqual(self.result["dirty_p5_hits"], [])
        self.assertEqual(self.result["status"], "p5r_verify_mask_descriptor_validated")

    def test_allowlist_is_bounded(self) -> None:
        self.assertEqual(set(self.result["allowed_files"]), {"common/speculative.cpp", "docs/speculative.md"})

    def test_descriptor_fields_are_present(self) -> None:
        for token in ['JETSPEC_VERIFY_MASK_PHASE', 'JETSPEC_VERIFY_MASK_ROLLBACK_POINT', 'JETSPEC_VERIFY_MASK_DESCRIPTOR', 'verify_mask_descriptor_ready', 'verify_mask_descriptor_hash_last', 'n_verify_mask_descriptors', 'invalid_verify_mask_descriptor']:
            self.assertIn(token, self.impl)

    def test_descriptor_requires_prior_descriptors(self) -> None:
        self.assertIn('!tree_build_descriptor_ready || tree_build_descriptor_hash_last == 0', self.impl)

    def test_descriptor_hash_and_zero_actuals_are_bounded(self) -> None:
        for token in ['- `actual_verify_mask_entries=0`']:
            self.assertIn(token.strip('- `'), self.impl)
        self.assertIn("JETSPEC_VERIFY_MASK_PHASE", self.impl)

    def test_fail_closed_and_no_real_runtime(self) -> None:
        self.assertIn("disable_runtime_state(jetspec_runtime_failure::invalid_verify_mask_descriptor", self.impl)
        self.assertIn("// fail closed: do not emit draft tokens", self.impl)
        for token in ["llama_decode", "llama_graph", "tree_accept", "llama_kv_cache", "result->push_back", "seq_cp", "seq_rm", "seq_import_physical"]:
            self.assertNotIn(token, self.impl)
        for token in ["std::vector<llama_token> token_ids", "std::vector<int32_t> parent_indices", "std::vector<float> cum_logprob", "token_ids.resize", "parent_indices.resize"]:
            self.assertNotIn(token, self.impl)

    def test_trace_marks_descriptor_only_boundary(self) -> None:
        for token in ['verify_mask_descriptor_ready=%d', 'verify_mask_descriptor_hash=%016', 'actual_verify_mask_entries=0', 'no_real_verify_mask=1', 'no_verify_mask=1', 'no_accept=1', 'no_kv_mutation=1', 'no_publish=1', 'no_draft_tokens=1', 'rollback_point=after_verify_mask']:
            self.assertIn(token, self.impl)

    def test_docs_mark_boundaries(self) -> None:
        docs = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        for token in ['P5R verify-mask descriptor', 'build_verify_mask', 'verify-mask descriptor', 'actual_verify_mask_entries=0', 'no real verify mask', 'no mask tensor', 'no accept', 'no KV mutation', 'no publish', 'no draft tokens', 'no draft tokens']:
            self.assertIn(token, docs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
