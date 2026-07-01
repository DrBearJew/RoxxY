#!/usr/bin/env python3
"""Tests for P5AF top-k verify-mask ABI runtime validator."""

from __future__ import annotations

import pathlib
import unittest

import validate_p5af_topk_verify_mask_runtime as validator

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "common/speculative.cpp"
DOC = REPO_ROOT / "docs/speculative.md"
CANDIDATE = pathlib.Path(__file__).resolve().parent / "jetspec_p5af_topk_verify_mask_runtime_candidate.md"


class P5AFTopkVerifyMaskRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8", errors="replace")
        self.doc = DOC.read_text(encoding="utf-8", errors="replace")
        self.candidate = CANDIDATE.read_text(encoding="utf-8", errors="replace")

    def test_source_has_default_off_gate_and_dependencies(self) -> None:
        for token in [
            "LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY",
            "p5af_topk_verify_mask_enabled",
            "build_topk_verify_mask_runtime",
            "invalid_topk_verify_mask_runtime",
            "!p5ae_topk_tree_enabled || !topk_tree_runtime_ready",
            "if (p5af_topk_verify_mask_enabled && (!p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict()))",
        ]:
            self.assertIn(token, self.source)

    def test_source_materializes_ancestor_only_sparse_mask(self) -> None:
        for token in [
            "JETSPEC_TOPK_ABI_MASK_ENTRIES = 5",
            "actual_verify_mask_entries_last = JETSPEC_TOPK_ABI_MASK_ENTRIES",
            "root_verify_mask_rows[0] = 0",
            "root_verify_mask_cols[0] = 0",
            "root_verify_mask_rows[1] = 1",
            "root_verify_mask_cols[1] = 0",
            "root_verify_mask_rows[2] = 1",
            "root_verify_mask_cols[2] = 1",
            "root_verify_mask_rows[3] = 2",
            "root_verify_mask_cols[3] = 0",
            "root_verify_mask_rows[4] = 2",
            "root_verify_mask_cols[4] = 2",
            "root_verify_mask_values[i] != 1",
        ]:
            self.assertIn(token, self.source)
        branch = self.source[self.source.index("bool build_topk_verify_mask_runtime"):self.source.index("bool build_root_only_verify_mask_runtime")]
        for forbidden in ["llama_decode", "llama_kv_cache", "result->push_back", "tree_accept"]:
            self.assertNotIn(forbidden, branch)

    def test_trace_boundary_tokens_document_no_runtime_side_effects(self) -> None:
        for token in [
            "p5af_topk_verify_mask_runtime",
            "actual_tree_nodes=%d",
            "actual_verify_mask_entries=%d",
            "allowed_edges=[0:0,1:0,1:1,2:0,2:2]",
            "ancestor_only=1",
            "sibling_visible=0",
            "descendant_visible=0",
            "no_mask_tensor=1",
            "no_accept=1",
            "no_kv_mutation=1",
            "no_publish=1",
            "no_draft_tokens=1",
        ]:
            self.assertIn(token, self.source)

    def test_docs_and_candidate_explain_abi_only_scope(self) -> None:
        for text in [self.doc, self.candidate]:
            for token in [
                "P5AF top-k verify-mask ABI",
                "LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY=1",
                "actual_verify_mask_entries=5",
                "allowed_edges=[0:0,1:0,1:1,2:0,2:2]",
                "no mask tensor",
                "no draft tokens",
            ]:
                self.assertIn(token, text)

    def test_validator_passes(self) -> None:
        out = validator.validate_p5af_topk_verify_mask_runtime()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5af_topk_verify_mask_runtime_validated")
        self.assertFalse(out["draft_tokens_emitted"])


if __name__ == "__main__":
    unittest.main()
