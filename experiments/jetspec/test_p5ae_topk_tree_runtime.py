#!/usr/bin/env python3
"""Tests for P5AE top-k tree ABI runtime validator."""

from __future__ import annotations

import pathlib
import unittest

import validate_p5ae_topk_tree_runtime as validator

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "common/speculative.cpp"
DOC = REPO_ROOT / "docs/speculative.md"
CANDIDATE = pathlib.Path(__file__).resolve().parent / "jetspec_p5ae_topk_tree_runtime_candidate.md"


class P5AETopkTreeRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8", errors="replace")
        self.doc = DOC.read_text(encoding="utf-8", errors="replace")
        self.candidate = CANDIDATE.read_text(encoding="utf-8", errors="replace")

    def test_source_has_default_off_gate_and_fail_closed_dependencies(self) -> None:
        for token in [
            "LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY",
            "p5ae_topk_tree_enabled",
            "build_topk_tree_runtime",
            "invalid_topk_tree_runtime",
            "topk_abi_root_tail_conflict()",
            "if (p5ae_topk_tree_enabled && (!p5x_root_tree_enabled || topk_abi_root_tail_conflict()))",
        ]:
            self.assertIn(token, self.source)

    def test_source_materializes_three_node_synthetic_tree_only(self) -> None:
        for token in [
            "JETSPEC_TOPK_ABI_NODES = 3",
            "tree_build_actual_nodes_last = JETSPEC_TOPK_ABI_NODES",
            "tree_token_ids[1] = (pre_round_root_token_last + 1) % JETSPEC_QWEN36_VOCAB_SIZE",
            "tree_token_ids[2] = (pre_round_root_token_last + 2) % JETSPEC_QWEN36_VOCAB_SIZE",
            "tree_parent_indices[1] = 0",
            "tree_parent_indices[2] = 0",
            "tree_depth[1] = 1",
            "tree_depth[2] = 1",
            "tree_rank[1] = 0",
            "tree_rank[2] = 1",
            "tree_cum_logprob[1] = -0.1f",
            "tree_cum_logprob[2] = -0.3f",
            "tree_build_node_budget_last = std::max(tree_build_node_budget_last, JETSPEC_TOPK_ABI_NODES)",
            "tree_build_node_budget_last < JETSPEC_TOPK_ABI_NODES",
            "std::array<llama_token, JETSPEC_QWEN36_DRAFT_BLOCK_SIZE>",
        ]:
            self.assertIn(token, self.source)
        branch = self.source[self.source.index("bool build_topk_tree_runtime"):self.source.index("bool build_topk_verify_mask_runtime")]
        for forbidden in ["llama_decode", "llama_kv_cache", "result->push_back", "tree_accept"]:
            self.assertNotIn(forbidden, branch)

    def test_trace_boundary_tokens_document_no_runtime_side_effects(self) -> None:
        for token in [
            "p5ae_topk_tree_runtime",
            "topk_logprob_source=%s",
            "actual_tree_nodes=%d",
            "tree_parent_indices=[%d,%d,%d]",
            "parent_before_child=1",
            "non_root_nodes=%d",
            "no_draft_logits=1",
            "no_verify_mask=1",
            "no_accept=1",
            "no_kv_mutation=1",
            "no_publish=1",
            "no_draft_tokens=1",
        ]:
            self.assertIn(token, self.source)

    def test_docs_and_candidate_explain_abi_only_scope(self) -> None:
        for text in [self.doc, self.candidate]:
            for token in [
                "P5AE top-k tree ABI",
                "LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY=1",
                "synthetic_full_vocab_softmax",
                "actual_tree_nodes=3",
                "tree_parent_indices=[-1,0,0]",
                "no draft tokens",
            ]:
                self.assertIn(token, text)

    def test_validator_passes(self) -> None:
        out = validator.validate_p5ae_topk_tree_runtime()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5ae_topk_tree_runtime_validated")
        self.assertFalse(out["draft_tokens_emitted"])


if __name__ == "__main__":
    unittest.main()
