#!/usr/bin/env python3
"""Tests for P5AG top-k accept-boundary ABI runtime validator."""

from __future__ import annotations

import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import validate_p5ag_topk_accept_boundary_runtime as validator

REPO_ROOT = HERE.parent.parent
SOURCE = REPO_ROOT / "common/speculative.cpp"
DOC = REPO_ROOT / "docs/speculative.md"
CANDIDATE = HERE / "jetspec_p5ag_topk_accept_boundary_runtime_candidate.md"


class P5AGTopKAcceptBoundaryRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8", errors="replace")
        self.doc = DOC.read_text(encoding="utf-8", errors="replace")
        self.candidate = CANDIDATE.read_text(encoding="utf-8", errors="replace")

    def test_source_has_default_off_gate_and_fail_closed_dependencies(self) -> None:
        for token in [
            "LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY",
            "p5ag_topk_accept_boundary_enabled",
            "build_topk_accept_boundary_runtime",
            "invalid_topk_accept_boundary_runtime",
            "!p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled",
            "topk_abi_root_tail_conflict()",
            "if (p5ag_topk_accept_boundary_enabled)",
            "if (!build_topk_accept_boundary_runtime())",
        ]:
            self.assertIn(token, self.source)

    def test_source_materializes_accept_boundary_metadata_only(self) -> None:
        for token in [
            "JETSPEC_TOPK_ACCEPT_BOUNDARY_RUNTIME_PHASE",
            "JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_LOGITS",
            "topk_accept_boundary_runtime_ready",
            "topk_accept_candidate_nodes_last = JETSPEC_TOPK_ABI_NON_ROOT_NODES",
            "topk_accept_boundary_verified_edges_last = actual_verify_mask_entries_last",
            "topk_actual_verified_logits_rows_last = 0",
            "accept_path_len_last = 0",
            "actual_accepted_nodes_last = 0",
            "correction_token_present_last = 0",
            "topk_accept_candidate_nodes_last != 2",
            "topk_accept_boundary_verified_edges_last != 5",
            "topk_actual_verified_logits_rows_last != 0",
        ]:
            self.assertIn(token, self.source)
        branch = self.source[self.source.index("bool build_topk_accept_boundary_runtime"):self.source.index("bool build_root_only_verify_mask_runtime")]
        for forbidden in [
            "llama_decode",
            "llama_graph",
            "llama_kv_cache",
            "result->push_back",
            "common_sampler_sample",
            "llama_sampler",
            "tree_accept",
            "build_token_commit_descriptor",
            "build_hidden_kv_survivor_commit_descriptor",
            "build_rejected_branch_discard_descriptor",
            "build_publish_gate_descriptor",
        ]:
            self.assertNotIn(forbidden, branch)

    def test_trace_boundary_tokens_document_no_runtime_side_effects(self) -> None:
        for token in [
            "p5ag_topk_accept_boundary_runtime",
            "phase=%s",
            "topk_accept_boundary_runtime_ready=%d",
            "actual_tree_nodes=%d",
            "actual_verify_mask_entries=%d",
            "accept_boundary_candidate_nodes=%d",
            "accept_boundary_verified_edges=%d",
            "actual_verified_logits_rows=%d",
            "accept_decision_source=%s",
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
            self.assertIn(token, self.source)

    def test_docs_and_candidate_record_boundary(self) -> None:
        for text in [self.doc, self.candidate]:
            for token in [
                "LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1",
                "topk_accept_boundary_runtime_ready=1",
                "actual_tree_nodes=3",
                "actual_verify_mask_entries=5",
                "accept_boundary_candidate_nodes=2",
                "accept_boundary_verified_edges=5",
                "actual_verified_logits_rows=0",
                "accept_decision_source=none_no_logits",
                "actual_accepted_nodes=0",
                "correction_token_present=0",
                "no target logits walk",
                "no target accept walk",
                "no KV mutation",
                "no draft tokens",
            ]:
                self.assertIn(token, text)

    def test_validator_passes(self) -> None:
        out = validator.validate_p5ag_topk_accept_boundary_runtime()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5ag_topk_accept_boundary_runtime_validated")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["draft_tokens_emitted"])
        self.assertFalse(out["kv_mutated"])
        self.assertFalse(out["published_visible_state"])


if __name__ == "__main__":
    unittest.main()
