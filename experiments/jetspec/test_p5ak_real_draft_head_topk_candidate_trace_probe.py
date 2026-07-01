#!/usr/bin/env python3
"""Tests for P5AK real draft-head top-k candidate ABI trace probe and validator."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import probe_p5ak_real_draft_head_topk_candidate_trace as probe
import validate_p5ak_real_draft_head_topk_candidate_runtime as validator

REPO_ROOT = HERE.parent.parent
SOURCE = REPO_ROOT / "common/speculative.cpp"
DOC = REPO_ROOT / "docs/speculative.md"
CANDIDATE = HERE / "jetspec_p5ak_real_draft_head_topk_candidate_runtime_candidate.md"


class P5AKRealDraftHeadTopKCandidateTraceProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8", errors="replace")
        self.doc = DOC.read_text(encoding="utf-8", errors="replace")
        self.candidate = CANDIDATE.read_text(encoding="utf-8", errors="replace")

    def test_probe_passes_without_live_log(self) -> None:
        out = probe.probe_p5ak_real_draft_head_topk_candidate_trace()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5ak_real_draft_head_topk_candidate_trace_contract_verified")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["model_loaded"])
        self.assertFalse(out["draft_tokens_emitted"])
        self.assertTrue(out["source_contract"]["ok"])

    def test_trace_line_requires_real_topk_candidate_metadata_and_no_tokens(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE)
        self.assertTrue(parsed["ok"], parsed["errors"])
        self.assertEqual(parsed["ctx_dft_present"], 1)
        self.assertEqual(parsed["decode_rc"], 0)
        self.assertEqual(parsed["logits_width"], 248320)
        self.assertEqual(parsed["logits_rows"], parsed["actual_verified_logits_rows"])
        self.assertEqual(parsed["topk_k"], 2)
        self.assertEqual(parsed["parent_node"], 0)
        self.assertEqual(parsed["candidate_nodes"], 2)
        self.assertEqual(parsed["candidate_ids"], [46746, 128519])
        self.assertGreater(parsed["candidate_logits"][0], parsed["candidate_logits"][1])
        self.assertEqual(parsed["actual_accepted_nodes"], 0)
        self.assertEqual(parsed["correction_token_present"], 0)

    def test_trace_line_rejects_sampler_target_and_token_side_effects(self) -> None:
        bad = probe.SELF_TEST_TRACE + " sampler target_logits no_token_commit=0 no_draft_tokens=0"
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        joined = "\n".join(parsed["errors"])
        self.assertIn("sampler", joined)
        self.assertIn("target_logits", joined)
        self.assertIn("no_token_commit=0", joined)
        self.assertIn("no_draft_tokens=0", joined)

    def test_probe_accepts_live_trace_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "server.log"
            path.write_text("noise\n" + probe.SELF_TEST_TRACE + "\n", encoding="utf-8")
            out = probe.probe_p5ak_real_draft_head_topk_candidate_trace(path)
        self.assertTrue(out["ok"], out["errors"])
        self.assertTrue(out["runtime_executed"])
        self.assertIsNotNone(out["live_trace"])
        self.assertEqual(out["live_trace"]["candidate_nodes"], 2)
        self.assertEqual(out["live_trace"]["actual_accepted_nodes"], 0)

    def test_probe_rejects_missing_live_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "server.log"
            path.write_text("no p5ak trace here\n", encoding="utf-8")
            out = probe.probe_p5ak_real_draft_head_topk_candidate_trace(path)
        self.assertFalse(out["ok"])
        self.assertIn("trace log does not contain draft-jetspec p5ak_real_draft_head_topk_candidate_runtime", out["errors"])

    def test_source_has_default_off_gate_and_fail_closed_chain(self) -> None:
        for token in [
            "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY",
            "p5ak_real_draft_head_topk_candidate_enabled",
            "build_real_draft_head_topk_candidate_runtime",
            "invalid_real_draft_head_topk_candidate_runtime",
            "!p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled",
            "topk_abi_root_tail_conflict()",
            "if (p5ak_real_draft_head_topk_candidate_enabled)",
            "if (!build_real_draft_head_topk_candidate_runtime())",
        ]:
            self.assertIn(token, self.source)

    def test_source_materializes_p5aj_candidates_after_p5ag_boundary_only(self) -> None:
        for token in [
            "JETSPEC_REAL_DRAFT_HEAD_TOPK_CANDIDATE_RUNTIME_PHASE",
            "JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE",
            "JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS",
            "real_draft_head_logits_canary_ready",
            "topk_accept_boundary_runtime_ready",
            "topk_verify_mask_runtime_ready",
            "topk_tree_runtime_ready",
            "real_draft_head_canary_topk_k_last != JETSPEC_TOPK_ABI_WIDTH",
            "real_draft_head_topk_candidate_ids[0] = real_draft_head_canary_top1_id_last",
            "real_draft_head_topk_candidate_ids[1] = real_draft_head_canary_top2_id_last",
            "real_draft_head_topk_candidate_logits[0] = real_draft_head_canary_top1_logit_last",
            "real_draft_head_topk_candidate_logits[1] = real_draft_head_canary_top2_logit_last",
            "accept_path_len_last != 0 || actual_accepted_nodes_last != 0 || correction_token_present_last != 0",
        ]:
            self.assertIn(token, self.source)
        branch = self.source[self.source.index("bool build_real_draft_head_topk_candidate_runtime"):self.source.index("bool build_pre_round_snapshot")]
        for forbidden in [
            "llama_decode",
            "llama_graph",
            "llama_kv_cache",
            "result->push_back",
            "common_sampler_sample",
            "llama_sampler",
            "tree_accept",
        ]:
            self.assertNotIn(forbidden, branch)

    def test_trace_boundary_tokens_document_no_runtime_side_effects(self) -> None:
        for token in [
            "p5ak_real_draft_head_topk_candidate_runtime",
            "real_topk_candidate_runtime_ready=%d",
            "topk_accept_boundary_runtime_ready=%d",
            "logits_source=%s",
            "ctx_dft_present=%d",
            "decode_rc=%d",
            "logits_rows=%d",
            "logits_width=%d",
            "actual_verified_logits_rows=%d",
            "topk_k=%d",
            "parent_node=%d",
            "candidate_nodes=%d",
            "candidate_ids=[%d,%d]",
            "candidate_logits=[%.6g,%.6g]",
            "rank_semantics=%s",
            "accept_path_len=%d",
            "actual_accepted_nodes=%d",
            "correction_token_present=%d",
            "no_external_logits_walk=1",
            "no_target_accept_walk=1",
            "no_accept=1",
            "no_token_commit=1",
            "no_hidden_kv_commit=1",
            "no_rejected_branch_discard=1",
            "no_publish=1",
            "no_visible_state_change=1",
            "no_kv_mutation=1",
            "no_draft_tokens=1",
        ]:
            self.assertIn(token, self.source)

    def test_docs_and_candidate_record_boundary(self) -> None:
        for text in [self.doc, self.candidate]:
            for token in [
                "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY=1",
                "LLAMA_JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY=1",
                "LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY=1",
                "real_topk_candidate_runtime_ready=1",
                "logits_source=draft_head_full_vocab_logits",
                "actual_verified_logits_rows=1",
                "topk_k=2",
                "parent_node=0",
                "candidate_nodes=2",
                "rank_semantics=rank_stable_descending_logit",
                "P5AJ top1/top2",
                "P5AG",
                "no sampler",
                "no target logits walk",
                "no target accept walk",
                "no KV mutation",
                "no draft tokens",
            ]:
                self.assertIn(token, text)

    def test_validator_passes(self) -> None:
        out = validator.validate_p5ak_real_draft_head_topk_candidate_runtime()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5ak_real_draft_head_topk_candidate_runtime_validated")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["model_loaded"])
        self.assertFalse(out["draft_tokens_emitted"])
        self.assertFalse(out["kv_mutated"])
        self.assertFalse(out["published_visible_state"])


if __name__ == "__main__":
    unittest.main()
