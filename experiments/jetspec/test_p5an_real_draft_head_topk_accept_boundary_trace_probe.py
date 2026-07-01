#!/usr/bin/env python3
"""Tests for P5AN real draft-head top-k accept-boundary no-model trace probe and validator."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import probe_p5an_real_draft_head_topk_accept_boundary_trace as probe
import validate_p5an_real_draft_head_topk_accept_boundary_runtime as validator

REPO_ROOT = HERE.parent.parent
DOC = REPO_ROOT / "docs/speculative.md"
README = HERE / "README.md"
CANDIDATE = HERE / "jetspec_p5an_real_draft_head_topk_accept_boundary_runtime_candidate.md"


class P5ANRealDraftHeadTopKAcceptBoundaryTraceProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = DOC.read_text(encoding="utf-8", errors="replace")
        self.readme = README.read_text(encoding="utf-8", errors="replace")
        self.candidate = CANDIDATE.read_text(encoding="utf-8", errors="replace")

    def test_probe_passes_without_live_log(self) -> None:
        out = probe.probe_p5an_real_draft_head_topk_accept_boundary_trace()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5an_real_draft_head_topk_accept_boundary_trace_contract_verified")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["model_loaded"])
        self.assertFalse(out["draft_tokens_emitted"])
        self.assertTrue(out["contract_documents"]["ok"])

    def test_trace_line_requires_real_tree_mask_and_accept_boundary_metadata(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE)
        self.assertTrue(parsed["ok"], parsed["errors"])
        self.assertEqual(parsed["ctx_dft_present"], 1)
        self.assertEqual(parsed["decode_rc"], 0)
        self.assertEqual(parsed["logits_rows"], 1)
        self.assertEqual(parsed["actual_verified_logits_rows"], 1)
        self.assertEqual(parsed["actual_tree_nodes"], 3)
        self.assertEqual(parsed["candidate_ids"], [76279, 46746])
        self.assertEqual(parsed["real_tree_token_ids"], [13, 76279, 46746])
        self.assertEqual(parsed["real_tree_token_ids"][1:], parsed["candidate_ids"])
        self.assertEqual(parsed["real_tree_parent_indices"], [-1, 0, 0])
        self.assertEqual(parsed["real_tree_depth"], [0, 1, 1])
        self.assertEqual(parsed["real_tree_rank"], [-1, 0, 1])
        self.assertEqual(parsed["real_tree_logits"][1:], parsed["candidate_logits"])
        self.assertEqual(parsed["actual_verify_mask_entries"], 5)
        self.assertEqual(parsed["real_verify_mask_rows"], [0, 1, 1, 2, 2])
        self.assertEqual(parsed["real_verify_mask_cols"], [0, 0, 1, 0, 2])
        self.assertEqual(parsed["real_verify_mask_values"], [1, 1, 1, 1, 1])
        self.assertEqual(parsed["accept_boundary_candidate_nodes"], 2)
        self.assertEqual(parsed["accept_boundary_verified_edges"], 5)
        self.assertEqual(parsed["accept_path_len"], 0)
        self.assertEqual(parsed["actual_accepted_nodes"], 0)
        self.assertEqual(parsed["correction_token_present"], 0)
        self.assertEqual(parsed["actual_committed_tokens"], 0)
        self.assertEqual(parsed["actual_survivor_pages_committed"], 0)
        self.assertEqual(parsed["actual_pages_discarded"], 0)
        self.assertEqual(parsed["actual_publish_visible_state"], 0)

    def test_trace_line_rejects_missing_commit_publish_zero_evidence(self) -> None:
        bad = probe.SELF_TEST_TRACE
        for token in [
            "actual_committed_tokens=0 ",
            "actual_survivor_pages_committed=0 ",
            "actual_pages_discarded=0 ",
            "actual_publish_visible_state=0 ",
        ]:
            bad = bad.replace(token, "")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        joined = "\n".join(parsed["errors"])
        self.assertIn("actual_committed_tokens must be 0", joined)
        self.assertIn("actual_survivor_pages_committed must be 0", joined)
        self.assertIn("actual_pages_discarded must be 0", joined)
        self.assertIn("actual_publish_visible_state must be 0", joined)

    def test_trace_line_rejects_candidate_tree_mismatch(self) -> None:
        bad = probe.SELF_TEST_TRACE.replace("real_tree_token_ids=[13,76279,46746]", "real_tree_token_ids=[13,21429,46746]")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("real_tree_token_ids[1:] must equal P5AK candidate_ids", "\n".join(parsed["errors"]))

    def test_trace_line_rejects_bad_mask_edges(self) -> None:
        bad = probe.SELF_TEST_TRACE.replace("real_verify_mask_cols=[0,0,1,0,2]", "real_verify_mask_cols=[0,0,2,0,2]")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("real_verify_mask_cols must be [0,0,1,0,2]", "\n".join(parsed["errors"]))

    def test_trace_line_rejects_target_walk_accept_and_token_side_effects(self) -> None:
        bad = probe.SELF_TEST_TRACE + " sampler  target_logits  no_target_logits_walk=0 no_target_accept_walk=0 no_accept=0 no_token_commit=0 no_kv_mutation=0 no_draft_tokens=0 actual_accepted_nodes=1 correction_token_present=1 actual_committed_tokens=1 actual_survivor_pages_committed=1 actual_pages_discarded=1 actual_publish_visible_state=1"
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        joined = "\n".join(parsed["errors"])
        self.assertIn("sampler", joined)
        self.assertIn("target_logits", joined)
        self.assertIn("no_target_logits_walk=0", joined)
        self.assertIn("no_target_accept_walk=0", joined)
        self.assertIn("no_accept=0", joined)
        self.assertIn("no_token_commit=0", joined)
        self.assertIn("no_kv_mutation=0", joined)
        self.assertIn("no_draft_tokens=0", joined)
        self.assertIn("actual_accepted_nodes=1", joined)
        self.assertIn("correction_token_present=1", joined)
        self.assertIn("actual_committed_tokens=1", joined)
        self.assertIn("actual_survivor_pages_committed=1", joined)
        self.assertIn("actual_pages_discarded=1", joined)
        self.assertIn("actual_publish_visible_state=1", joined)

    def test_probe_accepts_live_trace_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "server.log"
            path.write_text("noise\n" + probe.SELF_TEST_TRACE + "\n", encoding="utf-8")
            out = probe.probe_p5an_real_draft_head_topk_accept_boundary_trace(path)
        self.assertTrue(out["ok"], out["errors"])
        self.assertTrue(out["runtime_executed"])
        self.assertIsNotNone(out["live_trace"])
        self.assertEqual(out["live_trace"]["candidate_ids"], [76279, 46746])
        self.assertEqual(out["live_trace"]["actual_accepted_nodes"], 0)

    def test_probe_rejects_missing_live_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "server.log"
            path.write_text("no p5an trace here\n", encoding="utf-8")
            out = probe.probe_p5an_real_draft_head_topk_accept_boundary_trace(path)
        self.assertFalse(out["ok"])
        self.assertIn("trace log does not contain draft-jetspec p5an_real_draft_head_topk_accept_boundary_runtime", out["errors"])

    def test_docs_candidate_and_readme_record_contract_boundary(self) -> None:
        for text in [self.doc, self.readme, self.candidate]:
            for token in [
                "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_BOUNDARY_ABI_ONLY=1",
                "P5AN real draft-head top-k accept-boundary ABI",
                "p5an_real_draft_head_topk_accept_boundary_runtime",
                "phase=real_draft_head_topk_accept_boundary_ready",
                "actual_verified_logits_rows=1",
                "allowed_edges=[0:0,1:0,1:1,2:0,2:2]",
                "accept_path_len=0",
                "actual_accepted_nodes=0",
                "correction_token_present=0",
                "no target logits walk",
                "no target accept walk",
                "no token commit",
                "no KV mutation",
                "no draft tokens",
            ]:
                self.assertIn(token, text)

    def test_validator_passes(self) -> None:
        out = validator.validate_p5an_real_draft_head_topk_accept_boundary_runtime()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5an_real_draft_head_topk_accept_boundary_contract_wiring_validated")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["model_loaded"])
        self.assertFalse(out["draft_tokens_emitted"])
        self.assertFalse(out["kv_mutated"])
        self.assertFalse(out["published_visible_state"])
        self.assertEqual(out["actual_verified_logits_rows"], 1)
        self.assertEqual(out["accept_path_len"], 0)


if __name__ == "__main__":
    unittest.main()
