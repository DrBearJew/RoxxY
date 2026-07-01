#!/usr/bin/env python3
"""Tests for P5AP real draft-head top-k token-commit no-op trace probe and validator."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import probe_p5ap_real_draft_head_topk_token_commit_noop_trace as probe
import validate_p5ap_real_draft_head_topk_token_commit_noop_runtime as validator

REPO_ROOT = HERE.parent.parent
DOC = REPO_ROOT / "docs/speculative.md"
README = HERE / "README.md"
CANDIDATE = HERE / "jetspec_p5ap_real_draft_head_topk_token_commit_noop_runtime_candidate.md"


class P5APRealDraftHeadTopKTokenCommitNoopTraceProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = DOC.read_text(encoding="utf-8", errors="replace")
        self.readme = README.read_text(encoding="utf-8", errors="replace")
        self.candidate = CANDIDATE.read_text(encoding="utf-8", errors="replace")

    def test_probe_passes_without_live_log(self) -> None:
        out = probe.probe_p5ap_real_draft_head_topk_token_commit_noop_trace()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5ap_real_draft_head_topk_token_commit_noop_trace_contract_verified")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["model_loaded"])
        self.assertFalse(out["draft_tokens_emitted"])
        self.assertTrue(out["contract_documents"]["ok"])

    def test_trace_line_requires_token_commit_noop_and_zero_side_effects(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE)
        self.assertTrue(parsed["ok"], parsed["errors"])
        self.assertEqual(parsed["real_topk_token_commit_noop_runtime_ready"], 1)
        self.assertEqual(parsed["real_topk_accept_path_descriptor_runtime_ready"], 1)
        self.assertEqual(parsed["real_topk_accept_boundary_runtime_ready"], 1)
        self.assertEqual(parsed["real_topk_verify_mask_runtime_ready"], 1)
        self.assertEqual(parsed["real_topk_tree_runtime_ready"], 1)
        self.assertEqual(parsed["real_topk_candidate_runtime_ready"], 1)
        self.assertEqual(parsed["ctx_dft_present"], 1)
        self.assertEqual(parsed["decode_rc"], 0)
        self.assertEqual(parsed["logits_rows"], 1)
        self.assertEqual(parsed["actual_verified_logits_rows"], 1)
        self.assertEqual(parsed["topk_k"], 2)
        self.assertEqual(parsed["actual_tree_nodes"], 3)
        self.assertEqual(parsed["candidate_nodes"], 2)
        self.assertEqual(parsed["candidate_ids"], [76279, 46746])
        self.assertEqual(parsed["real_tree_token_ids"], [13, 76279, 46746])
        self.assertEqual(parsed["real_tree_token_ids"][1:], parsed["candidate_ids"])
        self.assertEqual(parsed["actual_verify_mask_entries"], 5)
        self.assertEqual(parsed["accept_boundary_candidate_nodes"], 2)
        self.assertEqual(parsed["accept_boundary_verified_edges"], 5)
        self.assertEqual(parsed["accept_path_descriptor_len"], 0)
        self.assertEqual(parsed["actual_accepted_nodes"], 0)
        self.assertEqual(parsed["correction_token_present"], 0)
        self.assertEqual(parsed["token_commit_noop"], 1)
        self.assertEqual(parsed["reuse_p5ao_accept_path_descriptor"], 1)
        self.assertEqual(parsed["actual_committed_tokens"], 0)
        self.assertEqual(parsed["actual_survivor_pages_committed"], 0)
        self.assertEqual(parsed["actual_pages_discarded"], 0)
        self.assertEqual(parsed["actual_publish_visible_state"], 0)
        for key in [
            "no_target_logits_walk", "no_target_accept_walk", "no_accept",
            "no_real_token_commit", "no_visible_token_publish", "no_hidden_kv_commit",
            "no_rejected_branch_discard", "no_publish", "no_visible_state_change",
            "no_kv_mutation", "no_draft_tokens",
        ]:
            self.assertEqual(parsed[key], 1)

    def test_parse_helpers_require_key_boundaries(self) -> None:
        self.assertIsNone(probe._parse_scalar("actual_verified_logits_rows=1", "logits_rows"))
        self.assertIsNone(probe._parse_scalar("xlogits_rows=1", "logits_rows"))
        self.assertEqual(probe._parse_scalar("logits_rows=1", "logits_rows"), 1)
        self.assertIsNone(probe._parse_int_list("not_candidate_ids=[1,2]", "candidate_ids"))
        self.assertEqual(probe._parse_int_list("candidate_ids=[1,2]", "candidate_ids"), [1, 2])

    def test_trace_line_rejects_missing_required_zero_counters(self) -> None:
        bad = probe.SELF_TEST_TRACE
        for token in [
            "accept_path_descriptor_len=0 ",
            "actual_accepted_nodes=0 ",
            "correction_token_present=0 ",
            "actual_committed_tokens=0 ",
            "actual_survivor_pages_committed=0 ",
            "actual_pages_discarded=0 ",
            "actual_publish_visible_state=0 ",
        ]:
            bad = bad.replace(token, "")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        joined = "\n".join(parsed["errors"])
        self.assertIn("accept_path_descriptor_len must be 0", joined)
        self.assertIn("actual_accepted_nodes must be 0", joined)
        self.assertIn("correction_token_present must be 0", joined)
        self.assertIn("actual_committed_tokens must be 0", joined)
        self.assertIn("actual_survivor_pages_committed must be 0", joined)
        self.assertIn("actual_pages_discarded must be 0", joined)
        self.assertIn("actual_publish_visible_state must be 0", joined)

    def test_trace_line_rejects_nonzero_side_effects(self) -> None:
        bad = (
            probe.SELF_TEST_TRACE
            + " accept_path_descriptor_len=1 actual_accepted_nodes=1 correction_token_present=1"
            + " actual_committed_tokens=1 actual_survivor_pages_committed=1 actual_pages_discarded=1 actual_publish_visible_state=1"
            + " token_commit_noop=0 reuse_p5ao_accept_path_descriptor=0 no_target_logits_walk=0 no_target_accept_walk=0 no_accept=0"
            + " no_real_token_commit=0 no_visible_token_publish=0 no_hidden_kv_commit=0 no_rejected_branch_discard=0"
            + " no_publish=0 no_visible_state_change=0 no_kv_mutation=0 no_draft_tokens=0"
        )
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        joined = "\n".join(parsed["errors"])
        for token in [
            "accept_path_descriptor_len=1", "actual_accepted_nodes=1", "correction_token_present=1",
            "actual_committed_tokens=1", "actual_survivor_pages_committed=1", "actual_pages_discarded=1",
            "actual_publish_visible_state=1", "token_commit_noop=0", "reuse_p5ao_accept_path_descriptor=0",
            "no_target_logits_walk=0", "no_target_accept_walk=0", "no_accept=0",
            "no_real_token_commit=0", "no_visible_token_publish=0", "no_hidden_kv_commit=0",
            "no_rejected_branch_discard=0", "no_publish=0", "no_kv_mutation=0", "no_draft_tokens=0",
        ]:
            self.assertIn(token, joined)

    def test_trace_line_rejects_generic_root_and_downstream_readiness(self) -> None:
        bad = (
            probe.SELF_TEST_TRACE
            + " token_commit_descriptor_ready=1 root_token_commit_noop_runtime_ready=1"
            + " hidden_kv_survivor_commit_descriptor_ready=1 rejected_branch_discard_descriptor_ready=1"
            + " publish_gate_descriptor_ready=1 #gen drafts = 1 #gen tokens = 1"
        )
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        joined = "\n".join(parsed["errors"])
        for token in [
            "token_commit_descriptor_ready=1",
            "root_token_commit_noop_runtime_ready=1",
            "hidden_kv_survivor_commit_descriptor_ready=1",
            "rejected_branch_discard_descriptor_ready=1",
            "publish_gate_descriptor_ready=1",
            "#gen drafts = 1",
            "#gen tokens = 1",
        ]:
            self.assertIn(token, joined)

    def test_trace_line_rejects_candidate_tree_mismatch(self) -> None:
        bad = probe.SELF_TEST_TRACE.replace("real_tree_token_ids=[13,76279,46746]", "real_tree_token_ids=[13,21429,46746]")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("real_tree_token_ids[1:] must equal P5AK candidate_ids", "\n".join(parsed["errors"]))

    def test_probe_accepts_live_trace_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "server.log"
            path.write_text("noise\n" + probe.SELF_TEST_TRACE + "\n", encoding="utf-8")
            out = probe.probe_p5ap_real_draft_head_topk_token_commit_noop_trace(path)
        self.assertTrue(out["ok"], out["errors"])
        self.assertTrue(out["runtime_executed"])
        self.assertIsNotNone(out["live_trace"])
        self.assertEqual(out["live_trace"]["candidate_ids"], [76279, 46746])
        self.assertEqual(out["live_trace"]["actual_committed_tokens"], 0)

    def test_probe_rejects_missing_live_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "server.log"
            path.write_text("no p5ap trace here\n", encoding="utf-8")
            out = probe.probe_p5ap_real_draft_head_topk_token_commit_noop_trace(path)
        self.assertFalse(out["ok"])
        self.assertIn("trace log does not contain draft-jetspec p5ap_real_draft_head_topk_token_commit_noop_runtime", out["errors"])

    def test_docs_candidate_and_readme_record_contract_boundary(self) -> None:
        for text in [self.doc, self.readme, self.candidate]:
            for token in [
                "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TOKEN_COMMIT_NOOP_ABI_ONLY=1",
                "P5AP real draft-head top-k token-commit no-op ABI",
                "p5ap_real_draft_head_topk_token_commit_noop_runtime",
                "phase=real_draft_head_topk_token_commit_noop_ready",
                "accept_path_descriptor_len=0",
                "actual_accepted_nodes=0",
                "correction_token_present=0",
                "token_commit_noop=1",
                "reuse_p5ao_accept_path_descriptor=1",
                "actual_committed_tokens=0",
                "actual_survivor_pages_committed=0",
                "actual_pages_discarded=0",
                "actual_publish_visible_state=0",
                "no target logits walk",
                "no target accept walk",
                "no accept",
                "no real token commit",
                "no visible token publish",
                "no hidden/KV commit",
                "no rejected-branch discard",
                "no KV mutation",
                "no draft tokens",
                "not the generic P5T",
                "not the root P5AA",
                "returns before downstream hidden/KV/discard/publish",
            ]:
                self.assertIn(token, text)

    def test_validator_passes(self) -> None:
        out = validator.validate_p5ap_real_draft_head_topk_token_commit_noop_runtime()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5ap_real_draft_head_topk_token_commit_noop_contract_wiring_validated")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["model_loaded"])
        self.assertFalse(out["draft_tokens_emitted"])
        self.assertFalse(out["kv_mutated"])
        self.assertFalse(out["published_visible_state"])
        self.assertEqual(out["cmake_hits"], [])
        self.assertEqual(out["forbidden_wiring_hits"], [])
        self.assertTrue(out["source_hook_limited_to_common_speculative_cpp"])
        self.assertFalse(out["source_implementation_required_in_this_lane"])
        self.assertEqual(out["actual_verified_logits_rows"], 1)
        self.assertEqual(out["accept_path_descriptor_len"], 0)
        self.assertEqual(out["committed_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
