#!/usr/bin/env python3
"""Tests for P5AM real draft-head top-k verify-mask ABI trace probe and validator."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import probe_p5am_real_draft_head_topk_verify_mask_trace as probe
import validate_p5am_real_draft_head_topk_verify_mask_runtime as validator

REPO_ROOT = HERE.parent.parent
SOURCE = REPO_ROOT / "common/speculative.cpp"
DOC = REPO_ROOT / "docs/speculative.md"
CANDIDATE = HERE / "jetspec_p5am_real_draft_head_topk_verify_mask_runtime_candidate.md"


class P5AMRealDraftHeadTopKVerifyMaskTraceProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8", errors="replace")
        self.doc = DOC.read_text(encoding="utf-8", errors="replace")
        self.candidate = CANDIDATE.read_text(encoding="utf-8", errors="replace")

    def test_probe_passes_without_live_log(self) -> None:
        out = probe.probe_p5am_real_draft_head_topk_verify_mask_trace()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5am_real_draft_head_topk_verify_mask_trace_contract_verified")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["model_loaded"])
        self.assertFalse(out["draft_tokens_emitted"])
        self.assertTrue(out["source_contract"]["ok"])

    def test_trace_line_requires_shadow_tree_and_mask_metadata(self) -> None:
        parsed = probe.validate_trace_line(probe.SELF_TEST_TRACE)
        self.assertTrue(parsed["ok"], parsed["errors"])
        self.assertEqual(parsed["ctx_dft_present"], 1)
        self.assertEqual(parsed["decode_rc"], 0)
        self.assertEqual(parsed["logits_rows"], 1)
        self.assertEqual(parsed["logits_width"], 248320)
        self.assertEqual(parsed["actual_verified_logits_rows"], 1)
        self.assertEqual(parsed["actual_tree_nodes"], 3)
        self.assertEqual(parsed["candidate_ids"], [46746, 128519])
        self.assertEqual(parsed["real_tree_token_ids"], [10240, 46746, 128519])
        self.assertEqual(parsed["real_tree_token_ids"][1:], parsed["candidate_ids"])
        self.assertEqual(parsed["real_tree_parent_indices"], [-1, 0, 0])
        self.assertEqual(parsed["real_tree_depth"], [0, 1, 1])
        self.assertEqual(parsed["real_tree_rank"], [-1, 0, 1])
        self.assertEqual(parsed["real_tree_logits"][1:], parsed["candidate_logits"])
        self.assertEqual(parsed["actual_verify_mask_entries"], 5)
        self.assertEqual(parsed["real_verify_mask_rows"], [0, 1, 1, 2, 2])
        self.assertEqual(parsed["real_verify_mask_cols"], [0, 0, 1, 0, 2])
        self.assertEqual(parsed["real_verify_mask_values"], [1, 1, 1, 1, 1])
        self.assertEqual(parsed["accept_path_len"], 0)
        self.assertEqual(parsed["actual_accepted_nodes"], 0)

    def test_trace_line_rejects_candidate_tree_mismatch(self) -> None:
        bad = probe.SELF_TEST_TRACE.replace("real_tree_token_ids=[10240,46746,128519]", "real_tree_token_ids=[10240,21429,46746]")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("real_tree_token_ids[1:] must equal P5AK candidate_ids", "\n".join(parsed["errors"]))

    def test_trace_line_rejects_bad_mask_edges(self) -> None:
        bad = probe.SELF_TEST_TRACE.replace("real_verify_mask_cols=[0,0,1,0,2]", "real_verify_mask_cols=[0,0,2,0,2]")
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        self.assertIn("real_verify_mask_cols must be [0,0,1,0,2]", "\n".join(parsed["errors"]))

    def test_trace_line_rejects_sampler_target_and_token_side_effects(self) -> None:
        bad = probe.SELF_TEST_TRACE + " sampler  target_logits  no_mask_tensor=0 no_accept=0 no_token_commit=0 no_kv_mutation=0 no_draft_tokens=0"
        parsed = probe.validate_trace_line(bad)
        self.assertFalse(parsed["ok"])
        joined = "\n".join(parsed["errors"])
        self.assertIn("sampler", joined)
        self.assertIn("target_logits", joined)
        self.assertIn("no_mask_tensor=0", joined)
        self.assertIn("no_accept=0", joined)
        self.assertIn("no_token_commit=0", joined)
        self.assertIn("no_kv_mutation=0", joined)
        self.assertIn("no_draft_tokens=0", joined)

    def test_probe_accepts_live_trace_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "server.log"
            path.write_text("noise\n" + probe.SELF_TEST_TRACE + "\n", encoding="utf-8")
            out = probe.probe_p5am_real_draft_head_topk_verify_mask_trace(path)
        self.assertTrue(out["ok"], out["errors"])
        self.assertTrue(out["runtime_executed"])
        self.assertIsNotNone(out["live_trace"])
        self.assertEqual(out["live_trace"]["real_verify_mask_rows"], [0, 1, 1, 2, 2])
        self.assertEqual(out["live_trace"]["actual_accepted_nodes"], 0)

    def test_probe_rejects_missing_live_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "server.log"
            path.write_text("no p5am trace here\n", encoding="utf-8")
            out = probe.probe_p5am_real_draft_head_topk_verify_mask_trace(path)
        self.assertFalse(out["ok"])
        self.assertIn("trace log does not contain draft-jetspec p5am_real_draft_head_topk_verify_mask_runtime", out["errors"])

    def test_source_has_default_off_gate_and_fail_closed_chain(self) -> None:
        for token in [
            "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_VERIFY_MASK_ABI_ONLY",
            "p5am_real_draft_head_topk_verify_mask_enabled",
            "build_real_draft_head_topk_verify_mask_runtime",
            "invalid_real_draft_head_topk_verify_mask_runtime",
            "if (p5am_real_draft_head_topk_verify_mask_enabled && (!p5al_real_draft_head_topk_tree_enabled",
            "if (!build_real_draft_head_topk_verify_mask_runtime())",
            "real_draft_head_canary_logits_rows_last != 1",
        ]:
            self.assertIn(token, self.source)

    def test_source_uses_shadow_mask_not_canonical_mutation(self) -> None:
        for token in [
            "real_draft_head_topk_verify_mask_rows[0] = 0",
            "real_draft_head_topk_verify_mask_cols[4] = 2",
            "real_draft_head_topk_verify_mask_values[4] = 1",
            "real_verify_mask_rows=[%d,%d,%d,%d,%d]",
            "allowed_edges=[0:0,1:0,1:1,2:0,2:2]",
            "no_synthetic_mask_mutation=1",
        ]:
            self.assertIn(token, self.source)
        builder = self.source[self.source.index("bool build_real_draft_head_topk_verify_mask_runtime"):self.source.index("bool build_pre_round_snapshot")]
        for forbidden in [
            "llama_decode", "llama_graph", "llama_kv_cache", "result->push_back",
            "common_sampler_sample", "llama_sampler", "tree_accept",
            "\n        root_verify_mask_rows[", "\n        root_verify_mask_cols[", "\n        root_verify_mask_values[",
        ]:
            self.assertNotIn(forbidden, builder)

    def test_docs_and_candidate_record_shadow_mask_boundary(self) -> None:
        for text in [self.doc, self.candidate]:
            for token in [
                "LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_VERIFY_MASK_ABI_ONLY=1",
                "shadow real-tree verify-mask ABI",
                "allowed_edges=[0:0,1:0,1:1,2:0,2:2]",
                "no mask tensor",
                "no accept",
                "no token commit",
                "no KV mutation",
                "no draft tokens",
            ]:
                self.assertIn(token, text)

    def test_validator_passes(self) -> None:
        out = validator.validate_p5am_real_draft_head_topk_verify_mask_runtime()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5am_real_draft_head_topk_verify_mask_runtime_validated")
        self.assertFalse(out["runtime_executed"])
        self.assertFalse(out["model_loaded"])
        self.assertFalse(out["draft_tokens_emitted"])
        self.assertFalse(out["kv_mutated"])
        self.assertFalse(out["published_visible_state"])


if __name__ == "__main__":
    unittest.main()
