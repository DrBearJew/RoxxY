#!/usr/bin/env python3
"""Tests for the P5O JetSpec pre-round snapshot validator."""

from __future__ import annotations

import unittest

import validate_p5o_pre_round_snapshot as validator


REPO_ROOT = validator.REPO_ROOT


class P5OPreRoundSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validator.validate_p5o_pre_round_snapshot()
        cls.source = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        start = cls.source.index("struct common_speculative_impl_draft_jetspec")
        end = cls.source.index("struct common_speculative_impl_draft_mtp")
        cls.impl = cls.source[start:end]

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["cmake_hits"], [])
        self.assertEqual(self.result["status"], "p5o_pre_round_snapshot_validated")

    def test_allowlist_is_bounded(self) -> None:
        self.assertEqual(set(self.result["allowed_files"]), {"common/speculative.cpp", "docs/speculative.md"})

    def test_snapshot_fields_are_present(self) -> None:
        for token in [
            "JETSPEC_PRE_ROUND_SNAPSHOT_PHASE",
            "pre_round_snapshot_ready",
            "pre_round_snapshot_hash_last",
            "pre_round_prompt_hash_last",
            "pre_round_seq_id_last",
            "pre_round_prompt_tokens_last",
            "n_pre_round_snapshots",
            "invalid_pre_round_snapshot",
        ]:
            self.assertIn(token, self.impl)

    def test_snapshot_is_built_in_begin_and_validates_seq(self) -> None:
        self.assertIn("build_pre_round_snapshot(seq_id, prompt)", self.impl)
        self.assertIn("seq_id < 0 || (uint32_t) seq_id >= n_seq", self.impl)
        self.assertIn("common_speculative_fnv1a64(prompt.data(), prompt.size() * sizeof(prompt[0]))", self.impl)
        self.assertIn("transaction_phase=%s", self.impl)
        self.assertIn("JETSPEC_PRE_ROUND_SNAPSHOT_PHASE", self.impl)

    def test_transaction_scaffold_requires_snapshot(self) -> None:
        self.assertIn("if (!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0)", self.impl)
        self.assertIn("plan_words.push_back((int64_t) pre_round_snapshot_hash_last)", self.impl)
        self.assertIn("plan_words.push_back((int64_t) pre_round_seq_id_last)", self.impl)
        self.assertIn("plan_words.push_back((int64_t) pre_round_prompt_tokens_last)", self.impl)
        self.assertIn("plan_words.push_back((int64_t) pre_round_prompt_hash_last)", self.impl)

    def test_fail_closed_and_still_no_next_phase_runtime(self) -> None:
        self.assertIn("disable_runtime_state(jetspec_runtime_failure::invalid_pre_round_snapshot", self.impl)
        self.assertIn("// fail closed: do not emit draft tokens", self.impl)
        for token in ["llama_decode", "llama_graph", "tree_accept", "llama_kv_cache", "result->push_back"]:
            self.assertNotIn(token, self.impl)

    def test_trace_marks_no_next_phase_work(self) -> None:
        for token in [
            "pre_round_snapshot_ready=%d",
            "pre_round_snapshot_hash=%016",
            "pre_round_seq_id=%d",
            "pre_round_prompt_tokens=%zu",
            "pre_round_prompt_hash=%016",
            "no_reserve=1",
            "no_tree_build=1",
            "no_verify_mask=1",
            "no_kv_mutation=1",
            "no_publish=1",
            "no_draft_tokens=1",
        ]:
            self.assertIn(token, self.impl)

    def test_docs_mark_snapshot_boundaries(self) -> None:
        docs = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        self.assertIn("P5O pre-round snapshot descriptor", docs)
        self.assertIn("sequence id", docs)
        self.assertIn("prompt token count", docs)
        self.assertIn("prompt hash", docs)
        self.assertIn("no reserve", docs)
        self.assertIn("no tree build", docs)
        self.assertIn("no verify mask", docs)
        self.assertIn("does not publish", docs)
        self.assertIn("mutate KV", docs)
        self.assertIn("dispatch CUDA", docs)
        self.assertIn("emits no draft tokens", docs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
