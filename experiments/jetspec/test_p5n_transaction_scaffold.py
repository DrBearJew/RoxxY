#!/usr/bin/env python3
"""Tests for the P5N JetSpec transaction-plan scaffold validator."""

from __future__ import annotations

import pathlib
import unittest

import validate_p5n_transaction_scaffold as validator


REPO_ROOT = validator.REPO_ROOT


class P5NTransactionScaffoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validator.validate_p5n_transaction_scaffold()
        cls.source = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        start = cls.source.index("struct common_speculative_impl_draft_jetspec")
        end = cls.source.index("struct common_speculative_impl_draft_mtp")
        cls.impl = cls.source[start:end]

    def test_validator_passes(self) -> None:
        self.assertTrue(self.result["ok"], self.result["errors"])
        self.assertEqual(self.result["cmake_hits"], [])
        self.assertEqual(self.result["status"], "p5n_transaction_scaffold_validated")

    def test_allowlist_is_bounded(self) -> None:
        self.assertEqual(set(self.result["allowed_files"]), {"common/speculative.cpp", "docs/speculative.md"})

    def test_transaction_phase_and_rollback_constants_are_present(self) -> None:
        self.assertIn("JETSPEC_TRANSACTION_PHASE_COUNT  = 9", self.source)
        self.assertIn("JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT = 7", self.source)
        for token in [
            "snapshot_pre_round",
            "reserve_transient_tree_pages",
            "build_tree",
            "build_verify_mask",
            "accept_path",
            "commit_tokens",
            "commit_hidden_kv_survivors",
            "discard_rejected_branches",
            "publish_post_commit_state",
            "after_reserve",
            "after_build_tree",
            "after_verify_mask",
            "after_accept",
            "after_token_commit",
            "after_hidden_kv_commit",
            "after_rejected_discard",
        ]:
            self.assertIn(token, self.source)

    def test_scaffold_builds_hash_after_target_taps(self) -> None:
        self.assertIn("build_transaction_plan_scaffold", self.impl)
        self.assertIn("transaction_plan_hash_last", self.impl)
        self.assertIn("push_back(JETSPEC_TRANSACTION_PHASE_COUNT)", self.impl)
        self.assertIn("push_back(JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT)", self.impl)
        self.assertIn("target_tap_row_state.size() != n_target_tap_rows_cached", self.impl)
        self.assertIn("runtime_phase = jetspec_runtime_phase::transaction_plan_scaffold_ready", self.impl)

    def test_scaffold_fails_closed_and_still_does_not_draft(self) -> None:
        self.assertIn("invalid_transaction_plan", self.impl)
        self.assertIn("disable_runtime_state(jetspec_runtime_failure::invalid_transaction_plan", self.impl)
        self.assertIn("// fail closed: do not emit draft tokens", self.impl)
        self.assertNotIn("result->push_back", self.impl)
        self.assertNotIn("llama_decode", self.impl)
        self.assertNotIn("llama_graph", self.impl)
        self.assertNotIn("tree_accept", self.impl)
        self.assertNotIn("llama_kv_cache", self.impl)

    def test_trace_marks_no_publish_no_kv_mutation_no_draft(self) -> None:
        self.assertIn("transaction_plan_ready=%d", self.impl)
        self.assertIn("transaction_plan_hash=%016", self.impl)
        self.assertIn("no_kv_mutation=1", self.impl)
        self.assertIn("no_publish=1", self.impl)
        self.assertIn("no_draft_tokens=1", self.impl)

    def test_docs_mark_runtime_boundaries(self) -> None:
        docs = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        self.assertIn("P5N transaction-plan scaffold", docs)
        self.assertIn("does not publish", docs)
        self.assertIn("mutate KV", docs)
        self.assertIn("dispatch CUDA", docs)
        self.assertIn("emits no draft tokens", docs)
        self.assertIn("transaction plan readiness", docs)
        self.assertIn("transaction plan hash", docs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
