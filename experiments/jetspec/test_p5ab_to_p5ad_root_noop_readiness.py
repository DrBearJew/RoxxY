#!/usr/bin/env python3
"""Tests for P5AB-P5AD root no-op readiness validator."""

from __future__ import annotations

import pathlib
import unittest

import validate_p5ab_to_p5ad_root_noop_readiness as validator

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


class P5ABToP5ADRootNoopReadinessTests(unittest.TestCase):
    def test_validator_passes(self) -> None:
        out = validator.validate_p5ab_to_p5ad_root_noop_readiness()
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(out["status"], "p5ab_to_p5ad_root_noop_readiness_validated")

    def test_source_has_all_gates_and_terminal_readiness(self) -> None:
        text = (REPO_ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
        for token in [
            "LLAMA_JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_ONLY",
            "LLAMA_JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_ONLY",
            "LLAMA_JETSPEC_ROOT_PUBLISH_GATE_NOOP_ONLY",
            "build_root_hidden_kv_commit_noop_runtime",
            "build_root_rejected_branch_discard_noop_runtime",
            "build_root_publish_gate_noop_runtime",
            "root_runtime_ready_for_real_test_last = 1",
            "actual_survivor_pages_committed_last = 0",
            "actual_pages_discarded_last = 0",
            "rejected_branch_pages_reachable_after_discard_last = 0",
            "actual_publish_visible_state_last = 0",
        ]:
            self.assertIn(token, text)

    def test_docs_define_stop_condition(self) -> None:
        text = (REPO_ROOT / "docs/speculative.md").read_text(encoding="utf-8")
        for token in [
            "P5AB/P5AC/P5AD complete the root-only no-op round tail",
            "root_runtime_ready_for_real_test=1",
            "ready to start a real root-only test",
            "does not itself run that test",
            "no real hidden/KV commit",
            "no real rejected-branch discard",
            "no real publish",
            "no KV mutation",
            "no draft tokens",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
