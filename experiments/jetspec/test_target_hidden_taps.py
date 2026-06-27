#!/usr/bin/env python3
"""Tests for inert JetSpec target-hidden tap contract helpers."""

from __future__ import annotations

import copy
import json
import pathlib
import unittest

import target_hidden_taps as taps


HERE = pathlib.Path(__file__).resolve().parent
PLAN_PATH = HERE / "conversion_plans/JetSpec_jetspec-Qwen3.6-35B-A3B_gguf_plan.json"
FIXTURE_PATH = HERE / "fixtures/target_hidden_taps_smoke.json"
EXPECTED_PATH = HERE / "fixtures/target_hidden_taps_smoke.out.json"
PARITY_FIXTURE_PATH = HERE / "fixtures/target_hidden_tap_parity_smoke.json"
PARITY_EXPECTED_PATH = HERE / "fixtures/target_hidden_tap_parity_smoke.out.json"


class TargetHiddenTapTests(unittest.TestCase):
    def test_plan_defines_valid_qwen36_tap_spec(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        spec = taps.spec_from_plan(plan)

        self.assertEqual(taps.validate_tap_spec(spec), [])
        self.assertEqual(spec.target_layer_ids, (1, 10, 19, 28, 37))
        self.assertEqual(spec.target_hidden_size, 2048)
        self.assertEqual(spec.draft_fc_input_size, 10240)
        self.assertEqual(spec.concatenated_width, 10240)
        self.assertEqual(spec.draft_depth, 15)

    def test_fixture_concatenates_in_target_layer_order(self) -> None:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
        actual = taps.evaluate_fixture(data)

        self.assertEqual(actual["capture_layer_ids"], expected["capture_layer_ids"])
        self.assertEqual(actual["block_output_ids"], expected["block_output_ids"])
        self.assertEqual(actual["target_hidden_concat"], expected["target_hidden_concat"])
        self.assertEqual(actual["target_hidden_shape"], expected["target_hidden_shape"])

    def test_normal_block_output_ids_are_anchor_plus_masks(self) -> None:
        spec = taps.TargetTapSpec(target_layer_ids=(1,), target_hidden_size=2, num_target_layers=4, block_size=5, mask_token_id=99, draft_fc_input_size=2)

        self.assertEqual(taps.block_output_ids(7, spec), (7, 99, 99, 99, 99))
        self.assertEqual(taps.block_output_ids(7, spec, fill_tokens=[8, 9, 10, 11, 12]), (7, 8, 9, 10, 11))

    def test_capture_fails_closed_for_missing_layer(self) -> None:
        spec = taps.TargetTapSpec(target_layer_ids=(1, 3), target_hidden_size=2, num_target_layers=4, block_size=3, mask_token_id=99, draft_fc_input_size=4)

        with self.assertRaisesRegex(taps.TapContractError, "missing target hidden tap for layer 3"):
            taps.capture_hidden_taps({1: [1.0, 2.0]}, spec)

    def test_capture_fails_closed_for_wrong_width(self) -> None:
        spec = taps.TargetTapSpec(target_layer_ids=(1,), target_hidden_size=2, num_target_layers=4, block_size=3, mask_token_id=99, draft_fc_input_size=2)

        with self.assertRaisesRegex(taps.TapContractError, "width mismatch"):
            taps.capture_hidden_taps({1: [1.0, 2.0, 3.0]}, spec)

    def test_spec_rejects_unsorted_or_duplicate_layers(self) -> None:
        unsorted = taps.TargetTapSpec(target_layer_ids=(10, 1), target_hidden_size=2, num_target_layers=40, block_size=3, mask_token_id=99, draft_fc_input_size=4)
        duplicate = taps.TargetTapSpec(target_layer_ids=(1, 1), target_hidden_size=2, num_target_layers=40, block_size=3, mask_token_id=99, draft_fc_input_size=4)

        self.assertTrue(any("sorted" in error for error in taps.validate_tap_spec(unsorted)))
        self.assertTrue(any("unique" in error for error in taps.validate_tap_spec(duplicate)))

    def test_capture_returns_immutable_copy_not_greedy_path_buffer(self) -> None:
        spec = taps.TargetTapSpec(target_layer_ids=(1,), target_hidden_size=2, num_target_layers=4, block_size=3, mask_token_id=99, draft_fc_input_size=2)
        source = {1: [1.0, 2.0]}
        captured = taps.capture_hidden_taps(source, spec)
        source[1][0] = 999.0

        self.assertEqual(captured, ((1, (1.0, 2.0)),))
        with self.assertRaises(TypeError):
            captured[0][1][0] = 3.0  # type: ignore[index]

    def test_concatenate_rejects_wrong_order(self) -> None:
        spec = taps.TargetTapSpec(target_layer_ids=(1, 3), target_hidden_size=1, num_target_layers=4, block_size=3, mask_token_id=99, draft_fc_input_size=2)

        with self.assertRaisesRegex(taps.TapContractError, "tap order mismatch"):
            taps.concatenate_taps(((3, [3.0]), (1, [1.0])), spec)

    def test_extract_context_feature_uses_hf_hidden_states_layer_plus_one(self) -> None:
        spec = taps.TargetTapSpec(target_layer_ids=(1, 3), target_hidden_size=2, num_target_layers=4, block_size=3, mask_token_id=99, draft_fc_input_size=4)
        hidden_states = [
            [0, 0],
            [10, 11],
            [20, 21],
            [30, 31],
            [40, 41],
        ]

        indices, fc_input = taps.extract_context_feature_from_hidden_states(hidden_states, spec)

        self.assertEqual(indices, ((1, 2), (3, 4)))
        self.assertEqual(fc_input, (20.0, 21.0, 40.0, 41.0))

    def test_parity_fixture_preserves_logits_and_greedy_output(self) -> None:
        data = json.loads(PARITY_FIXTURE_PATH.read_text(encoding="utf-8"))
        expected = json.loads(PARITY_EXPECTED_PATH.read_text(encoding="utf-8"))
        actual = taps.evaluate_fixture(data)

        self.assertEqual(actual, expected)
        self.assertEqual(actual["target_hidden_shape"], [1, 1, 10240])
        self.assertTrue(actual["fallback_matches_hook_capture"])
        self.assertTrue(actual["capture_is_side_channel_only"])
        self.assertEqual(actual["baseline_greedy_token_ids"], [3, 2])
        self.assertEqual(actual["capture_greedy_token_ids"], [3, 2])

    def test_parity_fails_closed_when_capture_logits_change(self) -> None:
        data = json.loads(PARITY_FIXTURE_PATH.read_text(encoding="utf-8"))
        data["capture_logits"] = [[9.9, 0.3, 0.2, 0.9, 0.4], [1.2, 0.8, 1.5, 1.1, 0.0]]
        result = taps.build_parity_result(data)

        self.assertFalse(result["ok"])
        self.assertFalse(result["logits_match"])
        self.assertFalse(result["greedy_output_match"])
        self.assertTrue(any("changed normal target logits" in error for error in result["errors"]))
        self.assertTrue(any("changed greedy output" in error for error in result["errors"]))

    def test_parity_fails_closed_when_hook_uses_wrong_hidden_state_index(self) -> None:
        spec = {
            "target_layer_ids": [1, 3],
            "target_hidden_size": 2,
            "num_target_layers": 4,
            "block_size": 3,
            "mask_token_id": 99,
            "draft_fc_input_size": 4,
        }
        data = {
            "mode": "target_hidden_tap_parity",
            "spec": spec,
            "hf_hidden_states": [[0, 0], [10, 11], [20, 21], [30, 31], [40, 41]],
            "hook_layer_outputs": {"1": [10, 11], "3": [30, 31]},
            "baseline_logits": [[0.0, 1.0]],
        }
        result = taps.build_parity_result(data)

        self.assertFalse(result["ok"])
        self.assertFalse(result["fallback_matches_hook_capture"])
        self.assertTrue(any("hook capture differs" in error for error in result["errors"]))

    def test_parity_fails_closed_for_target_hidden_size_mismatch(self) -> None:
        data = json.loads(PARITY_FIXTURE_PATH.read_text(encoding="utf-8"))
        data["target_model"]["hidden_size"] = 1024
        result = taps.build_parity_result(data)

        self.assertFalse(result["ok"])
        self.assertTrue(any("target hidden_size mismatch" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
