#!/usr/bin/env python3
"""Validate the inert JetSpec promotion checklist / ADR."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any


DEFAULT_CHECKLIST = "jetspec_promotion_checklist.md"

REQUIRED_SECTIONS = [
    "# JetSpec promotion checklist / ADR",
    "## Current hard boundary",
    "## Promotion phases",
    "### P0: inert contract baseline",
    "### P1: isolated loader prototype, still outside production CMake",
    "### P2: BF16 payload converter/loader parity fixture",
    "### P3: target hidden tap parity fixture",
    "### P4: tree verify and rollback parity fixture",
    "### P5: production-path candidate, default-off only",
    "### P6: performance and promotion gate",
    "## Path touch policy",
    "## Evidence gates",
    "## Fail-closed blockers",
    "## Rollback and demotion policy",
    "## Next allowed work",
]

REQUIRED_FORBIDDEN_PATHS = [
    "src/models/*.cpp",
    "`common/` files other than the approved P5C/P5D/P5E/P5F/P5N/P5O/P5P/P5Q hook set",
    "non-P5C/P5D/P5E/P5F/P5N/P5O/P5P/P5Q `common/` files",
    "tools/server/",
    "repository `tests/`",
    "examples/",
    "pocs/",
    "ggml/src/",
    "top-level `CMakeLists.txt`",
]

REQUIRED_TOKENS = [
    "python3 run_all_jetspec_contracts.py",
    "draft_head_loader_prototype.py --self-test",
    "test_draft_head_loader_prototype.py",
    "bf16_payload_parity.py --fixture fixtures/bf16_payload_parity_smoke.json",
    "bf16_payload_parity.py --self-test",
    "test_bf16_payload_parity.py",
    "raw_bf16_no_transform",
    "target_hidden_taps.py --fixture fixtures/target_hidden_tap_parity_smoke.json --json",
    "fixtures/target_hidden_tap_parity_smoke.out.json",
    "capture_is_side_channel_only=true",
    "jetspec_round_contract.py --fixture fixtures/jetspec_round_parity_smoke.json",
    "fixtures/jetspec_round_parity_smoke.out.json",
    "greedy_output_matches_baseline=true",
    "jetspec_p5_default_off_plan.md",
    "jetspec_p5a_loader_candidate.md",
    "validate_p5_plan.py",
    "validate_p5a_loader_candidate.py",
    "jetspec_p5b_target_taps_candidate.md",
    "validate_p5b_target_taps.py",
    "jetspec_p5c_speculative_type_candidate.md",
    "validate_p5c_speculative_type.py",
    "jetspec_p5d_target_tap_ingestion_candidate.md",
    "validate_p5d_target_tap_ingestion.py",
    "jetspec_p5e_runtime_state_candidate.md",
    "validate_p5e_runtime_state.py",
    "jetspec_p5f_binding_preflight_candidate.md",
    "validate_p5f_binding_preflight.py",
    "validate_p5f_artifact_binding.py",
    "probe_p5f_loader_gate.py",
    "jetspec_p5g_tree_runtime_readiness.md",
    "jetspec_tree_runtime_readiness.py",
    "validate_p5g_tree_runtime_readiness.py",
    "test_p5g_tree_runtime_readiness.py",
    "tree_runtime_readiness_verified_not_executed",
    "jetspec_p5h_kv_commit_readiness.md",
    "jetspec_kv_commit_readiness.py",
    "validate_p5h_kv_commit_readiness.py",
    "test_p5h_kv_commit_readiness.py",
    "kv_commit_readiness_verified_not_executed",
    "jetspec_p5i_tree_runtime_approval_packet.md",
    "tree_runtime_approval_matrix.py",
    "validate_p5i_tree_runtime_approval_packet.py",
    "test_p5i_tree_runtime_approval_packet.py",
    "tree_runtime_approval_packet_verified_not_executed",
    "jetspec_p5j_kv_primitive_audit.md",
    "kv_primitive_audit.py",
    "validate_p5j_kv_primitive_audit.py",
    "test_p5j_kv_primitive_audit.py",
    "kv_primitive_audit_verified_not_executed",
    "jetspec_p5k_kv_ownership_primitive_design.md",
    "kv_ownership_primitive_design.py",
    "validate_p5k_kv_ownership_primitive_design.py",
    "test_p5k_kv_ownership_primitive_design.py",
    "kv_ownership_primitive_design_verified_not_executed",
    "jetspec_p5l_page_map_ownership_oracle.md",
    "page_map_ownership_oracle.py",
    "validate_p5l_page_map_ownership_oracle.py",
    "test_p5l_page_map_ownership_oracle.py",
    "page_map_ownership_oracle_verified_not_executed",
    "P5L page-map ownership oracle status",
    "page_map_oracle_only",
    "accepted survivor page ownership",
    "rejected branch page unreachability",
    "cross-sequence page isolation",
    "QBlock/PageAttention descriptor lessons",
    "accepted path pages map to `[root | accepted]` only",
    "accepted path physical gather/compact explicit",
    "rejected transient pages unreachable after commit",
    "accepted path cannot read rejected siblings or descendants",
    "rollback restores pre-round page snapshot",
    "other-sequence pages unchanged",
    "no duplicate mutable physical page ownership",
    "rollback preserves other sequences",
    "llama_kv_cache_jetspec_validate_page_ownership_oracle_candidate",
    "llama_kv_cache_jetspec_reserve_transient_tree_pages_candidate",
    "llama_kv_cache_jetspec_commit_page_survivor_path_candidate",
    "llama_kv_cache_jetspec_discard_rejected_tree_pages_candidate",
    "identity maps are only parity/oracle cases",
    "visible noncanonical owned overlays fail closed",
    "full current-K map",
    "canonical write-through remains required",
    "jetspec_p5m_transaction_plan_oracle.md",
    "transaction_plan_oracle.py",
    "validate_p5m_transaction_plan_oracle.py",
    "test_p5m_transaction_plan_oracle.py",
    "transaction_plan_oracle_verified_not_executed",
    "P5M transaction/failpoint plan oracle status",
    "transaction_plan_oracle_only",
    "snapshot_pre_round",
    "reserve_transient_tree_pages",
    "build_tree",
    "build_verify_mask",
    "accept_path",
    "commit_tokens",
    "commit_hidden_kv_survivors",
    "discard_rejected_branches",
    "publish_post_commit_state",
    "rollback points after reserve",
    "all committed token, hidden/KV, and page-map visibility hidden before publish",
    "post-publish tokens `[accepted draft tokens | correction]`",
    "hidden/KV survivors `[root | accepted]`",
    "pre-round snapshot restoration at every failpoint",
    "llama_kv_cache_jetspec_rollback_tree_transaction_candidate",
    "jetspec_p5n_transaction_scaffold_candidate.md",
    "validate_p5n_transaction_scaffold.py",
    "test_p5n_transaction_scaffold.py",
    "P5N transaction-plan scaffold status",
    "JETSPEC_TRANSACTION_PHASE_ORDER",
    "JETSPEC_TRANSACTION_ROLLBACK_POINTS",
    "transaction_plan_scaffold_ready",
    "transaction_plan_hash_last",
    "invalid_transaction_plan",
    "transaction_plan_ready",
    "transaction_plan_hash",
    "no_kv_mutation=1",
    "no_publish=1",
    "no_draft_tokens=1",
    "no draft-head graph execution",
    "no draft tokens emitted",
    "no real KV mutation",
    "no CUDA dispatch",
    "no public API",
    "P5N consumed the first explicit tree-runtime approval",
    "separate explicit approval decision for a real tree build/verify/accept/commit slice",
    "jetspec_p5o_pre_round_snapshot_candidate.md",
    "validate_p5o_pre_round_snapshot.py",
    "test_p5o_pre_round_snapshot.py",
    "P5O pre-round snapshot status",
    "JETSPEC_PRE_ROUND_SNAPSHOT_PHASE",
    "pre_round_snapshot_ready",
    "pre_round_snapshot_hash_last",
    "pre_round_snapshot_prompt_tokens_last",
    "pre_round_snapshot_seq_id_last",
    "invalid_pre_round_snapshot",
    "snapshot_ready",
    "snapshot_hash",
    "snapshot_prompt_tokens",
    "transaction_phase=snapshot_pre_round",
    "no_reserve=1",
    "no_tree_build=1",
    "no_verify_mask=1",
    "pre-round snapshot descriptor",
    "no reserve, tree build, verify mask, accept, commit, KV mutation, CUDA, server, public API, or CMake work",
    "P5O pre-round snapshot descriptor approved",
    "P5O pre-round snapshot hook set",
    "jetspec_p5p_transient_reservation_descriptor_candidate.md",
    "validate_p5p_transient_reservation_descriptor.py",
    "test_p5p_transient_reservation_descriptor.py",
    "P5P transient-reservation descriptor status",
    "JETSPEC_TRANSIENT_RESERVATION_PHASE",
    "JETSPEC_TRANSIENT_RESERVATION_ROLLBACK_POINT",
    "transient_reservation_descriptor_ready",
    "transient_reservation_hash_last",
    "invalid_transient_reservation_descriptor",
    "transient_reservation_ready",
    "transient_reservation_hash",
    "transient_tree_node_budget",
    "actual_pages_reserved=0",
    "pre_publish_visible_state_unmodified=1",
    "no real page reservation",
    "no llama_kv_cache primitive",
    "P5P transient-reservation descriptor approved",
    "P5P transient-reservation descriptor hook set",
    "jetspec_p5q_tree_build_descriptor_candidate.md",
    "validate_p5q_tree_build_descriptor.py",
    "test_p5q_tree_build_descriptor.py",
    "P5Q tree-build descriptor status",
    "JETSPEC_TREE_BUILD_PHASE",
    "JETSPEC_TREE_BUILD_ROLLBACK_POINT",
    "tree_build_descriptor_ready",
    "tree_build_descriptor_hash_last",
    "invalid_tree_build_descriptor",
    "tree_build_descriptor_hash",
    "planned_tree_node_budget",
    "actual_tree_nodes=0",
    "no real tree build",
    "no tree arrays",
    "P5Q tree-build descriptor approved",
    "P5Q tree-build descriptor hook set",
    "non-P5C/P5D/P5E/P5F/P5N/P5O/P5P/P5Q `common/` files",
    "P5G readiness status",
    "P5H readiness status",
    "P5I approval-packet status",
    "P5J primitive-audit status",
    "DraftTree parent-before-child ABI",
    "full-vocab top-k logprobs without top-k-only renormalization",
    "hidden other-sequence tree columns",
    "root-inclusive `accepted_path`",
    "deterministic duplicate-child overwrite",
    "max_len + accepted_path",
    "no target/draft `llama_context`",
    "no KV cache mutation",
    "no server route",
    "abstract transient tree slots",
    "model hidden/KV rows trailing committed tokens by one",
    "rejected tree nodes unreachable after commit",
    "past_len + accepted_path",
    "cross-sequence slot isolation",
    "explicit `missing primitive` ownership mapping",
    "no real KV cache mutation",
    "validated_by_p5g",
    "validated_by_p5h",
    "blocked_pending_explicit_approval",
    "hidden/KV survivor commit",
    "rollback/fail-closed disable",
    "runtime execution claims",
    "implicit primitives like `seq_cp`/`seq_rm`",
    "performance claims",
    "promotion claims",
    "baseline benchmark comparison",
    "read-only KV/runtime primitive audit",
    "P5K primitive-design status",
    "llama_kv_cache_jetspec_commit_survivor_path_candidate",
    "llama_kv_cache_jetspec_discard_rejected_tree_candidate",
    "llama_kv_cache_jetspec_assert_cross_sequence_isolation_candidate",
    "accepted-path physical gather/compact",
    "correction hidden deferred",
    "committed tail compact",
    "rejected transient tree slots unreachable",
    "rollback restores pre-round state",
    "other-sequence slots unchanged",
    "no shared-slot corruption",
    "rollback preserving other sequences",
    "seq_import_physical",
    "seq_keep",
    "find_slot",
    "apply_ubatch",
    "exact_missing_primitive",
    "not exact JetSpec accepted-path tree gather/compact/discard ownership primitives",
    "artifact_verified_not_runtime_executed",
    "loader_gate_verified_preflight_still_blocked",
    "preview_not_allowed",
    "unsupported_runtime",
    "common_speculative_jetspec_preflight",
    "fc.weight",
    "requires_target_embeddings=true",
    "requires_target_lm_head=true",
    "P5A loader-registration candidate approved",
    "P5B target-hidden tap capture approved",
    "P5C speculative type parsing approved",
    "P5E runtime-state bookkeeping approved",
    "P5F binding preflight approved",
    "P5N transaction-plan scaffold, P5O pre-round snapshot descriptor, P5P transient-reservation descriptor, and P5Q tree-build descriptor have explicit approval",
    "P5N transaction-plan scaffold hook set",
    "P5O pre-round snapshot hook set",
    "P5P transient-reservation descriptor hook set",
    "P5Q tree-build descriptor hook set",
    "non-P5C/P5D/P5E/P5F/P5N/P5O/P5P/P5Q `common/` files",
    "validation-only fail-closed production hooks",
    "private side-channel hooks",
    "fail-closed non-executable route hooks",
    "non-drafting runtime slice",
    "state-only runtime slice",
    "preflight-only runtime slice",
    "binding preflight",
    "metadata/shape checks",
    "target tap count/width checks",
    "runtime phase",
    "failure state",
    "row metadata",
    "target tap rows",
    "FNV-1a hash",
    "LLAMA_JETSPEC_TAP_TRACE=1",
    "LLAMA_JETSPEC_STATE_TRACE=1",
    "no draft tokens",
    "0-or-91 tensor inventory",
    "no graph construction",
    "fixed tap layout",
    "graph-reuse guard",
    "no public `include/llama.h`",
    "approved downstream connector only",
    "COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC",
    "draft-jetspec",
    "no silent fallback to `draft-simple`",
    "LLAMA_JETSPEC_EXPERIMENTAL=1",
    "LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1",
    "--spec-type draft-jetspec",
    "runtime_supported=false",
    "no CMake references",
    "91 expected BF16 tensors",
    "metadata-only GGUF preview",
    "explicit experimental flag",
    "target hidden taps",
    "hidden_states[layer_id + 1]",
    "[1, 10, 19, 28, 37]",
    "10240",
    "ancestor-only tree visibility",
    "[root | accepted]",
    "[accepted draft tokens | correction]",
    "rejected branch",
    "default-off",
    "A/B fixture",
    "benchmark",
    "baseline",
    "not change default behavior",
]

REQUIRED_PHASE_ORDER = ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]

FORBIDDEN_PROMOTION_CLAIMS = [
    r"current status:\s*P[5-6]",
    r"runtime_supported=true",
    r"default-enable",
    r"production-ready now",
]


class PromotionChecklistError(ValueError):
    """Raised when the promotion checklist is invalid."""


def _line_for(text: str, needle: str) -> int | None:
    for idx, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return idx
    return None


def validate_checklist(path: pathlib.Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lower = text.lower()
    errors: list[str] = []

    for section in REQUIRED_SECTIONS:
        if section not in text:
            errors.append(f"missing required section: {section}")

    phase_positions: list[tuple[str, int]] = []
    for phase in REQUIRED_PHASE_ORDER:
        match = re.search(rf"###\s+{phase}\b", text)
        if not match:
            errors.append(f"missing phase heading: {phase}")
        else:
            phase_positions.append((phase, match.start()))
    if phase_positions != sorted(phase_positions, key=lambda item: item[1]):
        errors.append(f"phase headings are out of order: {phase_positions}")

    for path_token in REQUIRED_FORBIDDEN_PATHS:
        if path_token not in text:
            errors.append(f"missing forbidden path policy: {path_token}")

    for token in REQUIRED_TOKENS:
        if token not in text:
            errors.append(f"missing required promotion token: {token}")

    if "Current status: P4 tree verify and rollback parity evidence staged; P5 default-off plan drafted; P5A loader-registration candidate approved and implemented as validation-only fail-closed production hooks; P5B target-hidden tap capture approved and implemented as private side-channel hooks; P5C speculative type parsing approved and implemented as fail-closed non-executable route hooks; P5D target-tap ingestion approved and implemented as a non-drafting runtime slice; P5E runtime-state bookkeeping approved and implemented as a state-only runtime slice; P5F binding preflight approved and implemented as a preflight-only runtime slice; P5N transaction-plan scaffold approved and implemented as a non-drafting transaction scaffold; P5O pre-round snapshot descriptor approved and implemented as a non-drafting snapshot descriptor; P5P transient-reservation descriptor approved and implemented as a non-drafting descriptor-only reservation slice; P5Q tree-build descriptor approved and implemented as a non-drafting descriptor-only tree-build slice; executable JetSpec drafting remains forbidden until separate explicit approval." not in text:
        errors.append("checklist must state current status is P4 evidence with approved P5A/P5B/P5C/P5D/P5E/P5F/P5N/P5O/P5P/P5Q hooks and executable drafting still forbidden")
    if "separate explicit approval decision for a real tree build/verify/accept/commit slice" not in text:
        errors.append("checklist must keep next allowed work at separate explicit tree-runtime approval decision")

    for pattern in FORBIDDEN_PROMOTION_CLAIMS:
        if re.search(pattern, lower):
            errors.append(f"forbidden promotion claim matched: {pattern}")

    p5_line = _line_for(text, "### P5:")
    p5a_line = _line_for(text, "| P5A production hook set | validation-only candidate approved | P5A |")
    p5b_line = _line_for(text, "| P5B private target-tap hook set | side-channel candidate approved | P5B |")
    p5c_line = _line_for(text, "| P5C speculative type hook set | fail-closed parser candidate approved | P5C |")
    p5d_line = _line_for(text, "| P5D target-tap ingestion hook set | non-drafting runtime slice approved | P5D |")
    p5e_line = _line_for(text, "| P5E runtime-state hook set | state-only runtime slice approved | P5E |")
    p5f_line = _line_for(text, "| P5F binding-preflight hook set | preflight-only runtime slice approved | P5F |")
    p5n_line = _line_for(text, "| P5N transaction-plan scaffold hook set | non-drafting transaction scaffold approved | P5N |")
    p5o_line = _line_for(text, "| P5O pre-round snapshot hook set | non-drafting snapshot descriptor approved | P5O |")
    p5p_line = _line_for(text, "| P5P transient-reservation descriptor hook set | descriptor-only reservation slice approved | P5P |")
    p5q_line = _line_for(text, "| P5Q tree-build descriptor hook set | descriptor-only tree-build slice approved | P5Q |")
    src_line = _line_for(text, "| non-P5A/P5B `src/models/*.cpp` | forbidden | tree-runtime approval |")
    if p5_line is None or p5a_line is None or p5b_line is None or p5c_line is None or p5d_line is None or p5e_line is None or p5f_line is None or p5n_line is None or p5o_line is None or p5p_line is None or p5q_line is None or src_line is None:
        errors.append("path touch table must mark P5A/P5B/P5C/P5D/P5E/P5F/P5N/P5O/P5P/P5Q hook sets approved and keep non-P5A/P5B src/models/*.cpp forbidden")

    return {
        "ok": not errors,
        "errors": errors,
        "path": str(path),
        "sections_checked": len(REQUIRED_SECTIONS),
        "forbidden_paths_checked": len(REQUIRED_FORBIDDEN_PATHS),
        "tokens_checked": len(REQUIRED_TOKENS),
        "phase_order": [phase for phase, _ in phase_positions],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checklist", type=pathlib.Path, default=here / DEFAULT_CHECKLIST)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        out = validate_checklist(args.checklist.resolve())
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
