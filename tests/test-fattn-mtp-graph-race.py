#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List

from dp16_fa_matrix_common import (
    LANE_ORDER,
    make_lane_calculation_table,
    make_route_capture_matrix,
    make_runtime_acceptance_matrix,
    make_summary_markdown,
    run_graph_key_proofs,
    run_rollback_proofs,
    run_scenario,
    scenario_name,
    write_json,
)


def parse_capture_values(value: str) -> List[bool]:
    if value in ("0", "false", "nocapture"):
        return [False]
    if value in ("1", "true", "capture"):
        return [True]
    return [False, True]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=128)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--artifact-dir", type=str, default="")
    parser.add_argument("--capture", choices=["0", "1", "both", "false", "true", "nocapture", "capture"], default="both")
    parser.add_argument("--lane", choices=["all"] + LANE_ORDER, default="all")
    args = parser.parse_args()

    proofs = run_graph_key_proofs()
    rollback = run_rollback_proofs()

    lanes = LANE_ORDER if args.lane == "all" else [args.lane]
    capture_values = parse_capture_values(args.capture)
    results = [run_scenario(lane, capture, args.repeats) for lane in lanes for capture in capture_values]

    route_capture_matrix = make_route_capture_matrix(results if args.lane == "all" else [run_scenario(lane, capture, args.repeats) for lane in LANE_ORDER for capture in [False, True]])
    runtime_matrix = make_runtime_acceptance_matrix()
    lane_calc_table = make_lane_calculation_table(results if args.lane == "all" else [run_scenario(lane, capture, args.repeats) for lane in LANE_ORDER for capture in [False, True]], proofs, rollback)

    artifact_dir = args.artifact_dir.strip()
    if artifact_dir:
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        default_results = os.path.join(artifact_dir, "results.json")
        if not args.json_out:
            args.json_out = default_results
        write_json(os.path.join(artifact_dir, "lane-calculation-table.json"), lane_calc_table)
        write_json(os.path.join(artifact_dir, "route-capture-matrix.json"), route_capture_matrix)
        write_json(os.path.join(artifact_dir, "runtime-acceptance-matrix.json"), runtime_matrix)
        write_json(os.path.join(artifact_dir, "graph-key-proofs.json"), proofs)
        write_json(os.path.join(artifact_dir, "rollback-proof.json"), rollback)
        with open(os.path.join(artifact_dir, "matrix-calculation-summary.md"), "w", encoding="utf-8") as f:
            f.write(make_summary_markdown(artifact_dir, results if args.lane == "all" else [run_scenario(lane, capture, args.repeats) for lane in LANE_ORDER for capture in [False, True]], proofs, rollback, route_capture_matrix, runtime_matrix))

    payload = json.dumps(results, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")
    print(payload)
    return 0 if all(not r["negative_markers"] for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
