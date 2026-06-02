#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from dp16_fa_matrix_common import LANE_ORDER, make_runtime_acceptance_matrix, scenario_name, write_json

ROUTE_RE = re.compile(r"(MTP_FA_ROUTE:.*|ROUTE:.*|dp16_fa_plan[^\n]*|rocm_[A-Za-z0-9_]+)")


def parse_capture_values(value: str) -> List[bool]:
    if value in ("0", "false", "nocapture"):
        return [False]
    if value in ("1", "true", "capture"):
        return [True]
    return [False, True]


def filter_rows(rows: List[Dict[str, Any]], lane: str, capture: str) -> List[Dict[str, Any]]:
    wanted_lanes = LANE_ORDER if lane == "all" else [lane]
    wanted_capture = set(parse_capture_values(capture))
    return [r for r in rows if r["lane"] in wanted_lanes and r["capture"] in wanted_capture]


def run_live_row(row: Dict[str, Any], command_template: str, artifact_dir: str) -> Dict[str, Any]:
    row_id = row["id"]
    row_dir = Path(artifact_dir) / row_id
    row_dir.mkdir(parents=True, exist_ok=True)
    command = command_template.format(
        lane=row["lane"],
        capture="capture" if row["capture"] else "nocapture",
        row_id=row_id,
        artifact_dir=artifact_dir,
    )
    env = os.environ.copy()
    env.update(row["enable_env"])
    proc = subprocess.run(
        command,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    stdout_path = row_dir / "stdout.log"
    stderr_path = row_dir / "stderr.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    route_lines = ROUTE_RE.findall(proc.stdout + "\n" + proc.stderr)
    return {
        **row,
        "live_status": "executed",
        "command": command,
        "returncode": proc.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "route_matches": route_lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry-run", "live"], default="dry-run")
    parser.add_argument("--lane", choices=["all"] + LANE_ORDER, default="all")
    parser.add_argument("--capture", choices=["0", "1", "both", "false", "true", "nocapture", "capture"], default="both")
    parser.add_argument("--artifact-dir", type=str, default="")
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--command", type=str, default="")
    args = parser.parse_args()

    rows = filter_rows(make_runtime_acceptance_matrix(), args.lane, args.capture)
    payload_rows: List[Dict[str, Any]]

    artifact_dir = args.artifact_dir.strip()
    if artifact_dir:
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "live":
        if not args.command:
            raise SystemExit("--command is required in --mode live")
        payload_rows = [run_live_row(row, args.command, artifact_dir or ".") for row in rows]
    else:
        payload_rows = [{**row, "live_status": "dry_run_contract_only"} for row in rows]

    payload = {
        "mode": args.mode,
        "rows": payload_rows,
    }

    if artifact_dir:
        write_json(os.path.join(artifact_dir, "hip-replay-plan.json"), payload)
    if args.json_out:
        write_json(args.json_out, payload)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
