#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

RE_INPUT = re.compile(r"MTP_INPUT_REAL: .*bad_h=(?P<bad_h>\d+) checked_h=(?P<checked_h>\d+)")
RE_TEACHER = re.compile(r"MTP_TEACHER_PROBE: .*output=(?P<output>\d+).*expected_rank=(?P<rank>\d+).*bad_logits=(?P<bad>\d+)")
RE_ACCEPT = re.compile(r"draft acceptance rate =\s*(?P<rate>[0-9.]+) \(\s*(?P<acc>\d+) accepted /\s*(?P<gen>\d+) generated\)")
RE_DEPTH = re.compile(r"statistics (?P<type>[^:]+)-depth: (?P<body>.*)")
RE_DEPTH_ITEM = re.compile(r"d(?P<depth>\d+)=(?P<acc>\d+)/(?:\s*)?(?P<gen>\d+)")


def summarize(path: Path):
    bad_h = checked_h = bad_logits = 0
    accept = {"rate": None, "accepted": None, "generated": None}
    ranks = {}
    depth = {}

    for line in path.read_text(errors="replace").splitlines():
        if m := RE_INPUT.search(line):
            bad_h += int(m.group("bad_h"))
            checked_h += int(m.group("checked_h"))
        if m := RE_TEACHER.search(line):
            out = int(m.group("output"))
            ranks.setdefault(out, []).append(int(m.group("rank")))
            bad_logits += int(m.group("bad"))
        if m := RE_ACCEPT.search(line):
            accept = {"rate": float(m.group("rate")), "accepted": int(m.group("acc")), "generated": int(m.group("gen"))}
        if m := RE_DEPTH.search(line):
            for dm in RE_DEPTH_ITEM.finditer(m.group("body")):
                d = int(dm.group("depth"))
                depth[d] = {"accepted": int(dm.group("acc")), "generated": int(dm.group("gen"))}

    rank_summary = {}
    for out, vals in sorted(ranks.items()):
        vals_sorted = sorted(vals)
        rank_summary[out] = {
            "count": len(vals),
            "rank1": sum(v == 1 for v in vals),
            "rank5": sum(v <= 5 for v in vals),
            "median": statistics.median(vals),
            "p90": vals_sorted[min(len(vals_sorted) - 1, int(0.9 * (len(vals_sorted) - 1)))] if vals_sorted else None,
        }

    return {
        "log": str(path),
        "bad_h": bad_h,
        "checked_h": checked_h,
        "bad_logits": bad_logits,
        "acceptance": accept,
        "teacher_rank_by_output": rank_summary,
        "acceptance_by_depth": depth,
    }


def main():
    ap = argparse.ArgumentParser(description="Summarize MTP acceptance, teacher ranks, and per-depth counters from llama-server logs")
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    rows = [summarize(p) for p in args.logs]
    print(json.dumps(rows, indent=2))

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["log", "bad_h", "checked_h", "bad_logits", "accepted", "generated", "rate", "d1", "d2", "d3", "out0_rank1", "out1_rank1", "out2_rank1"])
            for r in rows:
                acc = r["acceptance"]
                d = r["acceptance_by_depth"]
                tr = r["teacher_rank_by_output"]
                w.writerow([
                    r["log"], r["bad_h"], r["checked_h"], r["bad_logits"],
                    acc.get("accepted"), acc.get("generated"), acc.get("rate"),
                    f"{d.get(1,{}).get('accepted',0)}/{d.get(1,{}).get('generated',0)}",
                    f"{d.get(2,{}).get('accepted',0)}/{d.get(2,{}).get('generated',0)}",
                    f"{d.get(3,{}).get('accepted',0)}/{d.get(3,{}).get('generated',0)}",
                    f"{tr.get(0,{}).get('rank1',0)}/{tr.get(0,{}).get('count',0)}",
                    f"{tr.get(1,{}).get('rank1',0)}/{tr.get(1,{}).get('count',0)}",
                    f"{tr.get(2,{}).get('rank1',0)}/{tr.get(2,{}).get('count',0)}",
                ])

if __name__ == "__main__":
    main()
