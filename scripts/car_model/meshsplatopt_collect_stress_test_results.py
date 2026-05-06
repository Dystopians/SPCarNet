#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect SCE14 stress-test result JSON files into one CSV.")
    parser.add_argument("--result_json", action="append", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    rows = []
    for path in args.result_json:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.extend(payload.get("rows", []))
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "defects_repaired", "defects_total", "repair_rate", "certificate_violation_rate", "false_repair_rate", "passes_gate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print({"rows": len(rows), "output_csv": str(out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

