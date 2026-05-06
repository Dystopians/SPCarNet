#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.sce_local_surgery import propose_sce_local_surgery, write_sce_local_surgery_outputs  # noqa: E402


def _rows_from_csv(path: Path) -> list[dict]:
    return [dict(r) for r in csv.DictReader(path.open("r", encoding="utf-8"))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Make SCE local surgery proposals from ECG cluster summary or synthetic JSON.")
    parser.add_argument("--cluster_csv", default="")
    parser.add_argument("--cluster_json", default="")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if args.cluster_json:
        payload = json.loads(Path(args.cluster_json).read_text(encoding="utf-8"))
        rows = payload.get("clusters", payload.get("cluster_summary", []))
    else:
        rows = _rows_from_csv(Path(args.cluster_csv))
    proposals = propose_sce_local_surgery(rows)
    write_sce_local_surgery_outputs(proposals, args.output_dir)
    print({"proposals": len(proposals), "accepted": sum(int(p["accepted"]) for p in proposals)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

