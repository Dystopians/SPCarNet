#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


TOPOLOGY_ACTIONS = {"SPLIT_ALLOCATE", "FILL_PATCH_LOCAL", "DELETE_OR_COLLAPSE"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize low-risk certificate edit plan entries into executable metadata packages.")
    parser.add_argument("--plan_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--allow_topology_edits", action="store_true")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan_json).read_text(encoding="utf-8"))
    selected = []
    rejected = []
    for row in plan.get("plans", []):
        action = str(row.get("action", ""))
        if action == "REJECT_UNOBSERVED":
            rejected.append({"row": row, "reason": "reject_unobserved"})
            continue
        if action in TOPOLOGY_ACTIONS and not args.allow_topology_edits:
            rejected.append({"row": row, "reason": "topology_edits_disabled"})
            continue
        selected.append(row)
        if len(selected) >= int(args.top_k):
            break
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_edits": selected,
        "rejected_candidates": rejected[:50],
        "top_k": int(args.top_k),
        "allow_topology_edits": bool(args.allow_topology_edits),
        "status": "READY" if selected else "NO_MATERIALIZABLE_EDITS",
    }
    (out / "real_surgery_plan.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Real Surgery Materialization Report", "", f"- status: `{payload['status']}`", f"- selected edits: `{len(selected)}`", ""]
    for row in selected:
        lines.append(f"- `{row.get('action')}` clusters `{row.get('target_cluster_ids')}` risk `{row.get('expected_risk')}`")
    if not selected:
        lines.append("No edit was materialized because the certificate planner found no action that satisfies the configured safety constraints.")
    (out / "real_surgery_materialization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": payload["status"], "selected": len(selected), "output_dir": str(out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

