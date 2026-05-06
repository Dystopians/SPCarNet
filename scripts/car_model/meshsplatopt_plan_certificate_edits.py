#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.certificate_edit_planner import plan_certificate_edits, write_certificate_edit_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan certificate-carrying mesh edits from an Evidence Conflict Graph.")
    parser.add_argument("--ecg_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    ecg = json.loads(Path(args.ecg_json).read_text(encoding="utf-8"))
    plan = plan_certificate_edits(ecg)
    write_certificate_edit_plan(plan, args.output_dir)
    print({"plans": len(plan["plans"]), "action_types": plan["action_types"], "top": plan["plans"][0] if plan["plans"] else None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

