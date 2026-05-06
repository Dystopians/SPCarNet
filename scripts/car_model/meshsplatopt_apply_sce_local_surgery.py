#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.sce_local_surgery import apply_synthetic_sce_local_surgery  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply synthetic SCE local surgery proposal.")
    parser.add_argument("--proposal_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sentinel_error", type=float, default=1.0)
    args = parser.parse_args()
    payload = json.loads(Path(args.proposal_json).read_text(encoding="utf-8"))
    proposal = payload.get("proposal", payload.get("proposals", [{}])[0])
    result = apply_synthetic_sce_local_surgery(proposal, float(args.sentinel_error))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sce_local_surgery_apply_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

