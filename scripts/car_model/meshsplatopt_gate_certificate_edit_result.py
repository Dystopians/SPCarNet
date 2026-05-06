#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


LOWER = {"lpips", "absrel", "depth_mae", "normal"}
HIGHER = {"psnr", "ssim"}


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate a certificate edit result against parent metrics.")
    parser.add_argument("--parent_metrics_json", required=True)
    parser.add_argument("--candidate_metrics_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()
    parent = _load(args.parent_metrics_json)
    candidate = _load(args.candidate_metrics_json)
    checks = {}
    for key in sorted(LOWER | HIGHER):
        if key not in parent or key not in candidate:
            continue
        p = float(parent[key])
        c = float(candidate[key])
        checks[key] = c <= p + args.tolerance if key in LOWER else c + args.tolerance >= p
    passed = bool(checks) and all(checks.values())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"passed": passed, "checks": checks, "tolerance": float(args.tolerance)}
    (out / "real_surgery_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

