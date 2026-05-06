#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.certified_model_selection import (  # noqa: E402
    CertifiedSelectionConfig,
    select_certified_render_candidate,
)


def _load_method_metrics(path: str, method: str) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if method not in payload:
        raise KeyError(f"Method {method!r} not found in {path}")
    return dict(payload[method])


def main() -> int:
    parser = argparse.ArgumentParser(description="Select parent or recovered model with a render Pareto certificate.")
    parser.add_argument("--parent_metrics_json", required=True)
    parser.add_argument("--parent_method", required=True)
    parser.add_argument("--candidate_metrics_json", required=True)
    parser.add_argument("--candidate_method", required=True)
    parser.add_argument("--psnr_tolerance", type=float, default=0.0)
    parser.add_argument("--ssim_tolerance", type=float, default=0.0)
    parser.add_argument("--lpips_tolerance", type=float, default=0.0)
    parser.add_argument("--min_score_delta", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = CertifiedSelectionConfig(
        psnr_tolerance=float(args.psnr_tolerance),
        ssim_tolerance=float(args.ssim_tolerance),
        lpips_tolerance=float(args.lpips_tolerance),
        min_score_delta=float(args.min_score_delta),
    )
    decision = select_certified_render_candidate(
        parent=_load_method_metrics(args.parent_metrics_json, args.parent_method),
        candidate=_load_method_metrics(args.candidate_metrics_json, args.candidate_method),
        cfg=cfg,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
