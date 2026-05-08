#!/usr/bin/env python3
"""Collect ECSR Phase-D surface residual delta smoke and policy-val results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SCENES = ("bicycle", "flowers", "treehill", "garden")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delta_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_d/surface_residual_delta_smoke"),
    )
    parser.add_argument(
        "--policy_val_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_d/surface_residual_delta_policy_val"),
    )
    parser.add_argument(
        "--method_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k"),
    )
    parser.add_argument(
        "--doc_out",
        type=Path,
        default=Path("docs/car_model/5-8-ECSR-PhaseD-SurfaceResidualDeltaSmoke.md"),
    )
    parser.add_argument("--policy_tag", default="sor_adaptive_geo")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--policy_l1_epsilon", type=float, default=0.0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def result_metric(path: Path, key: str) -> dict[str, float] | None:
    if not path.exists():
        return None
    payload = load_json(path)
    value = payload.get(key)
    if not isinstance(value, dict):
        return None
    return {k: float(value[k]) for k in ("PSNR", "SSIM", "LPIPS") if k in value}


def mean_policy_l1(path: Path) -> tuple[float | None, list[int]]:
    if not path.exists():
        return None, []
    payload = load_json(path)
    views = payload.get("view_summaries", [])
    if not views:
        return None, []
    mean_l1 = sum(float(v["mean_l1_error"]) for v in views) / float(len(views))
    indices = [int(v["view_index"]) for v in views]
    return mean_l1, indices


def fmt_delta(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def main() -> int:
    args = parse_args()
    scenes = [x.strip() for x in args.scenes.split(",") if x.strip()]
    records = []
    for scene in scenes:
        source_model = args.method_root / scene / args.policy_tag / "compact_model"
        delta_model = args.delta_root / scene / "model"
        audit_path = delta_model / "surface_residual_delta_audit.json"
        compact_metric = result_metric(source_model / "results.json", f"ours_{args.iteration}")
        delta_metric = result_metric(delta_model / "results.json", f"ours_{args.iteration}")
        compact_ela_metric = result_metric(
            source_model / "results.json",
            f"ours_{args.iteration}_{args.policy_tag}_compact_ela",
        )
        compact_l1, compact_indices = mean_policy_l1(
            args.policy_val_root / "compact" / scene / "surface_evidence_summary.json"
        )
        delta_l1, delta_indices = mean_policy_l1(
            args.policy_val_root / "delta" / scene / "surface_evidence_summary.json"
        )
        audit = load_json(audit_path) if audit_path.exists() else {}
        d_policy_l1 = None if compact_l1 is None or delta_l1 is None else delta_l1 - compact_l1
        accepted_policy = d_policy_l1 is not None and d_policy_l1 <= float(args.policy_l1_epsilon)
        if compact_metric is not None and delta_metric is not None:
            dpsnr = delta_metric["PSNR"] - compact_metric["PSNR"]
            dssim = delta_metric["SSIM"] - compact_metric["SSIM"]
            dlpips = delta_metric["LPIPS"] - compact_metric["LPIPS"]
            test_safe = dpsnr >= 0.0 and dssim >= 0.0 and dlpips <= 0.0
        else:
            dpsnr = dssim = dlpips = None
            test_safe = False
        if not accepted_policy:
            status = "REJECT_POLICY_VAL"
        elif test_safe:
            status = "ACCEPT_SMOKE"
        else:
            status = "POLICY_ACCEPT_TEST_REGRESSION"
        records.append(
            {
                "scene": scene,
                "status": status,
                "policy_accepted": bool(accepted_policy),
                "test_safe": bool(test_safe),
                "policy_val_indices": delta_indices or compact_indices,
                "compact_policy_l1": compact_l1,
                "delta_policy_l1": delta_l1,
                "d_policy_l1": d_policy_l1,
                "compact_metric": compact_metric,
                "delta_metric": delta_metric,
                "compact_ela_metric": compact_ela_metric,
                "dpsnr_vs_compact": dpsnr,
                "dssim_vs_compact": dssim,
                "dlpips_vs_compact": dlpips,
                "faces_used": audit.get("faces_used"),
                "vertices_modified": audit.get("vertices_modified"),
                "delta_rgb_abs_mean": audit.get("delta_rgb_abs_mean"),
                "delta_rgb_abs_max": audit.get("delta_rgb_abs_max"),
                "topology_unchanged": audit.get("topology_before") == {
                    "triangles": audit.get("topology_after", {}).get("triangles"),
                    "vertices": audit.get("topology_after", {}).get("vertices"),
                }
                if audit
                else False,
            }
        )

    payload = {
        "protocol": {
            "delta_root": str(args.delta_root),
            "policy_val_root": str(args.policy_val_root),
            "method_root": str(args.method_root),
            "test_usage": "test metrics are final diagnostics only; policy status is from train policy-val L1",
            "acceptance_rule": "delta policy-val mean L1 <= compact policy-val mean L1 + epsilon",
            "policy_l1_epsilon": float(args.policy_l1_epsilon),
        },
        "records": records,
        "policy_accepted": sum(1 for r in records if r.get("policy_accepted")),
        "accepted": sum(1 for r in records if r["status"] == "ACCEPT_SMOKE"),
        "total": len(records),
    }
    args.delta_root.mkdir(parents=True, exist_ok=True)
    (args.delta_root / "phase_d_surface_residual_delta_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for record in records:
        metric = record.get("delta_metric") or {}
        rows.append(
            [
                record["scene"],
                record["status"],
                ",".join(str(x) for x in record["policy_val_indices"][:8]),
                f"{record['faces_used']}" if record.get("faces_used") is not None else "n/a",
                f"{record['vertices_modified']}" if record.get("vertices_modified") is not None else "n/a",
                f"{record['delta_rgb_abs_mean']:.6f}" if record.get("delta_rgb_abs_mean") is not None else "n/a",
                f"{record['d_policy_l1']:+.6f}" if record.get("d_policy_l1") is not None else "n/a",
                f"{metric.get('PSNR', 0.0):.4f}" if metric else "n/a",
                fmt_delta(record["dpsnr_vs_compact"]) if record.get("dpsnr_vs_compact") is not None else "n/a",
                fmt_delta(record["dssim_vs_compact"]) if record.get("dssim_vs_compact") is not None else "n/a",
                fmt_delta(record["dlpips_vs_compact"]) if record.get("dlpips_vs_compact") is not None else "n/a",
            ]
        )
    policy_accepted = int(payload["policy_accepted"])
    accepted = int(payload["accepted"])
    md = [
        "# ECSR Phase-D Surface Residual Delta Smoke",
        "",
        "This is the first Version-2 representation-attached residual test. It",
        "writes a bounded residual RGB delta into per-vertex SH DC coefficients",
        "using only Phase-A train evidence. It does not edit rendered images and",
        "does not use held-out test views for policy acceptance.",
        "",
        md_table(
            [
                "scene",
                "policy status",
                "policy-val views",
                "faces",
                "vertices",
                "mean delta RGB",
                "dPolicy L1",
                "test PSNR",
                "dPSNR",
                "dSSIM",
                "dLPIPS",
            ],
            rows,
        ),
        "",
        f"Policy-val accepted: `{policy_accepted} / {len(records)}`",
        f"Final smoke accepted after held-out diagnostic: `{accepted} / {len(records)}`",
        "",
        "## Interpretation",
        "",
        "The checkpoint-level residual attachment path is now implemented and",
        "renderer-compatible. The current naive DC-only rule is intentionally",
        "bounded and fixed across scenes. The smoke exposes a stronger problem:",
        "mean train-policy L1 is too weak as the only gate, because it accepts",
        "several tiny local deltas that still regress held-out RGB. The next",
        "representation-level recovery needs local-mask policy metrics, strength",
        "selection, and a least-squares or learned residual solve with smoothness",
        "instead of a direct top-support DC offset.",
        "",
    ]
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text("\n".join(md), encoding="utf-8")
    print(f"[ECSR] wrote {args.doc_out}")
    print(f"[ECSR] wrote {args.delta_root / 'phase_d_surface_residual_delta_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
