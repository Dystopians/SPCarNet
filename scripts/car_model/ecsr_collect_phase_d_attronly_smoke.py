#!/usr/bin/env python3
"""Collect Phase-D attribute-only recovery smoke results.

The collector is intentionally strict: a completed run is not automatically an
accepted ECSR candidate. It compares the recovered checkpoint against the source
compact-only checkpoint and the archived Compact-ELA result, then records the
accept/reject status for the Phase-D smoke milestone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RECOVERIES = (
    "bicycle_C0001",
    "kitchen_C0019",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recovery_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_d/attronly_smoke"),
    )
    parser.add_argument(
        "--materialized_summary",
        type=Path,
        default=Path(
            "outputs/carnet/meshsplatopt/ecsr_phase_c/materialized_static_pass/"
            "phase_c_materialized_static_pass_summary.json"
        ),
    )
    parser.add_argument(
        "--doc_out",
        type=Path,
        default=Path("docs/car_model/5-8-ECSR-PhaseD-AttributeOnlySmoke.md"),
    )
    parser.add_argument("--recoveries", default=",".join(DEFAULT_RECOVERIES))
    parser.add_argument("--policy_psnr_epsilon", type=float, default=0.0)
    parser.add_argument("--policy_ssim_epsilon", type=float, default=0.0)
    parser.add_argument("--policy_lpips_epsilon", type=float, default=0.0)
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


def metric(results: dict[str, Any], key: str) -> dict[str, float] | None:
    value = results.get(key)
    if not isinstance(value, dict):
        return None
    return {k: float(value[k]) for k in ("PSNR", "SSIM", "LPIPS") if k in value}


def find_wandb_url(model_path: Path, project: str, fallback_name: str) -> str:
    wandb_root = model_path / "wandb"
    if wandb_root.exists():
        runs = sorted(p for p in wandb_root.iterdir() if p.is_dir() and p.name.startswith("run-"))
        if runs:
            run_id = runs[-1].name.rsplit("-", 1)[-1]
            return f"https://wandb.ai/karamazovaniki-university-of-southern-california/{project}/runs/{run_id}"
    return f"https://wandb.ai/karamazovaniki-university-of-southern-california/{project}/runs/{fallback_name}"


def geometry_summary(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    payload = load_json(path)
    depth = payload.get("depth", {})
    normal = payload.get("normal", {})
    return {
        "depth_abs_rel": float(depth.get("abs_rel", 0.0)),
        "depth_mae": float(depth.get("mae", 0.0)),
        "normal_mean_ang_deg": float(normal.get("mean_ang_deg", 0.0)),
    }


def fmt_delta(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def candidate_key(scene: str, candidate_id: str) -> str:
    return f"{scene}_{candidate_id}"


def main() -> int:
    args = parse_args()
    recoveries = [x.strip() for x in args.recoveries.split(",") if x.strip()]
    materialized = load_json(args.materialized_summary)
    materialized_by_key = {
        candidate_key(str(rep["scene"]), str(rep["candidate_id"])): rep
        for rep in materialized.get("reports", [])
    }
    records = []
    for recovery in recoveries:
        out_dir = args.recovery_root / recovery
        summary_path = out_dir / "recovery_summary.json"
        topology_path = out_dir / "topology_audit.json"
        if not summary_path.exists() or not topology_path.exists():
            records.append({"recovery": recovery, "status": "missing", "out_dir": str(out_dir)})
            continue
        summary = load_json(summary_path)
        topology = load_json(topology_path)
        model_path = Path(summary["output_path"])
        final_iteration = int(summary["final_iteration"])
        source_iteration = int(summary["load_iteration"])
        scene = Path(summary["source_path"]).name
        candidate_id = recovery.split("_", 1)[1]
        materialized_report = materialized_by_key.get(candidate_key(scene, candidate_id), {})
        source_model = Path(materialized_report.get("source_model", ""))
        final_results = load_json(model_path / "results.json")
        source_results = load_json(source_model / "results.json") if source_model.exists() else {}
        final_metric = metric(final_results, f"ours_{final_iteration}")
        compact_metric = metric(source_results, f"ours_{source_iteration}")
        compact_ela_metric = metric(source_results, f"ours_{source_iteration}_sor_adaptive_geo_compact_ela")
        final_geometry = geometry_summary(model_path / "geometry_eval_colmap" / f"iter_{final_iteration}_max500.json")
        compact_geometry = geometry_summary(
            source_model / "geometry_eval_colmap" / f"iter_{source_iteration}_max500.json"
        )
        if final_metric is None or compact_metric is None:
            status = "missing_metrics"
            deltas = {}
            accepted = False
        else:
            deltas = {
                "dpsnr_vs_compact": final_metric["PSNR"] - compact_metric["PSNR"],
                "dssim_vs_compact": final_metric["SSIM"] - compact_metric["SSIM"],
                "dlpips_vs_compact": final_metric["LPIPS"] - compact_metric["LPIPS"],
            }
            accepted = (
                bool(topology.get("topology_unchanged", False))
                and deltas["dpsnr_vs_compact"] >= -float(args.policy_psnr_epsilon)
                and deltas["dssim_vs_compact"] >= -float(args.policy_ssim_epsilon)
                and deltas["dlpips_vs_compact"] <= float(args.policy_lpips_epsilon)
            )
            status = "ACCEPT_SMOKE" if accepted else "REJECT_SMOKE_REGRESSION"
        extra_triangle_reduction = None
        topo = materialized_report.get("topology_audit", {})
        if topo:
            pre_tri = max(1, int(topo.get("pre_triangles", 1)))
            extra_triangle_reduction = 100.0 * float(topo.get("removed_triangles", 0)) / float(pre_tri)
        records.append(
            {
                "recovery": recovery,
                "scene": scene,
                "candidate_id": candidate_id,
                "status": status,
                "accepted": accepted,
                "model_path": str(model_path),
                "source_model": str(source_model),
                "source_iteration": source_iteration,
                "final_iteration": final_iteration,
                "wandb_url": find_wandb_url(
                    model_path,
                    str(summary.get("wandb_project", "spcarnet_meshprior")),
                    str(summary.get("wandb_name", "")),
                ),
                "topology_unchanged": bool(topology.get("topology_unchanged", False)),
                "extra_triangle_reduction_percent": extra_triangle_reduction,
                "metrics": {
                    "final": final_metric,
                    "compact_only": compact_metric,
                    "compact_ela": compact_ela_metric,
                    "deltas": deltas,
                },
                "geometry": {
                    "final": final_geometry,
                    "compact_only": compact_geometry,
                },
            }
        )

    payload = {
        "protocol": {
            "recovery_root": str(args.recovery_root),
            "materialized_summary": str(args.materialized_summary),
            "test_usage": "final smoke reporting only; not used for policy selection",
            "acceptance_rule": "topology unchanged and no RGB metric regression vs compact-only",
            "policy_epsilons": {
                "psnr": float(args.policy_psnr_epsilon),
                "ssim": float(args.policy_ssim_epsilon),
                "lpips": float(args.policy_lpips_epsilon),
            },
        },
        "records": records,
        "accepted": sum(1 for r in records if r.get("accepted")),
        "total": len(records),
    }
    args.recovery_root.mkdir(parents=True, exist_ok=True)
    (args.recovery_root / "phase_d_attronly_smoke_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for record in records:
        final_metric = record.get("metrics", {}).get("final") or {}
        deltas = record.get("metrics", {}).get("deltas") or {}
        rows.append(
            [
                record["recovery"],
                record.get("status", "missing"),
                "yes" if record.get("topology_unchanged") else "no",
                f"{record.get('extra_triangle_reduction_percent', 0.0):.6f}%"
                if record.get("extra_triangle_reduction_percent") is not None
                else "n/a",
                f"{final_metric.get('PSNR', 0.0):.4f}" if final_metric else "n/a",
                f"{final_metric.get('SSIM', 0.0):.4f}" if final_metric else "n/a",
                f"{final_metric.get('LPIPS', 0.0):.4f}" if final_metric else "n/a",
                fmt_delta(float(deltas.get("dpsnr_vs_compact", 0.0))) if deltas else "n/a",
                fmt_delta(float(deltas.get("dssim_vs_compact", 0.0))) if deltas else "n/a",
                fmt_delta(float(deltas.get("dlpips_vs_compact", 0.0))) if deltas else "n/a",
            ]
        )
    geom_rows = []
    for record in records:
        final_geom = record.get("geometry", {}).get("final")
        compact_geom = record.get("geometry", {}).get("compact_only")
        if not final_geom or not compact_geom:
            continue
        geom_rows.append(
            [
                record["recovery"],
                fmt_delta(final_geom["depth_abs_rel"] - compact_geom["depth_abs_rel"], 6),
                fmt_delta(final_geom["depth_mae"] - compact_geom["depth_mae"], 4),
                fmt_delta(final_geom["normal_mean_ang_deg"] - compact_geom["normal_mean_ang_deg"], 4),
            ]
        )
    wandb_rows = [[record["recovery"], record.get("wandb_url", "n/a")] for record in records]
    accepted = int(payload["accepted"])
    md = [
        "# ECSR Phase-D Attribute-Only Recovery Smoke",
        "",
        "This report covers the first executable Version-1 surface-attached",
        "appearance recovery smoke after Phase-C static contraction. The runs",
        "freeze topology and vertices, optimize only appearance attributes, sync",
        "to W&B, render the held-out test split once, and evaluate sparse COLMAP",
        "geometry. The held-out test metrics here are diagnostics; they were not",
        "used to select a policy or tune a scene-specific setting.",
        "",
        md_table(
            [
                "recovery",
                "status",
                "topology",
                "extra tri red.",
                "PSNR",
                "SSIM",
                "LPIPS",
                "dPSNR vs compact",
                "dSSIM vs compact",
                "dLPIPS vs compact",
            ],
            rows,
        ),
        "",
        f"Accepted by smoke rule: `{accepted} / {len(records)}`",
        "",
        "## Geometry Delta Vs Compact-Only",
        "",
        md_table(["recovery", "dAbsRel", "dDepthMAE", "dNormalDeg"], geom_rows) if geom_rows else "No geometry files found.",
        "",
        "## W&B Runs",
        "",
        md_table(["recovery", "url"], wandb_rows),
        "",
        "## Interpretation",
        "",
        "The infrastructure is now real: materialized contraction checkpoints can",
        "be loaded, topology can remain frozen through recovery, W&B logging works,",
        "and RGB/geometry metrics are produced. However, this Version-1 smoke is",
        "not accepted as a final method because it regresses held-out RGB metrics",
        "relative to the compact-only source checkpoints. The next Phase-D step",
        "must add policy-val controlled early stopping or a representation-attached",
        "residual/delta mechanism instead of treating longer attribute fine-tuning",
        "as automatically beneficial.",
        "",
    ]
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text("\n".join(md), encoding="utf-8")
    print(f"[ECSR] wrote {args.doc_out}")
    print(f"[ECSR] wrote {args.recovery_root / 'phase_d_attronly_smoke_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
