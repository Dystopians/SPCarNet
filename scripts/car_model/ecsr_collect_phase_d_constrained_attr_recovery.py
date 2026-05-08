#!/usr/bin/env python3
"""Collect constrained topology-frozen attribute-recovery experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SCENES = ("bicycle", "flowers", "treehill", "garden")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recovery_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_d/constrained_attr_recovery_v2_cachefix"),
    )
    parser.add_argument(
        "--method_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k"),
    )
    parser.add_argument(
        "--doc_out",
        type=Path,
        default=Path("docs/car_model/5-8-ECSR-PhaseD-ConstrainedAttributeRecovery.md"),
    )
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--policy_tag", default="sor_adaptive_geo")
    parser.add_argument("--compact_source", choices=("method_root", "recovery"), default="method_root")
    parser.add_argument("--source_iteration", type=int, default=26000)
    parser.add_argument("--final_iteration", type=int, default=27000)
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


def metric(path: Path, method: str) -> dict[str, float] | None:
    if not path.exists():
        return None
    payload = load_json(path)
    value = payload.get(method)
    if not isinstance(value, dict):
        return None
    return {k: float(value[k]) for k in ("PSNR", "SSIM", "LPIPS") if k in value}


def geometry(path: Path) -> dict[str, float] | None:
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


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmtd(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def find_wandb_url(model_path: Path, project: str, fallback_name: str) -> str:
    wandb_root = model_path / "wandb"
    if wandb_root.exists():
        runs = sorted(p for p in wandb_root.iterdir() if p.is_dir() and p.name.startswith("run-"))
        if runs:
            run_id = runs[-1].name.rsplit("-", 1)[-1]
            return f"https://wandb.ai/karamazovaniki-university-of-southern-california/{project}/runs/{run_id}"
    return f"https://wandb.ai/karamazovaniki-university-of-southern-california/{project}/runs/{fallback_name}"


def main() -> int:
    args = parse_args()
    scenes = [x.strip() for x in args.scenes.split(",") if x.strip()]
    records: list[dict[str, Any]] = []
    for scene in scenes:
        out_dir = args.recovery_root / scene
        model_path = out_dir / "model"
        summary_path = out_dir / "recovery_summary.json"
        topology_path = out_dir / "topology_audit.json"
        source_model = (
            model_path
            if args.compact_source == "recovery"
            else args.method_root / scene / args.policy_tag / "compact_model"
        )
        if not summary_path.exists() or not topology_path.exists():
            records.append({"scene": scene, "status": "missing", "model_path": str(model_path)})
            continue
        summary = load_json(summary_path)
        topology = load_json(topology_path)
        final_metric = metric(model_path / "results.json", f"ours_{args.final_iteration}")
        compact_metric = metric(source_model / "results.json", f"ours_{args.source_iteration}")
        compact_ela_metric = metric(
            source_model / "results.json",
            f"ours_{args.source_iteration}_{args.policy_tag}_compact_ela",
        )
        final_geom = geometry(model_path / "geometry_eval_colmap" / f"iter_{args.final_iteration}_max500.json")
        compact_geom = geometry(source_model / "geometry_eval_colmap" / f"iter_{args.source_iteration}_max500.json")
        if final_metric is None or compact_metric is None:
            status = "missing_metrics"
            deltas: dict[str, float | None] = {"dpsnr": None, "dssim": None, "dlpips": None}
            accepted = False
        else:
            deltas = {
                "dpsnr": final_metric["PSNR"] - compact_metric["PSNR"],
                "dssim": final_metric["SSIM"] - compact_metric["SSIM"],
                "dlpips": final_metric["LPIPS"] - compact_metric["LPIPS"],
            }
            accepted = (
                bool(topology.get("topology_unchanged", False))
                and float(deltas["dpsnr"]) >= -float(args.policy_psnr_epsilon)
                and float(deltas["dssim"]) >= -float(args.policy_ssim_epsilon)
                and float(deltas["dlpips"]) <= float(args.policy_lpips_epsilon)
            )
            status = "ACCEPT_POLICY_DIAGNOSTIC" if accepted else "REJECT_RGB_REGRESSION"
        ela_gap = {}
        if final_metric is not None and compact_ela_metric is not None:
            ela_gap = {
                "dpsnr_vs_compact_ela": final_metric["PSNR"] - compact_ela_metric["PSNR"],
                "dssim_vs_compact_ela": final_metric["SSIM"] - compact_ela_metric["SSIM"],
                "dlpips_vs_compact_ela": final_metric["LPIPS"] - compact_ela_metric["LPIPS"],
            }
        geom_delta = {}
        if final_geom is not None and compact_geom is not None:
            geom_delta = {
                "d_absrel": final_geom["depth_abs_rel"] - compact_geom["depth_abs_rel"],
                "d_depth_mae": final_geom["depth_mae"] - compact_geom["depth_mae"],
                "d_normal": final_geom["normal_mean_ang_deg"] - compact_geom["normal_mean_ang_deg"],
            }
        records.append(
            {
                "scene": scene,
                "status": status,
                "accepted": accepted,
                "model_path": str(model_path),
                "source_model": str(source_model),
                "topology_unchanged": bool(topology.get("topology_unchanged", False)),
                "load_topology": topology.get("load"),
                "final_topology": topology.get("final"),
                "summary": summary,
                "metrics": {
                    "final": final_metric,
                    "compact_only": compact_metric,
                    "compact_ela": compact_ela_metric,
                    "delta_vs_compact": deltas,
                    "delta_vs_compact_ela": ela_gap,
                },
                "geometry": {
                    "final": final_geom,
                    "compact_only": compact_geom,
                    "delta_vs_compact": geom_delta,
                },
                "wandb_url": find_wandb_url(
                    model_path,
                    str(summary.get("wandb_project", "spcarnet_meshprior")),
                    str(summary.get("wandb_name", "")),
                ),
            }
        )

    accepted_count = sum(1 for r in records if r.get("accepted"))
    payload = {
        "protocol": {
            "recovery_root": str(args.recovery_root),
            "method_root": str(args.method_root),
            "policy_tag": args.policy_tag,
            "compact_source": args.compact_source,
            "source_iteration": int(args.source_iteration),
            "final_iteration": int(args.final_iteration),
            "test_usage": "held-out test metrics are final diagnostics only, not strength selection",
            "acceptance_rule": "topology unchanged and no RGB regression vs compact-only",
            "policy_epsilons": {
                "psnr": float(args.policy_psnr_epsilon),
                "ssim": float(args.policy_ssim_epsilon),
                "lpips": float(args.policy_lpips_epsilon),
            },
        },
        "records": records,
        "accepted": accepted_count,
        "total": len(records),
    }
    args.recovery_root.mkdir(parents=True, exist_ok=True)
    (args.recovery_root / "phase_d_constrained_attr_recovery_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    geom_rows = []
    wandb_rows = []
    for record in records:
        final_metric = record.get("metrics", {}).get("final") or {}
        deltas = record.get("metrics", {}).get("delta_vs_compact") or {}
        ela_gap = record.get("metrics", {}).get("delta_vs_compact_ela") or {}
        rows.append(
            [
                record["scene"],
                record.get("status", "missing"),
                "yes" if record.get("topology_unchanged") else "no",
                fmt(final_metric.get("PSNR")),
                fmt(final_metric.get("SSIM")),
                fmt(final_metric.get("LPIPS")),
                fmtd(deltas.get("dpsnr")),
                fmtd(deltas.get("dssim")),
                fmtd(deltas.get("dlpips")),
                fmtd(ela_gap.get("dpsnr_vs_compact_ela")),
                fmtd(ela_gap.get("dssim_vs_compact_ela")),
                fmtd(ela_gap.get("dlpips_vs_compact_ela")),
            ]
        )
        geom_delta = record.get("geometry", {}).get("delta_vs_compact") or {}
        if geom_delta:
            geom_rows.append(
                [
                    record["scene"],
                    fmtd(geom_delta.get("d_absrel"), 6),
                    fmtd(geom_delta.get("d_depth_mae"), 4),
                    fmtd(geom_delta.get("d_normal"), 4),
                ]
            )
        wandb_rows.append([record["scene"], record.get("wandb_url", "n/a")])

    md = [
        "# ECSR Phase-D Constrained Attribute Recovery",
        "",
        "This report collects topology-frozen representation-level recovery runs.",
        "The checkpoint topology is fixed, the rendered images are not edited, and",
        "W&B is enabled during training. A run is accepted only if the recovered",
        "checkpoint is topology-stable and does not regress PSNR, SSIM, or LPIPS",
        "against compact-only. Compact-ELA is reported as the image-space teacher",
        "or upper bound, not as the accepted representation-level method.",
        "",
        md_table(
            [
                "scene",
                "status",
                "topology",
                "PSNR",
                "SSIM",
                "LPIPS",
                "dPSNR vs compact",
                "dSSIM vs compact",
                "dLPIPS vs compact",
                "dPSNR vs ELA",
                "dSSIM vs ELA",
                "dLPIPS vs ELA",
            ],
            rows,
        ),
        "",
        f"Accepted by strict diagnostic rule: `{accepted_count} / {len(records)}`.",
        "",
        "## Geometry Delta Vs Compact-Only",
        "",
        md_table(["scene", "dAbsRel", "dDepthMAE", "dNormalDeg"], geom_rows) if geom_rows else "No geometry deltas found.",
        "",
        "## W&B Runs",
        "",
        md_table(["scene", "url"], wandb_rows),
        "",
        "## Interpretation",
        "",
        "This is a strict Phase-D diagnostic, not a headline method unless it",
        "passes the table above. Negative rows are useful because they separate",
        "representation-level recovery failures from image-space ELA gains and",
        "prevent us from promoting a method that only looks good after test-time",
        "post-render correction.",
        "",
    ]
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text("\n".join(md), encoding="utf-8")
    print(f"[ECSR] wrote {args.doc_out}")
    print(f"[ECSR] wrote {args.recovery_root / 'phase_d_constrained_attr_recovery_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
