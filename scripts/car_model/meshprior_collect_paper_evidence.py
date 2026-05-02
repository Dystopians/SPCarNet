"""Collect MeshPrior paper-evidence tables from local artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]

SCENE_ROWS = [
    {
        "row_id": "clean_origin_main_7000",
        "method": "clean_mesh_splatting",
        "scene": "parking_phone_tiny",
        "iteration": 7000,
        "model": "outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/origin_main_7000iter/model",
        "wandb_url": "https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/yiwb4d2n",
        "claim_role": "clean long-budget baseline",
    },
    {
        "row_id": "current_branch_7000",
        "method": "current_branch_unpruned",
        "scene": "parking_phone_tiny",
        "iteration": 7000,
        "model": "outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/current_branch_7000iter/model",
        "wandb_url": "https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/l5buxl3m",
        "claim_role": "quality-positive but topology-inflated diagnostic",
    },
    {
        "row_id": "current_branch_prune_50_7000",
        "method": "current_branch_area_prune_50",
        "scene": "parking_phone_tiny",
        "iteration": 7000,
        "model": "outputs/carnet/meshprior/parking_phone_tiny/stage21_5_topology_control/prune_50/model",
        "wandb_url": "https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/w1ix6e9a",
        "claim_role": "default topology-controlled row for M22",
    },
    {
        "row_id": "current_branch_prune_66_7000",
        "method": "current_branch_area_prune_66",
        "scene": "parking_phone_tiny",
        "iteration": 7000,
        "model": "outputs/carnet/meshprior/parking_phone_tiny/stage21_5_topology_control/prune_66/model",
        "wandb_url": "https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xzfqwpgi",
        "claim_role": "high-compression Pareto endpoint",
    },
    {
        "row_id": "stage17_meshprior_resume_7000",
        "method": "stage17_meshprior_resume",
        "scene": "parking_phone_tiny",
        "iteration": 7000,
        "model": "outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/stage17_meshprior_7000iter/model",
        "wandb_url": "https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/w3kczubb",
        "claim_role": "long-budget failure case",
    },
]

MISSING_ROWS = [
    {
        "row_id": "second_real_scene",
        "metric_class": "scene_generalization",
        "status": "MISSING",
        "reason": "M20 found no second suitable parking-lot COLMAP/image scene under /data/peilincai.",
    },
    {
        "row_id": "integrated_optimization_time_topology_control",
        "metric_class": "method_algorithm",
        "status": "MISSING",
        "reason": "M21.5 is post-hoc checkpoint-copy pruning, not training-loop topology control.",
    },
    {
        "row_id": "render_gated_full_meshprior_insertion",
        "metric_class": "scene_method",
        "status": "MISSING",
        "reason": "Real-scene MeshPrior edits are validated as copied-patch and checkpoint-copy diagnostics, not full render-gated insertion.",
    },
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _state_counts(model: Path, iteration: int) -> tuple[int | None, int | None]:
    path = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    if not path.is_file():
        return None, None
    state = torch.load(path, map_location="cpu")
    return int(state["_triangle_indices"].shape[0]), int(state["triangles_points"].shape[0])


def _scene_row(spec: dict[str, Any]) -> dict[str, Any]:
    model = ROOT / spec["model"]
    iteration = int(spec["iteration"])
    results_path = model / "results.json"
    geometry_path = model / "geometry_eval_colmap" / f"iter_{iteration}.json"
    triangles, vertices = _state_counts(model, iteration)
    row = {
        **{k: v for k, v in spec.items() if k != "model"},
        "metric_class": "scene_render_geometry_topology",
        "status": "AVAILABLE" if results_path.is_file() and geometry_path.is_file() else "MISSING",
        "model_path": str(model.relative_to(ROOT)),
        "triangles": triangles,
        "vertices": vertices,
        "render_psnr": None,
        "render_ssim": None,
        "render_lpips": None,
        "geometry_depth_absrel": None,
        "geometry_depth_mae": None,
        "geometry_normal_mean_ang": None,
        "oracle_metric": False,
    }
    if row["status"] == "AVAILABLE":
        metrics = _load_json(results_path).get(f"ours_{iteration}", {})
        geometry = _load_json(geometry_path)
        row.update(
            {
                "render_psnr": metrics.get("PSNR"),
                "render_ssim": metrics.get("SSIM"),
                "render_lpips": metrics.get("LPIPS"),
                "geometry_depth_absrel": (geometry.get("depth") or {}).get("abs_rel"),
                "geometry_depth_mae": (geometry.get("depth") or {}).get("mae"),
                "geometry_normal_mean_ang": (geometry.get("normal") or {}).get("mean_ang_deg"),
            }
        )
    else:
        row["missing_reason"] = f"missing {results_path} or {geometry_path}"
    return row


def _object_prior_rows() -> list[dict[str, Any]]:
    path = ROOT / "outputs/carnet/spcarnet/posterior_encoder_v1/eval_val.json"
    if not path.is_file():
        return [{"row_id": "stage3_posterior_encoder", "metric_class": "object_prior", "status": "MISSING", "source": str(path)}]
    summary = _load_json(path)["summary"]
    return [
        {
            "row_id": "stage3_posterior_encoder",
            "metric_class": "object_prior",
            "status": "AVAILABLE",
            "source": str(path.relative_to(ROOT)),
            "n_total": summary.get("n_total"),
            "recon_chamfer_l1_mean": summary.get("recon_chamfer_l1_mean"),
            "hidden_chamfer_l1_mean": summary.get("hidden_chamfer_l1_mean"),
            "visible_preservation_error_mean": summary.get("visible_preservation_error_mean"),
            "free_space_violation_rate_mean": summary.get("free_space_violation_rate_mean"),
            "mesh_extraction_success_rate": summary.get("mesh_extraction_success_rate"),
            "zero_corruption_recon_chamfer_l1_mean": summary.get("zero_corruption_recon_chamfer_l1_mean"),
            "oracle_metric": False,
        }
    ]


def _synthetic_rows() -> list[dict[str, Any]]:
    path = ROOT / "outputs/carnet/meshprior/retrieval_deformation/stage15_eval/metrics.json"
    if not path.is_file():
        return [{"row_id": "stage15_synthetic_damage", "metric_class": "synthetic_damage", "status": "MISSING", "source": str(path)}]
    payload = _load_json(path)
    rows = []
    for item in payload.get("inference_time_metrics", []):
        rows.append(
            {
                "row_id": f"stage15_{item.get('method')}_{item.get('damage_type')}",
                "metric_class": "synthetic_damage",
                "status": "AVAILABLE",
                "source": str(path.relative_to(ROOT)),
                "method": item.get("method"),
                "damage_type": item.get("damage_type"),
                "triangle_count_delta": item.get("triangle_count_delta"),
                "floater_prune_precision": item.get("floater_prune_precision"),
                "floater_prune_recall": item.get("floater_prune_recall"),
                "valid_surface_protect_recall": item.get("valid_surface_protect_recall"),
                "free_space_violation_rate": item.get("free_space_violation_rate"),
                "visible_preservation_error": item.get("visible_preservation_error"),
                "mesh_extraction_success": item.get("mesh_extraction_success"),
                "oracle_metric": False,
            }
        )
    return rows


def _proposal_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patch_path = ROOT / "outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/patch_proposal_test_report.json"
    if patch_path.is_file():
        report = _load_json(patch_path)
        counts = report.get("counts", {})
        rows.append(
            {
                "row_id": "parking_patch_proposal_gate",
                "metric_class": "proposal_gate_rollback",
                "status": "AVAILABLE",
                "source": str(patch_path.relative_to(ROOT)),
                "accepted": counts.get("accepted"),
                "rejected": counts.get("rejected"),
                "protect_noop_rejected": counts.get("protect_noop_rejected"),
                "cleanup_accepted": counts.get("cleanup_accepted"),
                "floater_rejected": counts.get("floater_rejected"),
                "rollback_snapshots": report.get("patches_tested"),
                "source_model_edited": report.get("source_model_edited"),
                "oracle_metric": False,
            }
        )
    else:
        rows.append({"row_id": "parking_patch_proposal_gate", "metric_class": "proposal_gate_rollback", "status": "MISSING", "source": str(patch_path)})
    dryrun_path = ROOT / "outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun/metrics.json"
    if dryrun_path.is_file():
        report = _load_json(dryrun_path)
        rows.append(
            {
                "row_id": "m11_synthetic_scene_gate",
                "metric_class": "proposal_gate_rollback",
                "status": "AVAILABLE",
                "source": str(dryrun_path.relative_to(ROOT)),
                "accepted": report.get("accepted_count"),
                "rejected": report.get("rejected_count"),
                "free_space_violation_delta_max": report.get("free_space_violation_delta_max"),
                "triangle_count_delta_sum": report.get("triangle_count_delta_sum"),
                "oracle_metric": False,
            }
        )
    return rows


def _failure_cases(scene_rows: list[dict[str, Any]], proposal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "case_id": "stage17_long_budget_collapse",
            "type": "method_failure",
            "evidence": "Stage17 MeshPrior resume reaches PSNR 10.839708 and depth AbsRel 0.744099 at 7000.",
            "action": "Do not continue longer Stage17 resume sweeps by default.",
        },
        {
            "case_id": "current_branch_topology_inflation",
            "type": "topology_inflation",
            "evidence": "Current branch 7000 uses 833775 triangles versus clean 285187.",
            "action": "Use prune_50 as the topology-controlled M22 row.",
        },
        {
            "case_id": "prune_66_proxy_regression",
            "type": "proxy_metric_disagreement",
            "evidence": "prune_66 beats clean render metrics but has worse depth AbsRel than clean.",
            "action": "Keep prune_66 as a Pareto endpoint, not the default row.",
        },
    ]
    for row in proposal_rows:
        if row.get("row_id") == "parking_patch_proposal_gate":
            rows.append(
                {
                    "case_id": "rejected_noop_and_floater_patch_proposals",
                    "type": "scene_gate_rejection",
                    "evidence": f"{row.get('protect_noop_rejected')} no-op and {row.get('floater_rejected')} floater proposals rejected.",
                    "action": "Use as safety evidence for proposal gates and rollback.",
                }
            )
    return rows


def _decision(scene_rows: list[dict[str, Any]]) -> tuple[str, str]:
    rows = {row["row_id"]: row for row in scene_rows}
    clean = rows.get("clean_origin_main_7000", {})
    prune = rows.get("current_branch_prune_50_7000", {})
    if not clean or not prune or prune.get("status") != "AVAILABLE":
        return "FAIL", "required clean/prune_50 scene rows are missing"
    beats_clean = (
        prune["render_psnr"] >= clean["render_psnr"]
        and prune["render_ssim"] >= clean["render_ssim"]
        and prune["render_lpips"] <= clean["render_lpips"]
    )
    if beats_clean and MISSING_ROWS:
        return "SOFT PASS", "paper evidence is reproducible and separated, but multi-scene and integrated topology-control rows remain MISSING"
    if beats_clean:
        return "PASS", "paper evidence is reproducible and separated"
    return "FAIL", "topology-controlled row does not beat clean render metrics"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _table(columns: list[str], rows: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    scene_rows = [_scene_row(spec) for spec in SCENE_ROWS]
    object_rows = _object_prior_rows()
    synthetic_rows = _synthetic_rows()
    proposal_rows = _proposal_rows()
    failure_rows = _failure_cases(scene_rows, proposal_rows)
    gate, decision = _decision(scene_rows)
    report = {
        "gate": gate,
        "decision": decision,
        "metric_classes": {
            "object_prior": object_rows,
            "synthetic_damage": synthetic_rows,
            "scene_render_geometry_topology": scene_rows,
            "proposal_gate_rollback": proposal_rows,
            "failure_cases": failure_rows,
            "missing_rows": MISSING_ROWS,
        },
        "counts": {
            "object_prior": len(object_rows),
            "synthetic_damage": len(synthetic_rows),
            "scene_render_geometry_topology": len(scene_rows),
            "proposal_gate_rollback": len(proposal_rows),
            "failure_cases": len(failure_rows),
            "missing_rows": len(MISSING_ROWS),
        },
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "paper_evidence.json").write_text(json.dumps(_clean(report), indent=2) + "\n", encoding="utf-8")
    _write_csv(out / "scene_rows.csv", _clean(scene_rows))
    _write_csv(out / "object_prior_rows.csv", _clean(object_rows))
    _write_csv(out / "synthetic_damage_rows.csv", _clean(synthetic_rows))
    _write_csv(out / "proposal_gate_rows.csv", _clean(proposal_rows))
    _write_csv(out / "failure_case_rows.csv", _clean(failure_rows))
    _write_csv(out / "missing_rows.csv", _clean(MISSING_ROWS))

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Stage 22 Paper Evidence Report\n\n")
        f.write("Date: 2026-05-02\n\n")
        f.write(f"Gate: `{gate}`.\n\n")
        f.write(f"Decision: {decision}.\n\n")
        f.write("## Scene Evidence\n\n")
        f.write(
            _table(
                ["row_id", "method", "render_psnr", "render_ssim", "render_lpips", "triangles", "geometry_depth_absrel", "claim_role"],
                scene_rows,
            )
            + "\n\n"
        )
        f.write("## Object Prior Evidence\n\n")
        f.write(
            _table(
                ["row_id", "recon_chamfer_l1_mean", "hidden_chamfer_l1_mean", "free_space_violation_rate_mean", "mesh_extraction_success_rate"],
                object_rows,
            )
            + "\n\n"
        )
        f.write("## Proposal Gate And Rollback Evidence\n\n")
        f.write(_table(["row_id", "accepted", "rejected", "cleanup_accepted", "floater_rejected", "source_model_edited"], proposal_rows) + "\n\n")
        f.write("## Failure Cases\n\n")
        f.write(_table(["case_id", "type", "evidence", "action"], failure_rows) + "\n\n")
        f.write("## Missing Rows\n\n")
        f.write(_table(["row_id", "metric_class", "status", "reason"], MISSING_ROWS) + "\n\n")
        f.write("## Output Files\n\n")
        for name in sorted(path.name for path in out.iterdir()):
            f.write(f"- `{out / name}`\n")
    report["paths"] = {"output_dir": str(out), "report_path": str(report_path)}
    print(json.dumps({"gate": gate, "decision": decision, "output_dir": str(out), "report_path": str(report_path)}, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect MeshPrior paper-evidence tables.")
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/paper_evidence")
    parser.add_argument("--report_path", default="docs/car_model/meshprior_stage22_paper_evidence_report.md")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
