"""Collect Stage 21.5 topology-control ablation rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


WAND_B_URLS = {
    "clean_origin_main_7000": "https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/yiwb4d2n",
    "current_branch_7000": "https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/l5buxl3m",
    "prune_25": "https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/evid1gbt",
    "prune_50": "https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/w1ix6e9a",
    "prune_66": "https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xzfqwpgi",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _state_counts(path: Path) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu")
    return int(state["_triangle_indices"].shape[0]), int(state["triangles_points"].shape[0])


def _row(label: str, method: str, model_path: Path, iteration: int) -> dict[str, Any]:
    state_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    results_path = model_path / "results.json"
    geometry_path = model_path / "geometry_eval_colmap" / f"iter_{iteration}.json"
    triangles, vertices = _state_counts(state_path)
    metrics = _load_json(results_path).get(f"ours_{iteration}", {})
    geometry = _load_json(geometry_path)
    depth = geometry.get("depth", {})
    normal = geometry.get("normal", {})
    triangle_units = max(triangles / 100000.0, 1e-12)
    return {
        "label": label,
        "method": method,
        "model_path": str(model_path),
        "wandb_url": WAND_B_URLS.get(label, ""),
        "iteration": int(iteration),
        "triangles": int(triangles),
        "vertices": int(vertices),
        "render_psnr": float(metrics.get("PSNR", float("nan"))),
        "render_ssim": float(metrics.get("SSIM", float("nan"))),
        "render_lpips": float(metrics.get("LPIPS", float("nan"))),
        "render_psnr_per_100k_triangles": float(metrics.get("PSNR", float("nan"))) / triangle_units,
        "geometry_depth_absrel": float(depth.get("abs_rel", float("nan"))),
        "geometry_depth_mae": float(depth.get("mae", float("nan"))),
        "geometry_normal_mean_ang": float(normal.get("mean_ang_deg", float("nan"))),
    }


def _add_deltas(rows: list[dict[str, Any]]) -> None:
    by_label = {row["label"]: row for row in rows}
    clean = by_label["clean_origin_main_7000"]
    current = by_label["current_branch_7000"]
    for row in rows:
        for base_name, base in (("clean", clean), ("current", current)):
            row[f"delta_psnr_vs_{base_name}"] = row["render_psnr"] - base["render_psnr"]
            row[f"delta_ssim_vs_{base_name}"] = row["render_ssim"] - base["render_ssim"]
            row[f"delta_lpips_vs_{base_name}"] = row["render_lpips"] - base["render_lpips"]
            row[f"delta_depth_absrel_vs_{base_name}"] = row["geometry_depth_absrel"] - base["geometry_depth_absrel"]
            row[f"triangle_ratio_vs_{base_name}"] = float(row["triangles"]) / max(float(base["triangles"]), 1.0)


def _decision(rows: list[dict[str, Any]]) -> tuple[str, str]:
    by_label = {row["label"]: row for row in rows}
    clean = by_label["clean_origin_main_7000"]
    current = by_label["current_branch_7000"]
    candidates = [row for row in rows if row["label"].startswith("prune_")]
    strong = [
        row
        for row in candidates
        if row["triangles"] <= current["triangles"] * 0.70
        and row["render_psnr"] >= clean["render_psnr"]
        and row["render_ssim"] >= clean["render_ssim"]
        and row["render_lpips"] <= clean["render_lpips"]
        and row["geometry_depth_absrel"] <= clean["geometry_depth_absrel"] * 1.05
    ]
    if strong:
        best = min(strong, key=lambda row: row["triangles"])
        return "PASS", f"{best['label']} preserves clean-beating render quality with controlled geometry and reduced topology"
    useful = [
        row
        for row in candidates
        if row["triangles"] <= current["triangles"] * 0.70
        and row["render_psnr"] >= clean["render_psnr"]
        and row["render_ssim"] >= clean["render_ssim"]
        and row["render_lpips"] <= clean["render_lpips"]
    ]
    if useful:
        best = min(useful, key=lambda row: row["triangles"])
        return "SOFT PASS", f"{best['label']} preserves clean-beating render quality but geometry proxy regresses"
    return "FAIL", "no topology-control ablation preserves clean-beating render quality under the requested topology reduction"


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    rows = [
        _row(
            "clean_origin_main_7000",
            "clean_mesh_splatting",
            Path(args.clean_model),
            int(args.iteration),
        ),
        _row(
            "current_branch_7000",
            "current_branch_unpruned",
            Path(args.current_model),
            int(args.iteration),
        ),
        _row("prune_25", "area_prune_smallest_25pct", root / "prune_25/model", int(args.iteration)),
        _row("prune_50", "area_prune_smallest_50pct", root / "prune_50/model", int(args.iteration)),
        _row("prune_66", "area_prune_smallest_66pct", root / "prune_66/model", int(args.iteration)),
    ]
    _add_deltas(rows)
    gate, decision = _decision(rows)
    report = {
        "iteration": int(args.iteration),
        "gate": gate,
        "decision": decision,
        "rows": rows,
        "notes": [
            "This collector uses independent render metrics and COLMAP proxy geometry, not training-internal metrics.",
            "Area-prune ablations are checkpoint-copy post-processing results; they do not overwrite the source current-branch model.",
            "A positive single-scene result is a topology-control diagnostic, not multi-scene paper evidence.",
        ],
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "topology_control_ablation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0].keys())
    with (out / "topology_control_ablation.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "topology_control_ablation.md").open("w", encoding="utf-8") as f:
        f.write("# Stage21.5 Topology Control Ablation\n\n")
        f.write(f"Gate: `{gate}`\n\n")
        f.write(f"Decision: {decision}\n\n")
        f.write("| label | PSNR | SSIM | LPIPS | triangles | triangle ratio vs clean | depth AbsRel | W&B |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                "| {label} | {render_psnr:.6f} | {render_ssim:.6f} | {render_lpips:.6f} | {triangles} | {triangle_ratio_vs_clean:.3f} | {geometry_depth_absrel:.6f} | {wandb_url} |\n".format(
                    **row
                )
            )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Stage21.5 topology-control ablation metrics.")
    parser.add_argument("--root", default="outputs/carnet/meshprior/parking_phone_tiny/stage21_5_topology_control")
    parser.add_argument(
        "--clean_model",
        default="outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/origin_main_7000iter/model",
    )
    parser.add_argument(
        "--current_model",
        default="outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/current_branch_7000iter/model",
    )
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument(
        "--output_dir",
        default="outputs/carnet/meshprior/parking_phone_tiny/stage21_5_topology_control/comparison",
    )
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps({"gate": report["gate"], "decision": report["decision"], "rows": len(report["rows"])}, indent=2))


if __name__ == "__main__":
    main()
