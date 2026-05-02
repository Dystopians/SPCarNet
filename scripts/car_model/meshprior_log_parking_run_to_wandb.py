"""Log an externally produced parking MeshPrior run summary to wandb."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import wandb


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _state_counts(path: Path) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu")
    return int(state["_triangle_indices"].shape[0]), int(state["triangles_points"].shape[0])


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model_path)
    results_path = model_path / "results.json"
    metrics = _load_json(results_path).get(f"ours_{args.iteration}", {})
    state_path = model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud_state_dict.pt"
    triangles, vertices = _state_counts(state_path)
    payload = {
        "iteration": int(args.iteration),
        "render/SSIM": float(metrics.get("SSIM", float("nan"))),
        "render/PSNR": float(metrics.get("PSNR", float("nan"))),
        "render/LPIPS": float(metrics.get("LPIPS", float("nan"))),
        "mesh/triangles": triangles,
        "mesh/vertices": vertices,
        "baseline/is_paper_baseline_candidate": bool(args.paper_baseline_candidate),
    }
    run_obj = wandb.init(
        project=args.project,
        group=args.group,
        name=args.name,
        mode=args.mode,
        config={
            "model_path": str(model_path),
            "results_path": str(results_path),
            "state_path": str(state_path),
            "source": args.source,
            "note": args.note,
            "external_wandb_log": True,
        },
    )
    wandb.log(payload, step=int(args.iteration))
    artifact = wandb.Artifact(name=args.artifact_name, type="parking_run_summary")
    artifact.add_file(str(results_path))
    if (model_path / "cfg_args").is_file():
        artifact.add_file(str(model_path / "cfg_args"))
    run_obj.log_artifact(artifact)
    run_obj.summary.update(payload)
    run_obj.finish()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Log parking run metrics and summary artifacts to wandb.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--project", default="spcarnet_meshprior")
    parser.add_argument("--group", default="parking_baselines")
    parser.add_argument("--name", required=True)
    parser.add_argument("--artifact_name", required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--mode", default="online")
    parser.add_argument("--paper_baseline_candidate", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
