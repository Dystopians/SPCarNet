#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FullBudgetJob:
    job_id: str
    scene: str
    method: str
    source_path: str
    model_path: str
    output_path: str
    load_iteration: int
    final_iteration: int
    prune_fraction: float | None = None
    wandb_project: str = "MeshSplatOpt-NeurIPS-Repair"
    wandb_group: str = "full_budget_sweep"
    notes: str = ""
    extra_train_args: list[str] = field(default_factory=list)

    def train_command(self) -> list[str]:
        cmd = [
            "python",
            "train.py",
            "-s",
            self.source_path,
            "-m",
            self.output_path,
            "--images",
            "images",
            "--resolution",
            "4",
            "--eval",
            "--load_iteration",
            str(self.load_iteration),
            "--iterations",
            str(self.final_iteration),
            "--test_iterations",
            str(self.final_iteration),
            "--save_iterations",
            str(self.final_iteration),
            "--checkpoint_iterations",
            str(self.final_iteration),
            "--densify_until_iter",
            str(self.load_iteration),
            "--skip_restricted_delaunay",
            "--freeze_topology_updates",
            "--enable_wandb",
            "--wandb_project",
            self.wandb_project,
            "--wandb_group",
            self.wandb_group,
            "--wandb_name",
            self.job_id,
            "--wandb_image_log_interval",
            "1000",
            "--wandb_scalar_log_interval",
            "50",
        ]
        return cmd + list(self.extra_train_args)

    def render_command(self) -> list[str]:
        return [
            "python",
            "render.py",
            "-s",
            self.source_path,
            "-m",
            self.output_path,
            "--images",
            "images",
            "--resolution",
            "4",
            "--eval",
            "--iteration",
            str(self.final_iteration),
            "--skip_train",
        ]

    def metrics_command(self) -> list[str]:
        return ["python", "metrics.py", "-m", self.output_path]

    def geometry_command(self) -> list[str]:
        return [
            "python",
            "evaluate_geometry_colmap.py",
            "-s",
            self.source_path,
            "-m",
            self.output_path,
            "--images",
            "images",
            "--eval",
            "--iteration",
            str(self.final_iteration),
            "--max_points_per_view",
            "500",
            "--output",
            f"{self.output_path}/geometry_eval_colmap/iter_{self.final_iteration}_max500.json",
        ]


def default_jobs() -> list[FullBudgetJob]:
    parking_source = "outputs/carnet/meshprior/parking_phone_tiny/dataset_view"
    return [
        FullBudgetJob(
            job_id="R53_full_parking_prune70_22000to26000",
            scene="parking_phone_tiny",
            method="clean_to_compact_prune70",
            source_path=parking_source,
            model_path="outputs/carnet/meshsplatopt/stageR53_clean22k_area_compaction/prune70/model",
            output_path="outputs/carnet/meshsplatopt/stageR53_01_prune70_clean_recovery_22000to26000/recovery_model",
            load_iteration=22000,
            final_iteration=26000,
            prune_fraction=0.70,
            notes="Validated headline row; rerun only when reproducing.",
        ),
        FullBudgetJob(
            job_id="R55_full_parking_prune65_22000to26000",
            scene="parking_phone_tiny",
            method="clean_to_compact_prune65",
            source_path=parking_source,
            model_path="outputs/carnet/meshsplatopt/stageR55_clean22k_area_compaction/prune65/model",
            output_path="outputs/carnet/meshsplatopt/stageR55_01_prune65_clean_recovery_22000to26000/recovery_model",
            load_iteration=22000,
            final_iteration=26000,
            prune_fraction=0.65,
            notes="Validated LPIPS/normal Pareto row.",
        ),
    ]


def _shell(cmd: list[str]) -> str:
    return " ".join(cmd)


def _job_payload(job: FullBudgetJob) -> dict[str, Any]:
    payload = asdict(job)
    payload["train_command"] = _shell(job.train_command())
    payload["render_command"] = _shell(job.render_command())
    payload["metrics_command"] = _shell(job.metrics_command())
    payload["geometry_command"] = _shell(job.geometry_command())
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Write or execute MeshSplatOpt full-budget sweep manifests.")
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/full_budget_sweep")
    parser.add_argument("--write-shell", action="store_true", help="Also write a runnable shell script.")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = default_jobs()
    payload = [_job_payload(job) for job in jobs]
    (out_dir / "full_budget_jobs.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.write_shell:
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", "export WANDB_MODE=online"]
        for job in jobs:
            lines.extend([
                "",
                f"# {job.job_id}: {job.notes}",
                _shell(job.train_command()),
                _shell(job.render_command()),
                _shell(job.metrics_command()),
                _shell(job.geometry_command()),
            ])
        script = out_dir / "run_full_budget_jobs.sh"
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script.chmod(0o755)
    print(f"Wrote {len(jobs)} full-budget job specs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

