#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import shlex


ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/peilincai/micromamba/envs/mesh_splatting/bin/python"


@dataclass(frozen=True)
class ParkingParetoJob:
    selector: str
    prune_fraction: float
    source_model: str = "outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_7000to22000/model"
    source_path: str = "outputs/carnet/meshprior/parking_phone_tiny/dataset_view"
    load_iteration: int = 22000
    final_iteration: int = 26000
    images: str = "images"
    resolution: int = 4

    @property
    def fraction_label(self) -> str:
        return f"prune{int(round(self.prune_fraction * 100))}"

    @property
    def run_name(self) -> str:
        return f"finalF7_parking_{self.selector}_{self.fraction_label}_{self.load_iteration}to{self.final_iteration}"

    @property
    def root(self) -> str:
        return f"outputs/carnet/meshsplatopt/final_stageF7_parking_pareto/{self.selector}/{self.fraction_label}"

    @property
    def compact_model(self) -> str:
        return f"{self.root}/compact_model"

    @property
    def recovery_model(self) -> str:
        return f"{self.root}/recovery_model"

    @property
    def contract_dir(self) -> str:
        return f"{self.root}/recovery_contract"

    def compaction_command(self) -> list[str]:
        return [
            PYTHON,
            "scripts/car_model/meshsplatopt_apply_compaction_to_checkpoint.py",
            "--source_model",
            self.source_model,
            "--iteration",
            str(self.load_iteration),
            "--output_model",
            self.compact_model,
            "--selector_mode",
            self.selector,
            "--target_prune_fraction",
            str(self.prune_fraction),
        ]

    def strict_contract_command(self) -> list[str]:
        return [
            PYTHON,
            "scripts/car_model/meshsplatopt_run_strict_compact_recovery.py",
            "--source_path",
            self.source_path,
            "--output_path",
            self.recovery_model,
            "--load_iteration",
            str(self.load_iteration),
            "--final_iteration",
            str(self.final_iteration),
            "--images",
            self.images,
            "--resolution",
            str(self.resolution),
            "--preset",
            "compact_render_only",
            "--wandb_group",
            "finalF7_parking_pareto",
            "--wandb_name",
            self.run_name,
            "--contract_out_dir",
            self.contract_dir,
            "--allow_missing_final",
        ]

    def train_command(self) -> list[str]:
        return [
            PYTHON,
            "train.py",
            "-s",
            self.source_path,
            "-m",
            self.recovery_model,
            "--images",
            self.images,
            "--resolution",
            str(self.resolution),
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
            "spcarnet_meshprior",
            "--wandb_group",
            "finalF7_parking_pareto",
            "--wandb_name",
            self.run_name,
            "--wandb_image_log_interval",
            "1000",
            "--wandb_scalar_log_interval",
            "50",
        ]

    def render_command(self) -> list[str]:
        return [
            PYTHON,
            "render.py",
            "-s",
            self.source_path,
            "-m",
            self.recovery_model,
            "--images",
            self.images,
            "--resolution",
            str(self.resolution),
            "--eval",
            "--iteration",
            str(self.final_iteration),
            "--skip_train",
        ]

    def metrics_command(self) -> list[str]:
        return [
            PYTHON,
            "scripts/car_model/meshsplatopt_eval_render_metrics_single_iteration.py",
            "-m",
            self.recovery_model,
            "--iteration",
            str(self.final_iteration),
        ]

    def geometry_command(self) -> list[str]:
        return [
            PYTHON,
            "evaluate_geometry_colmap.py",
            "-s",
            self.source_path,
            "-m",
            self.recovery_model,
            "--images",
            self.images,
            "--eval",
            "--iteration",
            str(self.final_iteration),
            "--max_points_per_view",
            "500",
            "--output",
            f"{self.recovery_model}/geometry_eval_colmap/iter_{self.final_iteration}_max500.json",
        ]


def _shell(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _payload(job: ParkingParetoJob) -> dict:
    out = asdict(job)
    out.update(
        {
            "run_name": job.run_name,
            "compact_model": job.compact_model,
            "recovery_model": job.recovery_model,
            "compaction_command": _shell(job.compaction_command()),
            "strict_contract_command": _shell(job.strict_contract_command()),
            "train_command": _shell(job.train_command()),
            "render_command": _shell(job.render_command()),
            "metrics_command": _shell(job.metrics_command()),
            "geometry_command": _shell(job.geometry_command()),
        }
    )
    return out


def default_jobs() -> list[ParkingParetoJob]:
    selectors = [
        "area_smallest",
        "csef_low_evidence_boundary_protected",
        "pareto_area_csef",
        "random_same_count",
    ]
    fractions = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]
    return [ParkingParetoJob(selector, fraction) for selector in selectors for fraction in fractions]


def main() -> int:
    parser = argparse.ArgumentParser(description="Write final F7 parking compact Pareto sweep manifest.")
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/final_stageF7_parking_pareto")
    parser.add_argument("--write-shell", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = default_jobs()
    payload = [_payload(job) for job in jobs]
    (out_dir / "parking_pareto_jobs.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.write_shell:
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", "export WANDB_PROJECT=spcarnet_meshprior", "export WANDB_MODE=online"]
        for job in jobs:
            lines.extend(
                [
                    "",
                    f"# {job.run_name}",
                    _shell(job.compaction_command()),
                    f"rsync -a {shlex.quote(job.compact_model)}/ {shlex.quote(job.recovery_model)}/",
                    _shell(job.strict_contract_command()),
                    _shell(job.train_command()),
                    _shell(job.render_command()),
                    _shell(job.metrics_command()),
                    _shell(job.geometry_command()),
                ]
            )
        script = out_dir / "run_parking_pareto_jobs.sh"
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script.chmod(0o755)
    print(f"Wrote {len(jobs)} parking Pareto jobs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
