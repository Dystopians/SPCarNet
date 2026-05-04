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
class SceneConfig:
    scene: str
    source_path: str
    images: str
    resolution: int
    clean_model: str
    clean_iteration: int
    final_iteration: int
    baseline_status: str
    stage35_model: str = ""
    sparse_only_model: str = ""


@dataclass(frozen=True)
class CrossSceneJob:
    scene: str
    selector: str
    prune_fraction: float
    source_path: str
    images: str
    resolution: int
    clean_model: str
    clean_iteration: int
    final_iteration: int
    baseline_status: str

    @property
    def fraction_label(self) -> str:
        return f"prune{int(round(self.prune_fraction * 100))}"

    @property
    def run_name(self) -> str:
        return f"finalF8_{self.scene}_{self.selector}_{self.fraction_label}_{self.clean_iteration}to{self.final_iteration}"

    @property
    def root(self) -> str:
        return f"outputs/carnet/meshsplatopt/final_stageF8_cross_scene_compact_pilot/{self.scene}/{self.selector}/{self.fraction_label}"

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
            self.clean_model,
            "--iteration",
            str(self.clean_iteration),
            "--output_model",
            self.compact_model,
            "--selector_mode",
            self.selector,
            "--target_prune_fraction",
            str(self.prune_fraction),
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
            str(self.clean_iteration),
            "--iterations",
            str(self.final_iteration),
            "--test_iterations",
            str(self.final_iteration),
            "--save_iterations",
            str(self.final_iteration),
            "--checkpoint_iterations",
            str(self.final_iteration),
            "--densify_until_iter",
            str(self.clean_iteration),
            "--skip_restricted_delaunay",
            "--freeze_topology_updates",
            "--enable_wandb",
            "--wandb_project",
            "spcarnet_meshprior",
            "--wandb_group",
            "finalF8_cross_scene_compact_pilot",
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

    def topology_command(self) -> list[str]:
        return [
            PYTHON,
            "scripts/car_model/meshsplatopt_run_strict_compact_recovery.py",
            "--source_path",
            self.source_path,
            "--output_path",
            self.recovery_model,
            "--load_iteration",
            str(self.clean_iteration),
            "--final_iteration",
            str(self.final_iteration),
            "--images",
            self.images,
            "--resolution",
            str(self.resolution),
            "--preset",
            "compact_render_only",
            "--wandb_group",
            "finalF8_cross_scene_compact_pilot",
            "--wandb_name",
            self.run_name,
            "--contract_out_dir",
            self.contract_dir,
        ]


def scene_configs() -> list[SceneConfig]:
    return [
        SceneConfig(
            scene="bonsai",
            source_path="/data/peilincai/mesh_datasets/mipnerf360/bonsai",
            images="images_4",
            resolution=4,
            clean_model="outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000",
            clean_iteration=22000,
            final_iteration=26000,
            baseline_status="CLEAN_LONG_RUNNING",
            stage35_model="outputs/carnet/meshprior/stage35_retained_refresh/mipnerf360_bonsai_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter_retry1/model",
            sparse_only_model="outputs/carnet/meshprior/stage26_cross_scene/mipnerf360_bonsai_baseline_sparse_depth_2000iter/model",
        ),
        SceneConfig(
            scene="courtyard",
            source_path="/data/peilincai/mesh_datasets/eth3d_colmap/courtyard",
            images="images",
            resolution=8,
            clean_model="outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000",
            clean_iteration=22000,
            final_iteration=26000,
            baseline_status="MISSING_BASELINE",
            stage35_model="outputs/carnet/meshprior/stage35_retained_refresh/eth3d_courtyard_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model",
            sparse_only_model="outputs/carnet/meshprior/stage26_cross_scene/eth3d_courtyard_baseline_sparse_depth_2000iter/model",
        ),
        SceneConfig(
            scene="room",
            source_path="/data/peilincai/mesh_datasets/tandt_db/tandt/training/room",
            images="images",
            resolution=4,
            clean_model="outputs/carnet/meshsplatopt/finalF3_room_clean_long_9000to22000",
            clean_iteration=22000,
            final_iteration=26000,
            baseline_status="MISSING_BASELINE",
        ),
        SceneConfig(
            scene="counter",
            source_path="/data/peilincai/mesh_datasets/tandt_db/tandt/training/counter",
            images="images",
            resolution=4,
            clean_model="outputs/carnet/meshsplatopt/finalF3_counter_clean_long_9000to22000",
            clean_iteration=22000,
            final_iteration=26000,
            baseline_status="MISSING_BASELINE",
        ),
    ]


def _shell(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _payload(job: CrossSceneJob) -> dict:
    payload = asdict(job)
    payload.update(
        {
            "run_name": job.run_name,
            "compact_model": job.compact_model,
            "recovery_model": job.recovery_model,
            "compaction_command": _shell(job.compaction_command()),
            "copy_compact_to_recovery_command": f"rsync -a {shlex.quote(job.compact_model)}/ {shlex.quote(job.recovery_model)}/",
            "train_command": _shell(job.train_command()),
            "render_command": _shell(job.render_command()),
            "metrics_command": _shell(job.metrics_command()),
            "geometry_command": _shell(job.geometry_command()),
            "topology_command": _shell(job.topology_command()),
        }
    )
    return payload


def build_jobs(selectors: list[str], fractions: list[float]) -> list[CrossSceneJob]:
    jobs: list[CrossSceneJob] = []
    for scene in scene_configs():
        for selector in selectors:
            for fraction in fractions:
                jobs.append(
                    CrossSceneJob(
                        scene=scene.scene,
                        selector=selector,
                        prune_fraction=fraction,
                        source_path=scene.source_path,
                        images=scene.images,
                        resolution=scene.resolution,
                        clean_model=scene.clean_model,
                        clean_iteration=scene.clean_iteration,
                        final_iteration=scene.final_iteration,
                        baseline_status=scene.baseline_status,
                    )
                )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description="Write final F8 cross-scene compact pilot manifest.")
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/final_stageF8_cross_scene_compact_pilot")
    parser.add_argument("--selectors", nargs="+", default=["csef_low_evidence_boundary_protected", "area_smallest"])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.50, 0.60, 0.70, 0.80])
    parser.add_argument("--write-shell", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [asdict(scene) for scene in scene_configs()]
    jobs = build_jobs(args.selectors, args.fractions)
    payload = [_payload(job) for job in jobs]
    (out_dir / "cross_scene_configs.json").write_text(json.dumps(configs, indent=2) + "\n", encoding="utf-8")
    (out_dir / "cross_scene_jobs.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.write_shell:
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", "export WANDB_PROJECT=spcarnet_meshprior", "export WANDB_MODE=online"]
        for job in jobs:
            lines.extend(
                [
                    "",
                    f"# {job.run_name}",
                    _shell(job.compaction_command()),
                    f"rsync -a {shlex.quote(job.compact_model)}/ {shlex.quote(job.recovery_model)}/",
                    _shell(job.train_command()),
                    _shell(job.render_command()),
                    _shell(job.metrics_command()),
                    _shell(job.geometry_command()),
                    _shell(job.topology_command()),
                ]
            )
        script = out_dir / "run_cross_scene_jobs.sh"
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script.chmod(0o755)
    print(f"Wrote {len(jobs)} F8 cross-scene jobs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
