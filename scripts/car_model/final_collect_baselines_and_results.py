#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.evaluation_contracts import (  # noqa: E402
    MetricTargets,
    clean_json,
    compare_to_baseline,
    load_geometry_metrics,
    load_render_metrics,
)


@dataclass(frozen=True)
class RegistryRow:
    row_id: str
    scene: str
    method_label: str
    source_checkpoint: str
    training_start_iteration: int
    final_iteration: int
    triangle_count: int | None
    vertex_count: int | None
    wandb_url: str
    exact_command_path: str
    metric_source_path: str
    metric_source_type: str
    topology_frozen: bool
    sparse_depth_loss_enabled: bool
    sparse_sampling_mode: str = ""
    sparse_sampling_fraction: float | None = None
    sparse_lambda: float | None = None
    sparse_decay: str = ""
    edit_primitives_applied: bool = False
    edit_class: str = "none"
    prior_only_flag: bool = False
    decision: str = ""
    role: str = ""
    notes: str = ""
    manual_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalResult:
    scene: str
    method_label: str
    source_checkpoint: str
    training_start_iteration: int
    final_iteration: int
    triangle_count: int | None
    vertex_count: int | None
    independent_psnr: float
    independent_ssim: float
    independent_lpips: float
    sparse_abs_rel: float
    sparse_depth_mae: float
    sparse_normal_angle: float
    wandb_url: str
    exact_command_path: str
    metric_source_path: str
    metric_source_type: str
    topology_frozen: bool
    sparse_depth_loss_enabled: bool
    sparse_sampling_mode: str
    sparse_sampling_fraction: float | None
    sparse_lambda: float | None
    sparse_decay: str
    edit_primitives_applied: bool
    edit_class: str
    prior_only_flag: bool
    decision: str
    row_id: str
    role: str
    status: str
    checkpoint_triangle_count: int | None = None
    checkpoint_vertex_count: int | None = None
    integrity_warnings: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _manual_metrics(**kwargs: float) -> dict[str, float]:
    return dict(kwargs)


def _wandb(run_id: str) -> str:
    if not run_id:
        return ""
    return f"https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/{run_id}"


def _rows() -> list[RegistryRow]:
    report_log = "docs/car_model/SPCarNet_research_log.md"
    return [
        RegistryRow(
            "parking.clean7k",
            "parking_phone_tiny",
            "clean MeshSplatting 7k historical baseline",
            "outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/current_branch_7000iter/model",
            0,
            7000,
            833775,
            1071408,
            "",
            "docs/car_model/parking_best_clean_long_vs_method_long_report.md",
            "outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/current_branch_7000iter/model/results.json",
            "independent",
            False,
            False,
            decision="historical_weak_clean_not_headline",
            role="historical_clean_baseline",
        ),
        RegistryRow(
            "parking.clean22k",
            "parking_phone_tiny",
            "clean MeshSplatting long baseline",
            "outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_7000to22000/model",
            7000,
            22000,
            8548242,
            2286499,
            _wandb("uus7fi39"),
            "docs/car_model/parking_best_clean_long_vs_method_long_report.md",
            "outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_7000to22000/model/results.json",
            "independent",
            False,
            False,
            decision="strongest_clean_render_baseline",
            role="headline_clean_baseline",
        ),
        RegistryRow(
            "parking.clean30k",
            "parking_phone_tiny",
            "clean MeshSplatting 30k continuation",
            "outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_22000to30000/model",
            22000,
            30000,
            8548242,
            2286499,
            _wandb("2q807xuf"),
            "docs/car_model/parking_best_clean_long_vs_method_long_report.md",
            "outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_22000to30000/model/results.json",
            "independent",
            False,
            False,
            decision="clean_long_continuation_baseline",
            role="clean_long_baseline",
        ),
        RegistryRow(
            "R44.01",
            "parking_phone_tiny",
            "sparse-depth decay long-horizon recovery",
            "outputs/carnet/meshsplatopt/stageR44_01_parking_decay_sparse_frac0p50_lam0p001_16000to22000/recovery_model",
            16000,
            22000,
            782982,
            820107,
            _wandb("c1rxa6q6"),
            "docs/car_model/parking_best_clean_long_vs_method_long_report.md",
            "outputs/carnet/meshsplatopt/stageR44_01_parking_decay_sparse_frac0p50_lam0p001_16000to22000/recovery_model/results.json",
            "independent",
            True,
            True,
            "mixed_low_error",
            0.50,
            0.001,
            "16000->20000 to 0",
            False,
            "sparse_recovery",
            decision="render_losing_vs_clean22k_topology_normal_pareto",
            role="negative_control",
        ),
        RegistryRow(
            "R43.01b",
            "parking_phone_tiny",
            "low-lambda sparse-depth long continuation",
            "outputs/carnet/meshsplatopt/stageR43_01b_parking_mixed_frac0p50_lam0p001_16000to30000_long/recovery_model",
            16000,
            30000,
            782982,
            820107,
            _wandb("mhz6t8ps"),
            "docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md",
            "outputs/carnet/meshsplatopt/stageR43_01b_parking_mixed_frac0p50_lam0p001_16000to30000_long/recovery_model/results.json",
            "independent",
            True,
            True,
            "mixed_low_error",
            0.50,
            0.001,
            "",
            False,
            "sparse_recovery",
            decision="long_horizon_failure",
            role="negative_control",
        ),
        RegistryRow(
            "R48.01",
            "parking_phone_tiny",
            "clean-to-compact prune80 recovery",
            "outputs/carnet/meshsplatopt/stageR48_01_prune80_clean_recovery_22000to26000/recovery_model",
            22000,
            26000,
            1709648,
            1322214,
            _wandb("1n6jv232"),
            "docs/car_model/parking_clean_to_compact_repair_report.md",
            "outputs/carnet/meshsplatopt/stageR48_01_prune80_clean_recovery_22000to26000/recovery_model/results.json",
            "independent",
            True,
            False,
            edit_primitives_applied=True,
            edit_class="area_prune80",
            decision="compact_pareto_lpips_slightly_worse",
            role="compact_pareto",
        ),
        RegistryRow(
            "R50.01",
            "parking_phone_tiny",
            "R48 true fixed-topology continuation",
            "outputs/carnet/meshsplatopt/stageR50_01_prune80_true_fixed_topology_26000to30000/recovery_model",
            26000,
            30000,
            1709648,
            1322214,
            _wandb("zwafhpte"),
            "docs/car_model/parking_clean_to_compact_repair_report.md",
            "outputs/carnet/meshsplatopt/stageR50_01_prune80_true_fixed_topology_26000to30000/recovery_model/results.json",
            "independent",
            True,
            False,
            edit_primitives_applied=True,
            edit_class="area_prune80",
            decision="continuation_rejected",
            role="negative_control",
        ),
        RegistryRow(
            "R53.01",
            "parking_phone_tiny",
            "clean-to-compact prune70 recovery",
            "outputs/carnet/meshsplatopt/stageR53_01_prune70_clean_recovery_22000to26000/recovery_model",
            22000,
            26000,
            2564473,
            1661616,
            _wandb("q15qg2b8"),
            "docs/car_model/parking_clean_to_compact_repair_report.md",
            "outputs/carnet/meshsplatopt/stageR53_01_prune70_clean_recovery_22000to26000/recovery_model/results.json",
            "independent",
            True,
            False,
            edit_primitives_applied=True,
            edit_class="area_prune70",
            decision="headline_quality_dominating",
            role="headline_method",
        ),
        RegistryRow(
            "R55.01",
            "parking_phone_tiny",
            "clean-to-compact prune65 recovery",
            "outputs/carnet/meshsplatopt/stageR55_01_prune65_clean_recovery_22000to26000/recovery_model",
            22000,
            26000,
            2991885,
            1783669,
            _wandb("ja7t57cx"),
            "docs/car_model/parking_clean_to_compact_repair_report.md",
            "outputs/carnet/meshsplatopt/stageR55_01_prune65_clean_recovery_22000to26000/recovery_model/results.json",
            "independent",
            True,
            False,
            edit_primitives_applied=True,
            edit_class="area_prune65",
            decision="lpips_normal_pareto",
            role="pareto_method",
        ),
        RegistryRow(
            "R56.01",
            "parking_phone_tiny",
            "R53 true fixed-topology continuation",
            "outputs/carnet/meshsplatopt/stageR56_01_prune70_true_fixed_topology_26000to28000/recovery_model",
            26000,
            28000,
            2564473,
            1661616,
            _wandb("bwf2up51"),
            "docs/car_model/parking_clean_to_compact_repair_report.md",
            "docs/car_model/parking_clean_to_compact_repair_report.md",
            "training-time",
            True,
            False,
            edit_primitives_applied=True,
            edit_class="area_prune70",
            decision="continuation_rejected_training_eval_only",
            role="negative_control",
            manual_metrics=_manual_metrics(psnr=18.356278, ssim=0.623526, lpips=0.367352),
        ),
        RegistryRow(
            "R40.02",
            "courtyard",
            "low-lambda sparse-depth recovery",
            "outputs/carnet/meshsplatopt/stageR40_02_courtyard_mixed_frac0p625_lam0p002_2000to7000/recovery_model",
            2000,
            7000,
            410254,
            444301,
            _wandb("coqls9rm"),
            "docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md",
            "outputs/carnet/meshsplatopt/stageR40_02_courtyard_mixed_frac0p625_lam0p002_2000to7000/recovery_model/results.json",
            "independent",
            True,
            True,
            "mixed_low_error",
            0.625,
            0.002,
            "",
            False,
            "sparse_recovery",
            decision="courtyard_all_metric_sparse_recovery",
            role="sparse_recovery_pareto",
        ),
        RegistryRow(
            "R43.02b",
            "courtyard",
            "courtyard low-lambda long continuation",
            "outputs/carnet/meshsplatopt/stageR43_02b_courtyard_mixed_frac0p625_lam0p002_7000to20000_long/recovery_model",
            7000,
            20000,
            410254,
            444301,
            _wandb("cla3utia"),
            report_log,
            "outputs/carnet/meshsplatopt/stageR43_02b_courtyard_mixed_frac0p625_lam0p002_7000to20000_long/recovery_model/results.json",
            "independent",
            True,
            True,
            "mixed_low_error",
            0.625,
            0.002,
            "",
            False,
            "sparse_recovery",
            decision="render_best_depth_tradeoff",
            role="sparse_recovery_pareto",
        ),
        RegistryRow(
            "R44.02",
            "courtyard",
            "courtyard sparse-depth decay long recovery",
            "outputs/carnet/meshsplatopt/stageR44_02_courtyard_decay_sparse_frac0p625_lam0p002_7000to20000/recovery_model",
            7000,
            20000,
            410254,
            444301,
            _wandb("5tleod3c"),
            report_log,
            "outputs/carnet/meshsplatopt/stageR44_02_courtyard_decay_sparse_frac0p625_lam0p002_7000to20000/recovery_model/results.json",
            "independent",
            True,
            True,
            "mixed_low_error",
            0.625,
            0.002,
            "7000->14000 to 0.25",
            False,
            "sparse_recovery",
            decision="courtyard_decay_long_partial_pass",
            role="sparse_recovery_pareto",
        ),
        RegistryRow(
            "R31.03",
            "bonsai",
            "bonsai random sparse-depth recovery",
            "outputs/carnet/meshsplatopt/stageR31_03_bonsai_stage35_sparse_depth_lam0p005_2000to7000/recovery_model",
            2000,
            7000,
            2487474,
            2478890,
            _wandb("3wygm9u4"),
            "docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md",
            "outputs/carnet/meshsplatopt/stageR31_03_bonsai_stage35_sparse_depth_lam0p005_2000to7000/recovery_model/results.json",
            "independent",
            True,
            True,
            "random",
            None,
            0.005,
            "",
            False,
            "sparse_recovery",
            decision="bonsai_sparse_recovery_baseline",
            role="sparse_recovery_baseline",
        ),
        RegistryRow(
            "R41.01",
            "bonsai",
            "bonsai low-lambda render breakthrough",
            "outputs/carnet/meshsplatopt/stageR41_01_bonsai_mixed_frac0p50_lam0p002_2000to7000/recovery_model",
            2000,
            7000,
            2487474,
            2478890,
            _wandb("poh8k4be"),
            report_log,
            "outputs/carnet/meshsplatopt/stageR41_01_bonsai_mixed_frac0p50_lam0p002_2000to7000/recovery_model/results.json",
            "independent",
            True,
            True,
            "mixed_low_error",
            0.50,
            0.002,
            "",
            False,
            "sparse_recovery",
            decision="render_breakthrough_geometry_tradeoff",
            role="sparse_recovery_pareto",
        ),
        RegistryRow(
            "R42.01",
            "bonsai",
            "bonsai fraction repair check",
            "outputs/carnet/meshsplatopt/stageR42_01_bonsai_mixed_frac0p625_lam0p002_2000to7000/recovery_model",
            2000,
            7000,
            2487474,
            2478890,
            _wandb("l2inxutg"),
            report_log,
            "outputs/carnet/meshsplatopt/stageR42_01_bonsai_mixed_frac0p625_lam0p002_2000to7000/recovery_model/results.json",
            "independent",
            True,
            True,
            "mixed_low_error",
            0.625,
            0.002,
            "",
            False,
            "sparse_recovery",
            decision="fraction_repair_boundary",
            role="negative_control",
        ),
        RegistryRow(
            "Stage35.courtyard",
            "courtyard",
            "Stage35 PRISM retained relaxed baseline",
            "outputs/carnet/meshprior/stage35_retained_refresh/eth3d_courtyard_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model",
            0,
            2000,
            410254,
            444301,
            "",
            "docs/car_model/meshprior_stage35_retained_refresh_report.md",
            "outputs/carnet/meshprior/stage35_retained_refresh/eth3d_courtyard_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model/results.json",
            "independent",
            False,
            False,
            decision="stage35_public_baseline",
            role="stage35_baseline",
        ),
        RegistryRow(
            "Stage35.bonsai",
            "bonsai",
            "Stage35 PRISM retained relaxed baseline",
            "outputs/carnet/meshprior/stage35_retained_refresh/mipnerf360_bonsai_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter_retry1/model",
            0,
            2000,
            2487474,
            2478890,
            "",
            "docs/car_model/meshprior_stage35_retained_refresh_report.md",
            "outputs/carnet/meshprior/stage35_retained_refresh/mipnerf360_bonsai_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter_retry1/model/results.json",
            "independent",
            False,
            False,
            decision="stage35_public_baseline",
            role="stage35_baseline",
        ),
        RegistryRow(
            "R15.01",
            "courtyard",
            "freeze-densify medium baseline",
            "outputs/carnet/meshsplatopt/stageR15_01_courtyard_baseline_freeze_densify_skip_delaunay_2000to4000/recovery_model",
            2000,
            4000,
            410254,
            444301,
            _wandb("cvf6t7do"),
            "docs/car_model/meshsplatopt_stageR15_01_04_multiscene_freeze_medium_report.md",
            "outputs/carnet/meshsplatopt/stageR15_01_courtyard_baseline_freeze_densify_skip_delaunay_2000to4000/recovery_model/results.json",
            "independent",
            True,
            False,
            decision="medium_freeze_schedule_pass",
            role="medium_schedule",
        ),
        RegistryRow(
            "R15.02",
            "courtyard",
            "snap freeze-densify medium",
            "outputs/carnet/meshsplatopt/stageR15_02_courtyard_snap_freeze_densify_skip_delaunay_2000to4000/recovery_model",
            2000,
            4000,
            410254,
            444301,
            _wandb("d3h2ruj3"),
            "docs/car_model/meshsplatopt_stageR15_01_04_multiscene_freeze_medium_report.md",
            "outputs/carnet/meshsplatopt/stageR15_02_courtyard_snap_freeze_densify_skip_delaunay_2000to4000/recovery_model/results.json",
            "independent",
            True,
            False,
            edit_primitives_applied=True,
            edit_class="snap",
            decision="snap_selector_weak",
            role="snap_negative_control",
        ),
        RegistryRow(
            "R15.03",
            "parking_phone_tiny",
            "freeze-densify medium baseline",
            "outputs/carnet/meshsplatopt/stageR15_03_parking_baseline_freeze_densify_skip_delaunay_2000to4000/recovery_model",
            2000,
            4000,
            782982,
            820107,
            _wandb("evj36lvp"),
            "docs/car_model/meshsplatopt_stageR15_01_04_multiscene_freeze_medium_report.md",
            "outputs/carnet/meshsplatopt/stageR15_03_parking_baseline_freeze_densify_skip_delaunay_2000to4000/recovery_model/results.json",
            "independent",
            True,
            False,
            decision="medium_freeze_schedule_pass",
            role="medium_schedule",
        ),
        RegistryRow(
            "R15.04",
            "parking_phone_tiny",
            "snap freeze-densify medium",
            "outputs/carnet/meshsplatopt/stageR15_04_parking_snap_freeze_densify_skip_delaunay_2000to4000/recovery_model",
            2000,
            4000,
            782982,
            820107,
            _wandb("3r7inkj0"),
            "docs/car_model/meshsplatopt_stageR15_01_04_multiscene_freeze_medium_report.md",
            "outputs/carnet/meshsplatopt/stageR15_04_parking_snap_freeze_densify_skip_delaunay_2000to4000/recovery_model/results.json",
            "independent",
            True,
            False,
            edit_primitives_applied=True,
            edit_class="snap",
            decision="snap_selector_weak",
            role="snap_negative_control",
        ),
        RegistryRow(
            "R57.clean9k",
            "courtyard",
            "matched clean 7k-to-9k continuation",
            "outputs/carnet/meshsplatopt/stageR57_02_courtyard_clean_continue_7000to9000/recovery_model",
            7000,
            9000,
            410254,
            444301,
            _wandb("ucqyn1ym"),
            "docs/car_model/meshsplatopt_stageR57_R58_cross_scene_matched_report.md",
            "outputs/carnet/meshsplatopt/stageR57_02_courtyard_clean_continue_7000to9000/recovery_model/results.json",
            "independent",
            True,
            False,
            decision="matched_public_clean",
            role="matched_clean_baseline",
        ),
        RegistryRow(
            "R57.compact70",
            "courtyard",
            "matched prune70 compact recovery",
            "outputs/carnet/meshsplatopt/stageR57_01_courtyard_prune70_recovery_7000to9000/recovery_model",
            7000,
            9000,
            123076,
            190787,
            _wandb("kgazucjj"),
            "docs/car_model/meshsplatopt_stageR57_R58_cross_scene_matched_report.md",
            "outputs/carnet/meshsplatopt/stageR57_01_courtyard_prune70_recovery_7000to9000/recovery_model/results.json",
            "independent",
            True,
            False,
            edit_primitives_applied=True,
            edit_class="area_prune70",
            decision="public_scene_negative",
            role="negative_control",
        ),
        RegistryRow(
            "R58.clean9k",
            "bonsai",
            "matched clean 7k-to-9k continuation",
            "outputs/carnet/meshsplatopt/stageR58_02_bonsai_clean_continue_7000to9000/recovery_model",
            7000,
            9000,
            2487474,
            2478890,
            _wandb("ulv6dpku"),
            "docs/car_model/meshsplatopt_stageR57_R58_cross_scene_matched_report.md",
            "outputs/carnet/meshsplatopt/stageR58_02_bonsai_clean_continue_7000to9000/recovery_model/results.json",
            "independent",
            True,
            False,
            decision="matched_public_clean",
            role="matched_clean_baseline",
        ),
        RegistryRow(
            "R58.compact70",
            "bonsai",
            "matched prune70 compact recovery",
            "outputs/carnet/meshsplatopt/stageR58_01_bonsai_prune70_recovery_7000to9000/recovery_model",
            7000,
            9000,
            746242,
            720177,
            _wandb("82v2cg9z"),
            "docs/car_model/meshsplatopt_stageR57_R58_cross_scene_matched_report.md",
            "outputs/carnet/meshsplatopt/stageR58_01_bonsai_prune70_recovery_7000to9000/recovery_model/results.json",
            "independent",
            True,
            False,
            edit_primitives_applied=True,
            edit_class="area_prune70",
            decision="public_scene_all_metric_pass",
            role="public_method",
        ),
        RegistryRow(
            "R59.clean9k",
            "room",
            "matched clean 7k-to-9k continuation",
            "outputs/carnet/meshsplatopt/stageR59_03_room_clean_continue_7000to9000",
            7000,
            9000,
            196057,
            274368,
            _wandb("mfhilc6u"),
            "docs/car_model/final_stageF0_current_state_audit.md",
            "outputs/carnet/meshsplatopt/stageR59_03_room_clean_continue_7000to9000/results.json",
            "independent",
            True,
            False,
            decision="matched_room_clean",
            role="matched_clean_baseline",
        ),
        RegistryRow(
            "R59.compact70",
            "room",
            "matched prune70 compact recovery",
            "outputs/carnet/meshsplatopt/stageR59_02_room_prune70_recovery_7000to9000",
            7000,
            9000,
            58817,
            112733,
            _wandb("pappuah6"),
            "docs/car_model/final_stageF0_current_state_audit.md",
            "outputs/carnet/meshsplatopt/stageR59_02_room_prune70_recovery_7000to9000/results.json",
            "independent",
            True,
            False,
            edit_primitives_applied=True,
            edit_class="area_prune70",
            decision="render_positive_geometry_negative",
            role="public_pareto_mixed",
        ),
        RegistryRow(
            "R60.clean9k",
            "counter",
            "matched clean 7k-to-9k continuation",
            "outputs/carnet/meshsplatopt/stageR60_03_counter_clean_continue_7000to9000",
            7000,
            9000,
            161465,
            228601,
            _wandb("gq6iknaa"),
            "docs/car_model/final_stageF0_current_state_audit.md",
            "outputs/carnet/meshsplatopt/stageR60_03_counter_clean_continue_7000to9000/results.json",
            "independent",
            True,
            False,
            decision="matched_counter_clean",
            role="matched_clean_baseline",
        ),
        RegistryRow(
            "R60.compact70",
            "counter",
            "matched prune70 compact recovery",
            "outputs/carnet/meshsplatopt/stageR60_02_counter_prune70_recovery_7000to9000",
            7000,
            9000,
            48440,
            88447,
            _wandb("ks5cnaop"),
            "docs/car_model/final_stageF0_current_state_audit.md",
            "outputs/carnet/meshsplatopt/stageR60_02_counter_prune70_recovery_7000to9000/results.json",
            "independent",
            True,
            False,
            edit_primitives_applied=True,
            edit_class="area_prune70",
            decision="mixed_negative_public_scene",
            role="negative_control",
        ),
    ]


def _number(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _checkpoint_path(model_path: Path, iteration: int) -> Path:
    direct = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    if direct.is_file():
        return direct
    return model_path / "model" / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"


def _checkpoint_topology(model_path: Path, iteration: int) -> tuple[int | None, int | None, str | None]:
    checkpoint = _checkpoint_path(model_path, iteration)
    if not checkpoint.is_file():
        return None, None, f"missing_checkpoint:{checkpoint}"
    try:
        import torch

        state = torch.load(checkpoint, map_location="cpu")
        triangles = state.get("_triangle_indices")
        vertices = state.get("triangles_points")
        tri_count = int(triangles.shape[0]) if hasattr(triangles, "shape") else None
        vert_count = int(vertices.shape[0]) if hasattr(vertices, "shape") else None
        return tri_count, vert_count, None
    except Exception as exc:  # pragma: no cover - defensive for unusual checkpoint encodings
        return None, None, f"checkpoint_read_error:{type(exc).__name__}:{exc}"


def _load_metrics(row: RegistryRow) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    if row.manual_metrics:
        metrics = {
            "psnr": _number(row.manual_metrics.get("psnr")),
            "ssim": _number(row.manual_metrics.get("ssim")),
            "lpips": _number(row.manual_metrics.get("lpips")),
            "abs_rel": _number(row.manual_metrics.get("abs_rel")),
            "depth_mae": _number(row.manual_metrics.get("depth_mae")),
            "normal_mean_ang_deg": _number(row.manual_metrics.get("normal_mean_ang_deg")),
        }
        warnings.append("manual_or_training_metrics")
        return metrics, warnings

    model_path = ROOT / row.source_checkpoint
    render = load_render_metrics(model_path, row.final_iteration)
    geom = load_geometry_metrics(model_path, row.final_iteration)
    metrics = {**render, **geom}
    for key, value in metrics.items():
        if math.isnan(value):
            warnings.append(f"missing_{key}")
    return metrics, warnings


def _materialize(row: RegistryRow, check_topology: bool) -> FinalResult:
    metrics, warnings = _load_metrics(row)
    ckpt_triangles: int | None = None
    ckpt_vertices: int | None = None
    if check_topology and row.metric_source_type == "independent":
        ckpt_triangles, ckpt_vertices, ckpt_warning = _checkpoint_topology(ROOT / row.source_checkpoint, row.final_iteration)
        if ckpt_warning:
            warnings.append(ckpt_warning)
        if ckpt_triangles is not None and row.triangle_count is not None and ckpt_triangles != row.triangle_count:
            warnings.append(f"triangle_mismatch_reported_{row.triangle_count}_checkpoint_{ckpt_triangles}")
        if ckpt_vertices is not None and row.vertex_count is not None and ckpt_vertices != row.vertex_count:
            warnings.append(f"vertex_mismatch_reported_{row.vertex_count}_checkpoint_{ckpt_vertices}")
    if row.metric_source_type != "independent":
        warnings.append(f"non_independent_metric_source:{row.metric_source_type}")
    if row.training_start_iteration != row.final_iteration and not row.wandb_url:
        warnings.append("missing_wandb_url_for_training_run")
    if row.prior_only_flag:
        warnings.append("prior_only_diagnostic")
    status = "OK" if not warnings else "WARN"
    return FinalResult(
        scene=row.scene,
        method_label=row.method_label,
        source_checkpoint=row.source_checkpoint,
        training_start_iteration=row.training_start_iteration,
        final_iteration=row.final_iteration,
        triangle_count=row.triangle_count,
        vertex_count=row.vertex_count,
        independent_psnr=metrics["psnr"],
        independent_ssim=metrics["ssim"],
        independent_lpips=metrics["lpips"],
        sparse_abs_rel=metrics["abs_rel"],
        sparse_depth_mae=metrics["depth_mae"],
        sparse_normal_angle=metrics["normal_mean_ang_deg"],
        wandb_url=row.wandb_url,
        exact_command_path=row.exact_command_path,
        metric_source_path=row.metric_source_path,
        metric_source_type=row.metric_source_type,
        topology_frozen=row.topology_frozen,
        sparse_depth_loss_enabled=row.sparse_depth_loss_enabled,
        sparse_sampling_mode=row.sparse_sampling_mode,
        sparse_sampling_fraction=row.sparse_sampling_fraction,
        sparse_lambda=row.sparse_lambda,
        sparse_decay=row.sparse_decay,
        edit_primitives_applied=row.edit_primitives_applied,
        edit_class=row.edit_class,
        prior_only_flag=row.prior_only_flag,
        decision=row.decision,
        row_id=row.row_id,
        role=row.role,
        status=status,
        checkpoint_triangle_count=ckpt_triangles,
        checkpoint_vertex_count=ckpt_vertices,
        integrity_warnings=";".join(warnings),
        notes=row.notes,
    )


def _method_result_like(row: FinalResult):
    from ss3dm_prior.meshsplatopt.evaluation_contracts import MethodResult

    return MethodResult(
        row_id=row.row_id,
        scene=row.scene,
        method=row.method_label,
        model_path=row.source_checkpoint,
        iteration=row.final_iteration,
        role=row.role,
        wandb_run=row.wandb_url,
        psnr=row.independent_psnr,
        ssim=row.independent_ssim,
        lpips=row.independent_lpips,
        abs_rel=row.sparse_abs_rel,
        depth_mae=row.sparse_depth_mae,
        normal_mean_ang_deg=row.sparse_normal_angle,
        triangles=row.triangle_count,
        vertices=row.vertex_count,
        status=row.status,
        notes=row.notes,
    )


def _comparison_payload(rows: list[FinalResult]) -> dict[str, Any]:
    by_id = {row.row_id: row for row in rows}
    targets = MetricTargets(triangle_reduction_min=0.5)
    pairs = [
        ("R44.01", "parking.clean22k"),
        ("R48.01", "parking.clean22k"),
        ("R53.01", "parking.clean22k"),
        ("R55.01", "parking.clean22k"),
        ("R58.compact70", "R58.clean9k"),
        ("R57.compact70", "R57.clean9k"),
        ("R59.compact70", "R59.clean9k"),
        ("R60.compact70", "R60.clean9k"),
    ]
    comparisons = []
    for candidate_id, baseline_id in pairs:
        if candidate_id not in by_id or baseline_id not in by_id:
            continue
        candidate = _method_result_like(by_id[candidate_id])
        baseline = _method_result_like(by_id[baseline_id])
        comparisons.append(compare_to_baseline(candidate, baseline, targets).to_dict())

    integrity = {
        "r53_vs_clean22k_reproduced": any(
            item["candidate_id"] == "R53.01" and item["baseline_id"] == "parking.clean22k" and item["pass_all_targets"]
            for item in comparisons
        ),
        "r44_flagged_render_losing_vs_clean22k": any(
            item["candidate_id"] == "R44.01"
            and item["baseline_id"] == "parking.clean22k"
            and {"psnr", "ssim", "lpips"}.issubset(set(item["failed_targets"]))
            for item in comparisons
        ),
        "forbidden_long_method_vs_clean7k_headline": False,
    }
    for item in comparisons:
        if item["candidate_id"].startswith("R") and item["baseline_id"] == "parking.clean7k":
            integrity["forbidden_long_method_vs_clean7k_headline"] = True
    return {"comparisons": comparisons, "integrity": integrity}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def _write_md(path: Path, rows: list[FinalResult], comparisons: list[dict[str, Any]], integrity: dict[str, Any]) -> None:
    lines = [
        "# Final Baseline Registry",
        "",
        "All headline comparisons use independent metrics unless a row is explicitly flagged otherwise.",
        "",
        "## Integrity Summary",
        "",
    ]
    for key, value in integrity.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| row | scene | method | iter | metric source | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal | triangles | W&B | decision | warnings |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.row_id} | {row.scene} | {row.method_label} | {row.final_iteration} | {row.metric_source_type} | "
            f"{_fmt(row.independent_psnr)} | {_fmt(row.independent_ssim)} | {_fmt(row.independent_lpips)} | "
            f"{_fmt(row.sparse_abs_rel)} | {_fmt(row.sparse_depth_mae)} | {_fmt(row.sparse_normal_angle)} | "
            f"{_fmt(row.triangle_count)} | {row.wandb_url or ''} | {row.decision} | {row.integrity_warnings} |"
        )
    lines.extend(
        [
            "",
            "## Key Comparisons",
            "",
            "| candidate | baseline | pass | failed targets | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepthMAE | dNormal | triangle reduction |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in comparisons:
        delta = item["deltas"]
        lines.append(
            f"| {item['candidate_id']} | {item['baseline_id']} | {item['pass_all_targets']} | "
            f"{','.join(item['failed_targets'])} | {_fmt(delta['psnr'])} | {_fmt(delta['ssim'])} | "
            f"{_fmt(delta['lpips'])} | {_fmt(delta['abs_rel'])} | {_fmt(delta['depth_mae'])} | "
            f"{_fmt(delta['normal_mean_ang_deg'])} | {_fmt(delta['triangle_reduction'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect final baseline registry and integrity checks.")
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/final_baseline_registry")
    parser.add_argument("--skip-topology-check", action="store_true")
    args = parser.parse_args()

    rows = [_materialize(row, check_topology=not args.skip_topology_check) for row in _rows()]
    payload_rows = [row.to_dict() for row in rows]
    comparison_payload = _comparison_payload(rows)
    payload = {"rows": payload_rows, **comparison_payload}

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_results.json").write_text(json.dumps(clean_json(payload), indent=2) + "\n", encoding="utf-8")
    _write_csv(out_dir / "final_results.csv", [clean_json(row) for row in payload_rows])
    _write_md(out_dir / "final_results.md", rows, comparison_payload["comparisons"], comparison_payload["integrity"])

    print(f"Wrote {len(rows)} rows to {out_dir}")
    print(json.dumps(comparison_payload["integrity"], indent=2))
    required_pass = (
        comparison_payload["integrity"]["r53_vs_clean22k_reproduced"]
        and comparison_payload["integrity"]["r44_flagged_render_losing_vs_clean22k"]
        and not comparison_payload["integrity"]["forbidden_long_method_vs_clean7k_headline"]
    )
    return 0 if required_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
